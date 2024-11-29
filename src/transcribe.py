import os
import sys
import torch
import intel_extension_for_pytorch as ipex
import whisper
from whisper.audio import SAMPLE_RATE
import pykakasi
import re
from datetime import timedelta
from pathlib import Path
import argparse
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
import time
from langdetect import detect
import numpy as np
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Lock
import json
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass, asdict
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('transcription.log')
    ]
)

@dataclass
class TranscriptionConfig:
    model_size: str = "base"
    batch_size: int = 5
    max_workers: int = 4
    device: str = "auto"
    chunk_length: int = 30
    use_translation: bool = True
    output_format: str = "txt"
    language: Optional[str] = None

class Timer:
    """Context manager for timing operations"""
    def __init__(self, label: str):
        self.label = label
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        duration = time.perf_counter() - self.start_time
        logging.info(f"{self.label}: {duration:.2f} seconds")

class AudioTranscriber:
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.setup_device()
        self.kakasi = pykakasi.kakasi()
        self.load_models()

    def setup_device(self):
        """Set up the processing device based on availability"""
        if self.config.device == "auto":
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                self.device = torch.device("xpu")
                logging.info(f"Using Intel XPU: {torch.xpu.get_device_properties(0)}")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                logging.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                logging.info("Using CPU")
        else:
            self.device = torch.device(self.config.device)
            logging.info(f"Using specified device: {self.config.device}")

    def load_models(self):
        """Load and optimize the transcription and translation models"""
        with Timer("Loading Models"):
            # Load Whisper model
            self.model = whisper.load_model(self.config.model_size)
            self.model = self.model.to(self.device)
            
            # Set model to eval mode for inference
            self.model.eval()
                
            if self.device.type != "cpu":
                # Optimize for inference
                self.model = ipex.optimize(
                    self.model.eval(),  # Ensure model is in eval mode
                    dtype=torch.float32,
                    inplace=True,
                    auto_kernel_selection=True,
                    weights_prepack=True
                )

            # Load translation model if needed
            if self.config.use_translation:
                self.translation_model_name = 'Helsinki-NLP/opus-mt-ja-en'
                self.tokenizer = MarianTokenizer.from_pretrained(self.translation_model_name)
                self.translation_model = MarianMTModel.from_pretrained(self.translation_model_name)
                self.translation_model = self.translation_model.to(self.device)
                self.translation_model.eval()  # Set to eval mode

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Convert seconds to formatted timestamp"""
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        seconds = int(td.total_seconds() % 60)
        milliseconds = int((td.total_seconds() * 1000) % 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    def load_audio(self, file_path: str) -> np.ndarray:
        """Load audio file using FFmpeg"""
        try:
            cmd = [
                "ffmpeg",
                "-i", file_path,
                "-f", "s16le",
                "-acodec", "pcm_s16le",
                "-ac", "1",
                "-ar", str(SAMPLE_RATE),
                "-"
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            output, error = process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {error.decode()}")
                
            audio = np.frombuffer(output, np.int16).astype(np.float32) / 32768.0
            return audio
            
        except Exception as e:
            logging.error(f"Error loading audio: {e}")
            raise

    def process_audio_chunk(self, audio_chunk: np.ndarray, start_time: float) -> Dict:
        """Process a single audio chunk"""
        with torch.no_grad():
            result = self.model.transcribe(
                audio_chunk,
                language=self.config.language,
                task='transcribe',
                fp16=(self.device.type != 'cpu')
            )
            
            # Adjust timestamps
            for segment in result["segments"]:
                segment["start"] += start_time
                segment["end"] += start_time
                
            return result

    def translate_text(self, text: str) -> str:
        """Translate text using MarianMT"""
        try:
            inputs = self.tokenizer([text], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.translation_model.generate(**inputs)
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translation
        except Exception as e:
            logging.error(f"Translation error: {e}")
            return "(Translation Failed)"

    def transcribe(self, audio_path: str) -> Dict:
        """Main transcription function"""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logging.info(f"Processing: {audio_path.name}")
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        logging.info(f"File size: {file_size_mb:.2f}MB")

        try:
            # Load audio
            with Timer("Loading Audio"):
                audio = self.load_audio(str(audio_path))

            # Process in chunks
            audio_length = len(audio) / SAMPLE_RATE
            chunk_size = int(SAMPLE_RATE * self.config.chunk_length)
            chunks = range(0, len(audio), chunk_size)

            all_segments = []
            with tqdm(total=len(chunks), desc="Transcribing") as pbar:
                for i, chunk_start in enumerate(chunks):
                    chunk_end = min(chunk_start + chunk_size, len(audio))
                    audio_chunk = audio[chunk_start:chunk_end]
                    start_time = chunk_start / SAMPLE_RATE

                    result = self.process_audio_chunk(audio_chunk, start_time)
                    all_segments.extend(result["segments"])
                    pbar.update(1)

            # Combine results
            final_result = {
                "segments": all_segments,
                "language": self.config.language or result.get("language", "unknown")
            }

            return final_result

        except Exception as e:
            logging.error(f"Transcription error: {e}")
            traceback.print_exc()
            raise

    def save_transcript(self, result: Dict, output_file: str):
        """Save transcription results to file"""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=== Transcription ===\n\n")
            f.write(f"Language: {result['language'].upper()}\n\n")

            for segment in result["segments"]:
                start = self.format_timestamp(segment["start"])
                end = self.format_timestamp(segment["end"])
                text = segment["text"].strip()

                f.write(f"[{start} --> {end}]\n")
                f.write(f"Text: {text}\n")

                if result["language"] == "ja":
                    # Add romanization
                    romaji = " ".join(
                        item['hepburn'] for item in self.kakasi.convert(text)
                    )
                    f.write(f"Romaji: {romaji}\n")

                    # Add translation if enabled
                    if self.config.use_translation:
                        translation = self.translate_text(text)
                        f.write(f"Translation: {translation}\n")

                f.write("\n")

def main():
    parser = argparse.ArgumentParser(description="Advanced Audio Transcription System")
    parser.add_argument("--input", type=str, required=True, help="Input audio file or directory")
    parser.add_argument("--output", type=str, help="Output directory (default: ./transcripts)")
    parser.add_argument("--model", type=str, default="base", help="Whisper model size")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cpu/cuda/xpu)")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for directory processing")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--no-translation", action="store_true", help="Disable translation")
    parser.add_argument("--language", type=str, help="Force specific language")
    args = parser.parse_args()

    # Create configuration
    config = TranscriptionConfig(
        model_size=args.model,
        batch_size=args.batch_size,
        max_workers=args.workers,
        device=args.device,
        use_translation=not args.no_translation,
        language=args.language
    )

    # Initialize transcriber
    transcriber = AudioTranscriber(config)

    # Process input
    input_path = Path(args.input)
    output_dir = Path(args.output or "./transcripts")

    if input_path.is_file():
        # Process single file
        output_file = output_dir / f"{input_path.stem}_transcript.txt"
        result = transcriber.transcribe(str(input_path))
        transcriber.save_transcript(result, output_file)
        logging.info(f"Transcription saved to: {output_file}")
    else:
        # Process directory
        for audio_file in input_path.glob("**/*"):
            if audio_file.suffix.lower() in {'.mp3', '.wav', '.m4a', '.ogg'}:
                try:
                    rel_path = audio_file.relative_to(input_path)
                    output_file = output_dir / rel_path.with_suffix('.txt')
                    result = transcriber.transcribe(str(audio_file))
                    transcriber.save_transcript(result, output_file)
                    logging.info(f"Processed: {audio_file.name}")
                except Exception as e:
                    logging.error(f"Failed to process {audio_file.name}: {e}")

if __name__ == "__main__":
    main()