import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Lock
import json
import logging
from typing import List, Dict
from transcribe import AudioTranscriber, TranscriptionConfig
from tqdm import tqdm
import traceback

class BatchProcessor:
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.transcriber = AudioTranscriber(config)
        self.print_lock = Lock()
        self.results_lock = Lock()
        self.progress_queue = Queue()
        self.processed_files: Dict[str, Dict] = {}

    def process_file(self, audio_file: Path, output_file: Path) -> Dict:
        """Process a single audio file"""
        try:
            with self.print_lock:
                logging.info(f"Processing: {audio_file.name}")

            # Transcribe
            result = self.transcriber.transcribe(str(audio_file))
            
            # Save transcript
            self.transcriber.save_transcript(result, output_file)

            return {
                "status": "completed",
                "output_file": str(output_file),
                "language": result["language"],
                "segments": len(result["segments"])
            }

        except Exception as e:
            logging.error(f"Error processing {audio_file.name}: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def process_batch(self, audio_files: List[Path], output_dir: Path):
        """Process a batch of audio files"""
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            
            for audio_file in audio_files:
                output_file = output_dir / f"{audio_file.stem}_transcript.txt"
                future = executor.submit(self.process_file, audio_file, output_file)
                futures.append((future, audio_file))

            # Process results as they complete
            with tqdm(total=len(futures), desc="Processing files") as pbar:
                for future, audio_file in futures:
                    try:
                        result = future.result()
                        with self.results_lock:
                            self.processed_files[str(audio_file)] = result
                        pbar.update(1)
                    except Exception as e:
                        logging.error(f"Error processing {audio_file}: {e}")
                        pbar.update(1)

    def process_directory(self, input_dir: Path, output_dir: Path):
        """Process all audio files in directory"""
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get all audio files
        audio_files = list(input_dir.glob("**/*"))
        audio_files = [f for f in audio_files if f.suffix.lower() in {'.mp3', '.wav', '.m4a', '.ogg'}]

        if not audio_files:
            logging.info(f"No audio files found in {input_dir}")
            return

        logging.info(f"Found {len(audio_files)} audio files")

        # Process in batches
        for i in range(0, len(audio_files), self.config.batch_size):
            batch = audio_files[i:i + self.config.batch_size]
            self.process_batch(batch, output_dir)