# Audio Transcriber with GPU Support

I created this to use OpenAI's whisper model with Intel Arc. It provides an audio file's transcription and translation (albeit imperfect). I've only tested it with Japanese audio so far, but it should work with other languages

## Features

- GPU Acceleration (Intel Arc and NVIDIA)
- Automatic language detection
- Japanese romanization
- English translation
- Batch processing support
- Multi-threaded processing

## Requirements

- Python 3.10+
- FFmpeg
- One of:
  - Intel Arc GPU with latest drivers
  - NVIDIA GPU with CUDA support
  - CPU (fallback)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/audio-transcriber.git
cd audio-transcriber
```

2. Install dependencies:
```bash
# For Intel GPU
pip install torch torchvision torchaudio intel-extension-for-pytorch --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# For NVIDIA GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Common dependencies
pip install -r requirements.txt
```

## Usage

Run the interactive CLI:
```bash
python src/cli.py
```

### Features:
- Process single audio files
- Batch process entire directories
- Configure model settings
- Enable/disable translation
- Automatic device selection

## Configuration

Available settings:
- Model Size: tiny, base, small, medium, large
- Batch Size: 1-10
- Worker Threads: 1-8
- Translation: Enable/Disable

## Output Format

```
=== Transcription ===

Language: JA

[00:00:000 --> 00:12:000]
Text: [Japanese text]
Romaji: [Romanization]
Translation: [English translation]
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Acknowledgments

- OpenAI Whisper for transcription
- Intel Extension for PyTorch
- MarianMT for translation
- pykakasi for romanization
