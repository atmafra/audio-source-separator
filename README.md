# audio-source-separator

The project's objective is to be able take an input audio file (optimized for songs) and separate into its constituent sources or instruments, like guitar, bass, drums, vocals, violins, percussion, etc.

## What it Does

This script provides a command-line interface to separate audio sources using one of two popular tools: Spleeter or Demucs. You can choose which tool to use and specify input/output files.
The configuration for each tool (like the specific model to use) is managed using Pydantic models.

## Prerequisites

This project uses [Poetry](https://python-poetry.org/) for dependency management and packaging.

### Python Dependencies

First, ensure you have [Poetry installed](https://python-poetry.org/docs/#installation). Then, navigate to the project root directory and run:

```bash
poetry install
```

### System Dependencies

- **libsndfile**: Required by `torchaudio` (a dependency of Demucs) for audio file operations on some systems.
  On Debian/Ubuntu: `sudo apt-get install libsndfile1`
- **(Optional but Recommended) FFmpeg**: While `ffmpeg-python` is listed in requirements, having the `ffmpeg` command-line tool installed system-wide can be beneficial for broader compatibility.
  On Debian/Ubuntu: `sudo apt install ffmpeg`
