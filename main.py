"""
Audio Separation Tool using Spleeter and Demucs
This script provides functionality to separate audio files into their constituent stems
"""

import argparse
import logging
import os
import sys
from audio_separators import (
    AudioSeparator,
    SeparationTool,
    AudioSeparatorFactory,
)


# Get a logger instance for this module
logger = logging.getLogger(__name__)


def _parse_command_line_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the audio separation script.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    default_tool = SeparationTool.DEMUCS

    parser = argparse.ArgumentParser(
        description="Separate audio sources using Spleeter or Demucs."
    )
    parser.add_argument(
        "-t",
        "--tool",
        type=SeparationTool,
        choices=list(SeparationTool),
        default=default_tool,
        help=f"The separation tool to use (default: {SeparationTool.DEMUCS}).",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_audio_file",
        type=str,
        default=None,
        help="Path to the input audio file (default: sample_audio/tijucos-no-dia-que-de.mp3).",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_folder",
        type=str,
        default=None,
        help="Path to the output folder. If not provided, defaults to 'output_stems/<selected_tool_name>'.",
    )

    args = parser.parse_args()
    return args


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = _parse_command_line_args()

    output_folder = args.output_folder
    if output_folder is None:
        output_folder = f"output_stems/{args.tool.value}"

    try:
        separator: AudioSeparator = AudioSeparatorFactory.create_separator(args.tool)
        separator.separate(
            input_audio_path=args.input_audio_file,
            output_audio_folder=output_folder,
        )
        return os.EX_OK

    except ValueError as e:
        logger.critical(f"Terminating due to error: {e}")
        return os.EX_SOFTWARE


if __name__ == "__main__":
    sys.exit(main())
