"""
Audio Separation Tool using Spleeter and Demucs
This script provides functionality to separate audio files into their constituent stems
"""

import logging
import argparse
from audio_separators import (
    AudioSeparator,
    SeparationTool,
    separator_factory,
)


# Get a logger instance for this module
logger = logging.getLogger(__name__)


def _create_separator(separation_tool: SeparationTool) -> AudioSeparator:
    """
    Creates and returns an instance of the appropriate audio separator.

    Args:
        separation_tool: The type of separation tool to create.

    Returns:
        An instance of a class derived from AudioSeparator.

    Raises:
        ValueError: If the specified separation_tool is unsupported.
    """
    if separation_tool not in separator_factory:
        err_msg = f"Unsupported separation tool: '{separation_tool}'"
        logger.error(err_msg)
        raise ValueError(err_msg)

    SeparatorClass, SeparatorConfigClass = separator_factory[separation_tool]
    separator_config = SeparatorConfigClass()
    return SeparatorClass(config=separator_config)


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
        default="sample_audio/tijucos-no-dia-que-de.mp3",
        help="Path to the input audio file (default: sample_audio/tijucos-no-dia-que-de.mp3).",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_folder",
        type=str,
        default=f"output_stems/{default_tool}",
        help="Path to the output folder. If not provided, defaults to 'output_stems/[tool_name]'.",
    )

    args = parser.parse_args()
    return args


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = _parse_command_line_args()

    try:
        separator = _create_separator(args.tool)
        separator.separate(
            input_audio_path=args.input_audio_file,
            output_audio_folder=args.output_folder,
        )

    except ValueError as e:
        logger.critical(f"Terminating due to error: {e}")
        return


if __name__ == "__main__":
    main()
