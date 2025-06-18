"""
Audio Separation Tool using Spleeter and Demucs
This script provides functionality to separate audio files into their constituent stems
"""

import argparse
import logging
import os
import sys
from audio_source_separator.audio_separators import (
    AudioSeparator,
    AudioSeparatorFactory,
    SeparationTool,
)
from audio_source_separator.common_types import InstrumentStem
from audio_source_separator.instrument_classifier import (
    InstrumentClassifier,
    PlaceholderInstrumentClassifier,
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
    parser.add_argument(
        "--detect-instruments",
        action="store_true",
        help="Enable instrument detection to attempt to select a more appropriate model (primarily for Spleeter).",
    )

    args = parser.parse_args()
    return args


def main() -> int:
    """
    Main function to separate audio sources using Spleeter or Demucs.

    Returns:
        int: The exit code of the program.
    """
    logging.basicConfig(
        level=logging.INFO,
        # format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = _parse_command_line_args()

    output_folder = args.output_folder
    if output_folder is None:
        output_folder = f"output_stems/{args.tool.value}"

    detected_instruments_list: list[InstrumentStem] = []
    if args.detect_instruments:
        if not args.input_audio_file:
            logger.warning(
                "Instrument detection requires an input audio file. Please provide one with -i."
            )
        else:
            try:
                # In a real application, you might have a factory
                # or configuration for choosing the classifier
                instrument_classifier: InstrumentClassifier = (
                    PlaceholderInstrumentClassifier()
                )
                detected_instruments_list = instrument_classifier.classify_instruments(
                    args.input_audio_file
                )
            except (FileNotFoundError, OSError, RuntimeError) as e:
                logger.error("Instrument classification failed: %s", e, exc_info=True)
                logger.warning("Proceeding without instrument-based model selection.")

    try:
        separator: AudioSeparator = AudioSeparatorFactory.create_separator(
            args.tool, detected_instruments_list
        )
        separator.separate(
            input_audio_path=args.input_audio_file,
            output_audio_folder=output_folder,
        )
        return os.EX_OK

    except ValueError as e:
        logger.critical("Terminating due to error: %s", e)
        return os.EX_SOFTWARE


if __name__ == "__main__":
    sys.exit(main())
