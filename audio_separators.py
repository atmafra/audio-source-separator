"""
Defines the abstract base class for audio separators and concrete implementations
for Spleeter and Demucs, along with their Pydantic configuration models.
"""

import os
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from enum import StrEnum
from pydantic import BaseModel
import torch
from demucs.api import Separator as DemucsSeparator
from demucs.audio import save_audio

logger = logging.getLogger(__name__)


class SeparationTool(StrEnum):
    """Enumeration of available audio separation tools."""

    SPLEETER = "spleeter"
    DEMUCS = "demucs"


# --- Configuration Classes ---
class AudioSeparatorConfig(BaseModel):
    """Base configuration for any audio separator."""

    model_name: str


class SpleeterConfig(AudioSeparatorConfig):
    """Configuration specific to Spleeter."""

    model_name: str = "spleeter:5stems"


class DemucsConfig(AudioSeparatorConfig):
    """Configuration specific to Demucs."""

    model_name: str = "htdemucs"


# --- Abstract Base Class for Audio Separators ---
class AudioSeparator(ABC):
    """
    Abstract base class for audio separation tools.
    """

    def __init__(self, config: AudioSeparatorConfig):
        self.config = config

    def _check_input_file(self, input_audio_path: str) -> bool:
        """
        Checks if the specified input audio file exists on the filesystem.

        Args:
            input_audio_path: The path to the audio file to check.

        Returns:
            True if the file exists, False otherwise.
        """
        if not os.path.exists(input_audio_path):
            logger.error(f"Input audio file not found at {input_audio_path}")
            return False
        return True

    @abstractmethod
    def separate(self, input_audio_path: str, output_audio_folder: str):
        """
        Performs the audio separation.
        Subclasses must implement this method, interpreting output_audio_folder appropriately.
        """
        pass


# --- Spleeter Specific Implementation ---
class SpleeterAudioSeparator(AudioSeparator):
    """Audio separator using Spleeter."""

    def __init__(self, config: SpleeterConfig):
        super().__init__(config)

    def separate(self, input_audio_path: str, output_audio_folder: str):
        """Separates an audio file using Spleeter."""
        logger.info(f"--- Using Spleeter (model: {self.config.model_name}) ---")
        if not self._check_input_file(input_audio_path):
            return

        if not os.path.exists(output_audio_folder):
            os.makedirs(output_audio_folder)
            logger.info(f"Created output directory: {output_audio_folder}")

        from spleeter.separator import Separator as SpleeterLibSeparator

        spleeter_instance = SpleeterLibSeparator(self.config.model_name)
        logger.info(
            f"Processing {input_audio_path} with Spleeter model {self.config.model_name}..."
        )
        spleeter_instance.separate_to_file(input_audio_path, output_audio_folder)
        logger.info(
            f"Spleeter separation complete. Output files are in {output_audio_folder}"
        )


# --- Demucs Specific Implementation ---
class DemucsAudioSeparator(AudioSeparator):
    """Audio separator using the Demucs library."""

    def __init__(self, config: DemucsConfig):
        super().__init__(config)

    def separate(self, input_audio_path: str, output_audio_folder: str):
        """
        Separates an audio file using the Demucs library.
        Demucs typically separates into: drums, bass, other, vocals.
        """
        logger.info(f"--- Using Demucs library (model: {self.config.model_name}) ---")
        if not self._check_input_file(input_audio_path):
            return

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Demucs will use device: {device}")

            demucs_instance = DemucsSeparator(
                model=self.config.model_name, device=device
            )

            logger.info(
                f"Processing {input_audio_path} with Demucs model {self.config.model_name}..."
            )
            _, separated_sources = demucs_instance.separate_audio_file(
                Path(input_audio_path)
            )

            input_audio_path_basename = os.path.basename(input_audio_path)
            input_filename_base = os.path.splitext(input_audio_path_basename)[0]
            output_path_for_song = os.path.join(
                output_audio_folder, self.config.model_name, input_filename_base
            )

            if not os.path.exists(output_path_for_song):
                os.makedirs(output_path_for_song)
                logger.info(f"Created output directory: {output_path_for_song}")

            for stem_name, stem_tensor in separated_sources.items():
                stem_output_path = os.path.join(
                    output_path_for_song, f"{stem_name}.wav"
                )
                save_audio(
                    stem_tensor, stem_output_path, samplerate=demucs_instance.samplerate
                )
                logger.info(f"Saved {stem_name} to {stem_output_path}")

            logger.info(
                f"Demucs separation complete. Output files are in {output_path_for_song}"
            )

        except (RuntimeError, ValueError, IOError) as e:
            logger.error(f"Error during Demucs library processing: {e}", exc_info=True)


# Factory to map tool enum to its corresponding classes
separator_factory = {
    SeparationTool.SPLEETER: (SpleeterAudioSeparator, SpleeterConfig),
    SeparationTool.DEMUCS: (DemucsAudioSeparator, DemucsConfig),
}
