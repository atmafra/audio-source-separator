"""
Audio Separation Tool using Spleeter and Demucs
This script provides functionality to separate audio files into their constituent stems
"""

import os
import traceback

from spleeter.separator import Separator as SpleeterSeparator
from demucs.api import Separator as DemucsSeparator
from demucs.audio import save_audio
import torch
from enum import StrEnum
from abc import ABC, abstractmethod
from pydantic import BaseModel
from pathlib import Path


# --- Abstract Base Class for Audio Separators ---
class AudioSeparator(ABC):
    """
    Abstract base class for audio separation tools.
    """

    def __init__(self, config: "AudioSeparatorConfig"):
        self.config = config

    def _check_input_file(self, input_audio_path: str) -> bool:
        """Checks if the input audio file exists."""
        if not os.path.exists(input_audio_path):
            print(f"Error: Input audio file not found at {input_audio_path}")
            return False
        return True

    @abstractmethod
    def separate(self, input_audio_path: str, output_destination: str):
        """
        Performs the audio separation.
        Subclasses must implement this method, interpreting output_destination appropriately.
        """
        pass


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


# --- Spleeter Specific Implementation ---
class SpleeterAudioSeparator(AudioSeparator):
    """
    Audio separator using Spleeter.
    """

    def __init__(
        self,
        config: SpleeterConfig,
    ):
        super().__init__(config)

    def separate(self, input_audio_path: str, output_destination: str):
        """Separates an audio file using Spleeter."""
        print(f"\n--- Using Spleeter (model: {self.config.model_name}) ---")
        if not self._check_input_file(input_audio_path):
            return

        if not os.path.exists(output_destination):
            os.makedirs(output_destination)
            print(f"Created output directory: {output_destination}")

        spleeter_instance = SpleeterSeparator(self.config.model_name)
        print(f"Processing {input_audio_path} with model {self.config.model_name}...")
        spleeter_instance.separate_to_file(input_audio_path, output_destination)
        print(f"Separation complete. Output files are in {output_destination}")


# --- Demucs Specific Implementation ---
class DemucsAudioSeparator(AudioSeparator):
    """
    Audio separator using the Demucs library.
    """

    def __init__(self, config: DemucsConfig):
        super().__init__(config)

    def separate(self, input_audio_path: str, output_destination: str):
        """
        Separates an audio file using the Demucs library.
        Demucs typically separates into: drums, bass, other, vocals.
        """
        print(f"\n--- Using Demucs library (model: {self.config.model_name}) ---")
        if not self._check_input_file(input_audio_path):
            return

        try:
            # Determine device (CPU or GPU)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Demucs will use device: {device}")

            # This will download the model if not already cached
            demucs_instance = DemucsSeparator(
                model=self.config.model_name, device=device
            )

            print(
                f"Processing {input_audio_path} with Demucs model {self.config.model_name}..."
            )
            # The `separate_audio_file` method loads the audio, separates it, and returns tensors.
            # It returns:
            #   origin: Tensor of the original waveform [channels, samples]
            #   separated_sources: Dict[str, Tensor], where keys are stem names (e.g., "vocals", "drums")
            _, separated_sources = demucs_instance.separate_audio_file(
                Path(input_audio_path)
            )

            # Define the output directory structure (similar to Demucs CLI)
            input_audio_path_basename = os.path.basename(input_audio_path)
            input_filename_base = os.path.splitext(input_audio_path_basename)[0]
            output_path_for_song = os.path.join(
                output_destination, self.config.model_name, input_filename_base
            )

            if not os.path.exists(output_path_for_song):
                os.makedirs(output_path_for_song)
                print(f"Created output directory: {output_path_for_song}")

            # Save each separated stem
            for stem_name, stem_tensor in separated_sources.items():
                stem_output_path = os.path.join(
                    output_path_for_song, f"{stem_name}.wav"
                )
                save_audio(
                    stem_tensor, stem_output_path, samplerate=demucs_instance.samplerate
                )
                print(f"Saved {stem_name} to {stem_output_path}")

            print(
                f"Demucs separation complete. Output files are in {output_path_for_song}"
            )

        except (RuntimeError, ValueError, IOError) as e:
            print(f"Error during Demucs library processing: {e}")
            traceback.print_exc()


class Tool(StrEnum):
    SPLEETER = "spleeter"
    DEMUCS = "demucs"


def main():
    # --- Configuration ---
    input_audio_file = "sample_audio/tijucos-no-dia-que-de.mp3"

    # Choose your tool: Tool.SPLEETER or Tool.DEMUCS
    tool_to_use = Tool.DEMUCS

    demucs_default_output = "output_stems/demucs"
    spleeter_default_output = "output_stems/spleeter"

    # --- End Configuration ---

    separator_instance: AudioSeparator

    if tool_to_use == Tool.SPLEETER:
        spleeter_config = SpleeterConfig()
        separator_instance = SpleeterAudioSeparator(config=spleeter_config)
        separator_instance.separate(input_audio_file, spleeter_default_output)

    elif tool_to_use == Tool.DEMUCS:
        demucs_config = DemucsConfig()
        separator_instance = DemucsAudioSeparator(config=demucs_config)
        separator_instance.separate(input_audio_file, demucs_default_output)

    else:
        print(f"Error: Unknown tool '{tool_to_use}'. Choose 'spleeter' or 'demucs'.")
        return


if __name__ == "__main__":
    main()
