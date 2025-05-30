"""
Audio Separation Tool using Spleeter and Demucs
This script provides functionality to separate audio files into their constituent stems
"""

import os
import traceback

# import subprocess # No longer needed for Demucs CLI
from spleeter.separator import Separator

from demucs.api import Separator as DemucsAPISeparator
from demucs.audio import save_audio
import torch

# --- Spleeter Specific Function ---
def separate_audio_spleeter(
    input_audio_path, output_directory, model_name="spleeter:5stems"
):
    """
    Separates an audio file using Spleeter.
    """
    print(f"\n--- Using Spleeter (model: {model_name}) ---")
    if not os.path.exists(input_audio_path):
        print(f"Error: Input audio file not found at {input_audio_path}")
        return

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")

    separator = Separator(model_name)
    print(f"Processing {input_audio_path} with model {model_name}...")
    separator.separate_to_file(input_audio_path, output_directory)
    print(f"Separation complete. Output files are in {output_directory}")


# --- Demucs Specific Function ---
def separate_audio_demucs_library(
    input_audio_path, output_base_dir, demucs_model_name="htdemucs"
):
    """
    Separates an audio file using the Demucs library.
    Demucs typically separates into: drums, bass, other, vocals.
    """
    print(f"\n--- Using Demucs library (model: {demucs_model_name}) ---")
    if not os.path.exists(input_audio_path):
        print(f"Error: Input audio file not found at {input_audio_path}")
        return

    try:
        # Determine device (CPU or GPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Demucs will use device: {device}")

        # Create a Demucs Separator instance
        # This will download the model if not already cached
        separator = DemucsAPISeparator(model=demucs_model_name, device=device)

        print(f"Processing {input_audio_path} with Demucs model {demucs_model_name}...")
        # The `separate_audio_file` method loads the audio, separates it, and returns tensors.
        # It returns:
        #   origin: Tensor of the original waveform [channels, samples]
        #   separated_sources: Dict[str, Tensor], where keys are stem names (e.g., "vocals", "drums")
        #                        and values are tensors of the separated stems [channels, samples].
        origin, separated_sources = separator.separate_audio_file(input_audio_path)

        # Define the output directory structure (similar to Demucs CLI)
        # output_base_dir / model_name / input_filename_base / stem_name.wav
        input_filename_base = os.path.splitext(os.path.basename(input_audio_path))[0]
        output_path_for_song = os.path.join(
            output_base_dir, demucs_model_name, input_filename_base
        )

        if not os.path.exists(output_path_for_song):
            os.makedirs(output_path_for_song)
            print(f"Created output directory: {output_path_for_song}")

        # Save each separated stem
        for stem_name, stem_tensor in separated_sources.items():
            stem_output_path = os.path.join(output_path_for_song, f"{stem_name}.wav")
            save_audio(stem_tensor, stem_output_path, samplerate=separator.samplerate)
            print(f"Saved {stem_name} to {stem_output_path}")

        print(f"Demucs separation complete. Output files are in {output_path_for_song}")
    except Exception as e:
        print(f"Error during Demucs library processing: {e}")

        traceback.print_exc()


if __name__ == "__main__":
    # --- Configuration ---
    INPUT_AUDIO = "sample_audio/tijucos-no-dia-que-de.mp3"

    # Choose your tool: "spleeter" or "demucs"
    # TOOL_TO_USE = "spleeter"
    TOOL_TO_USE = "demucs"

    # Spleeter settings
    SPLEETER_OUTPUT_DIR = "output_stems/spleeter"
    SPLEETER_MODEL = "spleeter:5stems"

    # Demucs settings
    DEMUCS_OUTPUT_DIR = "output_stems/demucs"
    DEMUCS_MODEL = (
        "htdemucs"  # htdemucs is a good default. Others: mdx_extra, htdemucs_ft
    )
    # --- End Configuration ---

    if TOOL_TO_USE.lower() == "spleeter":
        separate_audio_spleeter(INPUT_AUDIO, SPLEETER_OUTPUT_DIR, SPLEETER_MODEL)
    elif TOOL_TO_USE.lower() == "demucs":
        separate_audio_demucs_library(INPUT_AUDIO, DEMUCS_OUTPUT_DIR, DEMUCS_MODEL)
    else:
        print(f"Error: Unknown tool '{TOOL_TO_USE}'. Choose 'spleeter' or 'demucs'.")
