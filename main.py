import os
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter


def separate_audio(input_audio_path, output_directory, model_name="spleeter:4stems"):
    """
    Separates an audio file into its constituent stems using Spleeter.

    Args:
        input_audio_path (str): Path to the input audio file.
        output_directory (str): Directory where the separated stems will be saved.
        model_name (str): Spleeter model to use (e.g., 'spleeter:2stems', 'spleeter:4stems', 'spleeter:5stems').
                          Defaults to 'spleeter:4stems'.
    """
    if not os.path.exists(input_audio_path):
        print(f"Error: Input audio file not found at {input_audio_path}")
        return

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")

    # Initialize Spleeter separator
    # Using a specific model (e.g., 'spleeter:2stems' for vocals/accompaniment,
    # 'spleeter:4stems' for vocals/drums/bass/other,
    # 'spleeter:5stems' for vocals/drums/bass/piano/other)
    separator = Separator(model_name)

    print(f"Processing {input_audio_path} with model {model_name}...")
    # Perform the separation
    # The input can be a filepath, a numpy array, or a librosa loaded audio waveform
    separator.separate_to_file(input_audio_path, output_directory)

    print(f"Separation complete. Output files are in {output_directory}")


if __name__ == "__main__":
    # --- Configuration ---
    INPUT_AUDIO = "sample_audio/tijucos-no-dia-que-de.mp3"
    OUTPUT_DIR = "output_stems"
    # --- End Configuration ---

    separate_audio(INPUT_AUDIO, OUTPUT_DIR)
