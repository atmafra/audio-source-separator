import os
import subprocess  # For calling Demucs CLI
from spleeter.separator import Separator


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
def separate_audio_demucs(input_audio_path, output_directory, demucs_model="htdemucs"):
    """
    Separates an audio file using Demucs CLI.
    Demucs typically separates into: drums, bass, other, vocals.
    Other models might offer different stems (e.g., htdemucs_ft for fine-tuned).
    """
    print(f"\n--- Using Demucs (model: {demucs_model}) ---")
    if not os.path.exists(input_audio_path):
        print(f"Error: Input audio file not found at {input_audio_path}")
        return

    # Demucs CLI typically creates a subdirectory with the model name in the output path.
    # We'll use the provided output_directory directly.
    # Example: demucs -n mdx_extra_q --two-stems --out ./separated_output "my_song.mp3"
    # For default 4 stems: demucs --out <output_dir> <input_audio_path>

    # Ensure output directory exists (Demucs might not create it if it's nested)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory for Demucs: {output_directory}")

    command = [
        "demucs",
        "-n",
        demucs_model,
        "--out",
        output_directory,
        input_audio_path,
    ]
    print(f"Executing Demucs command: {' '.join(command)}")

    try:
        subprocess.run(command, check=True)
        input_filename_without_ext = os.path.splitext(
            os.path.basename(input_audio_path)
        )[0]
        print(
            f"Demucs separation complete. Output files are in {output_directory}/{demucs_model}/{input_filename_without_ext}/"
        )
    except subprocess.CalledProcessError as e:
        print(f"Error during Demucs processing: {e}")
    except FileNotFoundError:
        print("Error: Demucs command not found. Is it installed and in your PATH?")


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
        separate_audio_demucs(INPUT_AUDIO, DEMUCS_OUTPUT_DIR, DEMUCS_MODEL)
    else:
        print(f"Error: Unknown tool '{TOOL_TO_USE}'. Choose 'spleeter' or 'demucs'.")
