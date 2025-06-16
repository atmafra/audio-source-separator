"""
Defines the abstract base class for audio separators and concrete implementations
for Spleeter and Demucs, along with their Pydantic configuration models.
"""

import os
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from enum import StrEnum  # type: ignore
from pydantic import BaseModel
from typing import List, Set, Dict
import yaml
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
    def separate(self, input_audio_path: str, output_audio_folder: str) -> None:
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

    def separate(self, input_audio_path: str, output_audio_folder: str) -> None:
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

    def separate(self, input_audio_path: str, output_audio_folder: str) -> None:
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


class ModelDefinition(BaseModel):
    """Defines a separation model and its supported stems."""

    name: str
    supported_stems: Set[str]
    # Optional: priority for tie-breaking, or other metadata


def _load_models_config(
    config_path: Path,
) -> Dict[SeparationTool, List[ModelDefinition]]:
    """Loads model definitions from a YAML configuration file."""
    available_models: Dict[SeparationTool, List[ModelDefinition]] = {}
    if not config_path.exists():
        logger.warning(
            f"Models configuration file not found at {config_path}. "
            "Instrument-based model selection will be limited."
        )
        return available_models

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        if not isinstance(config_data, dict):
            logger.error(
                f"Invalid format in {config_path}. Expected a dictionary at the root."
            )
            return available_models

        for tool_name, models_data in config_data.items():
            try:
                tool_enum = SeparationTool(tool_name.lower())
                model_defs = []
                if isinstance(models_data, list):
                    for model_entry in models_data:
                        if (
                            isinstance(model_entry, dict)
                            and "name" in model_entry
                            and "supported_stems" in model_entry
                        ):
                            # Ensure supported_stems is a set
                            stems = (
                                set(model_entry["supported_stems"])
                                if isinstance(model_entry["supported_stems"], list)
                                else set()
                            )
                            model_defs.append(
                                ModelDefinition(
                                    name=model_entry["name"], supported_stems=stems
                                )
                            )
                        else:
                            logger.warning(
                                f"Skipping invalid model entry for tool {tool_name} in {config_path}: {model_entry}"
                            )
                available_models[tool_enum] = model_defs
            except ValueError:
                logger.warning(
                    f"Unknown separation tool '{tool_name}' in {config_path}. Skipping."
                )

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_path}: {e}", exc_info=True)
    except IOError as e:
        logger.error(f"Error reading file {config_path}: {e}", exc_info=True)
    return available_models


# Determine the path to the models_config.yaml relative to this script
MODELS_CONFIG_PATH = Path(__file__).parent / "models_config.yaml"
AVAILABLE_MODELS: Dict[SeparationTool, List[ModelDefinition]] = _load_models_config(
    MODELS_CONFIG_PATH
)


class AudioSeparatorFactory:
    """
    Factory class for creating audio separator instances based on the specified tool.
    """

    _registry = {
        SeparationTool.SPLEETER: (SpleeterAudioSeparator, SpleeterConfig),
        SeparationTool.DEMUCS: (DemucsAudioSeparator, DemucsConfig),
    }

    @staticmethod
    def _calculate_jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
        """Calculates Jaccard similarity between two sets of strings."""
        if not set1 and not set2:  # Both empty
            return 1.0
        if not set1 or not set2:  # One empty
            return 0.0

        intersection_count = len(set1.intersection(set2))
        union_count = len(set1.union(set2))

        return intersection_count / union_count if union_count > 0 else 0.0

    @classmethod
    def create_separator(
        cls,
        separation_tool: SeparationTool,
        detected_instruments: list[str] | None = None,
    ) -> AudioSeparator:
        """
        Creates and returns an instance of the appropriate audio separator.

        Args:
            separation_tool: The type of separation tool to create.
            detected_instruments: An optional list of instrument names detected in the audio.

        Returns:
            An instance of a class derived from AudioSeparator.

        Raises:
            ValueError: If the specified separation_tool is unsupported.
        """
        if separation_tool not in cls._registry:
            err_msg = f"Unsupported separation tool: '{separation_tool}'"
            logger.error(err_msg)
            raise ValueError(err_msg)

        SeparatorClass, SeparatorConfigClass = cls._registry[separation_tool]

        # Start with default config (which includes the default model_name from the Config class)
        separator_config = SeparatorConfigClass()
        selected_model_name = separator_config.model_name  # Default model

        if detected_instruments and separation_tool in AVAILABLE_MODELS:
            detected_set = set(detected_instruments)
            logger.info(
                f"Attempting to select best model for {separation_tool.value} based on detected instruments: {detected_set}"
            )

            best_model_candidate: ModelDefinition | None = None
            highest_similarity_score: float = -1.0
            best_coverage_of_detected: int = -1

            tool_models = AVAILABLE_MODELS[separation_tool]

            for model_def in tool_models:
                similarity = cls._calculate_jaccard_similarity(
                    detected_set, model_def.supported_stems
                )
                # How many of the *detected* instruments are covered by this model's stems
                coverage_of_detected = len(
                    detected_set.intersection(model_def.supported_stems)
                )

                logger.debug(
                    f"Evaluating model: {model_def.name}, Stems: {model_def.supported_stems}, "
                    f"Jaccard: {similarity:.4f}, Detected Coverage: {coverage_of_detected}"
                )

                if similarity > highest_similarity_score:
                    highest_similarity_score = similarity
                    best_model_candidate = model_def
                    best_coverage_of_detected = coverage_of_detected
                elif similarity == highest_similarity_score:
                    # Tie-breaking:
                    # 1. Prefer model that covers more of the *detected* instruments.
                    if coverage_of_detected > best_coverage_of_detected:
                        best_model_candidate = model_def
                        best_coverage_of_detected = coverage_of_detected
                    # 2. If coverage is also equal, prefer model with more total stems (more granular)
                    elif (
                        coverage_of_detected == best_coverage_of_detected
                        and best_model_candidate
                        and len(model_def.supported_stems)
                        > len(best_model_candidate.supported_stems)
                    ):
                        best_model_candidate = model_def
                    # 3. If still tied, the one listed earlier in AVAILABLE_MODELS wins (implicit priority)

            # Ensure there's a meaningful match before overriding the default
            if (
                best_model_candidate and highest_similarity_score > 0.0
            ):  # Threshold can be adjusted
                selected_model_name = best_model_candidate.name
                logger.info(
                    f"Selected model based on instruments: {selected_model_name} (Jaccard Similarity: {highest_similarity_score:.4f}, Coverage: {best_coverage_of_detected})"
                )
            else:
                logger.info(
                    f"No sufficiently similar model found or no overlap with detected instruments. Using default model: {selected_model_name}"
                )

        # Update config with the selected model name if it changed from the default
        separator_config.model_name = selected_model_name

        return SeparatorClass(config=separator_config)
