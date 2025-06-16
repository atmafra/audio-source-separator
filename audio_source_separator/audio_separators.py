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
from audio_source_separator.common_types import InstrumentStem

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
    description: str
    supported_stems: Set[InstrumentStem]
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
                            parsed_stems: Set[InstrumentStem] = set()
                            if isinstance(model_entry["supported_stems"], list):
                                for stem_str in model_entry["supported_stems"]:
                                    try:
                                        # Convert string from YAML to InstrumentStem enum member
                                        parsed_stems.add(
                                            InstrumentStem(str(stem_str).lower())
                                        )
                                    except ValueError:
                                        logger.warning(
                                            f"Unknown stem '{stem_str}' in model '{model_entry['name']}' "
                                            f"for tool {tool_name} in {config_path}. Skipping this stem."
                                        )
                            model_defs.append(
                                ModelDefinition(
                                    name=model_entry.get("name", "Unnamed Model"),
                                    description=model_entry.get(
                                        "description", "No description provided."
                                    ),
                                    supported_stems=parsed_stems,
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


class ModelSelector:
    """
    Handles the logic for selecting the most appropriate separation model
    based on detected instruments and available model definitions.
    """

    @staticmethod
    def _calculate_jaccard_similarity(
        set1: Set[InstrumentStem], set2: Set[InstrumentStem]
    ) -> float:
        """Calculates Jaccard similarity between two sets of strings."""
        if not set1 and not set2:  # Both empty
            return 1.0

        if not set1 or not set2:
            return 0.0

        intersection_count = len(set1.intersection(set2))
        union_count = len(set1.union(set2))

        return intersection_count / union_count if union_count > 0 else 0.0

    def _find_best_match_by_jaccard(
        self,
        tool_models: List[ModelDefinition],
        detected_instruments_set: Set[InstrumentStem],
        default_model_name: str,
    ) -> str:
        """
        Selects the best model name from the available tool_models based on detected instruments.
        using Jaccard similarity and coverage heuristics. This is the core matching logic.

        Args:
            tool_models: A list of ModelDefinition objects for the current separation tool.
            detected_instruments_set: A set of detected instrument names.
            default_model_name: The default model name to return if no better match is found.

        Returns:
            The name of the selected model.
        """
        best_model_candidate: ModelDefinition | None = None
        highest_similarity_score: float = -1.0
        best_coverage_of_detected: int = -1

        for model_def in tool_models:
            similarity = self._calculate_jaccard_similarity(
                detected_instruments_set, model_def.supported_stems
            )
            coverage_of_detected = len(
                detected_instruments_set.intersection(model_def.supported_stems)
            )

            logger.debug(
                f"Evaluating model: {model_def.name}, Stems: {[s.value for s in model_def.supported_stems]}, "
                f"Jaccard: {similarity:.4f}, Detected Coverage: {coverage_of_detected}"
            )

            if similarity > highest_similarity_score:
                highest_similarity_score = similarity
                best_model_candidate = model_def
                best_coverage_of_detected = coverage_of_detected

            elif similarity == highest_similarity_score:
                if coverage_of_detected > best_coverage_of_detected:
                    best_model_candidate = model_def
                    best_coverage_of_detected = coverage_of_detected

                elif (
                    coverage_of_detected == best_coverage_of_detected
                    and best_model_candidate
                    and len(model_def.supported_stems)
                    > len(best_model_candidate.supported_stems)
                ):
                    best_model_candidate = model_def

        if best_model_candidate and highest_similarity_score > 0.0:
            logger.info(
                f"Selected model based on instruments: {best_model_candidate.name} "
                f"(Jaccard Similarity: {highest_similarity_score:.4f}, Coverage: {best_coverage_of_detected})"
            )
            return best_model_candidate.name

        else:
            logger.info(
                f"No sufficiently similar model found or no overlap with detected instruments. "
                f"Using default model: {default_model_name}"
            )
            return default_model_name

    def get_model_name_for_tool(
        self,
        separation_tool: SeparationTool,
        detected_instruments_list: List[InstrumentStem],
        all_available_models: Dict[SeparationTool, List[ModelDefinition]],
        default_model_name: str,
    ) -> str:
        """
        Determines the appropriate model name for a given separation tool based on detected instruments.
        This method orchestrates the selection by performing preliminary checks and then
        delegating to the core Jaccard-based matching logic.

        Args:
            separation_tool: The separation tool being used.
            detected_instruments_list: A list of detected InstrumentStem objects.
                                      Can be empty if no instruments were detected or if detection was not run.
            all_available_models: A dictionary containing all configured model definitions.
            default_model_name: The default model name to use if no better match is found.

        Returns:
            The selected model name.
        """
        if not detected_instruments_list:
            logger.info(
                "No instruments provided for model selection (list is empty). Using default model: %s",
                default_model_name,
            )
            return default_model_name

        if (
            separation_tool not in all_available_models
            or not all_available_models[separation_tool]
        ):
            logger.info(
                "No model definitions found for tool %s in configuration. Using default model: %s",
                separation_tool.value,
                default_model_name,
            )
            return default_model_name

        tool_specific_models = all_available_models[separation_tool]
        detected_stems_set = set(detected_instruments_list)

        logger.info(
            "Attempting to select best model for %s based on detected instruments: %s",
            separation_tool.value,
            {stem.value for stem in detected_stems_set},
        )
        return self._find_best_match_by_jaccard(
            tool_models=tool_specific_models,
            detected_instruments_set=detected_stems_set,
            default_model_name=default_model_name,
        )


class AudioSeparatorFactory:
    """
    Factory class for creating audio separator instances based on the specified tool.
    """

    _registry = {
        SeparationTool.SPLEETER: (SpleeterAudioSeparator, SpleeterConfig),
        SeparationTool.DEMUCS: (DemucsAudioSeparator, DemucsConfig),
    }

    @classmethod
    def create_separator(
        cls,
        separation_tool: SeparationTool,
        detected_instruments: list[InstrumentStem],  # Expects a list, can be empty
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
        current_default_model_name = separator_config.model_name

        # Use ModelSelector to determine the best model name
        selector = ModelSelector()
        selected_model_name = selector.get_model_name_for_tool(
            separation_tool=separation_tool,
            detected_instruments_list=detected_instruments,
            all_available_models=AVAILABLE_MODELS,
            default_model_name=current_default_model_name,
        )

        # Update config with the selected model name if it changed from the default
        separator_config.model_name = selected_model_name
        return SeparatorClass(config=separator_config)
