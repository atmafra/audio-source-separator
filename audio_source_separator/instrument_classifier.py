"""
Defines the abstract base class for instrument classification and
provides placeholder and mock implementations.
"""

from abc import ABC, abstractmethod
import logging
from typing import List
import random
from audio_source_separator.common_types import InstrumentStem

logger = logging.getLogger(__name__)


class InstrumentClassifier(ABC):
    """
    Abstract base class for instrument classification.
    """

    @abstractmethod
    def classify_instruments(self, audio_path: str) -> List[InstrumentStem]:
        """
        Classifies instruments present in the given audio file.

        Args:
            audio_path: Path to the input audio file.

        Returns:
            A list of detected `InstrumentStem` enum members.
        """


class PlaceholderInstrumentClassifier(InstrumentClassifier):
    """
    A very basic placeholder for instrument classification.
    This implementation provides a fixed, default set of instruments
    and should be replaced with a real instrument detection model for
    meaningful results.
    """

    def classify_instruments(self, audio_path: str) -> List[InstrumentStem]:
        logger.info(
            "PlaceholderInstrumentClassifier: Using default instruments for %s.",
            audio_path,
        )
        # Default assumption for many pop/rock songs, or a common set for Spleeter 4stems/Demucs
        detected_instrument_stems: List[InstrumentStem] = [
            InstrumentStem.VOCALS,
            InstrumentStem.DRUMS,
            InstrumentStem.BASS,
            InstrumentStem.OTHER,
        ]
        logger.info(
            "Placeholder: Detected instruments: %s",
            [stem.value for stem in detected_instrument_stems],
        )
        return detected_instrument_stems


class MockInstrumentClassifier(InstrumentClassifier):
    """
    A mock implementation for instrument classification that simulates
    detection based on simple heuristics or predefined rules.
    Useful for testing the model selection pipeline without a full ML model.
    """

    def classify_instruments(self, audio_path: str) -> List[InstrumentStem]:
        logger.info(
            "MockInstrumentClassifier: Simulating instrument detection for %s...",
            audio_path,
        )
        detected_stems: List[InstrumentStem] = []

        if "piano" in audio_path.lower():
            detected_stems.extend(
                [
                    InstrumentStem.VOCALS,
                    InstrumentStem.PIANO,
                    InstrumentStem.DRUMS,
                    InstrumentStem.BASS,
                ]
            )
        elif "acoustic" in audio_path.lower() or "guitar" in audio_path.lower():
            detected_stems.extend(
                [InstrumentStem.VOCALS, InstrumentStem.GUITAR, InstrumentStem.BASS]
            )
        else:
            # Use all available InstrumentStem members as possibilities
            possible_stems: List[InstrumentStem] = list(InstrumentStem)

            # Ensure vocals is usually present
            if InstrumentStem.VOCALS in possible_stems:
                detected_stems.append(InstrumentStem.VOCALS)

            # Determine remaining stems after potentially adding vocals
            remaining_stems = [
                stem for stem in possible_stems if stem not in detected_stems
            ]

            if remaining_stems:
                # Decide how many additional instruments to pick, up to 3 more or all remaining if fewer than 3
                num_additional_to_sample = random.randint(
                    1, min(3, len(remaining_stems))
                )
                detected_stems.extend(
                    random.sample(remaining_stems, num_additional_to_sample)
                )

        # Ensure uniqueness (though sampling without replacement should handle this)
        # and convert to list
        final_detected_stems = list(set(detected_stems))
        logger.info(
            "MockInstrumentClassifier: Detected instruments: %s",
            [stem.value for stem in final_detected_stems],
        )
        return final_detected_stems
