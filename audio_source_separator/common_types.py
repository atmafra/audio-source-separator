"""
Defines common enumerations and type definitions used across the
audio_source_separator package.
"""

from enum import StrEnum


class InstrumentStem(StrEnum):
    """Enumeration of known instrument stems."""

    VOCALS = "vocals"
    DRUMS = "drums"
    BASS = "bass"
    OTHER = "other"
    PIANO = "piano"
    GUITAR = "guitar"
    ACCOMPANIMENT = "accompaniment"
    # Add more instruments/stems as needed
    # e.g., STRINGS = "strings", BRASS = "brass", SYNTH = "synth"
