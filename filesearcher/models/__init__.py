"""Ładowanie modeli dla wybranego profilu."""
from dataclasses import dataclass

from .base import Describer, Embedder


@dataclass
class ModelBundle:
    embedder: Embedder
    describer: Describer | None


def load_models(profile, log=print) -> ModelBundle:
    raise NotImplementedError
