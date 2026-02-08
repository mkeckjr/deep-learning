from pathlib import Path

from .types import TokenizerType

def get_base_directory() -> str:
    return str(Path(__file__).parent.absolute())

__all__ = ['get_base_directory', 'TokenizerType']
