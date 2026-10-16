from .eliza_engine import Eliza, load_script, parse_script  # noqa: F401
from . import eliza_engine as engine  # noqa: F401
import os as _os
ENGINE_PATH = _os.path.join(_os.path.dirname(__file__), "eliza_engine.py")
