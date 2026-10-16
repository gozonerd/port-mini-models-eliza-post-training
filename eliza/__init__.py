"""The engine lives in eliza-engine.py (Krystal Martinez's file name). A hyphen is not importable by
`import`, so the module is loaded by path here and re-exported."""
import importlib.util as _ilu
import os as _os

_spec = _ilu.spec_from_file_location("eliza_engine", _os.path.join(_os.path.dirname(__file__), "eliza-engine.py"))
engine = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(engine)
Eliza, load_script, parse_script = engine.Eliza, engine.load_script, engine.parse_script
ENGINE_PATH = _spec.origin
