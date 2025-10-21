# utils_agent_build.py
import importlib.util
from pathlib import Path

# path to ai-agent.py (adjust if needed)
AGENT_PATH = Path(__file__).parent / "agenticai.py"

spec = importlib.util.spec_from_file_location("agenticai_dyn", str(AGENT_PATH))
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

# expose compiled graph as expected by server/app.py
compiled_app = module.app
