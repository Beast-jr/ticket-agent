import sys
from pathlib import Path

# Let tests import the project's modules (agent, llm, search, ...) from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
