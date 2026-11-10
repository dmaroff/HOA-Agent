from pathlib import Path

from dotenv import load_dotenv

# Load ANTHROPIC_API_KEY (and any other settings) from app/.env.
# Variables already set in the environment, e.g. by a run configuration, take precedence.
load_dotenv(Path(__file__).resolve().parent / ".env")
