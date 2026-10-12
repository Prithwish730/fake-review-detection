"""
pin_requirements.py

Writes a SMALL requirements.txt containing only what the dashboard needs, with every
package pinned to the version installed on THIS machine. Streamlit Community Cloud
installs from requirements.txt, and the saved model must be loaded with the same
scikit-learn / numpy versions it was trained with.

Your old requirements.txt (with torch, transformers, xgboost, ...) is kept as
requirements-train.txt for local development:  pip install -r requirements-train.txt

Run (from the project root, with your venv active):
    python deploy/pin_requirements.py
"""

import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARKER = "# Runtime dependencies for the dashboard"
PACKAGES = ["streamlit", "pandas", "numpy", "scikit-learn", "joblib", "nltk", "matplotlib"]


def write_requirements(root: Path = ROOT) -> Path:
    req = root / "requirements.txt"
    train = root / "requirements-train.txt"

    # keep the full development list (once), unless requirements.txt is already the slim one
    if req.exists() and MARKER not in req.read_text() and not train.exists():
        train.write_text(req.read_text())

    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    lines = [f"{MARKER} (used by Streamlit Community Cloud).",
             f"# Pinned to the versions this project was built with; choose Python {py} when deploying."]
    for pkg in PACKAGES:
        try:
            lines.append(f"{pkg}=={metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            lines.append(f"{pkg}   # not installed here: install it, then rerun this script")
    req.write_text("\n".join(lines) + "\n")
    return req


if __name__ == "__main__":
    path = write_requirements()
    print(f"Wrote {path.name}:\n")
    print(path.read_text())
    print(f"Your Python version is {sys.version_info.major}.{sys.version_info.minor}: "
          "pick the same one under 'Advanced settings' when you deploy.")
    if (ROOT / "requirements-train.txt").exists():
        print("Full development requirements are kept in requirements-train.txt")
