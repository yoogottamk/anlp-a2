import os
from pathlib import Path


DATASET_DIR = (
    Path(os.getenv("DATASET_DIR", str(Path(__file__).parent.parent.absolute())))
    / "dataset"
)
