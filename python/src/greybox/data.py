"""Built-in datasets for greybox."""

from pathlib import Path

import pandas as pd


def _get_data_path(filename: str) -> Path:
    return Path(__file__).parent / filename


mtcars = pd.read_csv(_get_data_path("mtcars.csv"))
