import pandas as pd
from pathlib import Path
import gcsfs

class DataLoader:
    """Handles loading data from local path or GCS bucket if needed."""
    def __init__(self, path: str):
        """Initialize the loader with the source path.

        :param path: File path or ``gs://`` URI pointing to the dataset.
        """
        self.path = path

    def load(self) -> pd.DataFrame:
        """Load the dataset into a DataFrame.

        :returns: DataFrame containing the campaign data. Supports local files or GCS URIs.
        """
        if self.path.startswith("gs://"):
            fs = gcsfs.GCSFileSystem()
            with fs.open(self.path) as f:
                df = pd.read_csv(f)
        else:
            df = pd.read_csv(Path(self.path))
        return df
