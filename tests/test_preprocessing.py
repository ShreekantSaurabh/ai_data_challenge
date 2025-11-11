import os
import sys

# Ensure src package is importable when running the test directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from src.preprocessing import Preprocessor

def test_preprocessor_fit_transform() -> None:
    """Confirm the preprocessor produces the expected output shape."""
    df = pd.DataFrame({'a':[1,2,None], 'b':[3,None,5]})
    pre = Preprocessor()
    X = pre.fit_transform(df)
    assert X.shape == (3, 2)
