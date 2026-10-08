"""Base-only: train once, transform unseen categories, save/load."""
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
import featm

name = "tabular.encoding.category_frequency"
train = pd.DataFrame({"category": ["A", "A", "B", "B", "B", "C"]})
test = pd.DataFrame({"category": ["A", "B", "C", "D"]})
fitted = featm.fit(train, features=[name], columns={name: "category"}, errors="raise")
result = fitted.transform(test)
np.testing.assert_allclose(result.to_numpy().ravel(), [2/6, 3/6, 1/6, 0])
with TemporaryDirectory() as tmp:
    path = Path(tmp) / "state.pkl"
    fitted.save(path)
    restored = featm.load_fitted(path).transform(test)
    np.testing.assert_array_equal(result.to_numpy(), restored.to_numpy())
print(result.to_pandas())
