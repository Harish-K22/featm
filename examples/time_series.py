"""Base-only: rows are samples and columns are time steps."""
import numpy as np
import featm

X = np.array([[1., 2., 3., 4.], [10., 20., 30., 40.]])
names = ["timeseries.statistical.mean", "timeseries.statistical.std"]
result = featm.extract(X, features=names, errors="raise")
np.testing.assert_allclose(result[names[0]], [2.5, 25.])
print(result)
