from . import features as _features  # register built-ins
from . import state as _state  # register Step 2 stateful two-pass encoders
from . import gpu  # Step 3: registers no CUDA import at module scope -- see featm.gpu.backend
from .engine import (
    AutoFeatures,
    create_feature,
    extract,
    feature_info,
    list_families,
    list_features,
    list_modalities,
    search,
)
from .extractor import FeatureExtractor, FittedFeatureExtractor, fit, load_fitted
from .distributed import distributed_extract
from .matrix import FeatureMatrix, MaterializationGuardError
from .registry import (
    BatchSemantics,
    Complexity,
    Cost,
    FeatureOutput,
    FeatureSpec,
    MemoryProfile,
    REGISTRY,
    RegistryValidationError,
    register_feature,
)
from .report import ExecutionReport
from .resources import ResourceBudget, parse_size
from .schema import DatasetProfile, SemanticType, infer_schema
from .sources import (
    ArrowSource,
    CSVSource,
    IteratorSource,
    JSONLSource,
    ParquetSource,
    PandasSource,
    Source,
    SourceNotReplayableError,
    resolve_source,
)

__all__ = [
    "ArrowSource",
    "AutoFeatures",
    "BatchSemantics",
    "CSVSource",
    "Complexity",
    "Cost",
    "DatasetProfile",
    "ExecutionReport",
    "FeatureExtractor",
    "FeatureMatrix",
    "FeatureOutput",
    "FeatureSpec",
    "FittedFeatureExtractor",
    "IteratorSource",
    "JSONLSource",
    "MaterializationGuardError",
    "MemoryProfile",
    "PandasSource",
    "ParquetSource",
    "REGISTRY",
    "RegistryValidationError",
    "ResourceBudget",
    "SemanticType",
    "Source",
    "SourceNotReplayableError",
    "create_feature",
    "distributed_extract",
    "extract",
    "feature_info",
    "fit",
    "gpu",
    "infer_schema",
    "list_families",
    "list_features",
    "list_modalities",
    "load_fitted",
    "parse_size",
    "register_feature",
    "resolve_source",
    "search",
]

__version__ = "0.5.7"
