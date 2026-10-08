# Import modules for registration side effects. Keep imports explicit so feature
# discovery is deterministic and does not depend on filesystem scanning.
from .tabular import numeric as _tabular_numeric
from .tabular import categorical as _tabular_categorical
from .tabular import datetime as _tabular_datetime
from .tabular import interactions as _tabular_interactions
from .tabular import full as _tabular_full

from .geometry import triangulation as _geometry
from .image import basic as _image_basic
from .image import full as _image_full
from .timeseries import basic as _timeseries_basic
from .timeseries import full as _timeseries_full
from .audio import full as _audio_full
from .signal import full as _signal_full
from .graph import full as _graph_full
from .pointcloud import full as _pointcloud_full
from .manifold import full as _manifold_full
from .spatial import full as _spatial_full
from .topology import full as _topology_full
from .information import full as _information_full
from .matrix import full as _matrix_full
from .sequence import full as _sequence_full
from .text import full as _text_full
from .video import full as _video_full
from .event import full as _event_full
from .multimodal import full as _multimodal_full
