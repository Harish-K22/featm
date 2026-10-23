"""Installation hints for exact missing optional roots, after execution fails."""

# Distribution/import names differ. Only extras actually declared by FEATM belong here.
_INSTALL = {
    "scipy": ("scipy", "scientific"),
    "sklearn": ("scikit-learn", "scientific"),
    "skimage": ("scikit-image", "vision"),
    "cv2": ("opencv-python", "vision"),
    "statsmodels": ("statsmodels", "scientific"),
    "librosa": ("librosa", "audio"),
    "networkx": ("networkx", "graph"),
    "holidays": ("holidays", "calendar"),
    "gudhi": ("gudhi", "topology"),
    "PIL": ("Pillow", "embeddings"),
}


def annotate_missing_dependency(spec, exc):
    """Preserve exception type, identity, name, path and original message.

    Do not guess about missing submodules, binary ImportErrors, transitive
    package failures, undeclared user-callable dependencies, or undeclared backends.
    No dependency is imported or checked before executing the feature.
    """
    if type(exc) is not ModuleNotFoundError or exc.name not in _INSTALL:
        return
    distribution, extra = _INSTALL[exc.name]
    if not any(value in spec.requires for value in (exc.name, distribution)):
        return
    hint = f'Feature {spec.name}: install "featm[{extra}]" for {distribution}.'
    if hint in str(exc):
        return
    message = f"{exc}. {hint}"
    exc.msg = message
    exc.args = (message, *exc.args[1:])
