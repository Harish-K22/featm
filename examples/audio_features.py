"""Audio extra. Synthetic default; optionally supply local ESC-10 files."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import featm

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esc-root", type=Path)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    sr = 22050
    if args.esc_root:
        import librosa
        metadata = pd.read_csv(args.esc_root / "meta" / "esc50.csv")
        metadata = metadata.loc[metadata["esc10"]].head(args.limit)
        clips = []
        for filename in metadata["filename"]:
            clip, loaded_sr = librosa.load(args.esc_root / "audio" / filename, sr=None, mono=True)
            if clips and loaded_sr != sr:
                raise ValueError("This simple batch requires a common sample rate")
            sr = loaded_sr
            clips.append(clip)
        if not clips or len({len(clip) for clip in clips}) != 1:
            raise ValueError("Select nonempty clips of equal length")
        X = np.stack(clips)
    else:
        t = np.arange(8192) / sr
        X = np.stack([np.sin(2*np.pi*220*t), np.sin(2*np.pi*440*t)])
    names = ["audio.perceptual.mfcc", "audio.waveform.rms", "audio.spectral.centroid"]
    parameters = {names[0]: {"n_mfcc": 13, "sr": sr}, names[2]: {"sr": sr}}
    result = featm.extract(X, features=names, params=parameters, errors="raise")
    assert result.shape == (len(X), 15)
    print(result)

if __name__ == "__main__":
    main()
