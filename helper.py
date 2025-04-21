from pathlib import Path
root=Path('/mnt/data/dwiepert/data/video_features')
paths = root.rglob('*.npz.npz')
for p in paths:
    print(p)
