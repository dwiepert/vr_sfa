from pathlib import Path
import os
root=Path('/mnt/data/dwiepert/data/video_features')
paths = root.rglob('*.npz.npz')
for p in paths:
    os.remove(p)
