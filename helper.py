from pathlib import Path
import shutil
import os
root1=Path('/Users/dwiepert/Documents/SCHOOL/Grad_School/Huth/data/video_sfa/model_innersz768_e2_iek5_d2_idk5_lr0.0001e500bs8_adamw_mse_tvl2_a0.001_earlystop_nonorm/encodings/')
root2 = Path('/Users/dwiepert/Documents/SCHOOL/Grad_School/Huth/data/video_features')
rootpath1 = Path('/Users/dwiepert/Documents/SCHOOL/Grad_School/Huth/data/video_sfa/model_innersz768_e2_iek5_d2_idk5_lr0.0001e500bs8_adamw_mse_tvl2_a0.001_earlystop_nonorm/encodings/')
remove_path1 = '/Users/dwiepert/Documents/SCHOOL/Grad_School/Huth/data/video_features/'

path1 = root1.rglob('*.npz')
path2 = root2.rglob('*.npz')

dict1 = {}
dict2 = {}
for p in path1:
    key = str(p.name)
    dict1[key] = p

for p in path2:
    key = str(p.name)
    val = Path(str(p).replace(remove_path1, '')).parent
    dict2[key] = val 

for p in dict1:
    to_add = dict2[p]
    src = dict1[p]
    dest = rootpath1 / to_add 
    dest.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
