from typing import Dict, Union, List
import argparse
import cv2 
from datasets import load_dataset
import numpy as np
from pathlib import Path
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.decomposition import PCA
import json
import os
import time 
import random
import pickle
##local
from _base_model import BaseModel


def collatefn(batch) -> np.ndarray:
    """
    Custom collate function to put batch together 

    :param batch: batch from DataLoader object
    :return: collated batch in proper format
    """

    feat_list = []
    file_list = []
    max_t = 0
    for b in batch:
        assert len(batch)==1
       # f = torch.transpose(b['features'],0,1)
       # if f.shape[-1] > max_t:
        #    max_t = f.shape[-1]

        feat_list.append(np.squeeze(b['features']))
        file_list.append(b['files'])



    return {'features':feat_list[0], 'files':file_list[0]}

def r2_score(true, pred):
    st = true**2
    sr = (true-pred)**2
    sstot = np.sum(st, axis=0)
    ssres = np.sum(sr, axis=0)
    r2 = 1 - np.mean(np.divide(ssres, sstot))
    return r2

def rmse(true, pred):
    return np.sqrt(np.mean((pred-true)**2))

class residualPCA():
    """
    :param iv: dict, independent variable(s), keys are stimulus names, values are array like features
    :param iv_type: str, type of feature for documentation purposes
    :param save_path: path like, path to save features to (can be cc path or local path)
    :param zscore: bool, indicate whether to zscore
    :param cci_features: cotton candy interface for saving
    :param overwrite: bool, indicate whether to overwrite values
    :param local_path: path like, path to save config to locally if save_path is not local
    """
    def __init__(self, iv:Dict[str,np.ndarray], iv_type:str, fnames:List[str], save_path:Union[str,Path], n_components:int=13,
                 cci_features=None, overwrite:bool=False, local_path:Union[str,Path]=None):
        
        self.n_components = n_components
        self.iv_type = iv_type
        self.model_type='pca'
        self.overwrite=overwrite
        self.fnames=fnames

        self.iv = iv

        ## local path
        self.save_path=Path(save_path)
        if local_path is None or self.cci_features is None:
            self.local_path = self.save_path
        else:
            self.local_path = Path(local_path)
        self.cci_features = cci_features
        if self.cci_features is None:
            self.local_path = self.save_path
        else:
            assert self.local_path is not None, 'Please give a local path to save config files to (json not cotton candy compatible)'

        self.config = self.config = {
             'iv_type': self.iv_type,'iv_shape': self.iv.shape,
             'train_fnames': self.fnames, 'overwrite': self.overwrite
         }

        config_name = self.local_path / f'{self.model_type}_config.json'
        os.makedirs(self.local_path, exist_ok=True)
        with open(str(config_name), 'w') as f:
            json.dump(self.config,f)
        
        if self.cci_features is None:
            self.new_path = self.save_path/'model'
        else:
            self.new_path = self.local_path/'model'

        config_name = self.new_path/ f'{self.model_type}_config.json'
        os.makedirs(self.new_path, exist_ok=True)
        with open(str(config_name), 'w') as f:
            json.dump(self.config,f)

        self.result_paths = {'model': self.new_path/'model', 'scaler': self.new_path/'scaler'}
        
        self.result_paths['train_eval'] = self.new_path / 'train_eval'
        self.result_paths['test_eval'] = self.new_path / 'test_eval'
        self.result_paths['metric'] = {}
        self.result_paths['eval'] = {}
        for f in self.fnames:
            self.result_paths['metric'][f] = self.save_path/f
            self.result_paths['eval'][f] = self.new_path/f"{f}_eval"
        #new_path = self.save_path / 'model'
        #self.result_paths['weights']= new_path /'weights'
        #self.result_paths['emawav'] = {}
        #for f in self.fnames:
        #     save = save_path / 'emawav'
        #     self.result_paths['emawav'][f] = save / f

        self._check_previous()
        st = time.time()
        self._fit()
        et = time.time()
        print(f'Model fit in {et-st} s')

    def _check_previous(self):
        """
        Check if previous models exist - for models saved as .pkl and using scalers
        """
        self.weights_exist = False
        if Path(str(self.result_paths['model'])+'.pkl').exists() and Path(str(self.result_paths['scaler'])+'.pkl').exists(): self.weights_exist=True

        if self.weights_exist and not self.overwrite:
            with open(str(self.result_paths['model'])+'.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open(str(self.result_paths['scaler'])+'.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            self.model=None
            self.scaler=None

    def _save_model(self, model:Union[LogisticRegressionCV, RidgeCV, PCA], scaler:StandardScaler):
        """
        Save a trained model/scaler and train evaluation metrics

        :param model: trained model (RidgeCV or LogisticRegressionCV)
        :param scaler: trained StandardScaler
        :param eval: dictionary with model evaluation metrics like RMSE and R^2 {str:float}
        """
        if self.cci_features:
            print('Model cannot be saved to cci_features. Saving to local path instead')

        # Save the model to a file using pickle
        os.makedirs(self.save_path, exist_ok=True)
        with open(str(self.result_paths['model'])+'.pkl', 'wb') as file:
            pickle.dump(model, file)
        with open(str(self.result_paths['scaler'])+'.pkl', 'wb') as file:
            pickle.dump(scaler, file)
        

    def _save_metrics(self, metric:Union[np.ndarray,dict], fname:str, name:str='metric'):
        """
        Save metrics for a given story

        :param metric: either a numpy array of the correlations,etc. or a dictionary of values
        """
        if fname not in self.result_paths[name]:
            if name == 'eval':
                self.result_paths[name][fname] = self.new_path / fname
            else:
                self.result_paths[name][fname] = self.save_path / fname

        if name == 'eval':
            os.makedirs(str(self.result_paths['eval'][fname].parent), exist_ok=True)
            with open(str(self.result_paths['eval'][fname])+'.json', 'w') as f:
                json.dump(metric,f)
            return 
        
        if self.cci_features:
            #print('features')
            self.cci_features.upload_raw_array(self.result_paths[name][fname], metric)
            #print(self.result_paths['residuals'][fname])
        else:
            #print('not features')
            os.makedirs(self.result_paths[name][fname].parent, exist_ok=True)
            np.save(str(self.result_paths[name][fname])+'.npy', metric)
    
    def eval_model(self, true:np.ndarray, pred:np.ndarray) -> Dict[str,float]:
        """
        Evaluate a model with R-squared and RMSE

        :param true: np.ndarray, true values
        :param pred: np.ndarray, predicted values
        :return metrics: dictionary of values
        """
        r2 = r2_score(true, pred)
        r = rmse(true, pred)
        #f1 = f1_score(true, pred)
        metrics = {'r2': float(r2), 'rmse':float(r)}
        return metrics

    def _fit(self):
        """
        Fit PCA
        """
        if self.weights_exist and not self.overwrite:
            print('Model already fitted and should not be overwritten')
        else:
            self.scaler = StandardScaler().fit(self.iv)
            self.model = PCA(n_components=self.n_components)
            self.model.fit(self.scaler.transform(self.iv))

            self._save_model(self.model, self.scaler)
            eval = {'explained_variance_ratio':[float(f) for f in list(self.model.explained_variance_ratio_)]}
            os.makedirs(str(self.result_paths['train_eval'].parent), exist_ok=True)
            with open(str(self.result_paths['train_eval'])+'.json', 'w') as f:
                json.dump(eval, f)
    
    def score(self, feature, fname:str) -> tuple[np.ndarray, Dict[str,np.ndarray]]:
        """
        Extract PCA features

        :param feats: dict, feature dictionary, stimulus names as keys
        :param ref_feats: dict, feature dictionary of ground truth predicted features, stimulus names as keys
        :param fname: str, name of stimulus to extract for
        :return pca: extracted pca features
        :return: Dictionary of true and predicted values 
        """
        #if self.cci_features is not None:
        #    if self.cci_features.exists_object(self.result_paths['metric'][fname]) and not self.overwrite:
        #        return 
        #else:
        #    if Path(str(self.result_paths['metric'][fname]) + '.npz').exists() and not self.overwrite:
        #        return 
        
        assert self.model is not None, 'PCA has not been run yet. Please do so.'
        
        f = feature

        pca = self.model.transform(self.scaler.transform(f))
        print(pca.shape)

        #save_pca = np.expand_dims(np.swapaxes(pca, 0,1),0)
        self._save_metrics(pca, fname, 'metric')
       

        return pca

class MAE_Extractor():

    def __init__(self, ckpt= "OpenGVLab/VideoMAEv2-Large", batch_size=16, device=torch.device('cuda')):
        self.device = device
        self.batch_size = batch_size
        self.config = AutoConfig.from_pretrained(ckpt, trust_remote_code=True)
        self.processor = VideoMAEImageProcessor.from_pretrained(ckpt)
        self.model = AutoModel.from_pretrained(ckpt, config=self.config, trust_remote_code=True).to(self.device).eval()
        self.target_size = (224, 224)
        self.fnum = 16
    
    def _frame_from_video(self,video):
        frame_count = 0
        while video.isOpened():

            success, frame = video.read()
            frame_count +=1
            if frame_count > 9000:
                break
            key = cv2.waitKey(1)
            if success:
                yield frame
            else:
                break    

    def _vid2list(self, path):
        video = cv2.VideoCapture(path)
        
        frames = video.get(cv2.CAP_PROP_FRAME_COUNT) 
        fps = video.get(cv2.CAP_PROP_FPS) 
        #print(f'FPS: {fps}')
        video.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Set buffer size to 1
        # calculate duration of the video 
        seconds = round(frames / fps) 
        if seconds < 5:
            return None
        
        #max_n_frames 
        #max_nframes = int(fps * (5*60))
        #print(max_nframes)


        frames = [x for x in self._frame_from_video(video)]
        video.release()
        #print('Finished getting frames')
        #if len(frames) > max_nframes:
        #    s = random.randrange((len(frames) - max_nframes))
        #    frames = frames[s:s+max_nframes]
        #    assert len(frames) == max_nframes

        assert (len(frames) >= self.fnum)

        vid_list = [cv2.resize(x[:,:,::-1], self.target_size) for x in frames]
        vid_tube = [np.expand_dims(x, axis=(0, 1)) for x in vid_list]
        vid_tube = np.concatenate(vid_tube, axis=1)
        vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
        vid_tube = np.squeeze(vid_tube)
        
        #vid_tube = np.lib.stride_tricks.sliding_window_view(vid_tube, self.fnum, axis=0)
        # vid_tube = np.transpose(vid_tube, (0,4,1,2,3))
        vid_tube = np.array_split(vid_tube, np.arange(self.fnum, len(vid_tube), self.fnum))

        #vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
        input_vids = [list(v) for v in vid_tube]
        if len(input_vids[-1]) != self.fnum:
            input_vids = input_vids[:-1]
        #for i in range(vid_tube.shape[0]):
        #    input_vids.append(list(np.squeeze(vid_tube[i,:,:,:,:])))
        
        return input_vids
    
    def __call__(self, path):
        vidlist = self._vid2list(path)

        if vidlist is None:
            return None 
        
    

        inputs = self.processor(vidlist, return_tensors="pt")
        
        del vidlist
        
        pixels = inputs['pixel_values'].permute(0, 2, 1, 3, 4)
        
       # print(f'Input shape: {pixels.shape}')
        batches = torch.split(pixels, self.batch_size, dim=0)
        del inputs
        batched_output = []
        with torch.no_grad():
            for b in batches:
                b = b.to(self.device)
                #print(b.shape)
                bo= self.model(pixel_values=b)
                batched_output.append(bo)
                del bo
        
        outputs = torch.cat(batched_output, dim=0).cpu().numpy()
        del batched_output
        del batches
        #print(f'Output shape: {outputs.shape}')
        return outputs
    
class ToTensor():
    """
    Convert sample features/times from numpy to tensors
    """
    def __call__(self, sample:Dict[str,np.ndarray]) -> Dict[str,torch.Tensor]:
        """
        Transform sample
        :param sample: dict, sample
        :return sample: dict, transformed sample
        """
        sample['features'] = torch.from_numpy(sample['features'])
        return sample
    
class Identity():
    """
    Convert sample features/times from numpy to tensors
    """
    def __call__(self, sample:Dict[str,np.ndarray]) -> Dict[str,torch.Tensor]:
        """
        Transform sample
        :param sample: dict, sample
        :return sample: dict, transformed sample
        """
        return sample
    
class Downsample3D():
    """
    """
    def __init__(self, method:str='uniform', step_size:int=5):
        self.method = method
        self.step_size = step_size
        

    def __call__(self, sample:Dict[str, Union[np.ndarray, torch.Tensor]]) -> Dict[str,torch.Tensor]:
        """
        Transform sample
        :param sample: dict, sample
        :return sample: dict, transformed sample
        """
        temp = sample['features']
        if self.method == 'uniform':
            temp = temp[:,:,::self.step_size]
        elif self.method == 'mean':
            temp = self._mean_pooling(temp)
        sample['features'] = temp
        return sample

    def _mean_pooling(self, input_array):
        """
            Performs mean pooling on a 3D NumPy array.

            Args:
                input_array (numpy.ndarray): The input array.
                kernel_size (int): The size of the pooling kernel (both height and width).
                stride (int, optional): The stride of the pooling operation. If None, it defaults to kernel_size.
                

            Returns:
                numpy.ndarray: The pooled output array.
        """
        batch, input_height, input_width = input_array.shape
        padded_array = input_array
        output_width = (input_width - self.step_size) // self.step_size + 1
        output_height = input_height

        output_array = np.zeros((batch, output_height, output_width))

        for i in range(output_width):
            start_row = i * self.step_size
            end_row = start_row + self.step_size
            output_array[:,:,i] = np.mean(padded_array[:,:,start_row:end_row],axis=2)

        return output_array

class Downsample():
    """
    """
    def __init__(self, method:str='uniform', step_size:int=5):
        self.method = method
        self.step_size = step_size
        

    def __call__(self, sample:Dict[str, Union[np.ndarray, torch.Tensor]]) -> Dict[str,torch.Tensor]:
        """
        Transform sample
        :param sample: dict, sample
        :return sample: dict, transformed sample
        """
        temp = sample['features']
        if self.method == 'uniform':
            temp = temp[::self.step_size,:]
        elif self.method == 'mean':
            temp = self._mean_pooling(temp)
        sample['features'] = temp
        return sample

    def _mean_pooling(self, input_array):
        """
            Performs mean pooling on a 3D NumPy array.

            Args:
                input_array (numpy.ndarray): The input array.
                kernel_size (int): The size of the pooling kernel (both height and width).
                stride (int, optional): The stride of the pooling operation. If None, it defaults to kernel_size.
                

            Returns:
                numpy.ndarray: The pooled output array.
        """
        input_height, input_width = input_array.shape
        padded_array = input_array
        output_height = (input_height - self.step_size) // self.step_size + 1
        output_width = input_width

        output_array = np.zeros((output_height, output_width))

        for i in range(output_height):
            start_row = i * self.step_size
            end_row = start_row + self.step_size
            output_array[i,:] = np.mean(padded_array[start_row:end_row,:],axis=0)

        return output_array
    
class VideoDataset(Dataset):
    def __init__(self, video_root:Path=Path(""), dataset='microsoft/TemporalBench', use_dataset:bool=False, split:List[Path] = None, features:Dict[str,np.ndarray]=None, access_token:str=None, 
                 feature_root:Path=Path("/mnt/data/dwiepert/data/video_features"), batch_size:int=16, ckpt:str="OpenGVLab/VideoMAEv2-Large", overwrite:bool=False, use_existing:bool=True,
                 downsample:bool=False, to_tensor:bool=False, cutoff_freq:float=0.2, downsample_method:str="uniform"):
        print('Loading dataset metadata ...')
        self.video_root = Path(video_root)
        if use_dataset:
            self.paths = load_dataset(dataset, token=access_token)['test']
        else:
            paths = self.video_root.rglob('*.mp4')
            #if split is not None:
            #paths = [p for p in paths if p in split]
            self.paths = [p.relative_to(self.video_root) for p in paths]
        print('Dataset metadata loaded.')
        self.feature_root = Path(feature_root)
        self.feature_root.mkdir(parents=True, exist_ok=True)
        self.split = split
        if features is None:
            self.features = {}
        else:
            self.features = features
        self.overwrite = overwrite
        self.downsample = downsample
        self.to_tensor = to_tensor

        self.use_existing=use_existing
        
        print('Loading features...')
        st = time.time()
        if features is None:
            self._load_features()
        if not self.use_existing:
            self.extractor = MAE_Extractor(ckpt=ckpt, batch_size=batch_size)
            self._extract_features() 
        #print(self.features)

        if self.split is not None:
            self._split_features()
        print('Features loaded.')
        et = time.time()
        print(f'Loading time: {et-st} s')
        self.files = list(self.features.keys())
        random.shuffle(self.files)
        print(f'# of files: {len(self.files)}')

        transforms = [Identity()]
        if self.downsample:
            transforms.append(Downsample(method=downsample_method, step_size=int(1/cutoff_freq)))
        if self.to_tensor:
            transforms.append(ToTensor())

        self.transforms = torchvision.transforms.Compose(transforms)
        
        self._get_maxt()
        print(self.features[self.files[0]].shape)

    def _load_features(self):
        paths1 = sorted(list(self.feature_root.rglob('*.npz')))
        paths2 = sorted(list(self.feature_root.rglob('*.npy')))
        paths1 = [p for p in paths1 if str(self.feature_root) in str(p)]
        paths2 = [p for p in paths2 if str(self.feature_root) in str(p)]

        paths = paths1 + paths2 
        print(len(paths))

        for f in paths:
            l = np.load(f)
            if '.npz' in str(f):
                key = list(l)[0]
                loaded = l[key]
            else:
                loaded = l
            self.features[str(f)]= loaded
            del loaded
            del l
        
        del paths
    
    def _split_features(self):
        new_features = {}
        for f in self.features:
            if f in self.split:
                new_features[f] = self.features[f]
        
        self.features = new_features

    def _save_feature(self, feature, new_path):
        dirs = new_path.parent
        dirs.mkdir(parents=True, exist_ok=True)
        np.savez(new_path, feature)
        self.features[str(new_path)] = feature

    def _get_feature(self, path, new_path):
        load_path = self.video_root / path
        if not load_path.exists():
            print(f'{str(load_path)} does not exist.')
            return
        features = self.extractor(str(load_path))
        if features is None:
            return
        self._save_feature(features, new_path)
        del features
    
    def _extract_features(self):
        for item in tqdm(self.paths):
            if isinstance(item, dict):
                path = Path(item["video_name"])
            else:
                path = item
            new_path = self.feature_root / path.with_suffix(".npz")
            run = True
            if new_path.exists() and not self.overwrite: run = False

            if run: self._get_feature(path, new_path)

    def _get_maxt(self):
        self.maxt=0 
        for f in self.features:
            feat = self.features[f]
            if feat.shape[0] > self.maxt:
                self.maxt = feat.shape[0]
            
    def __len__(self) -> int:
        """
        :return: int, length of data
        """
        return len(self.features)

    def __getitem__(self, idx:int) -> Dict[str,np.ndarray]:
        """
        Get item
        
        :param idx: int/List of ints/tensor of indices
        :return: dict, transformed sample - the features will be in form (TIME, FEATURE_DIM)
        """
        f = self.files[idx]
        sample = {'files':f, 'features': self.features[f]}
        #print(sample)
        transformed = self.transforms(sample)
        #print(transformed)
        return transformed
    
    def get_concatenated_features(self, keep=1000):
        feat_list = []
        for i in range(len(self.files)):
            if i >= keep:
                break
            f = self.files[i]
            sample = {'files':f, 'features': self.features[f]}
            feats = self.transforms(sample)
            feat_list.append(feats['features'])
        
        feats = np.concat(feat_list)
        return feats

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=Path, required=True,
                        help='Path to directory with video files.')
    parser.add_argument('--feature_dir', type=Path, required=True,
                        help='Path to directory to load/save features from.')
    parser.add_argument('--batch_sz', type=str, default=8,
                        help='Path to directory to load/save features from.')
    parser.add_argument('--model_ckpt',type=str, default="OpenGVLab/VideoMAEv2-Base")
    parser.add_argument('--dataset', type=str, default = 'microsoft/TemporalBench')
    parser.add_argument('--use_dataset', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--token', type=str, default=None)
    parser.add_argument('--downsample', action='store_true')
    parser.add_argument('--downsample_method', type=str, default='uniform', help="uniform or mean")
    parser.add_argument('--cutoff_freq', type=float, default=0.2)
    parser.add_argument('--to_tensor', action='store_true')
    parser.add_argument('--use_existing', action='store_true')
    parser.add_argument('--save_path', type=Path, default='')
    parser.add_argument('--n_components', type=int, default=50)
    args = parser.parse_args()

    args.save_path = Path(args.save_path)
    if args.use_dataset:
        assert args.token is not None

    pca_features = VideoDataset(video_root=args.root_dir, use_dataset=False,feature_root=args.feature_dir,
                                 overwrite=args.overwrite, use_existing=True,
                                downsample=True, downsample_method='uniform',
                                to_tensor=False)
    
    vid_features = VideoDataset(video_root=args.root_dir, features=pca_features.features, use_dataset=False,feature_root=args.feature_dir, 
                                batch_size=args.batch_sz, overwrite=args.overwrite, use_existing=True,
                                downsample=args.downsample, downsample_method=args.downsample_method,
                                to_tensor=args.to_tensor)
    
    feats = pca_features.get_concatenated_features(keep=5000)

    model = residualPCA(iv=feats,
                        fnames=pca_features.files,
                        iv_type='videomaev2g',
                        save_path=args.save_path,
                        n_components=args.n_components,
                        overwrite=args.overwrite #len(list(feats.keys()))
                        )
    
    #print('Model Trained')

    #model2 = residualPCA(iv=feats,
                                # iv_type='videomaev2g',
                                # save_path=args.save_path,
                                # n_components=args.n_components,
                                # overwrite=args.overwrite, keep=2000
                                # )
    
    #print('Model Trained')

    #model3 = residualPCA(iv=feats,
                                # iv_type='videomaev2g',
                                # save_path=args.save_path,
                                # n_components=args.n_components,
                                # overwrite=args.overwrite, keep=5000
                                # )
    

    print('Model Trained')


    loader = DataLoader(vid_features, batch_size=1, shuffle=False, num_workers=0, collate_fn=collatefn)
    for data in tqdm(loader):
        k = data['files']
        f = data['features']
        pk = Path(k).with_suffix("")
        bn = pk.name
        par = pk.parent.name 
        fname = os.path.join(par, bn)
        _ = model.score(f, fname)

    # example
    """
    vid_loader = DataLoader(vid_features, batch_size=1)
    for data in tqdm(vid_loader):
        print('Use this to iterate through video dataset')
    """