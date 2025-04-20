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
import random

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
            if frame_count > 18000:
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
        print('Finished getting frames')
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
            temp = temp[::self.step_size]
        elif self.method == 'mean':
            temp = self._mean_pooling(temp)
        sample['features'] = temp
        return sample

    def _mean_pooling(self, input_array):
        """
            Performs mean pooling on a 2D NumPy array.

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
            output_array[i,:] = np.mean(padded_array[start_row:end_row, :])

        return output_array
    
class VideoDataset(Dataset):
    def __init__(self, video_root:Path=Path("/mnt/data/dwiepert/data/temporalbench"), dataset='microsoft/TemporalBench', use_dataset:bool=True, split:List[Path] = None, access_token:str=None, 
                 feature_root:Path=Path("/mnt/data/dwiepert/data/video_features"), batch_size:int=16, ckpt:str="OpenGVLab/VideoMAEv2-Large", overwrite:bool=False, use_existing:bool=False,
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
        self.features = {}
        self.overwrite = overwrite
        self.downsample = downsample
        self.to_tensor = to_tensor

        self.use_existing=use_existing
        
        print('Loading features...')
        self._load_features()
        if not self.use_existing:
            self.extractor = MAE_Extractor(ckpt=ckpt, batch_size=batch_size)
            self._extract_features() 
        #print(self.features)

        if self.split is not None:
            self._split_features()
        print('Features loaded.')

        self.files = list(self.features.keys())
        print(f'# of files: {len(self.files)}')

        transforms = [Identity()]
        if self.downsample:
            transforms.append(Downsample(method=downsample_method, step_size=int(1/cutoff_freq)))
        if self.to_tensor:
            transforms.append(ToTensor())

        self.transforms = torchvision.transforms.Compose(transforms)
        
        self._get_maxt()

        s = self.features[self.files[0]].shape
        print(f'Feature shape: {s}')

    def _load_features(self):
        paths = sorted(list(self.feature_root.rglob('*.npz')))
        paths = [p for p in paths if str(self.feature_root) in str(p)]

        for f in paths:
            l = np.load(f)
            key = list(l)[0]
            loaded = l[key]
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
    args = parser.parse_args()

    if args.use_dataset:
        assert args.token is not None

    vid_features = VideoDataset(video_root=args.root_dir, dataset=args.dataset, use_dataset=args.use_dataset,feature_root=args.feature_dir, 
                                batch_size=args.batch_sz, ckpt=args.model_ckpt, overwrite=args.overwrite, use_existing=args.use_existing, access_token=args.token,
                                downsample=args.downsample, downsample_method=args.downsample_method, cutoff_freq=args.cutoff_freq,
                                to_tensor=args.to_tensor)

    # example
    """
    vid_loader = DataLoader(vid_features, batch_size=1)
    for data in tqdm(vid_loader):
        print('Use this to iterate through video dataset')
    """