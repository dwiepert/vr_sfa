"""
Base model for regression/classification

Author(s): Daniela Wiepert
Last modified: 12/04/2024
"""
#IMPORTS
##built-in
import json
import os
import pickle
from pathlib import Path
from typing import Union,Dict
import random
##third-party
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.decomposition import PCA

def r2_score(true, pred):
    st = true**2
    sr = (true-pred)**2
    sstot = np.sum(st, axis=0)
    ssres = np.sum(sr, axis=0)
    r2 = 1 - np.mean(np.divide(ssres, sstot))
    return r2

def rmse(true, pred):
    return np.sqrt(np.mean((pred-true)**2))

class BaseModel:
    """
    Base Model class
    Includes all shared parameters

    :param model_type: str, type of sklearn model
    :param iv: dict, independent variable(s), keys are stimulus names, values are array like features
    :param iv_type: str, type of feature for documentation purposes
    :param dv: dict, dependent variable(s), keys are stimulus names, values are array like features
    :param dv_type: str, type of feature for documentation purposes
    :param config: dict, dictionary of model specific parameters to include in saved configuration
    :param save_path: path like, path to save features to (can be cc path or local path)
    :param cci_features: cotton candy interface for saving
    :param overwrite: bool, indicate whether to overwrite values
    :param local_path: path like, path to save config to locally if save_path is not local
    """
    def __init__(self, model_type:str, iv:Dict[str,Dict[str,np.ndarray]], iv_type:str, dv:Dict[str,Dict[str,np.ndarray]], dv_type:str, config:dict, 
                 save_path:Union[str,Path], cci_features=None, overwrite:bool=False, local_path:Union[str,Path]=None, keep:int=1000):
        
        self.model_type = model_type
        self.iv_type = iv_type
        self.dv_type = dv_type
        self.fnames = list(iv.keys())
        random.shuffle(self.fnames)
        self.fnames = self.fnames[:keep]
        self.overwrite=overwrite

        self.iv, self.iv_rows = self._process_features(iv)
        self.dv, self.dv_rows = self._process_features(dv)
        self._check_rows()

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
             'iv_type': self.iv_type,'iv_shape': self.iv.shape, 'iv_rows': self.iv_rows, 
             'dv_type': self.dv_type, 'dv_shape': self.dv.shape, 'dv_rows':self.dv_rows, 
             'train_fnames': self.fnames, 'overwrite': self.overwrite
         }
        self.config.update(config)

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
        
        #self._check_previous()
    
    def _process_features(self, feat:Dict[str,np.ndarray]) -> tuple[np.ndarray, Dict[str,list], np.ndarray]:
        """
        Concatenate features from separate files into one and maintain information to undo concatenation
        
        :param feat: dict, feature dictionary, stimulus names as keys
        :return concat: np.ndarray, concatenated array
        :return nrows: dict, start/end indices for each stimulus
        :return concat_times: np.ndarray, concatenated tiems array
        """
        nrows = {}
        concat = np.concat(list(feat.values()))
        #print(concat.shape)
        # concat = None 
        # for f in self.fnames:
        #     #print(feat[f].shape)
        #     n = feat[f] #np.squeeze(np.swapaxes(feat[f], 1,2))
        #     if concat is None:
        #         concat = n 
        #         start_ind = 0
        #     else:
        #         if n.ndim ==1:
        #             concat = np.concatenate((concat, n))
        #         else:
        #             concat = np.vstack((concat, n))
        #         start_ind = concat.shape[0]-1
            
        #     end_ind = start_ind + n.shape[0]
        #     nrows[f] = [start_ind, end_ind]
        return concat, nrows

    def _unprocess_features(self, concat:np.ndarray, nrows: Dict[str,list]) -> Dict[str,np.ndarray]:
        """
        Undo concatenation process

        :param concat: np.ndarray, concatenated array
        :param nrows: dict, start/end indices for each stimulus
        :param concat_times: np.ndarray, concatenated tiems array
        :return feats: dict, feature dictionary, stimulus names as keys
        """
        feats = {}
        for f in nrows:
            inds = nrows[f]
            n = concat[inds[0]:inds[1],:]
            feats[f] = n
        return feats

    def _check_rows(self):
        """
        Check that all rows match
        """
        for s in list(self.iv_rows.keys()):
            assert all(np.equal(self.iv_rows[s], self.dv_rows[s])), f'Stimulus {s} has inconsistent sizes. Please check features.'
    
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
        
    