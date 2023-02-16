'''
Pipeline to standardize hyper-parameter tuning
'''
from abc import ABC, abstractmethod
import wandb
import pandas as pd
import numpy as np
import yaml
import sys
from sklearn.metrics import f1_score, roc_auc_score
from src.model.Model import FasterRisk
from sklearn.model_selection import KFold
from tqdm import tqdm
SEED = 474      # use this global seed for any randomness
# NOTE: DO NOT SET np.random.seed(474) inside MODULES, this should be done at the top level

# ----------------------------------------- HELPERS -----------------------------------------
def get_config(path: str):
    with open(path, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    return config
    
def check_np(X, y):
    assert type(X) == np.ndarray and type(y) == np.ndarray, 'must be np.array'
    assert len(X.shape) == 2 and len(y.shape) == 1, 'X needs to be 2D, y is 1D'

def train_test_split(X: np.ndarray, y: np.ndarray, test: float=0.25):
    """
    Create train test split for one set of data (X and y)

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray

    Returns
    -------
        X_train, y_tran, X_test, y_test
    """
    check_np(X, y)
    rands = np.random.permutation(X.shape[0])
    X, y = X[rands], y[rands]
    idx = int(test*X.shape[0])
    X_train, y_train, X_test, y_test = X[idx:], y[idx:], X[:idx], y[:idx]
    
    return X_train, y_train, X_test, y_test
# ------------------------------------------------------------------------------------------
    
class AbstractTuner(ABC):
    
    def __init__(self, config: dict, KF: bool, K: int=10, test: float=0.25, entity: str='dukeds-mimic2023'):
        pass
        
    @abstractmethod
    def tune(self):
        '''
        Used to tune a specifc model (according to the child class) using Weights & Biases
        '''
        pass


class FasterRiskTuner(AbstractTuner):
    
    def __init__(self, config: dict, KF: bool, K: int=10, test: float = 0.25, entity: str = 'dukeds-mimic2023'):
        super().__init__(config, KF, K, test, entity)
        wandb.init(
            entity=entity,
            project=config['project'],
            config=config,
        )
        config = wandb.config
        # initialize parameters
        params = {}
        params['k']=wandb.config.k
        params['select_top_m']=wandb.config.select_top_m
        params['lb']=wandb.config.lb
        params['ub']=wandb.config.ub
        params['gap_tolerance']=wandb.config.gap_tolerance
        params['parent_size']=wandb.config.parent_size
        params['child_size']=wandb.config.child_size
        params['maxAttempts']=wandb.config.maxAttempts
        params['num_ray_search']=wandb.config.num_ray_search
        params['lineSearch_early_stop_tolerance']=wandb.config.lineSearch_early_stop_tolerance
        self.params = params
        self.KF = KF
        self.K = K
        self.test = test
    
    def tune(self, X, y):
        if self.KF == True:
            f1_arr, auc_arr = [], []
            kf = KFold(n_splits=self.K, shuffle=True, random_state=SEED)
            
            for train_idx, test_idx in tqdm(kf.split(X), desc='Running KFold...', file=sys.stdout):
                X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
                model = FasterRisk(ModelHyperparams=self.params)
                model.train(X_train, y_train)
                y_pred = model.predict(X_test)
                f1 = f1_score(y_test, y_pred)
                y_prob = model.predict_proba(X_test)
                roc_auc = roc_auc_score(y_test, y_prob)
                f1_arr.append(f1)
                auc_arr.append(roc_auc)
                
            wandb.log({"F1" : np.mean(f1_arr)})
            wandb.log({"ROC AUC": np.mean(auc_arr)})
        else:   # if no kfold, just do train test split
            model = FasterRisk(ModelHyperparams=self.params)
            X_train, y_train, X_test, y_test = train_test_split(X, y, test=self.test)
            model.train(X_train, y_train)
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            y_prob = model.predict_proba(X_test)
            roc_auc = roc_auc_score(y_test, y_prob)
            wandb.log({"F1" : f1})
            wandb.log({"ROC AUC": roc_auc})



    
    
    
    
