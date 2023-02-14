"""
Author: Tony
Date: 02/12/2023
helper for loading datasets
"""
import pandas as pd
import dask.dataframe as dd
from abc import ABC, abstractmethod
#-------------------------------------- COLLECT METHODS ------------------------------------------
METHODS = {}

def pd_load(path, args):
    if type(args[0]) == str:
        result = pd.read_csv(f"{path}/{args[0]}.csv")
    else:
        result = [pd.read_csv(f"{path}/{arg}.csv") for arg in args[0]]
        
    return result

METHODS['pd'] = pd_load

def dd_load(path, args):
    if type(args[0]) == str:
        result = dd.read_csv(f"{path}/{args[0]}.csv", dtype='object')
    else:
        result = [dd.read_csv(f"{path}/{arg}.csv", dtype='object') for arg in args[0]]
        
    return result

METHODS['dd'] = dd_load

def oasis_load(path, args):
    return pd.read_pickle(path)

METHODS['oasis'] = oasis_load
# -----------------------------------------------------------------------------------------
class AbstractLoader():
    
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def __getitem__(self):
        pass

class Loader(AbstractLoader):
    """
    for helping with getting data from folders
    
    Parameters
    ----------
    path (str): 
        the path to which the dataset is stored, default to path in cluster "/usr/xtmp/mimic2023/mimic3"
    
    Usage Example
    -------------
    >>> loader = ClusterLoader()    # specify the path if you are running locally (different from cluster)
    >>> df = loader['CPTEVENTS']    # to obtain CPTEVENTS.csv table
    >>> df1, df2 = loader['ADMISSIONS', 'CPTEVENTS']    # to obtain ADMISSIONS.csv and CPTEVENTS.csv, respectively
    >>> # you can do as many as you want (but be careful with memory usage)
    >>> df1, df2, df3, df4 = loader['ADMISSIONS', 'CPTEVENTS', 'NOTEEVENTS', 'CALLOUT']
    >>> # or as a list
    >>> df_list = loader['ADMISSIONS', 'CPTEVENTS', 'NOTEEVENTS', 'CALLOUT']
    >>> # change mode
    >>> loader.mode('dd')    # change to dask.dataframe (fast for large datasets)
    >>> df = loader['CHARTEVENTS']  # load CHARTEVENTS using dask
    >>> # to check mode
    >>> print(loader.cur_mode)
    >>> 'dd'
    """
    def __init__(self, path: str='/usr/xtmp/mimic2023/mimic3', mode: str='pd') -> None:
        self.path = path
        self.cur_mode = mode
    
    
    def __getitem__(self, *args):
        if args[0] == 'OASIS':
            func = METHODS['oasis']
            result = func('/usr/xtmp/mimic2023/mimic3/oasis_df.pkl', args)
        else:
            func = METHODS[self.cur_mode]   # get right function according to current mode
            result = func(self.path, args)
        
        return result
    
    
    def mode(self, mode: str):
        """
        To alter mode of loading

        Args:
            mode (str): must be in {'pd', 'dd'}, which stands for switching to reading with pandas or dask
        """
        assert mode in ['pd', 'dd'], "mode must be 'pd' or 'dd'"
        self.cur_mode = mode

if __name__ == '__main__':
    # EXAMPLES
    loader = Loader()
    print(loader['OASIS'])