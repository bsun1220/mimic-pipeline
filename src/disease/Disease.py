'''
Author: Tony
Classes for each disease in order to find patients and relevant features
'''
from abc import ABC, abstractmethod
from src.loader.Loader import Loader
import pandas as pd

ICD = {}  # global map to store ICD codes
DIAG_DF = pd.read_csv('/usr/xtmp/mimic2023/mimic3/DIAGNOSES_ICD.csv')    
# this is the other table that would allow us to get subject ID

class Disease(ABC):
    
    def __init__(self, path: str=None):
        self.loader = Loader() if path == None else Loader(path)    # each class should have a way to load data
        self.main_df = self.loader['D_ICD_DIAGNOSES']       # each class should have the entire df
    
    @abstractmethod
    def subjectID(self):
        """
        Obtain all UNIQUE subjectIDs associated with one particular disease, type of disease
        is decided by the class
        
        Returns:
            np.1darray that contains all unique subjectIDs
        """
        pass


ICD['sepsis'] = '038'
class Sepsis(Disease):
    """
    For Septicemia
    """
    def __init__(self):
        super().__init__()
        code = ICD['sepsis']
        self.disease_df = self.main_df[self.main_df['ICD9_CODE'].str.startswith(code)]

    def subjectID(self):
        code = self.disease_df['ICD9_CODE']
        filtered = DIAG_DF[DIAG_DF['ICD9_CODE'].isin(code)]
        
        return filtered['SUBJECT_ID'].unique()
    
    
ICD['heart attack'] = '410'
class HeartAttack(Disease):
    """"
    For AMI (Acute Myocardial Infarction)
    """
    def __init__(self):
        super().__init__()
        code = ICD['heart attack']
        self.disease_df = self.main_df[self.main_df['ICD9_CODE'].str.startswith(code)]
    
    def subjectID(self):
        code = self.disease_df['ICD9_CODE']
        filtered = DIAG_DF[DIAG_DF['ICD9_CODE'].isin(code)]
        
        return filtered['SUBJECT_ID'].unique()


ICD['heart failure'] = '428'
class HeartFailure(Disease):
    
    def __init__(self):
        super().__init__()
        code = ICD['heart failure']
        self.disease_df = self.main_df[self.main_df['ICD9_CODE'].str.startswith(code)]
    
    def subjectID(self):
        code = self.disease_df['ICD9_CODE']
        filtered = DIAG_DF[DIAG_DF['ICD9_CODE'].isin(code)]
        
        return filtered['SUBJECT_ID'].unique()
    
    
if __name__=="__main__":
    loader = Loader()
    
    
    