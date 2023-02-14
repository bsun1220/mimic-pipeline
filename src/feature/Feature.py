from abc import ABC, abstractmethod, abstractproperty

class FeatureType(Enum):
    NUMERIC_TIME = 0
    CATEGORY_TIME = 1
    NUMERIC_STATIC = 2
    CATEGORY_STATIC = 3
    EVENT = 4

class Feature(ABC):

    @abstractmethod
    def get_features():
        """
        Args: None
        
        Returns:
        DataFrame: DataFrame(subject_id, adm_id, icustay_id, [feature columns])
            
        Each subject_id, adm_id, icustay_id will only occur once as a row.
        """
        ...
        

class ScoreFeatures(Feature):
    @abstractmethod
    def get_scores():
        ...
    
    @abstractmethod
    def get_score_prediction():
        ...
    
class OasisFeatures(ScoreFeatures):
    oasis_path = ""
    def __init__(self):
        ...
        
    def get_values(self):
        ...
        
    def get_scores(self):
        ...
        
        
    def get_score_predictions(self):
        ...

class TimeSeriesFeature(Feature):
    def __init__(self, window: int=24):
        self.window = window
        
        
class TimeSeriesNumericFeature(TimeSeriesFeature):
    
    def __init__():
        ...
        
    def get_features():
        ...
        
    def get_measurements():
        ...
    
    def bin_time_series():
        ...
        
    def summary_statistics_time_series():
        ...
        