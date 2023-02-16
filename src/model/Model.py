'''
For models that need to be tuned and trained (to make our work flow consistent), mostly writing 
wrappers for models so that we can just  do train(), predict() and predict_proba() on all 
of them in the pipeline. This may seem trivial, but it really isn't.
'''
from abc import ABC, abstractmethod
from fasterrisk.fasterrisk import RiskScoreOptimizer, RiskScoreClassifier
import numpy as np

class Model(ABC):
    '''Abstract class that's the parent of all other classes'''
    def __init__(self, ModelHyperparams: dict):
        ...
        pass
    
    @abstractmethod
    def train(self, X, y):
        pass
    
    @abstractmethod
    def predict_proba(self):
        pass


class FastSparse(Model):
    
    def __init__(self, ModelHyperparams):
        super().__init__(ModelHyperparams)


class FasterRisk(Model):
    """
    Wrapper for FasterRisk: https://arxiv.org/pdf/2210.05846.pdf

    Parameters
    ----------
    ModelHyperparams : dict
        dictionary containing the hyper-parameter names as keys and their corresponding values as items
    verbose : bool, optional
        whether prints out number of models after training, by default False
    
    Usage
    -----
    >>> model = FasterRisk()
    >>> model.train(X, y)
    >>> y_pred = model.predict(X)
    >>> y_prob = model.predict_proba(X)
    >>> # to get risk score cards (see Cynthia's paper)
    >>> model.print_model_card()
    >>> # print top K score cards
    >>> model.print_topK_model_cards(feature_names=df.columns, X_train, y_train,
                                    X_test, y_test, K)
    """
    def __init__(self, ModelHyperparams, verbose: bool=False):
        self.optimizer = RiskScoreOptimizer
        self.ModelHyperparams = ModelHyperparams
        self.classifier = None
        self.multipliers = None
        self.beta0 = None
        self.betas = None
        self.trained = False
        self.verbose = verbose
        
    
    def train(self, X: np.ndarray, y: np.ndarray):
        assert not self.trained, 'model already trained'
        dict = self.ModelHyperparams
        # initialzie optimizer, if parameter is None, use default value
        # NOTE: variable that comes after or is the default value
        for key in dict.keys():     # NOTE: this step is to deal with the fact that wandb doesn't allow null
            if dict[key] == 'None':
                dict[key] = None
        
        opt = self.optimizer(
            X=X, 
            y=y, 
            k=dict['k'],
            select_top_m=dict['select_top_m'] or 50,
            lb=dict['lb'] or -5,
            ub=dict['ub'] or 5,
            gap_tolerance=dict['gap_tolerance'] or 0.05,
            parent_size=dict['parent_size'] or 10,
            child_size=dict['child_size'] or None,
            maxAttempts=dict['maxAttempts'] or 50,
            num_ray_search=dict['num_ray_search'] or 20,
            lineSearch_early_stop_tolerance=dict['lineSearch_early_stop_tolerance'] or 0.001,
            )
        
        opt.optimize()      # train
        self.multipliers, self.beta0, self.betas = opt.get_models()     # obtain trained coefficients
        if self.verbose == True:
            print(f"Number of risk score models: {len(self.multipliers)}")
        self.trained = True

    
    def predict_proba(self, X: np.ndarray, model_idx: int=0):
        assert self.trained, 'model not trained yet'
        multiplier = self.multipliers[model_idx]
        beta0 = self.beta0[model_idx]
        betas = self.betas[model_idx]
        
        self.classifier = RiskScoreClassifier(multiplier=multiplier, intercept=beta0, coefficients=betas)
        y_prob = self.classifier.predict_prob(X)
        
        return y_prob
    
    
    def predict(self, X: np.ndarray, model_idx: int=0):
        assert self.trained, 'model not trained yet'
        multiplier = self.multipliers[model_idx]
        beta0 = self.beta0[model_idx]
        betas = self.betas[model_idx]
        
        self.classifier = RiskScoreClassifier(multiplier=multiplier, intercept=beta0, coefficients=betas)
        y_pred = self.classifier.predict(X)

        return y_pred
    
    
    def print_model_card(self, feature_names: list):     # NOTE: this is fasterrisk specific
        assert self.classifier is not None, 'need to do prediction first!'
        self.classifier.reset_featureNames(feature_names)
        self.classifier.print_model_card()
    
    
    def print_topK_model_cards(self, feature_names: list, X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray,  K: int=10):  
        # NOTE: this is fasterrisk specific
        assert type(self.betas) == list, 'model not trained yet'
        num_models = min(K, len(self.multipliers))
        
        for model_index in range(num_models):
            multiplier = self.multipliers[model_index]
            beta0 = self.beta0[model_index]
            betas = self.betas[model_index]

            tmp_classifier = RiskScoreClassifier(multiplier, beta0, betas)
            tmp_classifier.reset_featureNames(feature_names)
            tmp_classifier.print_model_card()

            train_loss = tmp_classifier.compute_logisticLoss(X_train, y_train)
            train_acc, train_auc = tmp_classifier.get_acc_and_auc(X_train, y_train)
            test_acc, test_auc = tmp_classifier.get_acc_and_auc(X_test, y_test)

            print("The logistic loss on the training set is {}".format(train_loss))
            print("The training accuracy and AUC are {:.3f}% and {:.3f}".format(train_acc*100, train_auc))
            print("The test accuracy and AUC are are {:.3f}% and {:.3f}\n".format(test_acc*100, test_auc))
    

class EBM(Model):
    
    def __init__(self, ModelHyperparams):
        super().__init__(ModelHyperparams)


if __name__ == "__main__":
    pass
    