from src.loader.Loader import Loader
from src.trainer.Tuner import FasterRiskTuner, get_config, train_test_split
import numpy as np

config = get_config('/home/users/mt361/mimic-pipeline/params/FasterRisk/sweep.yaml')
loader = Loader()
df = loader['OASIS']
df = df[df['diagnosis'] != 'NEWBORN']
df = df[df['admission_type'] != 'NEWBORN']
df.loc[df['admission_type'] == 'URGENT', 'admission_type'] = 'EMERGENCY'
df = df[['subject_id', 'icu_mortality', 'admission_type', 'admission_location',
         'insurance', 'age', 'gcs', 'heartrate', 'meanbp', 'resprate',
         'temp', 'urineoutput', 'mechvent', 'electivesurgery']]
df = df.dropna()
lst = ['CLINIC REFERRAL/PREMATURE', 'EMERGENCY ROOM ADMIT', 'TRANSFER FROM HOSP/EXTRAM', 'PHYS REFERRAL/NORMAL DELI']
df = df[df['admission_location'].isin(lst)]
df = df[df['insurance'] != 'Self Pay']
df['icu_mortality'] = df['icu_mortality'].replace({False: -1, True: 1})
df['admission_type'] = df['admission_type'].replace({'EMERGENCY':1, 'ELECTIVE': 0})
df['admission_location'] = df['admission_location'].replace({'CLINIC REFERRAL/PREMATURE': 0,
                                                            'PHYS REFERRAL/NORMAL DELI': 1,
                                                            'EMERGENCY ROOM ADMIT': 2,
                                                            'TRANSFER FROM HOSP/EXTRAM': 3})
df['insurance'] = df['insurance'].replace({'Medicare': 0, 'Private': 1, 'Medicaid': 2, 'Government': 3})
y = df['icu_mortality'].to_numpy()
X = df.drop('icu_mortality', axis=1).to_numpy()

# running hyper-parameter tuning
np.random.seed(474)
X_train, y_train, X_test, y_test = train_test_split(X, y)
config = get_config('/home/users/mt361/mimic-pipeline/params/FasterRisk/sweep.yaml')
tuner = FasterRiskTuner(config, KF=False)
tuner.tune(X_train, y_train)
