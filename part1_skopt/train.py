import pandas as pd

from utils import train_evaluate

N_ROWS=10000
TRAIN_PATH = '/mnt/ml-team/minerva/open-solutions/santander/data/train.csv'
MODEL_PARAMS = {'boosting': 'gbdt',
                'objective':'binary',
                'metric': 'auc',
                'num_threads': 12,
                'learning_rate': 0.3,
                }

data = pd.read_csv(TRAIN_PATH, nrows=N_ROWS)
    
X = data.drop(['ID_code', 'target'], axis=1)
y = data['target']
    
score = train_evaluate(X, y, MODEL_PARAMS)
print('Validation AUC: {}'.format(score))
    