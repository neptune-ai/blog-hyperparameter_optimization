from collections import OrderedDict

from hyperopt import hp, tpe, fmin, Trials
import neptune
import neptunecontrib.hpo.utils as hpo_utils
import neptunecontrib.monitoring.skopt as sk_utils
import numpy as np
import pandas as pd
from sklearn.externals import joblib

from utils import train_evaluate, axes2fig

neptune.init(project_qualified_name='jakub-czakon/blog-hpo')

N_ROWS = 10000
TRAIN_PATH = '/mnt/ml-team/minerva/open-solutions/santander/data/train.csv'
STATIC_PARAMS = {'boosting': 'gbdt',
                 'objective': 'binary',
                 'metric': 'auc',
                 'num_threads': 12,
                 }
HPO_PARAMS = {'max_evals': 100,
              }

SPACE = OrderedDict([('learning_rate', hp.loguniform('learning_rate', np.log(0.01), np.log(0.5))),
                     ('max_depth', hp.choice('max_depth', range(1, 30, 1))),
                     ('num_leaves', hp.choice('num_leaves', range(2, 100, 1))),
                     ('min_data_in_leaf', hp.choice('min_data_in_leaf', range(10, 1000, 1))),
                     ('feature_fraction', hp.uniform('feature_fraction', 0.1, 1.0)),
                     ('subsample', hp.uniform('subsample', 0.1, 1.0))
                     ])

data = pd.read_csv(TRAIN_PATH, nrows=N_ROWS)

X = data.drop(['ID_code', 'target'], axis=1)
y = data['target']


def objective(params):
    all_params = {**params, **STATIC_PARAMS}
    return -1.0 * train_evaluate(X, y, all_params)


experiment_params = {**STATIC_PARAMS,
                     **HPO_PARAMS,
                     'n_rows': N_ROWS
                     }

with neptune.create_experiment(name='hyperopt sweep',
                               params=experiment_params,
                               tags=['hyperopt', 'tpe'],
                               upload_source_files=['search_hyperopt_tpe.py',
                                                    'search_hyperopt_basic.py',
                                                    'utils.py']):
    trials = Trials()
    _ = fmin(objective, SPACE, trials=trials, algo=tpe.suggest, **HPO_PARAMS)

    results = hpo_utils.hyperopt2skopt(trials, SPACE)

    best_auc = -1.0 * results.fun
    best_params = results.x

    # log metrics
    print('Best Validation AUC: {}'.format(best_auc))
    print('Best Params: {}'.format(best_params))

    neptune.send_metric('validation auc', best_auc)

    # log results
    joblib.dump(trials, 'artifacts/hyperopt_trials.pkl')
    joblib.dump(results, 'artifacts/hyperopt_results.pkl')
    joblib.dump(SPACE, 'artifacts/hyperopt_space.pkl')

    neptune.send_artifact('artifacts/hyperopt_trials.pkl')
    neptune.send_artifact('artifacts/hyperopt_results.pkl')
    neptune.send_artifact('artifacts/hyperopt_space.pkl')

    # log runs
    sk_utils.send_runs(results)
    sk_utils.send_best_parameters(results)
    sk_utils.send_plot_convergence(results, channel_name='diagnostics')
    sk_utils.send_plot_evaluations(results, channel_name='diagnostics')