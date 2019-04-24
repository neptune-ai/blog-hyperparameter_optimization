import matplotlib.pyplot as plt
import neptune
import pandas as pd
import skopt
import skopt.plots
from sklearn.externals import joblib

from utils import train_evaluate, axes2fig

neptune.init(project_qualified_name='jakub-czakon/blog-hpo')

N_ROWS=10000
TRAIN_PATH = '/mnt/ml-team/minerva/open-solutions/santander/data/train.csv'
STATIC_PARAMS = {'boosting': 'gbdt',
                'objective':'binary',
                'metric': 'auc',
                'num_threads': 12,
                }
HPO_PARAMS = {'n_calls':100,
              'n_random_starts':10,
              'acq_func':'PI',
              'xi':0.02,
              'kappa':1.96,
             }

SPACE = [skopt.space.Real(0.01, 0.5, name='learning_rate', prior='log-uniform'),
         skopt.space.Integer(1, 30, name='max_depth'),
         skopt.space.Integer(2, 100, name='num_leaves'),
         skopt.space.Integer(10, 1000, name='min_data_in_leaf'),
         skopt.space.Real(0.1, 1.0, name='feature_fraction', prior='uniform'),
         skopt.space.Real(0.1, 1.0, name='subsample', prior='uniform'),
         ]

data = pd.read_csv(TRAIN_PATH, nrows=N_ROWS)
    
X = data.drop(['ID_code', 'target'], axis=1)
y = data['target']

@skopt.utils.use_named_args(SPACE)
def objective(**params):
    all_params = {**params, **STATIC_PARAMS}
    return -1.0 * train_evaluate(X, y, all_params)

experiment_params = {**STATIC_PARAMS, 
                     **HPO_PARAMS,
                     'n_rows': N_ROWS
                    }

def to_named_params(params):
    return([(dimension.name, param) for dimension, param in zip(SPACE, params)])

def monitor(res):
    neptune.send_metric('run_score', res.func_vals[-1])
    neptune.send_text('run_parameters', str(to_named_params(res.x_iters[-1])))

with neptune.create_experiment(name='skopt gbrt sweep',
                               params=experiment_params,
                               tags=['skopt', 'gbrt'],
                               upload_source_files=['search_gbrt.py', 
                                                    'basic_sweep.py', 
                                                    'utils.py']):
    results = skopt.gbrt_minimize(objective, SPACE,
                                    callback=[monitor],
                                    **HPO_PARAMS)
    best_auc = -1.0 * results.fun
    best_params = results.x
    
    # log metrics
    print('Best Validation AUC: {}'.format(best_auc))
    print('Best Params: {}'.format(best_params))
    
    neptune.send_metric('validation auc', best_auc)
    neptune.set_property('best_params', str(to_named_params(best_params)))
    
    # log results
    skopt.dump(results, 'artifacts/gbrt_results.pkl')
    joblib.dump(SPACE, 'artifacts/gbrt_space.pkl')

    neptune.send_artifact('artifacts/gbrt_results.pkl')
    neptune.send_artifact('artifacts/gbrt_space.pkl')
    
    # log diagnostic plots
    fig, ax = plt.subplots(figsize=(16,12))
    skopt.plots.plot_convergence(results, ax=ax)
    fig.savefig('plots/gbrt_convergence.png')
    
    neptune.send_image('diagnostics', 'plots/gbrt_convergence.png')
   
    axes = skopt.plots.plot_evaluations(results)
    fig = axes2fig(axes, figsize=(16,12))
    fig.savefig('plots/gbrt_evaluations.png')
    
    neptune.send_image('diagnostics', 'plots/gbrt_evaluations.png')

    axes = skopt.plots.plot_objective(results)
    fig = axes2fig(axes, figsize=(16,12))
    fig.savefig('plots/gbrt_objective.png')
    
    neptune.send_image('diagnostics', 'plots/gbrt_objective.png')
