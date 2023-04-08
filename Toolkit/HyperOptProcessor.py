import uproot as ur
import hyperopt as hopt
import platform
import os
import numpy as np

from .TrainProcessor import TrainProcessor
import xgboost_bin.eval as eval

def get_significance(score, label, weight):
    mask_sig, mask_bkg = (label==1), (label==0)
    score_sig, score_bkg = score[mask_sig], score[mask_bkg]
    weight_sig, weight_bkg = weight[mask_sig], weight[mask_bkg]

    
    bins = np.linspace(0, 1, num=200, endpoint=True)
    hist_sig, _ = np.histogram(score_sig, bins=bins, weights=weight_sig)
    hist_bkg, _ = np.histogram(score_bkg, bins=bins, weights=weight_bkg)
    s = np.cumsum(hist_sig[::-1])[::-1]
    b = np.cumsum(hist_bkg[::-1])[::-1]

    significance = (s / np.sqrt(s + b))
    significance[np.isnan(significance)] = 0
    significance_with_min_bkg = max([(y, x) for x, y in enumerate(significance) if b[x] > 1.0])
    
    return significance_with_min_bkg[0], max(significance)


class HyperOptProcessor:
    def __init__(self, config):
        self.train_param = config['optimization']['hyper_parameters']

        self.kfold = config['k_fold']
        self.variables = config['training_parameters']['variables']
        self.out_dir= config['output_path']

        self.trainers = [ TrainProcessor(os.path.join(config['input_path'], f'fold_{k}', f'original_hist_{k}.root'),
                                         self.out_dir, self.kfold, k, variables=self.variables ) for k in range(self.kfold)]

    def opt(self, max_evals: int = 10):
        def opt_train(args):
            # specify parameters
            train_param = {
                "svb_weight_ratio": float(args['svb_weight_ratio']),
                "max_depth": int(args['max_depth']),
                "eta": float(args['eta']),
            }
            # loss = 0
            scores, labels, weights = [], [], []
            for fold, trainer in enumerate(self.trainers):
                ls, _, score = trainer.train(train_param, if_save_result=False, verbose=0)
                scores.append(score)
                labels.append(trainer.label_test)
                weights.append(trainer.df_test['weight'].to_numpy())
                # loss += ls
            significance, _ = get_significance(np.concatenate(scores), np.concatenate(labels), np.concatenate(weights) )
            loss = -significance

            return {'loss': loss, 'status': hopt.STATUS_OK}


        space = {
            "svb_weight_ratio": hopt.hp.uniform('svb_weight_ratio', *self.train_param['svb_weight_ratio']),
            'max_depth': hopt.hp.quniform("max_depth", *self.train_param['max_depth']),
            'eta': hopt.hp.uniform('eta', *self.train_param['eta']),
        }

        # minimize the objective over the space
        trials = hopt.Trials()

        self.best_hyperparams = hopt.fmin(fn=opt_train,
                                space=space,
                                algo=hopt.tpe.suggest,
                                max_evals=max_evals,
                                trials=trials)

        print("The best hyper parameters are : ")
        print(self.best_hyperparams)

        return trials, self.best_hyperparams

    def train_all(self, args=None):
        args = args if (args is not None) else self.best_hyperparams
        train_param = {
            "svb_weight_ratio": float(args['svb_weight_ratio']),
            "max_depth": int(args['max_depth']),
            "eta": float(args['eta']),
        }

        loss = 0
        outfile_list = []
        for k, trainer in enumerate(self.trainers):
            trainer.out_dir = os.path.join(self.out_dir, f'fold_{k}')
            ls, _, _ = trainer.train(train_param, if_save_result=True, verbose=1)
            print(f'fold_{k} loss: {ls}')
            loss += ls
            outfile_list.append(os.path.join(trainer.out_dir, 'xgboost_output.root'))
        print(f'total loss: {loss}')

        # merge_target = os.path.join(self.out_dir, 'eval/xgboost_output.root')

        # # Loop over the input files and concatenate the TestTrees
        # files = [ur.open(file_name) for file_name in outfile_list]

        # with ur.recreate(merge_target) as output:
        #     output["TestTree"] = ur.concatenate([file['TestTree'] for file in files])
        #     output["TrainTree"] = ur.concatenate([file['TrainTree'] for file in files])






