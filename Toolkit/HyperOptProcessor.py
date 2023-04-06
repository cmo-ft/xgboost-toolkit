import uproot as ur
import hyperopt as hopt
import platform
import os

from .TrainProcessor import TrainProcessor
import xgboost_bin.eval as eval

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
            loss = 0
            for trainer in self.trainers:
                ls, _ = trainer.train(train_param, if_save_result=False, verbose=0)
                loss += ls
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
            ls, _ = trainer.train(train_param, if_save_result=True, verbose=1)
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






