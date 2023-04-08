import uproot as ur
import xgboost as xgb
import pandas as pd
import numpy as np
import os


class TrainProcessor:
    def __init__(
            self,
            ntuple: str,
            out_dir: str,
            k_fold: int = None,
            fold: int = None,
            variables: list = None,
    ):
        self.ntuple = ntuple
        self.fold_name = f'fold_{fold}'
        self.out_dir = os.path.join(out_dir, self.fold_name)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        self.k_fold = k_fold
        self.fold = fold
        self.variables = variables


        data_file = ur.open(self.ntuple)
        self.sig_train = data_file["sig_train"].arrays(library='pd')
        self.sig_test = data_file["sig_test"].arrays(library='pd')
        self.bkg_train = data_file["bkg_train"].arrays(library='pd')
        self.bkg_test = data_file["bkg_test"].arrays(library='pd')

        self.df_train = pd.concat([self.sig_train, self.bkg_train])
        self.label_train = np.array( [[1]*len(self.sig_train) + [0]*len(self.bkg_train)] )
        self.df_test = pd.concat([self.sig_test, self.bkg_test])
        self.label_test = np.array( [1]*len(self.sig_test) + [0]*len(self.bkg_test) )


    def train(self, hyper_parameters, if_save_result=True, verbose=1):

        self.weight_signal = float(hyper_parameters['svb_weight_ratio'])
        self.weight_bkg = 1

        weight_train = np.concatenate([[self.weight_signal]*len(self.sig_train) + [self.weight_bkg]*len(self.bkg_train)])
        # weight_train = weight_train / weight_train.sum()
        self.dtrain = xgb.DMatrix(data=self.df_train[self.variables].to_numpy(),
                              label=self.label_train,
                              weight=weight_train
                             )

        # no weight on test set
        # weight_test = np.concatenate([[self.weight_signal]*len(sig_test) + [self.weight_bkg]*len(bkg_test)])
        self.dtest = xgb.DMatrix(data=self.df_test[self.variables].to_numpy(),
                                  label=self.label_test,
                                #   weight=weight_test
                                  )

        # Default train param
        import platform
        train_param = {
            "tree_method": "gpu_hist" if platform.system() != 'Darwin' else 'hist',
            # "tree_method": 'hist',
            "objective": "binary:logistic",
            'eval_metric': ['logloss'],
        }
        # train_param.update(hyper_parameters)
        for i in hyper_parameters.keys():
            if i != 'svb_weight_ratio':
                train_param[i] = hyper_parameters[i]

        # Learning task parameters
        task_param = {
            'num_boost_round': 300,
            'early_stopping_rounds': 100,
            'evals': [(self.dtrain, "train"), (self.dtest, "eval")],
            'verbose_eval': False
            # 'verbose_eval': 50 if debug else False,
        }
        bst = xgb.train(train_param, self.dtrain, **task_param)


        # evaluation on test
        train_preds = bst.predict(self.dtrain, iteration_range=(0, bst.best_iteration + 1))
        test_preds = bst.predict(self.dtest, iteration_range=(0, bst.best_iteration + 1))

        loss = bst.eval(self.dtest)
        loss = float(loss.split(':')[1])
        # significance_with_min_bkg, _ = get_significance(test_preds, self.dtest.get_label(), df_test['weight'].to_numpy())
        # loss = -significance_with_min_bkg

        # save result
        if if_save_result:

            bst.save_model(os.path.join(self.out_dir, 'xgboost_model.json'))

            with ur.recreate(os.path.join(self.out_dir, 'xgboost_output.root')) as f:
                test_to_save = self.df_test.copy()
                test_to_save['scores'] = test_preds
                test_to_save['labels'] = self.dtest.get_label()
                f["TestTree"] = test_to_save

                train_to_save = self.df_train.copy()
                train_to_save['scores'] = train_preds
                train_to_save['labels'] = self.dtrain.get_label()
                f["TrainTree"] = train_to_save

            print(f"===> wrote trainning result {self.out_dir}\n")
        if verbose:
            print("===> xgboost training is done!\n")
            print(f"===> loss: {loss}") 
        return loss, bst, test_preds

