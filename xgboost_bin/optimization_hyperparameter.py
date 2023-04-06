import os

from Toolkit.HyperOptProcessor import HyperOptProcessor
import xgboost_bin.eval as eval

def opt_hyper_param(cfg):
    optimizer = HyperOptProcessor(cfg)

    result, best_param = optimizer.opt(100)
    optimizer.train_all(best_param)
    scores_loc = eval.merge_scores(cfg)
    eval.eval_scores(in_data=scores_loc, out_d=os.path.dirname(scores_loc) , do_plot=True)