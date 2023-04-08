import os
import json

from Toolkit.HyperOptProcessor import HyperOptProcessor
import xgboost_bin.eval as eval

def opt_hyper_param(cfg):
    optimizer = HyperOptProcessor(cfg)

    result, best_param = optimizer.opt(cfg['optimization']['max_evals'] if (cfg['optimization'].get('max_evals') is not None) else 100)
    # best_param = {'eta': 0.05593108861335488, 'max_depth': 8.0, 'svb_weight_ratio': 4.707355652268655}
    optimizer.train_all(best_param)
    scores_loc = eval.merge_scores(cfg)
    out_dir = os.path.dirname(scores_loc)
    with open(os.path.join(out_dir, 'best_param.json'), 'w') as f:
        json.dump(best_param, f, indent=4)

    eval.eval_scores(in_data=scores_loc, out_d=out_dir, do_plot=True)
