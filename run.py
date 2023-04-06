#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys
import os

from xgboost_bin.init import init
from Config.Control import load_config, cfg
from xgboost_bin.process import process_fold
from xgboost_bin.optimization_hyperparameter import opt_hyper_param
import xgboost_bin.eval as eval



if __name__=='__main__':
    base_dir = Path(__file__).parent.resolve().parent.absolute()
    sys.path.append(f'{base_dir}')

    par = argparse.ArgumentParser(prog='xgboost', description='main entrance for xgboost training')
    subparsers = par.add_subparsers(title='modules', help='sub-command help', dest='command')

    # parser for init
    par_init = subparsers.add_parser('init', help='initiate a workspace for xgboost')
    par_init.add_argument('-k', '--k_fold', metavar='n', type=int, required=True, help='define k in k-cv')
    par_init.add_argument('workspace_name', nargs='?', default='default_workspace', type=str, help="workspace name")

    # parser for k_fold_process
    par_single = subparsers.add_parser('single-process', help='run a training for single fold')
    par_single.add_argument('-c', '--config', type=str, help='path to config file')
    par_single.add_argument('-k', '--k_fold', metavar='n', type=int, help='the k-th fold to run')

    # parser for hyperparameter optimization
    par_opt_param = subparsers.add_parser('opt_hyper_param', help='build hyper parameter optimization workspace')
    par_opt_param.add_argument('-c', '--config', type=str, required=True, help='path to config file')

    # parser for evaluation
    par_eval = subparsers.add_parser('eval', help='eval test tree')
    par_eval.add_argument('-c', '--config', type=str, required=True, help='path to config file')
    par_eval.add_argument(
        '--do_plot', action=argparse.BooleanOptionalAction, default=True,
        help='flag for doing evaluation plots'
    )



    args = par.parse_args()

    if args.command == 'init':
        init(**{k: v for k, v in vars(args).items() if k != 'command'})

    if args.command == 'single-process':
        load_config(args.config)
        process_fold(config=cfg, fold=args.k_fold)

    if args.command == 'eval':
        load_config(args.config)
        scores_loc = eval.merge_scores(cfg)
        eval.eval_scores(in_data=scores_loc, out_d=os.path.dirname(scores_loc) , do_plot=True)


    if args.command == 'opt_hyper_param':
        load_config(args.config)
        opt_hyper_param(cfg=cfg)

    else:
        pass
