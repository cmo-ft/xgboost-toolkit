import os

from Toolkit.TrainProcessor import TrainProcessor



def process_fold(config: dict, fold: int):
    ntuple = os.path.join(config['input_path'], f"fold_{fold}", f"original_hist_{fold}.root")
    train = TrainProcessor(
        ntuple=ntuple, out_dir=config['output_path'],
        k_fold=config['k_fold'], fold=fold,
        variables=config['training_parameters']['variables'],
    )
    
    config['hyper_parameters']['svb_weight_ratio'] = float(config['hyper_parameters']['svb_weight_ratio'])
    config['hyper_parameters']['max_depth'] = int(config['hyper_parameters']['max_depth'])
    config['hyper_parameters']['eta'] = float(config['hyper_parameters']['eta'])
    train.train(config['hyper_parameters'])

    print('--> Done')
