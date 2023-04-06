import os

import yaml


cfg = {}


def load_config(file_str: str) -> None:
    global cfg

    cfg['_config_abs_path'] = os.path.abspath(file_str)

    with open(file_str) as f:
        cfg.update(yaml.safe_load(f))

        # read in run type
        # try:
        #     if isinstance(cfg['run_type'], str):
        #         cfg['run_type'] = DataType[cfg['run_type']]
        #     if isinstance(cfg['run_type'], int):
        #         cfg['run_type'] = DataType(cfg['run_type'])
        # except KeyError:
        #     type_helper(RunType)
        # except ValueError:
        #     type_helper(RunType)
        # read in k-fold
        if cfg['k_fold'] < 3:
            raise Exception("k-fold number must be greater than 2")


if __name__ == '__main__':
    load_config(r'scripts/config.yaml')

    print(yaml.dump(cfg, default_flow_style=False))
