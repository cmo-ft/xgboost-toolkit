import fileinput
import os
import shutil
from pathlib import Path


def init(k_fold: int, workspace_name: str):
    work_path = os.path.join(os.getcwd(), workspace_name)

    print(f'==> Init workspace...')
    print(f'==> Directory: {work_path}')
    print(f'==> k-fold: {k_fold}')

    base_dir = Path(__file__).parent.resolve().parent.absolute()

    if os.path.exists(work_path):
        reply = input(f'==> Remove {work_path}? [y/n]: ')
        if reply == 'y':
            shutil.rmtree(work_path)
    if not os.path.exists(work_path):
        os.makedirs(work_path)

    shutil.copy(os.path.join(base_dir, 'scripts', 'config.yaml'), os.path.join(work_path, 'config.yaml'))

    with fileinput.input((os.path.join(work_path, 'config.yaml')), inplace=True) as file:
        for line in file:
            print(line.replace('k_fold: 4', f'k_fold: {k_fold}'), end='')

    print('==> Init Done!')
    pass
