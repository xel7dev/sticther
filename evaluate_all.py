"""Run each benchmark config sequentially."""

import subprocess
import sys
from pathlib import Path

DATASET_CONFIGS = (
    'cfg_ade20k.py',
    'cfg_city_scapes.py',
    'cfg_coco_object.py',
    'cfg_coco_stuff164k.py',
    'cfg_context59.py',
    'cfg_context60.py',
    'cfg_voc20.py',
    'cfg_voc21.py',
)


def main():
    package_root = Path(__file__).resolve().parent
    evaluate_script = package_root / 'evaluation.py'
    for filename in DATASET_CONFIGS:
        config = package_root / 'configs' / 'datasets' / filename
        print(f'Running {config}', flush=True)
        subprocess.run(
            [sys.executable, str(evaluate_script), '--config', str(config)],
            cwd=package_root,
            check=True)


if __name__ == '__main__':
    main()
