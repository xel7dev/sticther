"""Command-line evaluation using MMEngine and MMSegmentation."""

import argparse
import json
import os
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

# Importing these modules registers the model and custom dataset types.
from ov_stitcher import datasets as _datasets  # noqa: F401
from ov_stitcher.models import OVStitcherSegmentation as _segmentor  # noqa: F401
from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate OV-Stitcher')
    parser.add_argument('--config', default='configs/datasets/cfg_voc21.py')
    parser.add_argument('--out', help='Save per-image predictions here')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch'],
                        default='none')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    return parser.parse_args(argv)


def resolve_config_path(config_arg):
    config_path = Path(config_arg)
    if config_path.is_absolute():
        return config_path.resolve()

    candidates = (
        Path.cwd() / config_path,
        PACKAGE_ROOT / config_path,
        PROJECT_ROOT / config_path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ', '.join(str(path) for path in candidates)
    raise FileNotFoundError(f'Config {config_arg!r} not found. Searched: {searched}')


def run_evaluation(args):
    """Resolve one config, build the MMSeg test runner, and evaluate it.

    The runner handles dataset iteration and metrics. The custom segmentor is
    intentionally batch-size one because each image has a different number of
    overlapping crops and SAM2 regions.
    """
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    config_path = resolve_config_path(args.config)
    os.chdir(PROJECT_ROOT)
    config = Config.fromfile(str(config_path))
    config.launcher = args.launcher
    if args.cfg_options:
        config.merge_from_dict(args.cfg_options)
    config.work_dir = str(Path('work_dirs') / config_path.stem)
    if args.out:
        config.test_evaluator['output_dir'] = args.out
        config.test_evaluator['keep_results'] = True

    runner = Runner.from_cfg(config)
    metrics = runner.test()
    if runner.rank == 0:
        result = {
            'config': str(config_path),
            'model': config.model.model_type,
            'clip': config.model.clip_type,
            'dino': config.model.dino_type,
            'dataset': config.dataset_type,
            'metrics': metrics,
        }
        result_path = Path(config.work_dir) / 'results.json'
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(f'Results saved to {result_path}')


def main(argv=None):
    run_evaluation(parse_args(argv))


if __name__ == '__main__':
    main()
