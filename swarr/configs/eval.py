"""Standalone evaluation configuration.

This reuses the evaluation datasets and prompt/task classes from the RL
experiment configuration. The model endpoint is supplied by scripts/eval.sh
(or by the --url / --model_name arguments of alloylm.algorithm.eval.run).
"""

from alloylm.algorithm.eval.base import EvalConfig

from swarr.configs.rl import eval_datasets

config = EvalConfig(
    datasets=eval_datasets,
    url="http://127.0.0.1:8000/v1",
    model_name="ALLOYLM",
    work_dir="work_dirs/eval",
)
