"""Standalone evaluation smoke test using the two-example smoke set."""

from alloylm.algorithm.eval.base import EvalConfig

from swarr.configs.smoke import eval_dataset

config = EvalConfig(
    datasets=[eval_dataset],
    url="http://127.0.0.1:8000/v1",
    model_name="ALLOYLM",
    work_dir="work_dirs/eval_smoke",
)
