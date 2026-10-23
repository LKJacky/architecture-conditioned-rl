"""Single-GPU smoke test: 0.5B model, four training examples, two eval examples."""

import copy

import torch
from alloylm.algorithm.base import InferArgs
from alloylm.algorithm.rl import rl_algo
from alloylm.algorithm.rl.rl_algo import RLAlgorithmConfig
from alloylm.algorithm.rl.rl_config import DEFAULT_MODEL_NAME
from alloylm.engine.infer_engine.engine import InferEngineConfig
from alloylm.engine.train_engine.train_engine import TrainEngineConfig
from alloylm.engine.train_engine.train_infer_engine import TrainInferEngineConfig
from alloylm.engine.train_engine.utils import FSDPConfig
from alloylm.impl.common import JsonlDatasetConfig
from alloylm.impl.engines.qwen.qwen2_modeling2 import FSDPQwen2ForCausalLMConfig
from alloylm.impl.rl import FilterSampler

from swarr.configs.rl import (
    EVAL_SAMPLE_ARGS,
    REASON_SYSTEM,
    TRAIN_SAMPLE_ARGS,
    EvalMathDataset,
    MathEvalTask,
    MathTrainTask,
    ThinkChatTemplate,
    chunk_loss_func,
    penalty_negative_advantages_per_optimize_step,
)

rl_algo.TaskSampler = FilterSampler

MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
WORK_DIR = "work_dirs/smoke"
INFER_LENGTH = 64
TRAIN_LENGTH = 512

train_sample_args = copy.deepcopy(TRAIN_SAMPLE_ARGS)
train_sample_args.update({"max_tokens": INFER_LENGTH, "timeout": 120})
train_sample_args["extra_body"].update({"max_entropy": 100})

eval_sample_args = copy.deepcopy(EVAL_SAMPLE_ARGS)
eval_sample_args.update({"max_tokens": INFER_LENGTH, "timeout": 120})

train_dataset = JsonlDatasetConfig(
    name="smoke_train",
    file_path="data/smoke/train.jsonl",
    task_cls=MathTrainTask,
    system_prompt=REASON_SYSTEM,
    dataset_cls=EvalMathDataset,
    infer_args=InferArgs(sample_args=train_sample_args),
)

eval_dataset = JsonlDatasetConfig(
    name="smoke_eval",
    file_path="data/smoke/eval.jsonl",
    task_cls=MathEvalTask,
    system_prompt=REASON_SYSTEM,
    dataset_cls=EvalMathDataset,
    infer_args=InferArgs(sample_args=eval_sample_args),
)

config = RLAlgorithmConfig(
    llm_config=FSDPQwen2ForCausalLMConfig(
        path=MODEL_PATH,
        tokenizer_path=MODEL_PATH,
        fsdp_config=FSDPConfig(
            train_mesh={
                "mesh_shape": (1, 1),
                "mesh_dim_names": ["dp", "sp"],
                "device_type": "cuda",
            },
            infer_mesh={
                "mesh_shape": (1, 1),
                "mesh_dim_names": ["dp", "tp"],
                "device_type": "cuda",
            },
            lm_head_dtype=torch.float32,
            shard_dtype=torch.bfloat16,
            recompute_ratio=1.0,
        ),
    ),
    engine_config=TrainInferEngineConfig(
        train_config=TrainEngineConfig(
            max_length=TRAIN_LENGTH,
            work_dir=WORK_DIR,
            num_workers=1,
            total_training_steps=1,
            num_optimize_per_step=1,
            lr=1e-6,
            h_clip=0.28,
            l_clip=0.2,
            loss_func=chunk_loss_func,
            chunk_loss_size=256,
            step_data_process_func=penalty_negative_advantages_per_optimize_step,
        ),
        infer_config=InferEngineConfig(
            model_name=DEFAULT_MODEL_NAME,
            max_prefill_length=128,
            memory_usage=0.5,
            chat_template=ThinkChatTemplate,
            tool_pattern=None,
            sampler_batch_size=2,
        ),
    ),
    datasets=[train_dataset],
    train_sample_ratios=[1.0],
    eval_datasets=[eval_dataset],
    eval_sample_ratio=[1.0],
    eval_interval=1,
    roll_out_bs=2,
    num_rl_group=2,
    max_length=INFER_LENGTH,
    total_training_steps=1,
    filter_group="none",
    async_rollout="task",
    work_dir=WORK_DIR,
    max_concurrency=4,
    checkpoint_interval=1,
    max_checkpoints=1,
    auto_resume=False,
)
