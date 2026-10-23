"""Open-source AlloyLM configuration for the math RL experiment."""

import copy
import os

import torch
from alloylm.algorithm.base import Dataset, DatasetConfig, InferArgs, TaskData, TaskItem
from alloylm.algorithm.rl import rl_algo
from alloylm.algorithm.rl.rl_algo import RLAlgorithmConfig, RLTrainer
from alloylm.algorithm.rl.rl_config import DEFAULT_MODEL_NAME
from alloylm.engine.infer_engine.engine import InferEngineConfig
from alloylm.engine.train_engine.train_engine import TrainEngineConfig
from alloylm.engine.train_engine.train_infer_engine import TrainInferEngineConfig
from alloylm.engine.train_engine.utils import FSDPConfig
from alloylm.impl.common import (
    JsonlDataset,
    JsonlDatasetConfig,
    OneStepTask,
    extract_boxed,
)
from alloylm.impl.engines.qwen.qwen2_modeling2 import FSDPQwen2ForCausalLMConfig
from alloylm.impl.rl import FilterSampler
from datasets import load_dataset
from math_verify import parse, verify
from torch import distributed as dist

rl_algo.TaskSampler = FilterSampler

INFER_LENGTH = 64 * 1024
TRAIN_LENGTH = 72 * 1024
ROLLOUT_BATCH_SIZE = int(os.environ.get("ROLLOUT_BATCH_SIZE", "128"))
RL_GROUP = 8
CHECKPOINT_INTERVAL = int(os.environ.get("CHECKPOINT_INTERVAL", "100"))
NUM_WORKERS = int(os.environ.get("GPU", "1")) * int(os.environ.get("WORLD_SIZE", "1"))
# Sequence-parallel (SP) degree. Keep the default at 1 unless activations do not fit:
# increase SP to 2/4/8 when GPU memory is tight. SP trades more communication for
# less per-GPU activation memory, and must divide NUM_WORKERS evenly.
SP_SIZE = int(os.environ.get("SP", "1"))
if SP_SIZE < 1:
    raise ValueError(f"SP must be >= 1 (got {SP_SIZE})")
if NUM_WORKERS % SP_SIZE != 0:
    raise ValueError(f"SP={SP_SIZE} must divide GPU * WORLD_SIZE={NUM_WORKERS}")
ENTROPY_LOSS_SCALE = float(os.environ.get("ENTROPY_LOSS_SCALE", "0.1"))
ENTROPY_TARGET = float(os.environ.get("ENTROPY_TARGET", "0.35"))
NEGATIVE_SCALE_FACTOR = float(os.environ.get("NEGATIVE_SCALE_FACTOR", "1.0"))
MODEL_PATH = os.environ.get("LLM", "KaiLiu1998/SWARR-SWA8K-1.5B-SFT")
WORK_DIR = os.environ.get("WORK_DIR", "work_dirs/default")
TOTAL_TRAINING_STEPS = 3000

REASON_SYSTEM = "You are an expert reasoner with extensive experience in reasoning. You approach problems through systematic thinking and rigorous reasoning. Your response should reflect deep understanding and precise logical thinking, making your solution path and reasoning clear to others. Please put your thinking process within <think>...</think> tags"

TRAIN_SAMPLE_ARGS = {
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": INFER_LENGTH,
    "stop": ["<|im_end|>"],
    "timeout": 24 * 60 * 60,
    "extra_body": {
        "for_training": True,
        "top_k": -1,  # disable top_k filtering
        "max_entropy": float(os.environ.get("MAX_ENTROPY", "4")),
    },
}
EVAL_SAMPLE_ARGS = copy.deepcopy(TRAIN_SAMPLE_ARGS)
EVAL_SAMPLE_ARGS.update({"temperature": 0.6})
EVAL_SAMPLE_ARGS["extra_body"].update({"top_k": 40, "max_entropy": 100, "for_training": False})


class ThinkChatTemplate:
    @classmethod
    def render(cls, messages, add_generation_prompt=False, **_kwargs):
        text = ""
        for message in messages:
            text += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n<think>"
        return text


class EvalMathDataset(JsonlDataset):
    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}")

        index %= len(self.data)
        data = self.data[index]
        messages = copy.deepcopy(data.get("messages", data.get("prompt")))
        answer = data.get("answer", data.get("reward_model", {}).get("ground_truth"))
        if self.config.system_prompt:
            system_message = {"role": "system", "content": self.config.system_prompt}
            if messages[0]["role"] == "system":
                messages[0] = system_message
            else:
                messages.insert(0, system_message)

        return TaskItem(
            task_data=TaskData(
                id=f"{self.config.name}_{index}",
                messages=messages,
                others={"origin_data": data, "answer": answer},
                infer_args=self.config.infer_args,
            ),
            task_cls=self.config.task_cls,
        )


class HuggingFaceMathDataset(Dataset):
    def __init__(self, config: "HuggingFaceMathDatasetConfig"):
        super().__init__(config)
        self.data = load_dataset(config.path, split=config.split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}")

        data = self.data[index]
        return TaskItem(
            task_data=TaskData(
                id=f"{self.config.name}_{index}",
                messages=[
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": data["problem"]},
                ],
                others={"origin_data": dict(data), "answer": data["answer"]},
                infer_args=self.config.infer_args,
            ),
            task_cls=self.config.task_cls,
        )


class HuggingFaceMathDatasetConfig(DatasetConfig):
    path: str
    split: str = "train"
    system_prompt: str = ""
    dataset_cls: object = HuggingFaceMathDataset


class MathEvalTask(OneStepTask):
    @classmethod
    async def eval(cls, task_data: TaskData) -> TaskData:
        assert task_data.messages[-1]["role"] == "assistant"
        prediction = extract_boxed(task_data.messages[-1]["content"])
        ground_truth = task_data.others["answer"]
        correct = verify(
            parse(f"\\boxed{{{ground_truth}}}", parsing_timeout=5),
            parse(f"\\boxed{{{prediction}}}", parsing_timeout=5),
            timeout_seconds=5,
        )
        task_data.metric = 1.0 if correct else 0.0
        return task_data


class MathTrainTask(MathEvalTask):
    @classmethod
    async def eval(cls, task_data: TaskData) -> TaskData:
        task_data = await super().eval(task_data)
        if task_data.finish_reason != "stop":
            task_data.metric = 0.0
        return task_data


train_datasets = [
    HuggingFaceMathDatasetConfig(
        name="ace_reason_math",
        path="nvidia/AceReason-Math",
        task_cls=MathTrainTask,
        system_prompt=REASON_SYSTEM,
        infer_args=InferArgs(sample_args=TRAIN_SAMPLE_ARGS),
    )
]

eval_datasets = [
    JsonlDatasetConfig(
        name=filename,
        file_path=f"data/eval/{filename}",
        task_cls=MathEvalTask,
        system_prompt=REASON_SYSTEM,
        dataset_cls=EvalMathDataset,
        repeat=repeat,
        infer_args=InferArgs(sample_args=EVAL_SAMPLE_ARGS),
    )
    for filename, repeat in [
        ("amc.jsonl", 8),
        ("math_500.jsonl", 2),
        ("olympiad.jsonl", 2),
        ("aime25.jsonl", 32),
        ("aime24.jsonl", 32),
    ]
]


def chunk_loss_func(
    logits: torch.Tensor,
    labels: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    loss_weight: torch.Tensor,
    policy_loss_cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert (labels >= 0).all(), "Labels must be non-negative for loss computation"
    log_probs = logits.log_softmax(dim=-1)
    gathered_logprob = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        log_diff = torch.clamp(gathered_logprob - old_logprobs, -20, 20)
        ratio = log_diff.exp()
        ratio_clip = ratio.clamp(max=1 + policy_loss_cfg["cliprange_high"]).detach()

    policy_loss = (-ratio_clip * advantages * gathered_logprob).sum() * loss_weight
    entropy = -(logits.softmax(dim=-1) * log_probs).sum(dim=-1)
    differentiable_entropy = torch.where(entropy < ENTROPY_TARGET, entropy, entropy.detach())
    global_entropy_mean = policy_loss_cfg.get("logprob_diff/entropy_mean", ENTROPY_TARGET)
    if global_entropy_mean < ENTROPY_TARGET:
        entropy_loss = (
            2
            * (global_entropy_mean - ENTROPY_TARGET)
            * differentiable_entropy.sum()
            * loss_weight
            * ENTROPY_LOSS_SCALE
        )
    else:
        entropy_loss = 0.0
    return policy_loss + entropy_loss, entropy


def penalty_negative_advantages_per_optimize_step(step_data):
    if not step_data:
        return step_data

    device = step_data[0]["advantages"].device
    positive_sum = torch.tensor(0.0, device=device)
    negative_sum = torch.tensor(0.0, device=device)
    for batch in step_data:
        masked_advantages = batch["advantages"][batch["labels"] != -100]
        positive_sum += masked_advantages[masked_advantages > 0].sum()
        negative_sum += masked_advantages[masked_advantages < 0].sum()

    if dist.is_available() and dist.is_initialized():
        if dist.get_backend() == dist.Backend.NCCL:
            # Step data can still be on CPU, but NCCL reductions require CUDA tensors.
            reduction_device = torch.device("cuda", torch.cuda.current_device())
            positive_sum = positive_sum.to(reduction_device)
            negative_sum = negative_sum.to(reduction_device)
        dist.all_reduce(positive_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(negative_sum, op=dist.ReduceOp.SUM)
    positive_sum_value = positive_sum.item()
    negative_sum_value = negative_sum.item()

    if positive_sum_value == 0:
        for batch in step_data:
            batch["advantages"] = batch["advantages"] * 0.0
        return step_data[:1]
    if negative_sum_value == 0:
        return step_data

    if NEGATIVE_SCALE_FACTOR < 1.0:
        negative_scale = positive_sum_value / -negative_sum_value
        for batch in step_data:
            batch["advantages"] = batch["advantages"].clone()
            negative = batch["advantages"] < 0
            batch["advantages"][negative] *= negative_scale * NEGATIVE_SCALE_FACTOR
    return step_data


config = RLAlgorithmConfig(
    llm_config=FSDPQwen2ForCausalLMConfig(
        path=MODEL_PATH,
        tokenizer_path=os.environ.get("TOKENIZER") or MODEL_PATH,
        fsdp_config=FSDPConfig(
            train_mesh={
                "mesh_shape": (NUM_WORKERS // SP_SIZE, SP_SIZE),
                "mesh_dim_names": ["dp", "sp"],
                "device_type": "cuda",
            },
            infer_mesh={
                "mesh_shape": (NUM_WORKERS, 1),
                "mesh_dim_names": ["dp", "tp"],
                "device_type": "cuda",
            },
            lm_head_dtype=torch.float32,
            shard_dtype=torch.float32,
            recompute_ratio=1.0,
        ),
    ),
    engine_config=TrainInferEngineConfig(
        train_config=TrainEngineConfig(
            max_length=TRAIN_LENGTH,
            work_dir=WORK_DIR,
            num_workers=NUM_WORKERS,
            total_training_steps=TOTAL_TRAINING_STEPS,
            num_optimize_per_step=int(os.environ.get("NUM_OPTIMIZE_PER_STEP", "4")),
            lr=float(os.environ.get("LR", "1e-6")),
            h_clip=0.28,
            l_clip=0.2,
            loss_func=chunk_loss_func,
            chunk_loss_size=2048,
            step_data_process_func=(
                penalty_negative_advantages_per_optimize_step
                if int(os.environ.get("ADV_NORMALIZATION", "1")) == 1
                else None
            ),
        ),
        infer_config=InferEngineConfig(
            model_name=DEFAULT_MODEL_NAME,
            max_prefill_length=8 * 1024,
            memory_usage=0.7,
            # chat_template=ThinkChatTemplate, 
            tool_pattern=None,
            sampler_batch_size=128,
        ),
    ),
    datasets=train_datasets,
    train_sample_ratios=[1.0],
    eval_datasets=eval_datasets,
    eval_sample_ratio=[1.0, 1.0, 1.0, 1.0, 1.0],
    eval_interval=CHECKPOINT_INTERVAL,
    roll_out_bs=ROLLOUT_BATCH_SIZE,
    num_rl_group=RL_GROUP,
    max_length=INFER_LENGTH,
    total_training_steps=TOTAL_TRAINING_STEPS,
    filter_group="discard",
    async_rollout="task",
    work_dir=WORK_DIR,
    max_concurrency=128 * NUM_WORKERS,
    # data_post_process_func=batch_normalized_advantages,
    checkpoint_interval=CHECKPOINT_INTERVAL,
    max_checkpoints=1,
    auto_resume=True,
)


def get_trainer() -> RLTrainer:
    return RLTrainer(config)
