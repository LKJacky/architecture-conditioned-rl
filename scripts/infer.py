#!/usr/bin/env python3
"""Serve a checkpoint with AlloyLM's OpenAI-compatible inference engine."""

import argparse
import asyncio
import os

import torch
from transformers import AutoTokenizer

from alloylm.engine.infer_engine.engine import InferEngine, InferEngineConfig
from alloylm.engine.spmd import init_dist
from alloylm.engine.train_engine.utils import FSDPConfig
from alloylm.impl.engines.qwen.qwen2_modeling2 import FSDPQwen2ForCausalLMConfig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF checkpoint directory or model id")
    parser.add_argument("--tokenizer", default=None, help="tokenizer path (defaults to --model)")
    parser.add_argument("--model-name", default="ALLOYLM", help="model name exposed by /v1/models")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-prefill-length", type=int, default=8192)
    parser.add_argument("--memory-usage", type=float, default=0.7)
    parser.add_argument("--sampler-batch-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=-1)
    return parser.parse_args()


async def main():
    args = parse_args()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "9001")
    torch.cuda.set_device(0)
    init_dist()

    model_config = FSDPQwen2ForCausalLMConfig(
        path=args.model,
        tokenizer_path=args.tokenizer or args.model,
        window_size=args.window_size,
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
            torch_compile=False,
            shard_dtype=torch.bfloat16,
        ),
    )
    model = model_config.build()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        use_fast=True,
        padding_side="right",
        trust_remote_code=True,
    )
    engine = InferEngine(
        model,
        tokenizer=tokenizer,
        engine_config=InferEngineConfig(
            model_name=args.model_name,
            port=args.port,
            max_prefill_length=args.max_prefill_length,
            memory_usage=args.memory_usage,
            sampler_batch_size=args.sampler_batch_size,
        ),
    )

    await engine.launch()
    print(f"serving {args.model_name} at {engine.url}", flush=True)
    try:
        await engine.wait_closed()
    finally:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
