#!/usr/bin/env python
"""
测试 eval 中的 infer_backend 到底是什么
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm.argument import EvalArguments
from swift.llm.infer.deploy import run_deploy

# 模拟用户的配置
args = EvalArguments(
    model='/mnt/data/huggingface_downloads/models/qwen/Qwen3-VL-8B-Instruct',
    eval_dataset=['gsm8k'],
    eval_limit=2,
    infer_backend='pt',  # 用户明确设置为 pt
    eval_backend='Native'
)

print("=" * 80)
print(f"EvalArguments.infer_backend = {args.infer_backend}")
print("=" * 80)

# 模拟 eval.py 中的 run_deploy 调用
print("\n开始启动 deploy (会启动一个服务器)...")
try:
    with run_deploy(args, return_url=True) as base_url:
        print(f"\n✅ 服务已启动: {base_url}")
        print("请查看上面的日志，看看使用的是什么 backend")
        import time
        time.sleep(2)  # 保持服务运行2秒以便查看日志
except Exception as e:
    print(f"\n❌ 启动失败: {e}")
    import traceback
    traceback.print_exc()
