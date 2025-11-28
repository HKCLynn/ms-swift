#!/usr/bin/env python
"""
测试 swift eval 中 run_deploy 的行为
"""
from dataclasses import asdict
import inspect
from swift.llm.argument import EvalArguments, DeployArguments

# 模拟用户的 eval 参数
eval_args = EvalArguments(
    model='/mnt/data/huggingface_downloads/models/qwen/Qwen3-VL-8B-Instruct',
    eval_dataset=['gsm8k'],
    eval_limit=10,
    infer_backend='pt',  # 用户明确设置为 pt
    eval_backend='Native'
)

print("=" * 60)
print("原始 EvalArguments:")
print(f"  infer_backend = {eval_args.infer_backend}")
print()

# 模拟 run_deploy 中的转换逻辑
print("=" * 60)
print("模拟 run_deploy 中的参数转换:")
print()

if isinstance(eval_args, DeployArguments) and eval_args.__class__.__name__ == 'DeployArguments':
    deploy_args = eval_args
    print("  直接使用 DeployArguments")
else:
    args_dict = asdict(eval_args)
    print(f"  转换前 args_dict 中的 infer_backend = {args_dict.get('infer_backend', 'NOT FOUND')}")

    parameters = inspect.signature(DeployArguments).parameters
    print(f"  DeployArguments 的参数列表包含 infer_backend: {'infer_backend' in parameters}")

    # 检查哪些参数会被移除
    removed_keys = []
    for k in list(args_dict.keys()):
        if k not in parameters or args_dict[k] is None:
            removed_keys.append(k)
            args_dict.pop(k)

    print(f"  被移除的参数: {removed_keys[:10]}...")  # 只显示前10个
    print(f"  转换后 args_dict 中的 infer_backend = {args_dict.get('infer_backend', 'NOT FOUND')}")

    deploy_args = DeployArguments(**args_dict)
    print()

print("=" * 60)
print("转换后的 DeployArguments:")
print(f"  infer_backend = {deploy_args.infer_backend}")
print()

# 检查 DeployArguments 的 __post_init__ 是否会修改 infer_backend
print("=" * 60)
print("检查继承链中的默认值:")
for cls in DeployArguments.__mro__:
    if hasattr(cls, '__dataclass_fields__') and 'infer_backend' in cls.__dataclass_fields__:
        field = cls.__dataclass_fields__['infer_backend']
        print(f"  {cls.__name__}.infer_backend 默认值 = {field.default}")
        break
