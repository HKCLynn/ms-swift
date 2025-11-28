# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from contextlib import nullcontext
from typing import List, Optional, Union

from evalscope.constants import EvalBackend, EvalType
from evalscope.run import TaskConfig, run_task
from evalscope.summarizer import Summarizer

from swift.utils import append_to_jsonl, get_logger
from .. import MediaResource
from ..argument import EvalArguments
from ..base import SwiftPipeline
from ..infer import run_deploy


from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from typing import Any, Dict, List
import base64
import json
import re

logger = get_logger()


POINT_CLICK_PROMPT = ". Output the coordinate in JSON format."


@register_benchmark(
    BenchmarkMeta(
        # 你自定义的 benchmark 名字
        name="point_click",
        pretty_name="PointClick-On-Mask",
        # 本地 jsonl 文件路径
        dataset_id="/mnt/data/datasets/knowin_datasets/washing_machine_1029/processed_washing_machine_data_all/qwen3_vl_test_bbox.jsonl",
        tags=[Tags.CUSTOM],
        description=(
            "Pointing benchmark: 模型输出坐标 (x, y)，"
            "在对应的 mask 图上，如果该像素值为 1 / 非 0 则计为命中。"
        ),
        subset_list=["default"],  # 本地文件使用 default
        metric_list=["acc"],      # 我们自己在 match_score 里返回 acc
        few_shot_num=0,
        train_split=None,
        eval_split=None,          # 本地文件不需要 split
    )
)

class PointClickAdapter(VisionLanguageAdapter):
    """
    一个自定义的打点评测适配器：
    - record_to_sample: 把原始数据记录转成 Sample
    - extract_answer: 从模型输出中解析坐标 (x, y)
    - match_score: 在 mask 图上检查该点是否命中（像素值==1 / 非0）
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _extract_coordinates_from_json(self, text: str, coord_keys: List[str] = None) -> list:
        """
        从文本中提取坐标，支持嵌套JSON格式。
        优雅地处理：[x, y] 或 [{"bbox_2d": [x, y], ...}] 等格式。
        """
        if coord_keys is None:
            coord_keys = ["point_2d", "bbox_2d", "coordinates", "position"]
        
        try:
            # 找到第一个 '[' 并使用 json.JSONDecoder 来正确解析完整的JSON
            start = text.find('[')
            if start == -1:
                return []
            
            decoder = json.JSONDecoder()
            parsed, _ = decoder.raw_decode(text[start:])
            
            # 如果是字典数组，提取坐标字段
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                for key in coord_keys:
                    if key in parsed[0]:
                        return parsed[0][key]
            # 如果是简单数组，直接返回
            elif isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        
        return []

    def load_from_disk(self, **kwargs):
        """重写此方法以支持本地 jsonl 文件加载"""
        # 如果是单个文件，用文件名作为 subset
        if os.path.isfile(self.dataset_id):
            file_name = os.path.basename(self.dataset_id)
            file_without_ext = os.path.splitext(file_name)[0]
            self.subset_list = [file_without_ext]
        return super().load_from_disk(use_local_loader=True)

    # ---------- 1. 数据读取：record_to_sample ----------

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        原始 record -> Sample

        构造 vLLM 可用的多模态格式：
        - content_list 包含 [ContentImage, ContentText]
        - 对应 vLLM 的 messages 格式：
          {
            "role": "user",
            "content": [
              {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
              {"type": "text", "text": "问题内容"}
            ]
          }
        """
        # 1. 提取 question
        messages = record.get("messages", [])
        question = ""
        for msg in messages:
            if msg.get("role") == "user":
                question = msg.get("content", "")
                question = question + POINT_CLICK_PROMPT
                # INSERT_YOUR_CODE
                # 删掉 <image>\n 这个词
                break

        # 2. 提取 image_path 并构造 content
        image_paths = record.get("images", [])
        base_dir = "/mnt/data/datasets/knowin_datasets/washing_machine_1029/processed_washing_machine_data_all"
        image_path = os.path.join(base_dir, image_paths[0]) if image_paths else None

        # 使用 evalscope 的 Content 类构造多模态内容
        content_list: List[Content] = [ContentText(text=question)]

        if image_path:
            # 读取图片并转换为 base64 data URI
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            image_data_uri = f"data:image/jpeg;base64,{base64_image}"
            content_list.append(ContentImage(image=image_data_uri))

        # 3. 提取 assistant 的 content 里的坐标作为 ground truth
        gt_points = []
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                gt_points = self._extract_coordinates_from_json(content)
                break

        # target 存放 ground truth 坐标点
        target = str(gt_points)

        metadata = {
            "id": record.get("id"),
            "image_path": image_path,  # 可选：保存图片路径用于调试
        }

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=target,
            subset_key=None,
            metadata=metadata,
        )

    # ---------- 2. 从模型输出中解析坐标：extract_answer ----------

    def extract_answer(self, prediction: str, task_state: TaskState) -> list:
        """
        从模型原始输出中提取点坐标，返回坐标列表 [x, y]。
        支持多种格式：
        - 简单数组: [x, y]
        - 嵌套格式: [{"bbox_2d": [x, y], ...}] 或 [{"point_2d": [x, y], ...}]
        如果没法解析出坐标，则返回空列表，让 match_score 判为 0 分。
        """
        return str(self._extract_coordinates_from_json(prediction))

    # ---------- 3. 自定义打分逻辑：match_score ----------

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: Any,
        reference: Any,
        task_state: TaskState,
    ) -> Score:
        """
        打分逻辑：
        - filtered_prediction 是 extract_answer() 的返回值，点坐标 [x, y]
        - reference 是 Sample.target，ground truth bbox [x1, y1, x2, y2]
        - 判断预测的点是否在reference的bbox内部
        """
        score = Score(
            extracted_prediction=str(filtered_prediction),
            prediction=original_prediction,
        )
        # 默认 0 分，避免异常时崩溃
        score.value = {"acc": 0.0}
        score.main_score_name = "acc"

        try:
            # 检查 filtered_prediction 是否有效 (点需要2个坐标)
            if not filtered_prediction or not isinstance(filtered_prediction, (list, tuple)) or len(filtered_prediction) < 2:
                return score

            # 检查 reference 是否有效 (bbox需要4个坐标)
            if not reference or not isinstance(reference, (list, tuple)) or len(reference) < 4:
                return score

            # 解析预测点坐标
            x_pred, y_pred = int(filtered_prediction[0]), int(filtered_prediction[1])
            # 解析真实bbox
            x1, y1, x2, y2 = int(reference[0]), int(reference[1]), int(reference[2]), int(reference[3])

            # 判断预测点是否在bbox内部
            if x1 <= x_pred <= x2 and y1 <= y_pred <= y2:
                is_hit = 1.0
            else:
                is_hit = 0.0

            score.value["acc"] = is_hit
        except Exception:
            # 任何解析错误都返回 0 分
            pass

        return score




class SwiftEval(SwiftPipeline):
    args_class = EvalArguments
    args: args_class

    def run(self):
        args = self.args
        eval_report = {}
        deploy_context = nullcontext() if args.eval_url else run_deploy(args, return_url=True)
        with deploy_context as base_url:
            base_url = args.eval_url or base_url

            task_cfg = self.get_task_cfg(args.eval_dataset, args.eval_backend, base_url)
            result = self.get_task_result(task_cfg)
            eval_report[args.eval_backend] = result

        eval_report.update({
            'time': args.time,
            'model': args.model,
            'adapters': args.adapters,
            'result_path': args.result_path,
            'eval_output_dir': args.eval_output_dir,
            'eval_limit': args.eval_limit
        })

        if args.result_jsonl:
            append_to_jsonl(args.result_jsonl, eval_report)
            logger.info(f'The eval result have been saved to result_jsonl: `{args.result_jsonl}`.')
        return eval_report

    def get_task_result(self, task_cfg: TaskConfig):
        run_task(task_cfg=task_cfg)
        reports = Summarizer.get_report_from_cfg(task_cfg=task_cfg)
        result = {}
        if task_cfg.eval_backend == EvalBackend.OPEN_COMPASS:
            for report in reports:
                if report[self.args.model_suffix] != '-':
                    result[report['dataset']] = {report['metric']: report[self.args.model_suffix]}
        elif task_cfg.eval_backend == EvalBackend.VLM_EVAL_KIT:
            for report in reports:
                splited_key = next(iter(report)).rsplit('_', 2)
                if len(splited_key) == 3:
                    _, dataset, metric = splited_key
                else:
                    dataset, metric = '-', '-'
                result[dataset] = {metric: list(report.values())[0]}
        else:
            result = reports
        return result

    def get_task_cfg(self, dataset: List[str], eval_backend: str, url: str):
        assert eval_backend in {EvalBackend.NATIVE, EvalBackend.OPEN_COMPASS, EvalBackend.VLM_EVAL_KIT}
        if eval_backend == EvalBackend.OPEN_COMPASS:
            if self.args.local_dataset:
                if os.path.exists('data'):
                    if not os.path.exists(os.path.join('data', 'CMB')):
                        raise RuntimeError('Opencompass need a `data` folder in your work dir('
                                           'which will be created automatically by swift eval), '
                                           'but a local path named `data` already exists, '
                                           'please consider moving the dir to another location.')
                else:
                    local_dir = MediaResource.download(
                        'https://modelscope.cn/datasets/'
                        'opencompass/OpenCompassDataComplete/'
                        'resolve/master/OpenCompassData-complete-20240207.zip', 'OpenCompassData')
                    os.symlink(os.path.join(local_dir, 'data'), 'data')

            task_cfg = self.get_opencompass_task_cfg(dataset, url)
        elif eval_backend == EvalBackend.VLM_EVAL_KIT:
            task_cfg = self.get_vlmeval_task_cfg(dataset, url)
        else:
            task_cfg = self.get_native_task_cfg(dataset, url)
        return task_cfg

    def get_native_task_cfg(self, dataset: List[str], url: str):
        args = self.args
        work_dir = os.path.join(args.eval_output_dir, 'native')
        return TaskConfig(
            model=args.model_suffix,
            eval_type=EvalType.SERVICE,
            api_url=url,
            api_key=args.api_key or 'EMPTY',
            datasets=dataset,
            work_dir=work_dir,
            limit=args.eval_limit,
            eval_batch_size=args.eval_num_proc,
            dataset_args=args.eval_dataset_args,
            generation_config=args.eval_generation_config,
            **args.extra_eval_args)

    def get_opencompass_task_cfg(self, dataset: List[str], url: str):
        # Must use chat/completion endpoint
        url = f"{url.rstrip('/')}/chat/completions"

        args = self.args
        work_dir = os.path.join(args.eval_output_dir, 'opencompass')
        return TaskConfig(
            eval_backend=EvalBackend.OPEN_COMPASS,
            eval_config={
                'datasets':
                dataset,
                'batch_size':
                args.eval_num_proc,
                'work_dir':
                work_dir,
                'models': [{
                    'path': args.model_suffix,
                    'openai_api_base': url,
                    'key': args.api_key or 'EMPTY',
                    'is_chat': args.use_chat_template
                }],
                'limit':
                args.eval_limit
            },
            work_dir=work_dir)

    def get_vlmeval_task_cfg(self, dataset: List[str], url: str):
        # Must use chat/completion endpoint
        url = f"{url.rstrip('/')}/chat/completions"

        args = self.args
        work_dir = os.path.join(args.eval_output_dir, 'vlmeval')
        return TaskConfig(
            eval_backend=EvalBackend.VLM_EVAL_KIT,
            eval_config={
                'data':
                dataset,
                'model': [{
                    'type': args.model_suffix,
                    'name': 'CustomAPIModel',
                    'api_base': url,
                    'key': args.api_key or 'EMPTY',
                    **args.eval_generation_config
                }],
                'nproc':
                args.eval_num_proc,
                'limit':
                args.eval_limit
            },
            work_dir=work_dir)


def eval_main(args: Optional[Union[List[str], EvalArguments]] = None):
    return SwiftEval(args).main()
