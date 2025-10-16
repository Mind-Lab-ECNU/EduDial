# # Copyright (c) OpenMMLab. All rights reserved.
# import argparse

# import torch
# from peft import PeftModel
# from transformers import (AutoModelForCausalLM, AutoTokenizer,
#                           CLIPImageProcessor, CLIPVisionModel)

# from xtuner.model.utils import LoadWoInit


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description='Merge a HuggingFace adapter to base model')
#     parser.add_argument('model_name_or_path', help='model name or path')
#     parser.add_argument('adapter_name_or_path', help='adapter name or path')
#     parser.add_argument(
#         'save_dir', help='the directory to save the merged model')
#     parser.add_argument(
#         '--max-shard-size',
#         type=str,
#         default='2GB',
#         help='Only applicable for LLM. The maximum size for '
#         'each sharded checkpoint.')
#     parser.add_argument(
#         '--is-clip',
#         action='store_true',
#         help='Indicate if the model is a clip model')
#     parser.add_argument(
#         '--safe-serialization',
#         action='store_true',
#         help='Indicate if using `safe_serialization`')
#     parser.add_argument(
#         '--device',
#         default='cuda',
#         choices=('cuda', 'cpu', 'auto'),
#         help='Indicate the device')

#     args = parser.parse_args()
#     return args


# def main():
#     args = parse_args()
#     if args.is_clip:
#         with LoadWoInit():
#             model = CLIPVisionModel.from_pretrained(
#                 args.model_name_or_path, device_map=args.device)
#         processor = CLIPImageProcessor.from_pretrained(args.model_name_or_path)
#     else:
#         with LoadWoInit():
#             model = AutoModelForCausalLM.from_pretrained(
#                 args.model_name_or_path,
#                 torch_dtype=torch.float16,
#                 low_cpu_mem_usage=True,
#                 device_map=args.device,
#                 trust_remote_code=True)
#         processor = AutoTokenizer.from_pretrained(
#             args.model_name_or_path, trust_remote_code=True)
#     model_unmerged = PeftModel.from_pretrained(
#         model,
#         args.adapter_name_or_path,
#         device_map=args.device,
#         is_trainable=False,
#         trust_remote_code=True)
#     model_merged = model_unmerged.merge_and_unload()
#     print(f'Saving to {args.save_dir}...')
#     model_merged.save_pretrained(
#         args.save_dir,
#         safe_serialization=args.safe_serialization,
#         max_shard_size=args.max_shard_size)
#     processor.save_pretrained(args.save_dir)
#     print('All done!')


# if __name__ == '__main__':
#     main()

# Copyright (c) OpenMMLab. All rights reserved.
# import argparse
# import os
# import torch
# import torch.distributed as dist
# from peft import PeftModel
# from transformers import (AutoModelForCausalLM, AutoTokenizer,
#                           CLIPImageProcessor, CLIPVisionModel)

# from xtuner.model.utils import LoadWoInit


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description='Merge a HuggingFace adapter to base model')
#     parser.add_argument('model_name_or_path', help='model name or path')
#     parser.add_argument('adapter_name_or_path', help='adapter name or path')
#     parser.add_argument(
#         'save_dir', help='the directory to save the merged model')
#     parser.add_argument(
#         '--max-shard-size',
#         type=str,
#         default='2GB',
#         help='Only applicable for LLM. The maximum size for '
#         'each sharded checkpoint.')
#     parser.add_argument(
#         '--is-clip',
#         action='store_true',
#         help='Indicate if the model is a clip model')
#     parser.add_argument(
#         '--safe-serialization',
#         action='store_true',
#         help='Indicate if using `safe_serialization`')
#     parser.add_argument(
#         '--device',
#         default='cuda',
#         choices=('cuda', 'cpu', 'auto'),
#         help='Indicate the device')

#     args, unknown = parser.parse_known_args()  # 修改为 parse_known_args
#     return args


# def initialize_distributed():
#     if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#         dist.init_process_group(backend='nccl')
#         local_rank = int(os.environ.get('LOCAL_RANK', 0))
#         torch.cuda.set_device(local_rank)
#         print(f"Distributed mode initialized. Local rank: {local_rank}")
#     else:
#         print("Distributed environment variables not found. Running on single GPU or CPU.")


# def main():
#     initialize_distributed()  # 初始化分布式环境
#     args = parse_args()
    
#     if args.is_clip:
#         with LoadWoInit():
#             model = CLIPVisionModel.from_pretrained(
#                 args.model_name_or_path, device_map=args.device)
#         processor = CLIPImageProcessor.from_pretrained(args.model_name_or_path)
#     else:
#         with LoadWoInit():
#             model = AutoModelForCausalLM.from_pretrained(
#                 args.model_name_or_path,
#                 torch_dtype=torch.float16,
#                 low_cpu_mem_usage=True,
#                 device_map=args.device,
#                 trust_remote_code=True)
#         processor = AutoTokenizer.from_pretrained(
#             args.model_name_or_path, trust_remote_code=True)
    
#     model_unmerged = PeftModel.from_pretrained(
#         model,
#         args.adapter_name_or_path,
#         device_map=args.device,
#         is_trainable=False,
#         trust_remote_code=True)
    
#     model_merged = model_unmerged.merge_and_unload()
#     print(f'Saving to {args.save_dir}...')
    
#     model_merged.save_pretrained(
#         args.save_dir,
#         safe_serialization=args.safe_serialization,
#         max_shard_size=args.max_shard_size)
    
#     processor.save_pretrained(args.save_dir)
#     print('All done!')


# if __name__ == '__main__':
#     main()

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import torch
from peft import PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          CLIPImageProcessor, CLIPVisionModel)

from xtuner.model.utils import LoadWoInit


def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge a HuggingFace adapter to base model')
    parser.add_argument('model_name_or_path', help='model name or path')
    parser.add_argument('adapter_name_or_path', help='adapter name or path')
    parser.add_argument(
        'save_dir', help='the directory to save the merged model')
    parser.add_argument(
        '--max-shard-size',
        type=str,
        default='2GB',
        help='Only applicable for LLM. The maximum size for '
             'each sharded checkpoint.')
    parser.add_argument(
        '--is-clip',
        action='store_true',
        help='Indicate if the model is a clip model')
    parser.add_argument(
        '--safe-serialization',
        action='store_true',
        help='Indicate if using `safe_serialization`')
    parser.add_argument(
        '--device',
        default='auto',  # 设置默认设备为 'auto' 以支持多 GPU
        choices=('cuda', 'cpu', 'auto'),
        help='Indicate the device')

    args, unknown = parser.parse_known_args()  # 使用 parse_known_args 以忽略未知参数
    return args


def main():
    args = parse_args()

    if args.is_clip:
        with LoadWoInit():
            model = CLIPVisionModel.from_pretrained(
                args.model_name_or_path, device_map=args.device)
        processor = CLIPImageProcessor.from_pretrained(args.model_name_or_path)
    else:
        with LoadWoInit():
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map=args.device,  # 'auto' 会自动分配到多个 GPU
                trust_remote_code=True)
        processor = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True)

    model_unmerged = PeftModel.from_pretrained(
        model,
        args.adapter_name_or_path,
        device_map=args.device,  # 确保 PEFT 模型也使用 'auto' 分配
        is_trainable=False,
        trust_remote_code=True)

    model_merged = model_unmerged.merge_and_unload()
    print(f'Saving to {args.save_dir}...')

    model_merged.save_pretrained(
        args.save_dir,
        safe_serialization=args.safe_serialization,
        max_shard_size=args.max_shard_size)

    processor.save_pretrained(args.save_dir)
    print('All done!')


if __name__ == '__main__':
    main()