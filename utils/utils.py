import os
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
import av
import matplotlib.pyplot as plt

from transformers import AutoProcessor 

from models.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from models.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info
from transformers.feature_extraction_utils import BatchFeature

def get_last_video_idx(input_ids, video_token_id):
    #reverse order 
    last_video_idx = -1
    for i in range(len(input_ids)-1, -1, -1):
        if input_ids[i] == video_token_id:
            last_video_idx = i
            break
    return last_video_idx

def load_model(model_type, target_model_path, draft_model_path):
    if model_type == 'qwen2_5_vl':
        processor = Qwen2_5_VLProcessor.from_pretrained(target_model_path, device_map="auto", torch_dtype=torch.float16)
        target_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            target_model_path, 
            device_map="auto", 
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            #attn_implementation = "flash_attention_2",
        )
        draft_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            draft_model_path, 
            device_map="auto", 
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            #attn_implementation = "flash_attention_2",
        )
    else:
        print("Not supported model type.")

    # video_token_id = model.config.video_token_id
    # print("video_token_id:",video_token_id)

    return target_model, draft_model, processor

def load_data(task, data_num, data_path):
    if task == "VideoDetailCaption":
        data_video = load_dataset(
                "/home/wmk/code/data/VideoDetailCaption",
                split="test",
                # cache_dir=cache_dir,
            ).shuffle(seed=42).select(range(data_num))
        
        def video_exists(example):
            video_path = os.path.join(video_dir, f"{example['video_name']}.mp4")
            return os.path.exists(video_path)

        video_dir = os.path.join(data_path, "Test_Videos")
        filtered_data = data_video.filter(video_exists)
        data_video = filtered_data
    elif task == 'MVBench':
        data_video_1 = load_dataset(
                "/home/wmk/code/data/MVBench",
                'action_sequence',
                split="train",
                #cache_dir=cache_dir,
            ).shuffle(seed=42).select(range(data_num))

        data_video_2 = load_dataset(
                "/home/wmk/code/data/MVBench",
                'action_prediction',
                split="train",
                #cache_dir=cache_dir,
            ).shuffle(seed=42).select(range(data_num))
        
        data_video = concatenate_datasets([data_video_1, data_video_2])
        data_video = data_video.shuffle(seed=42)

        def video_exists(example):
            video_path = os.path.join(video_dir, f"{example['video']}")
            return os.path.exists(video_path)
        
        video_dir = "/home/wmk/code/data/MVBench"
        filtered_data = data_video.filter(video_exists)
        data_video = filtered_data
    elif task == 'MVLU':
        data_video = load_dataset(
                "",
                split="train",
                cache_dir=cache_dir,
            ).shuffle(seed=42).select(range(data_num))
    elif task == 'LongVideoBench':
        data_video = load_dataset(
                "/home/wmk/code/data/LongVideoBench",
                split="test",
                #cache_dir=cache_dir,
            ).shuffle(seed=24).select(range(data_num))
    elif task == 'MMBench':
        data_video = load_dataset(
                "",
                split="train",
                cache_dir=cache_dir,
            ).shuffle(seed=42).select(range(data_num))
    elif task == 'COCO_caption':
        cache_dir = ''
        os.makedirs(cache_dir, exist_ok=True)
        data_video = load_dataset(
                "",
                split="test",
                cache_dir=cache_dir,
            ).shuffle(seed=42).select(range(100))
    else:
        data_video = None

    # print(data_video)
    return data_video

def decode_video(processor, task, data_instance, frame_num=8, model_type='qwen2_5_vl', data_path=None):
    def calculate_fps_for_target_frames(container, target_frames):
            video_stream = container.streams.video[0]
            duration = container.duration / 1000000
            if duration <= 0:
                return 1.0 
            
            required_fps = target_frames / duration
            print(f"INFO: Duration: {duration:.2f}s, frame_num: {target_frames}, fps: {required_fps:.2f}")
            return required_fps
    
    if model_type == 'qwen2_5_vl':
        if task == "VideoDetailCaption":
            video_path = os.path.join(data_path, "Test_Videos/")
            video_name = data_instance["video_name"]
            video_path = video_path + video_name + ".mp4"
            question = data_instance["question"]
        
        elif task == "MVBench":
            video_path = data_path
            video_name = data_instance["video"]
            video_path = video_path + video_name
            question = "Please provide a detailed description of the video, focusing on the main subjects, their actions, and the background scenes."
            
        elif task == 'LongVideoBench':
            video_path = data_path
            video_name = data_instance["video_path"]
            video_path = video_path + video_name
            question = "Please provide a detailed description of the video, focusing on the main subjects, their actions, and the background scenes."
            
        elif task == "MVLU":
            video_reader = data_instance['video']
            total_frames = len(video_reader)
            print("Total frames:", total_frames)

        
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        # print("Total frames:", total_frames)
        
        if total_frames == 0:
            return None

        fps = calculate_fps_for_target_frames(container, frame_num)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file://{video_path}",
                        "max_pixels": 448*448,  
                        "fps": fps, 
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        # messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "video",
        #                 "video":"/home/wmk/code/VLM/data/1408717315-1-192.mp4",
        #                 "fps": fps,
        #             },
        #             {"type": "text", "text": "Describe what happen in the video?"},
        #         ],
        #     }
        # ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to("cuda")
    
    print("INFO: Input length:", inputs['input_ids'].shape[1])
    return inputs, video_inputs

def video_chunk_prefill(whole_inputs, video_inputs, model, kvcache, video_group_size=8, sparse_cache=False):
    whole_inputs = whole_inputs.to(model.device)
    n_video_tokens = (whole_inputs['input_ids'] == model.config.video_token_id).sum().item()
    video_token_idxs = (whole_inputs['input_ids'] == model.config.video_token_id).nonzero(as_tuple=True)[1]
    first_video_token_id_idx = video_token_idxs[0].item()
    position_ids, rope_deltas = model.get_rope_index(
        whole_inputs['input_ids'],
        whole_inputs.get('image_grid_thw', None),
        whole_inputs.get('video_grid_thw', None),
        whole_inputs.get('second_per_grid_ts', None),
        whole_inputs['attention_mask'],
    )
    model.rope_deltas = rope_deltas
    
    temporal_patch_size = 2
    if not video_group_size % temporal_patch_size == 0:
        video_group_size += temporal_patch_size - (video_group_size % temporal_patch_size)
    if video_group_size is not None and video_group_size > 0:
        video_groups = video_inputs[0].split(video_group_size)
        assert all(len(group) % 2 == 0 for group in video_groups), "The video group size should be even."
        video_groups_tokens = [int(n_video_tokens * (len(group) / len(video_inputs[0]))) for group in video_groups]
        video_grid_thw = whole_inputs['video_grid_thw'][0]
        video_groups_grid_thw = []
        for group in video_groups:
            video_groups_grid_thw.append(
                torch.tensor(
                    [(len(group) -1 ) // temporal_patch_size + 1,
                    video_grid_thw[1],
                    video_grid_thw[2]]
                ).unsqueeze(0)
            )
        pixel_values_videos_group_size = round((video_group_size / len(video_inputs[0])) * whole_inputs['pixel_values_videos'].shape[0])
        pixel_values_videos_groups = whole_inputs['pixel_values_videos'].split(pixel_values_videos_group_size)
    
    # preprepare the chunk processing
    past_key_values = kvcache
    past_len = 0
    video_token_idxs = (whole_inputs['input_ids'] == model.config.video_token_id).nonzero(as_tuple=True)[1]
    first_video_token_id_idx = video_token_idxs[0].item()
    video_groups_tokens[0] += first_video_token_id_idx
        
    # start processing the video groups
    for i, pixel_values_videos_groups_i in tqdm(enumerate(pixel_values_videos_groups), 
        desc="Processing video groups", total=len(pixel_values_videos_groups), disable=not False):
        group_i_inputs = {
            "video_grid_thw": video_groups_grid_thw[i],
            "second_per_grid_ts": whole_inputs['second_per_grid_ts'],
            "pixel_values_videos": pixel_values_videos_groups_i,
        }
        group_i_inputs = BatchFeature(data=group_i_inputs)
        group_i_inputs['input_ids'] = whole_inputs['input_ids'][:, past_len:past_len + video_groups_tokens[i]]
        group_i_inputs['attention_mask'] = whole_inputs['attention_mask'][:, :past_len + video_groups_tokens[i]]
        group_i_inputs['past_key_values'] = past_key_values
        group_i_inputs['cache_position'] = torch.arange(group_i_inputs['input_ids'].shape[1], dtype=torch.int64, device=model.device) + past_len
        group_i_inputs['position_ids'] = position_ids[:, :, past_len:past_len + group_i_inputs['input_ids'].shape[1]]
        past_len += video_groups_tokens[i] # only the video group tokens are counted, prompt tokens are not counted
        group_i_inputs = group_i_inputs.to(model.device)
        group_i_inputs['use_cache'] = True
        with torch.no_grad():
            outputs = model(**group_i_inputs,)
    assert past_len < whole_inputs['input_ids'].shape[1], "The past length should be less than the final input length."   
    final_inputs = {
        "input_ids": whole_inputs['input_ids'][:, past_len:],
    }
    final_inputs = BatchFeature(data=final_inputs)
    final_inputs = final_inputs.to(model.device)
    final_inputs['past_key_values'] = past_key_values
    final_inputs['use_cache'] = True

    output = model( **final_inputs, sparse_cache=sparse_cache)
    return output

def convert_attention_to_score(attentions, input_ids, visual_token_id=151646, idx=None):
    # attentions: [num_layers, 1, num_heads, seq_len_q, seq_len_k]
    visual_token_mask = (input_ids == visual_token_id)
    visual_positions = torch.where(visual_token_mask[0])[0]
    n_image_tokens = len(visual_positions)
    print(f"INFO: Video Token Num: {n_image_tokens}")
    
    seq_len_k = attentions[0].shape[3]  
    device = attentions[0].device
    total_attention = torch.zeros(seq_len_k, device=device)
    
    num_layers = len(attentions)
    total_elements = num_layers
    
    for layer_attention in attentions:
        # layer_attention: [1, num_heads, seq_len_q, seq_len_k]
        layer_attention = layer_attention.mean(dim=1)  # [1, seq_len_q, seq_len_k]
        if idx != None:
            layer_attention = layer_attention[:,idx,:] #len(idx)>1
            layer_attention = layer_attention.mean(dim=1) 
        else:
            layer_attention = layer_attention.mean(dim=1)  # [1, seq_len_k]
        layer_attention = layer_attention.squeeze(0)  # [seq_len_k]
        total_attention += layer_attention
    
    avg_attention = total_attention / total_elements
    visual_attention = avg_attention[visual_positions]
    
    return visual_attention.tolist()

def drop_visual_tokens(attentions, inputs, drop_rate=0.5, visual_token_id=151647,output_scores=False, reverse=False, idx=None):
    scores = convert_attention_to_score(attentions, inputs['input_ids'], visual_token_id, idx)
    
    visual_token_mask = (inputs['input_ids'] == visual_token_id)
    visual_positions = torch.where(visual_token_mask[0])[0]
    n_image_tokens = len(visual_positions)
    
    tokens_to_keep = int(n_image_tokens * (1 - drop_rate))
    
    scores_tensor = torch.tensor(scores)
    if reverse == True:
        _ , indices = torch.sort(scores_tensor, descending=False)
    else:
        _, indices = torch.sort(scores_tensor, descending=True)
    keep_indices = indices[:tokens_to_keep]
    keep_indices, _ = torch.sort(keep_indices)
    keep_positions = visual_positions[keep_indices]
    
    non_visual_mask = ~visual_token_mask[0]
    final_mask = non_visual_mask.clone()
    final_mask[keep_positions] = True
    
    new_input_ids = inputs['input_ids'][:, final_mask]
    
    new_inputs = inputs.copy()
    new_inputs['input_ids'] = new_input_ids
    new_inputs['selected_indices'] = keep_indices
    new_inputs['attention_mask'] = None
    
    if output_scores:
        return new_inputs, scores
    return new_inputs
    