# Codebase: https://github.com/SafeAILab/EAGLE

import json
import time
import copy
import random
from typing import List, Tuple
import torch
import os

from cache.kv_cache import initialize_past_key_values
from cache.draft_cache import initialize_past_key_values_draft
from cache.sparse_cache import initialize_past_key_values_retrieval
from utils.utils_c import generate_tree_buffers_draft
from utils.sampling import sample, norm_logits
from utils.choices import mc_sim_7b_63
from utils.utils import *

TOPK = 10
video_group_size = 32

@torch.no_grad()
def Autoregressive(inputs, video_inputs, target_model, processor, max_new_tokens=128, top_k=-1, top_p=0.9, temperature=0.6):
    torch.cuda.synchronize()
    time1 = time.time()

    (
        past_key_values,
        past_key_values_data,
        current_length_data,
    ) = initialize_past_key_values(target_model)
    target_model.model.past_key_values = past_key_values
    target_model.model.past_key_values_data = past_key_values_data
    target_model.model.current_length_data = current_length_data

    target_model.model.tree_mask = None 

    input_ids = inputs['input_ids']
    batch_size = input_ids.shape[0]
    
    with torch.no_grad():
        output = video_chunk_prefill(inputs, video_inputs, target_model, processor, past_key_values, video_group_size, sparse_cache = True)
        logits = output.logits
        #attentions = output.attentions
        if temperature==0:
            next_token = torch.argmax(logits[:, -1])
            next_token = next_token[None, None]
        else:
            next_token = sample(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))

        generated = torch.cat([inputs['input_ids'], next_token], dim=1)

        torch.cuda.synchronize()
        time2 = time.time()
        
        for step in range(max_new_tokens - 1):
            new_inputs = {
                'input_ids': next_token,
                'past_key_values': past_key_values,
            }
            outputs = target_model(**new_inputs)

            if temperature==0:
                next_token = torch.argmax(outputs.logits[:, -1:], dim=-1)
            else:
                next_token = sample(norm_logits(outputs.logits, temperature=temperature ,top_k=top_k, top_p=top_p))
            generated = torch.cat([generated, next_token], dim=-1)

        torch.cuda.synchronize()
        time3 = time.time()

        result = {
            'output_ids': generated,
            'inference_time': time3 - time1,
            'decoding_time': time3 - time2,
        }
    _cleanup_model_inference_cache(target_model)
    return result

@torch.no_grad()
def speculative_decoding(
        inputs,
        video_inputs,
        target_model,
        draft_model,
        processor,
        max_new_tokens=512,
        log=False,
        tree_choices=mc_sim_7b_63,
        temperature=0.6,
        top_k=-1,
        top_p=0.9,
    ):
        torch.cuda.synchronize()
        infer_start = time.time()

        #Tree Structure
        tree_buffers = generate_tree_buffers(
                tree_choices, device=target_model.model.layers[-1].self_attn.q_proj.weight.device
            )
        tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
                target_model.lm_head.weight.device)
        target_model.tree_buffers = tree_buffers
        target_model.tree_choices = tree_choices

        #Draft Tree Structure
        draft_model.tree_buffer = generate_tree_buffers_draft(
            tree_choices, device=draft_model.model.layers[-1].self_attn.q_proj.weight.device)

        # Initialize the past key values
        (
                past_key_values,
                past_key_values_data,
                current_length_data,
        ) = initialize_past_key_values(target_model)
        target_model.model.past_key_values = past_key_values
        target_model.model.past_key_values_data = past_key_values_data
        target_model.model.current_length_data = current_length_data
        
        (
                draft_past_key_values,
                draft_past_key_values_data,
                draft_current_length_data,
        ) = initialize_past_key_values(draft_model)

        input_ids = inputs['input_ids']
        input_ids = input_ids.clone()
        input_len = input_ids.shape[1]

        #Init
        reset_tree_mode(target_model)
        reset_tree_mode(draft_model)

        #Prefill
        sample_token = initialize_tree(
            inputs,video_inputs, target_model, draft_model, processor, past_key_values, draft_past_key_values, temperature, top_k, top_p
        )

        torch.cuda.synchronize()
        decode_start = time.time()
        
        #First Draft
        first_id = sample_token.to(inputs['input_ids'].device)
        len_posi = inputs['input_ids'].shape[1] + 1
        tree_logits = tree_draft(first_id, draft_model, draft_past_key_values, len_posi)
        target_model.model.tree_mask = tree_buffers["tree_attn_mask"]
        #tree_logits:[11,10]

        new_token = 0
        accept_length_total = []
        for step in range(max_new_tokens):
            candidates, tree_candidates = generate_candidates(
                tree_logits,
                tree_buffers["tree_indices"],
                tree_buffers["retrieve_indices"],
                sample_token,
                processor,
            )

            logits, outputs = tree_decoding(
                target_model,
                tree_candidates,
                past_key_values,
                tree_buffers["tree_position_ids"],
                input_ids,
                tree_buffers["retrieve_indices"],
                )
            
            best_candidate, accept_length, sample_p = evaluate_posterior(
                    logits, candidates, temperature, top_k, top_p,
                )
            accept_length_total.append(accept_length)

            input_ids, tree_logits, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                tree_buffers["retrieve_indices"],
                logits,
                tree_logits,
                new_token,
                past_key_values_data,
                current_length_data,
                draft_model,
                draft_past_key_values,
                draft_past_key_values_data,
                draft_current_length_data,
                sample_p,
                temperature,
                top_k,
                top_p,
            )
            # Currently, we mannually set the generation length for fair comparison.
            if new_token >= max_new_tokens:
                break

        torch.cuda.synchronize()
        end = time.time()
        reset_tree_mode(target_model)
        reset_tree_mode(draft_model)
        if accept_length_total:
            mean_accept = sum(accept_length_total) / len(accept_length_total)
        else:
            mean_accept = torch.tensor(0, device=input_ids.device)
        result = {
            'output_ids': input_ids,
            'inference_time': end - infer_start,
            'decoding_time': end - decode_start,
            'mean_accept_length': mean_accept,
        }
        _cleanup_model_inference_cache(target_model, draft_model)
        return result

@torch.no_grad()
def SD_generate_with_pruning(
        inputs,
        video_inputs,
        model,
        draft_model,
        processor,
        method,
        drop_rate,
        video_token_id=151656,
        max_new_tokens=512,
        log=False,
        tree_choices=mc_sim_7b_63,
        idx=None,
        inputs_drop=None,
        threshold=None,
        percentage=None,
        similarity_threshold=0.95,
        temperature=0.6,
        top_k=-1,
        top_p=0.9,
    ):
        torch.cuda.synchronize()
        infer_start = time.time()

        #Tree Structure
        tree_buffers = generate_tree_buffers(
                tree_choices, device=model.model.layers[-1].self_attn.q_proj.weight.device
            )
        tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
                model.lm_head.weight.device)
        model.tree_buffers = tree_buffers
        model.tree_choices = tree_choices

        #Draft Tree Structure
        draft_model.tree_buffer = generate_tree_buffers_draft(
            tree_choices, device=draft_model.model.layers[-1].self_attn.q_proj.weight.device)

        # Initialize the past key values
        (
                past_key_values,
                past_key_values_data,
                current_length_data,
        ) = initialize_past_key_values(model)
        model.model.past_key_values = past_key_values
        model.model.past_key_values_data = past_key_values_data
        model.model.current_length_data = current_length_data
        
        (
                draft_past_key_values,
                draft_past_key_values_data,
                draft_current_length_data,
        ) = initialize_past_key_values(draft_model)


        #Init
        reset_tree_mode(model)
        reset_tree_mode(draft_model)

        scores = None
        sample_token, input_ids, draft_input_len, scores = initialize_tree_with_pruning(
            inputs, video_inputs, model, draft_model, processor, past_key_values, draft_past_key_values,
            method, video_token_id, drop_rate, idx=idx, inputs_drop=inputs_drop,
            threshold=threshold, percentage=percentage,similarity_threshold=similarity_threshold,
        )


        input_ids = input_ids.clone()
        input_len = input_ids.shape[1]


        torch.cuda.synchronize()
        decode_start = time.time()
        #First Draft
        first_id = sample_token.to(input_ids.device)
        len_posi = draft_input_len + 1
        # len_posi = inputs['input_ids'].shape[1] + 1
        tree_logits = tree_draft(first_id, draft_model, draft_past_key_values, len_posi)
        model.model.tree_mask = tree_buffers["tree_attn_mask"]


        new_token = 0
        accept_length_total = []
        for step in range(max_new_tokens):
            candidates, tree_candidates = generate_candidates(
                tree_logits,
                tree_buffers["tree_indices"],
                tree_buffers["retrieve_indices"],
                sample_token,
                processor,
            )

            logits, outputs = tree_decoding(
                model,
                tree_candidates,
                past_key_values,
                tree_buffers["tree_position_ids"],
                input_ids,
                tree_buffers["retrieve_indices"],
                )
            
            best_candidate, accept_length, sample_p = evaluate_posterior(
                    logits, candidates, temperature, top_k, top_p,
                )
            accept_length_total.append(accept_length)

            input_ids, tree_logits, new_token, hidden_state, sample_token, draft_input_len = update_inference_inputs_with_pruning(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                tree_buffers["retrieve_indices"],
                logits,
                tree_logits,
                new_token,
                past_key_values_data,
                current_length_data,
                draft_model,
                draft_past_key_values,
                draft_past_key_values_data,
                draft_current_length_data,
                sample_p,
                draft_input_len
            )

            # if processor.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            #     reset_tree_mode(model)
            #     reset_tree_mode(draft_model)
            #     torch.cuda.synchronize()
            #     end = time.time()
            #     return {
            #         'output_ids': input_ids,
            #         'inference_time': end - infer_start,
            #         'decoding_time': end - decode_start,
            #         'mean_accept_length': sum(accept_length_total) / len(accept_length_total),
            #     }

            # Currently, we mannually set the generation length for fair comparison.
            if new_token >= max_new_tokens:
                reset_tree_mode(model)
                reset_tree_mode(draft_model)
                torch.cuda.synchronize()
                end = time.time()

                _cleanup_model_inference_cache(model, draft_model)

                #scores = convert_attention_to_score(outputs.attentions, input_ids, video_token_id, idx) #len(idx)>1 
                scores = None
                return {
                    'output_ids': input_ids,
                    'inference_time': end - infer_start,
                    'decoding_time': end - decode_start,
                    'mean_accept_length': sum(accept_length_total) / len(accept_length_total),
                    'scores': scores,
                }

@torch.no_grad()
def sparse_speculative_decoding_TriVLM(
        inputs,
        video_inputs,
        target_model,
        draft_model,
        processor,
        max_new_tokens=512,
        log=False,
        tree_choices=mc_sim_7b_63,
        temperature=0.6,
        top_k=-1,
        top_p=0.9,
    ):
        torch.cuda.synchronize()
        infer_start = time.time()

        #Tree Structure
        tree_buffers = generate_tree_buffers(
                tree_choices, device=target_model.model.layers[-1].self_attn.q_proj.weight.device
            )
        tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
                target_model.lm_head.weight.device)
        target_model.tree_buffers = tree_buffers
        target_model.tree_choices = tree_choices

        #Draft Tree Structure
        draft_model.tree_buffer = generate_tree_buffers_draft(
            tree_choices, device=draft_model.model.layers[-1].self_attn.q_proj.weight.device)

        # Initialize the past key values
        (
                past_key_values,
                past_key_values_data,
                current_length_data,
        ) = initialize_past_key_values(target_model)
        target_model.model.past_key_values = past_key_values
        target_model.model.past_key_values_data = past_key_values_data
        target_model.model.current_length_data = current_length_data

        (
                retrieval_past_key_values,
                retrieval_past_key_values_data,
                retrieval_current_length_data,
        ) = initialize_past_key_values(draft_model)


        (
                draft_past_key_values,
                draft_past_key_values_data,
                draft_current_length_data,
        ) = initialize_past_key_values_retrieval(draft_model)


        #Init
        reset_tree_mode(target_model)
        reset_tree_mode(draft_model)

        scores = None
        sample_token, draft_input_len,= initialize_tree_with_TriVLM(
            inputs, video_inputs, target_model, draft_model, processor, past_key_values, retrieval_past_key_values, draft_past_key_values,
            temperature=temperature, top_k=top_k, top_p=top_p 
        )

        input_ids = inputs['input_ids']
        input_ids = input_ids.clone()
        input_len = input_ids.shape[1]

        torch.cuda.synchronize()
        decode_start = time.time()
        #First Draft
        first_id = sample_token.to(input_ids.device)
        len_posi = draft_input_len + 1
        # len_posi = inputs['input_ids'].shape[1] + 1
        tree_logits = tree_draft(first_id, draft_model, draft_past_key_values, len_posi)
        target_model.model.tree_mask = tree_buffers["tree_attn_mask"]


        new_token = 0
        accept_length_total = []
        for step in range(max_new_tokens):
            candidates, tree_candidates = generate_candidates(
                tree_logits,
                tree_buffers["tree_indices"],
                tree_buffers["retrieve_indices"],
                sample_token,
                processor,
            )

            logits, outputs = tree_decoding(
                target_model,
                tree_candidates,
                past_key_values,
                tree_buffers["tree_position_ids"],
                input_ids,
                tree_buffers["retrieve_indices"],
                )
            
            best_candidate, accept_length, sample_p = evaluate_posterior(
                    logits, candidates, temperature, top_k, top_p,
                )
            accept_length_total.append(accept_length)

            input_ids, tree_logits, new_token, hidden_state, sample_token, draft_input_len = update_inference_inputs_with_pruning(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                tree_buffers["retrieve_indices"],
                logits,
                tree_logits,
                new_token,
                past_key_values_data,
                current_length_data,
                draft_model,
                draft_past_key_values,
                draft_past_key_values_data,
                draft_current_length_data,
                sample_p,
                draft_input_len,
                temperature,
                top_k,
                top_p,
            )

            if new_token >= max_new_tokens:
                break

        torch.cuda.synchronize()
        end = time.time()
        reset_tree_mode(target_model)
        reset_tree_mode(draft_model)
        if accept_length_total:
            mean_accept = sum(accept_length_total) / len(accept_length_total)
        else:
            mean_accept = torch.tensor(0, device=input_ids.device)
        result = {
            'output_ids': input_ids,
            'inference_time': end - infer_start,
            'decoding_time': end - decode_start,
            'mean_accept_length': mean_accept,
            'scores': scores,
        }
        _cleanup_model_inference_cache(target_model, draft_model)
        return result

class Timer:
    def __init__(self,name):
        self.name = name
    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.perf_counter()


    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start
        #print(f'{self.name} took {elapsed} seconds')

def pad_path(path: List[int], length: int, pad_value: int = -2) -> List[int]:
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))

def generate_tree_buffers(tree_choices, device="cuda"):
    def custom_sort(lst):
        # sort_keys=[len(list)]
        sort_keys = []
        for i in range(len(lst)):
            sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
        return sort_keys
    with Timer("sort"):

        sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
        tree_len = len(sorted_tree_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
        depth_counts = []
        prev_depth = 0
        for path in sorted_tree_choices:
            depth = len(path)
            if depth != prev_depth:
                depth_counts.append(0)
            depth_counts[depth - 1] += 1
            prev_depth = depth

        tree_attn_mask = torch.eye(tree_len, tree_len)
        tree_attn_mask[:, 0] = 1
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                # retrieve ancestor position
                if len(cur_tree_choice) == 1:
                    continue
                ancestor_idx = []
                for c in range(len(cur_tree_choice) - 1):
                    ancestor_idx.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]) + 1)
                tree_attn_mask[j + start + 1, ancestor_idx] = 1
            start += depth_counts[i]

        tree_indices = torch.zeros(tree_len, dtype=torch.long)
        p_indices = [0 for _ in range(tree_len - 1)]
        b_indices = [[] for _ in range(tree_len - 1)]
        tree_indices[0] = 0
        start = 0
        bias = 0
        for i in range(len(depth_counts)):
            inlayer_bias = 0
            b = []
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                cur_parent = cur_tree_choice[:-1]
                if j != 0:
                    if cur_parent != parent:
                        bias += 1
                        inlayer_bias += 1
                        parent = cur_parent
                        b = []
                else:
                    parent = cur_parent
                tree_indices[start + j + 1] = cur_tree_choice[-1] + TOPK * (i + bias) + 1
                p_indices[start + j] = inlayer_bias
                if len(b) > 0:
                    b_indices[start + j] = copy.deepcopy(b)
                else:
                    b_indices[start + j] = []
                b.append(cur_tree_choice[-1] + TOPK * (i + bias) + 1)
            start += depth_counts[i]

        p_indices = [-1] + p_indices
        tree_position_ids = torch.zeros(tree_len, dtype=torch.long)
        start = 0
        for i in range(len(depth_counts)):
            tree_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
            start += depth_counts[i]

        retrieve_indices_nest = []
        retrieve_paths = []
        for i in range(len(sorted_tree_choices)):
            cur_tree_choice = sorted_tree_choices[-i - 1]
            retrieve_indice = []
            if cur_tree_choice in retrieve_paths:
                continue
            else:
                for c in range(len(cur_tree_choice)):
                    retrieve_indice.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]))
                    retrieve_paths.append(cur_tree_choice[:c + 1])
            retrieve_indices_nest.append(retrieve_indice)
        max_length = max([len(x) for x in retrieve_indices_nest])
        retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        retrieve_indices = retrieve_indices + 1
        retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices],
                                     dim=1)

        maxitem = retrieve_indices.max().item() + 5



        retrieve_indices = retrieve_indices.tolist()
        retrieve_indices = sorted(retrieve_indices, key=custom_sort)
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)



    # Aggregate the generated buffers into a dictionary
    tree_buffers = {
        "tree_attn_mask": tree_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": tree_indices,
        "tree_position_ids": tree_position_ids,
        "retrieve_indices": retrieve_indices,
    }

    # Move the tensors in the dictionary to the specified device
    tree_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v, device=device)
        for k, v in tree_buffers.items()
    }

    return tree_buffers

def reset_tree_mode(
        model,
):
    model.model.tree_mask = None
    model.model.tree_mode = None

def generate_candidates(tree_logits, tree_indices, retrieve_indices, sample_token, logits_processor):
    sample_token = sample_token.to(tree_indices.device)

    candidates_logit = sample_token[0]

    candidates_tree_logits = tree_logits

    candidates = torch.cat([candidates_logit, candidates_tree_logits.to(candidates_logit.device).view(-1)], dim=-1)

    tree_candidates = candidates[tree_indices]

    tree_candidates_ext = torch.cat(
        [tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device) - 1], dim=0)

    cart_candidates = tree_candidates_ext[retrieve_indices]


    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    return cart_candidates,  tree_candidates

def initialize_tree(inputs,video_inputs, target_model, draft_model, processor, past_key_values, draft_past_key_values,
                     temperature=0.6, top_k=-1, top_p=0.9):
    output = video_chunk_prefill(inputs, video_inputs, target_model, processor, past_key_values, video_group_size)
    logits = output.logits
    if temperature==0:
        sample_token = torch.argmax(logits[:, -1])
        sample_token = sample_token[None, None]
    else:
        sample_token = sample(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))
    output_draft = video_chunk_prefill(inputs, video_inputs, draft_model, processor, draft_past_key_values, video_group_size, sparse_cache=True)
    return sample_token

def initialize_tree_with_pruning(inputs, video_inputs, model, draft_model, processor, past_key_values, draft_past_key_values,
                              method=None, video_token_id=151656, drop_rate=None, idx=None, inputs_drop=None, threshold=None, percentage=None, similarity_threshold=0.95):
    # #Find the last video_token
    # last_video_idx = get_last_video_idx(inputs['input_ids'][0], video_token_id)
    # text_input_ids = inputs['input_ids'][:,last_video_idx+1:].clone()
    # text_attention_mask = inputs['attention_mask'][:,last_video_idx+1:].clone()

    input_ids = inputs['input_ids'].clone()
    # inputs['input_ids'] = inputs['input_ids'][:, :last_video_idx+1]
    # inputs['attention_mask'] = inputs['attention_mask'][:, :last_video_idx+1]

    # #First stage of prefilling video tokens
    # output1 = model(
    #     **inputs, past_key_values=past_key_values
    # )

    # #Second stage of prefilling text tokens
    # output2 = model(
    #     input_ids=text_input_ids, 
    #     past_key_values=past_key_values, 
    #     output_attentions=True,
    # )
    output = video_chunk_prefill(inputs, video_inputs, model, processor, past_key_values, video_group_size,output_attentions=True)
    logits = output.logits
    attentions = output.attentions
    # text_emb = output2.output_embeddings
    sample_token = torch.argmax(logits[:, -1])
    sample_token = sample_token[None, None]

    #Prefill of Draft Model
    # inputs['input_ids'] = torch.cat([inputs['input_ids'], text_input_ids], dim=1)

    scores = None

    if method == 'specvlm':
        inputs_drop = drop_visual_tokens_specvlm(attentions, inputs, drop_rate=drop_rate,
                                    visual_token_id=video_token_id, idx=idx, threshold=threshold, percentage=percentage)
    else:
        print("Method not supported")

    draft_input_len = inputs_drop['input_ids'].shape[1]

    #Prefill of Draft Model
    output_draft = draft_model(
        **inputs_drop, past_key_values=draft_past_key_values
    )
    # print("Target KV:",past_key_values[0][0].shape)
    # print("Draft KV:",draft_past_key_values[0][0].shape)

    return sample_token, input_ids, draft_input_len, scores

def initialize_tree_with_TriVLM(inputs, video_inputs, target_model, draft_model, processor, past_key_values,
                                retrieval_past_key_values, draft_past_key_values,
                                temperature=0.6, top_k=-1, top_p=0.9):
    _ = video_chunk_prefill(inputs, video_inputs, draft_model, processor, draft_past_key_values,  video_group_size, sparse_cache=True)
    draft_input_len = draft_past_key_values[0][0].shape[2]
    output = video_chunk_prefill(inputs, video_inputs, target_model, processor, past_key_values, video_group_size,)
    logits = output.logits
    if temperature==0:
        sample_token = torch.argmax(logits[:, -1])
        sample_token = sample_token[None, None]
    else:
        sample_token = sample(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))

    #Prefill of Draft Model

    return sample_token, draft_input_len,

@torch.no_grad()
def tree_draft(input_ids, draft_model, draft_past_key_values,len_posi):
    # input_ids = input_ids[:, 1:]
    input_ids = input_ids.to(draft_model.device)
    ss_token,ss_prob,ss_op = [],[],[]
    # len_posi = total_input_len + sample_token
    draft_model.model.tree_mask=None

    #old kv cache and new token(s)
    outputs = draft_model(input_ids=input_ids, past_key_values=draft_past_key_values)
    last_headout = outputs.logits[0, -1:] #[input_len, vocab_size]

    for i in range(len(draft_model.tree_buffer['tree_indices'])):
        top=torch.topk(last_headout, TOPK, dim=-1)
        topk_index,topk_prob = top.indices,top.values #[1,input_len,10]
        op=None
        ss_token.append(topk_index)
        # ss_prob.append(topk_prob)
        # ss_op.append(op)

        #flatten
        topk_index = topk_index.view(-1) #[input_len * 10]
        #Choose next input_ids
        select_index=topk_index[draft_model.tree_buffer['tree_indices'][i].to(topk_index.device)]
        input_ids=select_index[None,:] #[1,4]/[1,1]
        #Prepare next position_ids and attn_mask
        draft_model.model.tree_mask=draft_model.tree_buffer['attn_mask'][i]
        position_ids=len_posi+draft_model.tree_buffer["position_ids"][i]
        # outputs, past_key_values = draft_model(input_ids=input_ids, past_key_values=past_key_values,
        #                                     position_ids=position_ids,use_cache=True)
        outputs = draft_model(input_ids=input_ids, past_key_values=draft_past_key_values,position_ids=position_ids)
        len_posi += 1
        last_headout = outputs.logits[0] #[len_input, vocab_size]

    top = torch.topk(last_headout, TOPK, dim=-1)
    topk_index, topk_prob = top.indices, top.values
    op=None

    ss_token.append(topk_index)
    ss_prob.append(topk_prob)
    ss_op.append(op)

    # return (torch.cat(ss_token),torch.cat(ss_prob),ss_op)
    return torch.cat(ss_token).view(-1)

def tree_decoding(
        model,
        tree_candidates,
        past_key_values,
        tree_position_ids,
        input_ids,
        retrieve_indices,
):
    position_ids = tree_position_ids + input_ids.shape[1]

    outputs = model.model(
        tree_candidates,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )
    tree_logits = model.lm_head(outputs[0]) #[1,26,152128]
    logits = tree_logits[0, retrieve_indices.to(tree_logits.device)] #[15,6,152128]
    return logits, outputs

def evaluate_posterior(
        logits: torch.Tensor,
        candidates: torch.Tensor,
        temperature=0.6,
        top_k=-1,
        top_p=0.9,
):
    # Greedy decoding based on temperature value
    # Find the tokens that match the maximum logits for each position in the sequence
    if temperature==0:
        posterior_mask = (
                candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
    else:
        posterior_mask = (
            candidates[:, 1:].to(logits.device) == sample(norm_logits(logits[:, :-1] , temperature=temperature ,top_k=top_k, top_p=top_p))
        ).int()
    candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
    accept_length = candidates_accept_length.max()
    # Choose the best candidate
    if accept_length == 0:
        # Default to the first candidate if none are accepted
        best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
    else:
        best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
    return best_candidate, accept_length, logits[best_candidate, accept_length]


@torch.no_grad()
def update_inference_inputs(
        input_ids,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        logits,
        tree_logits,
        new_token,
        past_key_values_data_list,
        current_length_data,
        draft_model,
        draft_past_key_values,
        draft_past_key_values_data_list,
        draft_current_length_data,
        sample_p,
        temperature=0.6,
        top_k=-1,
        top_p=0.9,
):
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
            retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)], dim=-1
    )
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
        # Destination tensor where the relevant past information will be stored
        dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)
    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    #Updata Draft model past key values
    del_len = 1
    for i in draft_model.tree_buffer['tree_indices']:
        del_len += len(i)
    # print("del_len:",del_len)
    # del_len = 11 # 1+4+4+1+1
    for draft_past_key_values_data in draft_past_key_values_data_list:
        draft_past_key_values_data = draft_past_key_values_data[..., :-del_len, :]
    
    draft_current_length_data.fill_(prev_input_len)

    prob = sample_p.unsqueeze(0)
    if temperature==0:
        token = torch.argmax(prob)
        token = token[None, None]
    else:
        token = sample(norm_logits(prob, temperature=temperature ,top_k=top_k, top_p=top_p))
    
    len_posi = input_ids.shape[1] + 1
    tree_logits = tree_draft(input_ids=torch.cat([candidates[None, best_candidate, : accept_length + 1], token],dim=-1),
                              draft_model = draft_model, draft_past_key_values = draft_past_key_values, len_posi = len_posi)

    new_token += accept_length + 1

    return input_ids, tree_logits, new_token, None, token

@torch.no_grad()
def update_inference_inputs_with_pruning(
        input_ids,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        logits,
        tree_logits,
        new_token,
        past_key_values_data_list,
        current_length_data,
        draft_model,
        draft_past_key_values,
        draft_past_key_values_data_list,
        draft_current_length_data,
        sample_p,
        draft_input_len,
        temperature=0.6,
        top_k=-1,
        top_p=0.9,
):
    prev_input_len = input_ids.shape[1]
    prev_draft_input_len = draft_input_len
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
            retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)], dim=-1
    )
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
        # Destination tensor where the relevant past information will be stored
        dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)
    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    #Updata Draft model past key values
    del_len = 1
    for i in draft_model.tree_buffer['tree_indices']:
        del_len += len(i)
    # print("del_len:",del_len)
    # del_len = 11 # 1+4+4+1+1
    for draft_past_key_values_data in draft_past_key_values_data_list:
        draft_past_key_values_data = draft_past_key_values_data[..., :-del_len, :]
    draft_current_length_data.fill_(prev_draft_input_len)
    draft_input_len += accept_length + 1

    prob = sample_p
    if temperature==0:
        token = torch.argmax(prob)
        token = token[None, None]
    else:
        token = sample(norm_logits(prob.unsqueeze(0), temperature ,top_k, top_p))
    len_posi = draft_input_len + 1
    # len_posi = input_ids.shape[1] + 1
    tree_logits = tree_draft(input_ids=torch.cat([candidates[None, best_candidate, : accept_length + 1], token],dim=-1),
                              draft_model = draft_model, draft_past_key_values = draft_past_key_values, len_posi = len_posi)

    new_token += accept_length + 1

    return input_ids, tree_logits, new_token, None, token, draft_input_len

def _cleanup_model_inference_cache(*models):
    """Release KV caches and auxiliary buffers to free GPU memory between samples."""
    for model in models:
        if model is None:
            continue
        module = getattr(model, "model", None)
        if module is None:
            continue
        
        # Clean up past key values
        past_key_values = getattr(module, "past_key_values", None)
        if isinstance(past_key_values, (list, tuple)):
            for layer_cache in past_key_values:
                if isinstance(layer_cache, (list, tuple)):
                    for cache in layer_cache:
                        if hasattr(cache, "data"):
                            cache.data = None
                        if hasattr(cache, "current_length"):
                            cache.current_length = None
        
        # Clean up main model cache attributes
        if hasattr(module, "past_key_values"):
            module.past_key_values = None
        if hasattr(module, "past_key_values_data"):
            module.past_key_values_data = None
        if hasattr(module, "current_length_data"):
            module.current_length_data = None
            
        # Clean up draft model cache attributes (fixed the bugs here)
        if hasattr(module, "draft_past_key_values"):
            module.draft_past_key_values = None
        if hasattr(module, "draft_past_key_values_data"):
            module.draft_past_key_values_data = None
        if hasattr(module, "draft_current_length_data"):
            module.draft_current_length_data = None
            
        # Clean up rope and tree-related attributes
        if hasattr(module, "rope_deltas"):
            module.rope_deltas = None
        if hasattr(module, "tree_mask"):
            module.tree_mask = None
        if hasattr(module, "tree_mode"):
            module.tree_mode = None
            
        # Clean up model-level tree attributes
        if hasattr(model, "tree_buffers"):
            model.tree_buffers = None
        if hasattr(model, "tree_choices"):
            model.tree_choices = None
        if hasattr(model, "tree_buffer"):
            model.tree_buffer = None
            
        # Additional cleanup for any remaining cache states
        if hasattr(module, "retrieval_past_key_values"):
            module.retrieval_past_key_values = None
        if hasattr(module, "retrieval_past_key_values_data"):
            module.retrieval_past_key_values_data = None
        if hasattr(module, "retrieval_current_length_data"):
            module.retrieval_current_length_data = None
            
    if torch.cuda.is_available():
        torch.cuda.empty_cache()