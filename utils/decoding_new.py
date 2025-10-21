import json
import time
import copy
import random
from typing import List, Tuple
import torch
import os

from cache.kv_cache import FlashSimpleCache
from cache.draft_cache import DraftCache
from cache.sparse_cache import RetrievalCache
from utils.utils_c import generate_tree_buffers_draft
from utils.sampling import sample, norm_logits
from termcolor import colored
from utils.choices import mc_sim_7b_63
from utils.utils import *

video_group_size = 64

def TriVLM(inputs, video_inputs, target_model, draft_model, processor, max_new_tokens=128, top_k=-1, top_p=0.9, temperature=0.6, verbose=True):
    torch.cuda.synchronize()
    time1 = time.time()

    cache =FlashSimpleCache(target_model)
    retrieval_cache =FlashSimpleCache(target_model)
    draft_cache =DraftCache(draft_model)

    with torch.no_grad():
        output = video_chunk_prefill(inputs, video_inputs, target_model, processor, cache, retrieval_cache, video_group_size, sparse_cache = False)
        _ = video_chunk_prefill(inputs, video_inputs, draft_model, processor, draft_cache, None, video_group_size, sparse_cache = True)
        logits = output.logits

    resample_count = 0
    accepted_count = 0
    target_sample_count = 0
    draft_count = 0

    next_token = sample(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))
    if verbose:
        spec_stream(next_token[0], processor, 'cyan')
    
    acc_rate_middle_list = []
    n = 0
    time1 = time.time()
    while n < max_new_tokens:
        if next_token.shape == torch.Size([1]):
            next_token = next_token.unsqueeze(0)
        
        # speculative decoding for draft (68m) and retrieval 7b model
        pred_token_idx = next_token
        verify_tokens, speculation_probs, acc_rate_middle = Middle_Spec(pred_token_idx, target_model, draft_model, 
                                                                        retrieval_cache, draft_cache, gamma=6, 
                                                                        verbose=False, processor=processor,
                                                                        temperature=temperature, top_k=top_k, top_p=top_p,
                                                                        )


def Middle_Spec(next_token, target_model, draft_model,retrieval_cache, draft_cache, gamma, verbose, processor,temperature, top_k, top_p):
    n = 0
    resample_count = 0
    accepted_count = 0
    target_sample_count = 0
    draft_count = 0

    pred_token_idx = next_token

    return_generated_ids = []
    return_speculation_probs = []
    return_generated_ids.append(next_token.item())

    verify_tokens = torch.full((1, gamma + 1), 0, device=target_model.device)
    verify_tokens[:, 0] = next_token

    while n < gamma:
        new_inputs = {
                'input_ids': next_token,
                'past_key_values': draft_cache,
            }
        with torch.no_grad():
            speculation_prob = draft_model(**new_inputs).logits
            pred_token_idx = sample(norm_logits(speculation_prob, temperature=temperature ,top_k=top_k, top_p=top_p))
            draft_count += 1
            verify_tokens[:, n+1:n+2] = pred_token_idx

    new_verify_inputs = {
                'input_ids': next_token,
                'retrieval_past_key_values': retrieval_cache,
            }

    verify_prob = target_model(**new_verify_inputs).logits
    while n < gamma:   
        r = torch.rand(1, device = target_model.device)
        token_idx = verify_tokens[:, n+1:n+2].item()
        if r < torch.min(torch.tensor([1], device=r.device), (verify_prob[n, token_idx] / speculation_prob[token_idx])):

            return_speculation_probs.append(verify_prob[n])
            return_generated_ids.append(token_idx)
            if verbose:
                spec_stream(pred_token_idx, processor, 'green')
            accepted_count += 1
            n += 1
        
            pred_token_idx = sample(verify_prob[n])
            return_speculation_probs.append(verify_prob[n])
            return_generated_ids.append(pred_token_idx.item())
            if verbose:
                spec_stream(pred_token_idx, processor, 'blue')
            target_sample_count += 1
            n += 1

            verify_tokens[:, n:n+1] = pred_token_idx
        else:
            pred_token_idx = sample(verify_prob[n])
            return_speculation_probs.append(verify_prob[n])
            return_generated_ids.append(pred_token_idx.item())
            if verbose:
                spec_stream(pred_token_idx, processor, 'red')
            resample_count += 1
            n += 1
            verify_tokens[:, n:n+1] = pred_token_idx



def spec_stream(pred_token_idx, tokenizer, color='blue'):
    decoded_token = tokenizer.decode(
            pred_token_idx,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            # spaces_between_special_tokens=False,
        )

    decoded_token = decoded_token.replace("<0x0A>", "\n")

    print(colored(decoded_token, color), flush=True, end=" ")

