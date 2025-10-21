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
from utils.sampling import sample, norm_logits ,max_fn
from termcolor import colored
from utils.choices import mc_sim_7b_63
from utils.utils import *

video_group_size = 64

def TriVLM(inputs, video_inputs, target_model, draft_model, processor, max_new_tokens=128, top_k=-1, top_p=0.9, temperature=0.6, verbose=True):
    torch.cuda.synchronize()
    time1 = time.time()

    cache = FlashSimpleCache(target_model)
    retrieval_cache = FlashSimpleCache(target_model)
    # tem_cache = FlashSimpleCache(draft_model)
    # draft_cache = FlashSimpleCache(draft_model)

    with torch.no_grad():
        output = video_chunk_prefill(inputs, video_inputs, target_model, processor, cache, retrieval_cache, video_group_size, sparse_cache = False)
        #_ = video_chunk_prefill_draft(inputs, video_inputs, draft_model, processor, tem_cache, draft_cache, video_group_size, sparse_cache = True)
        logits = output.logits

    resample_count = 0
    accepted_count = 0
    target_sample_count = 0
    draft_count = 0

    next_token = sample(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))
    if verbose:
        spec_stream(next_token[0], processor, 'cyan')
    
    acc_rate_middle_list = []
    generated_tokens = []
    generated_tokens.append(next_token.item())
    
    gamma = 6
    token_count = 0
    
    while token_count < max_new_tokens:
        if next_token.shape == torch.Size([1]):
            next_token = next_token.unsqueeze(0)

        pred_token_idx = next_token
        
        speculation_probs_list = []
        verify_tokens = torch.full((1, gamma + 1), 0, device=target_model.device)
        verify_tokens[:, 0] = next_token

        # Draft phase: generate gamma tokens using target model
        draft_tokens = []
        for n in range(gamma):
            if token_count + n >= max_new_tokens:
                break
                
            new_inputs = {
                    'input_ids': pred_token_idx,
                    'retrieval_past_key_values': retrieval_cache,
                }
            with torch.no_grad():
                logits = target_model(**new_inputs).logits
                speculation_prob = norm_logits(logits[0], temperature=temperature ,top_k=top_k, top_p=top_p)
                speculation_probs_list.append(speculation_prob)
                pred_token_idx = sample(speculation_prob)
                draft_count += 1
                verify_tokens[:, n+1:n+2] = pred_token_idx
                draft_tokens.append(pred_token_idx.item())

        # Verification phase
        new_verify_inputs = {
                    'input_ids': verify_tokens,
                    'past_key_values': cache,
                }

        with torch.no_grad():
            logits = target_model(**new_verify_inputs).logits

        count = 0
        verify_probs = []
    
        probs = norm_logits(logits[0], temperature=temperature ,top_k=top_k, top_p=top_p)
        for i in range(len(draft_tokens) + 1):
            verify_probs.append(probs[i])
        
        # Verify each speculated token
        for i in range(len(draft_tokens)):
            r = torch.rand(1, device = target_model.device)
            token_id = draft_tokens[i]
            
            # Calculate acceptance probability
            accept_prob = torch.min(torch.tensor([1.0], device=r.device), 
                                  (verify_probs[i][token_id] / speculation_probs_list[i][0, token_id]))
            
            if r < accept_prob:
                count += 1
                accepted_count += 1
                generated_tokens.append(token_id)
                if verbose:
                    spec_stream(torch.tensor([[token_id]]), processor, 'green')
                
                if token_id == processor.tokenizer.eos_token_id:
                    break
            else:
                resample_count += 1
                # Resample from the difference distribution
                resampled_token = sample(max_fn(verify_probs[i+1] - speculation_probs_list[i][0]))
                generated_tokens.append(resampled_token.item())
                if verbose:
                    spec_stream(resampled_token, processor, 'red')
                
                if resampled_token.item() == processor.tokenizer.eos_token_id:
                    break

        # Update token count
        token_count += count + 1  # +1 for the initial token
        
        # Update cache
        cache.seen_tokens -= (len(draft_tokens) - count)
        cache.reset_cache()
        retrieval_cache.seen_tokens -= (len(draft_tokens) + 1) 
        retrieval_cache.reset_cache()
        retrieval_cache.spec_update(cache, len(draft_tokens) - count)

        # If all tokens were accepted, sample one more from the target model
        if count == len(draft_tokens) and token_count < max_new_tokens:
            target_sample_count += 1
            next_token = sample(verify_probs[-1])
            generated_tokens.append(next_token.item())
            if verbose:
                spec_stream(next_token, processor, 'blue')
            token_count += 1
        else:
            # Use the last resampled token or the last accepted token
            if count < len(draft_tokens):
                next_token = torch.tensor([[generated_tokens[-1]]]).to(target_model.device)
            else:
                next_token = torch.tensor([[generated_tokens[-1]]]).to(target_model.device)
        
        # Calculate acceptance rate for this round
        acc_rate = count / len(draft_tokens) if len(draft_tokens) > 0 else 0
        acc_rate_middle_list.append(acc_rate)
        print(f"\nRound acceptance rate: {acc_rate:.3f}")
        
        # Check for EOS token
        if generated_tokens[-1] == processor.tokenizer.eos_token_id:
            break

    # Final statistics
    torch.cuda.synchronize()
    time2 = time.time()
    
    total_time = time2 - time1
    tokens_per_second = len(generated_tokens) / total_time if total_time > 0 else 0

    print(f"\n=== TriVLM Decoding Statistics ===")
    print(f"Total tokens generated: {len(generated_tokens)}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Accepted tokens: {accepted_count}")
    print(f"Resampled tokens: {resample_count}")
    print(f"Target model samples: {target_sample_count}")
    print(f"Draft model samples: {draft_count}")

    return {
        'generated_tokens': generated_tokens,
        'total_time': total_time,
        'tokens_per_second': tokens_per_second,
        'accepted_count': accepted_count,
        'resample_count': resample_count,
        'target_sample_count': target_sample_count,
        'draft_count': draft_count,
        'acceptance_rate': sum(acc_rate_middle_list)/len(acc_rate_middle_list) if acc_rate_middle_list else 0
    }

def Middle_Spec(next_token, target_model, draft_model,retrieval_cache, draft_cache, gamma, verbose, processor,temperature, top_k, top_p):
    n = 0
    resample_count = 0
    accepted_count = 0
    target_sample_count = 0
    draft_count = 0

    pred_token_idx = next_token

    return_generated_ids = []
    return_speculation_probs = []
    speculation_probs_list = []
    return_generated_ids.append(next_token.item())

    verify_tokens = torch.full((1, gamma + 1), 0, device=target_model.device)
    verify_tokens[:, 0] = next_token

    while n < gamma:
        new_inputs = {
                'input_ids': pred_token_idx,
                'draft_past_key_values': draft_cache,
            }
        with torch.no_grad():
            speculation_prob = draft_model(**new_inputs).logits
            speculation_probs_list.append(speculation_prob)
            pred_token_idx = sample(norm_logits(speculation_prob, temperature=temperature ,top_k=top_k, top_p=top_p))
            draft_count += 1
            verify_tokens[:, n+1:n+2] = pred_token_idx
        n += 1


    new_verify_inputs = {
                'input_ids': verify_tokens,
                'retrieval_past_key_values': retrieval_cache,
            }

    verify_prob = target_model(**new_verify_inputs).logits
    n = 0
    while n < gamma:   
        r = torch.rand(1, device = target_model.device)
        token_idx = verify_tokens[:, n+1:n+2].item()
        if r < torch.min(torch.tensor([1], device=r.device), (verify_prob[0, n, token_idx] / speculation_probs_list[n][:,:,token_idx])):

            return_speculation_probs.append(verify_prob[:,n,:])
            return_generated_ids.append(token_idx)
            if verbose:
                spec_stream(pred_token_idx, processor, 'green')
            accepted_count += 1
            n += 1
        
            pred_token_idx = sample(norm_logits(verify_prob[:,n,:], temperature=temperature ,top_k=top_k, top_p=top_p))
            return_speculation_probs.append(verify_prob[:,n,:])
            return_generated_ids.append(pred_token_idx.item())
            if verbose:
                spec_stream(pred_token_idx, processor, 'blue')
            target_sample_count += 1
            n += 1

            verify_tokens[:, n:n+1] = pred_token_idx
        else:
            pred_token_idx = sample(norm_logits(verify_prob[:,n,:], temperature=temperature ,top_k=top_k, top_p=top_p))
            return_speculation_probs.append(verify_prob[0, n])
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

