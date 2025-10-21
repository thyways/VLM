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

    cache = FlashSimpleCache(target_model)
    retrieval_cache = FlashSimpleCache(target_model)
    tem_cache = FlashSimpleCache(draft_model)
    draft_cache = FlashSimpleCache(draft_model)

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
        
        # Reset counters for this speculation round
        round_resample_count = 0
        round_accepted_count = 0
        round_target_sample_count = 0
        round_draft_count = 0

        return_generated_ids = []
        return_speculation_probs = []
        speculation_probs_list = []
        return_generated_ids.append(next_token.item())

        verify_tokens = torch.full((1, gamma + 1), 0, device=target_model.device)
        verify_tokens[:, 0] = next_token

        # Draft phase: generate gamma tokens using target model
        n = 0
        while n < gamma and token_count + n < max_new_tokens:
            new_inputs = {
                    'input_ids': pred_token_idx,
                    'retrieval_past_key_values': retrieval_cache,
                }
            with torch.no_grad():
                speculation_prob = target_model(**new_inputs).logits
                speculation_probs_list.append(speculation_prob)
                pred_token_idx = sample(norm_logits(speculation_prob, temperature=temperature ,top_k=top_k, top_p=top_p))
                round_draft_count += 1
                verify_tokens[:, n+1:n+2] = pred_token_idx
            n += 1

        # Verification phase
        new_verify_inputs = {
                    'input_ids': verify_tokens,
                    'past_key_values': cache,
                }

        verify_prob = target_model(**new_verify_inputs).logits
        n = 0
        accepted_tokens = 0
        
        while n < gamma and token_count + n < max_new_tokens:   
            r = torch.rand(1, device = target_model.device)
            token_idx = verify_tokens[:, n+1:n+2].item()
            
            # Calculate acceptance probability
            acceptance_prob = torch.min(torch.tensor([1.0], device=r.device), 
                                     (verify_prob[0, n, token_idx] / speculation_probs_list[n][0, 0, token_idx]))
            
            if r < acceptance_prob:
                # Accept the speculated token
                return_speculation_probs.append(verify_prob[:,n,:])
                return_generated_ids.append(token_idx)
                if verbose:
                    spec_stream(torch.tensor([token_idx], device=target_model.device), processor, 'green')
                round_accepted_count += 1
                accepted_tokens += 1
                n += 1
                
                # Generate next token from target model
                if n < gamma and token_count + n < max_new_tokens:
                    pred_token_idx = sample(norm_logits(verify_prob[:,n,:], temperature=temperature ,top_k=top_k, top_p=top_p))
                    return_speculation_probs.append(verify_prob[:,n,:])
                    return_generated_ids.append(pred_token_idx.item())
                    if verbose:
                        spec_stream(pred_token_idx, processor, 'blue')
                    round_target_sample_count += 1
                    n += 1
                    verify_tokens[:, n:n+1] = pred_token_idx
            else:
                # Reject and resample
                pred_token_idx = sample(norm_logits(verify_prob[:,n,:], temperature=temperature ,top_k=top_k, top_p=top_p))
                return_speculation_probs.append(verify_prob[:,n,:])
                return_generated_ids.append(pred_token_idx.item())
                if verbose:
                    spec_stream(pred_token_idx, processor, 'red')
                round_resample_count += 1
                n += 1
                verify_tokens[:, n:n+1] = pred_token_idx
                break

        # Update global counters
        resample_count += round_resample_count
        accepted_count += round_accepted_count
        target_sample_count += round_target_sample_count
        draft_count += round_draft_count
        
        # Add accepted tokens to the final output
        generated_tokens.extend(return_generated_ids[1:accepted_tokens+1])  # Skip the first token as it's already added
        token_count += accepted_tokens + 1  # +1 for the resampled token
        
        # Update next_token for next iteration
        next_token = pred_token_idx
        
        # Calculate acceptance rate for this round
        if round_draft_count > 0:
            acc_rate = round_accepted_count / round_draft_count
            acc_rate_middle_list.append(acc_rate)
            if verbose:
                print(f"\nRound acceptance rate: {acc_rate:.3f}")

    # Final statistics
    torch.cuda.synchronize()
    time2 = time.time()
    
    total_time = time2 - time1
    tokens_per_second = len(generated_tokens) / total_time if total_time > 0 else 0
    
    if verbose:
        print(f"\n=== TriVLM Decoding Statistics ===")
        print(f"Total tokens generated: {len(generated_tokens)}")
        print(f"Total time: {total_time:.3f}s")
        print(f"Tokens per second: {tokens_per_second:.2f}")
        print(f"Accepted tokens: {accepted_count}")
        print(f"Resampled tokens: {resample_count}")
        print(f"Target model samples: {target_sample_count}")
        print(f"Draft model samples: {draft_count}")
        if acc_rate_middle_list:
            print(f"Average acceptance rate: {sum(acc_rate_middle_list)/len(acc_rate_middle_list):.3f}")

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

