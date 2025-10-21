import torch
import time
from transformers.cache_utils import (
    DynamicCache,
    Iterable,
    List,
    Dict,
    Optional,
    Any,
    Tuple,
)

class FlashSimpleCache(DynamicCache):
    def __init__(self, model,) -> None:
        # Initialize DynamicCache first
        super().__init__()
        
        self.seq_len = 0
        
        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_key_value_heads 
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        dtype = model.model.layers[0].self_attn.q_proj.weight.dtype

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        self.scores = []
        self.prompt_length = 0
        self.accum_attn_scores = {}
    
    def reset(self):
        self.seq_len=0
        with torch.inference_mode():
            self.key_cache: List[torch.Tensor] = []
            self.value_cache: List[torch.Tensor] = []
    
    def reset_kv_cache(self):
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][..., :self.seq_len, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][..., :self.seq_len, :]

    def set_prompt_length(self, prompt_length: int=0):
        """
        Set the prompt length for the cache.
        Args:
            prompt_length (int): The length of the prompt.
        """
        self.prompt_length = prompt_length
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.prompt_length:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)
        else:
            query_states = cache_kwargs["query_states"] # (bz, num_heads, Q, head_dim)
            query_states = query_states[:, :, -self.prompt_length:, :]
            key_states = key_states[:, :, :-self.prompt_length, :]
            value_states = value_states[:, :, :-self.prompt_length, :]
            super_result = super().update(key_states, value_states, layer_idx, cache_kwargs)
            # postprocess
            bsz, num_heads, q_len, head_dim = query_states.shape
            num_key_value_heads, k_len = key_states.shape[1:3]
            # attention scores of query to key
            key_states_repeated = repeat_kv(key_states, num_heads // num_key_value_heads)
            attn_scores = torch.einsum("bhqd,bhkd->bhqk", query_states, key_states_repeated) / (head_dim ** 0.5)
            attn_scores = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            ).detach() # # (bz, num_heads, Q, K)
            attn_scores = attn_scores.sum(-2).mean(1) # average over num_key_value_heads (bz, k_len)
            self.accum_attn_scores[layer_idx] = self.accum_attn_scores.get(layer_idx, [])
            self.accum_attn_scores[layer_idx].append(attn_scores)
            return super_result

    def reset_cache(self):
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][..., :self.seen_tokens, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][..., :self.seen_tokens, :]

    def spec_update(self, kv_cache, count):
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx],kv_cache.key_cache[layer_idx][..., kv_cache.seen_tokens-self.gamma-1+count:, :]], dim=-2)
            self.value_cache[layer_idx]= torch.cat([self.value_cache[layer_idx],kv_cache.value_cache[layer_idx][..., kv_cache.seen_tokens-self.gamma-1+count:, :]], dim=-2)
        self.seq_len = self.key_cache[layer_idx].shape[2]

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)