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

def TriVLM(inputs, video_inputs, target_model, processor, max_new_tokens=128, top_k=-1, top_p=0.9, temperature=0.6, verbose=True):
    torch.cuda.synchronize()
    time1 = time.time()

    cache =FlashSimpleCache(target_model)
    retrieval_cache =FlashSimpleCache(target_model)

    with torch.no_grad():
        output = video_chunk_prefill(inputs, video_inputs, target_model, processor, cache, retrieval_cache, video_group_size, sparse_cache = False)