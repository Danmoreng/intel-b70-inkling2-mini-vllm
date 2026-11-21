"""Batch-one vLLM adapter for the Intel XPU Inkling EXL3 model."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from inkling_trellis.inkling2.attn_triton import attention_prefill_esimd
from inkling_trellis.inkling2.sgl_model import CONV_RING, RING, Inkling2LoopForConditionalGeneration
from inkling_trellis.moe_offload import ops as moe_ops
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

from .exl3_experts import InklingEXL3Experts
from .paged_attention import decode_paged
from .prefix_attention import hybrid_prefill


class Inkling2B70ForCausalLM(Inkling2LoopForConditionalGeneration):
    # The dense tensors go through vLLM's weight iterator. EXL3 expert shards
    # are loaded one expert at a time into their final packed device arenas.
    allow_patterns_overrides = ["dense-*.safetensors"]

    def __init__(self, *, vllm_config, prefix=""):
        if vllm_config.scheduler_config.max_num_seqs != 1 or not vllm_config.model_config.enforce_eager:
            raise ValueError("Inkling XPU currently requires --max-num-seqs 1 --enforce-eager")
        cache = vllm_config.cache_config
        if cache.enable_prefix_caching or cache.cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
            raise ValueError("Inkling XPU requires prefix caching off and a BF16 or E4M3 KV cache")
        checkpoint = Path(vllm_config.model_config.model)
        raw = json.loads((checkpoint / "config.json").read_text())
        super().__init__(SimpleNamespace(inkling_text=raw["text_config"], inkling_vision=None), external_slots=1)
        self.checkpoint = checkpoint
        self.page_tokens = cache.block_size or 16
        self.logit_cap = self.tc["final_logit_softcapping"]
        self.cache_dtype = torch.float8_e4m3fn if cache.cache_dtype.startswith("fp8") else torch.bfloat16
        self.prefix_block = int(os.environ.get("INKLING2_ONEDNN_PREFIX_BLOCK", "0"))
        if self.prefix_block:
            if self.prefix_block < 0 or self.prefix_block % self.page_tokens:
                raise ValueError("The oneDNN prefix block must be a positive multiple of the KV page size")
            torch.ops.load_library("/opt/exl3xpu/exl3xpu/_C.so")
        root = f"{prefix}." if prefix else ""
        self.kv_layers = nn.ModuleList([
            # Reuse the metadata builder and allocator contract. This model
            # calls the SYCL attention directly using the bound page views.
            Attention(self.H, self.hd, self.hd ** -0.5, self.Hkv, cache_config=cache,
                      prefix=f"{root}global_attn.{index}", attn_backend=TritonAttentionBackend)
            for index in range(self.n_global)
        ])

    def load_weights(self, weights):
        device = torch.device("xpu", torch.xpu.current_device())
        min_dpas = int(os.environ.get("INKLING2_EXL3_PREFILL_MIN", "1025"))
        moe_ops().moe_set_prefill_min_m(min_dpas)
        experts = {layer: InklingEXL3Experts(self.checkpoint, layer, device, self.tc["n_routed_experts"])
                   for layer in range(self.n_layers)}
        super().load_weights(weights, experts=experts)

    def embed_input_ids(self, input_ids):
        return self.get_input_embeddings()(input_ids)

    def compute_logits(self, hidden_states):
        logits = self.lm_q8(hidden_states)
        return torch.tanh(logits / self.logit_cap) * self.logit_cap

    def _pages(self, visit, batch):
        cache = batch.caches[self.visit_slot[visit]]
        if cache.ndim != 4 or cache.shape[1:] != (self.Hkv, self.page_tokens, 2 * self.hd):
            raise ValueError(f"Unexpected vLLM cache view {tuple(cache.shape)}")
        # vLLM's allocator exposes FP8 cache storage as bytes. The kernel
        # dispatches on the logical E4M3 type and needs the original byte bits.
        if self.cache_dtype == torch.float8_e4m3fn and cache.dtype in (torch.uint8, torch.int8):
            cache = cache.view(self.cache_dtype)
        if cache.dtype != self.cache_dtype:
            raise ValueError(f"Expected {self.cache_dtype} in the vLLM KV cache, got {cache.dtype}")
        return cache.transpose(1, 2).split(self.hd, dim=-1)

    def _store_kv(self, key_cache, value_cache, key, value, rows):
        pages = rows // self.page_tokens
        positions = rows % self.page_tokens
        if key_cache.dtype == torch.float8_e4m3fn:
            key_cache.view(torch.uint8)[pages, positions] = key.to(key_cache.dtype).view(torch.uint8)
            value_cache.view(torch.uint8)[pages, positions] = value.to(value_cache.dtype).view(torch.uint8)
        else:
            key_cache[pages, positions] = key.to(key_cache.dtype)
            value_cache[pages, positions] = value.to(value_cache.dtype)

    def _decode_global(self, visit, q, k, v, pos, rel, proj, sink, batch):
        key_cache, value_cache = self._pages(visit, batch)
        self._store_kv(key_cache, value_cache, k, v, batch.out_cache_loc)
        splits = min(64, max(8, (batch.total_tokens + 1023) // 1024))
        return decode_paged(q, key_cache, value_cache, batch.block_table, batch.seq_lens, pos, rel, proj, sink,
                            queries_per_request=q.shape[0], num_splits=splits)

    def _prefill_global(self, visit, q, k, v, table, sink, reqs, batch):
        key_cache, value_cache = self._pages(visit, batch)
        self._store_kv(key_cache, value_cache, k, v, batch.out_cache_loc)

        def read_kv(start, end):
            page_start = start // self.page_tokens
            page_end = (end + self.page_tokens - 1) // self.page_tokens
            pages = batch.block_table[0, page_start:page_end].long()
            if key_cache.dtype == torch.float8_e4m3fn:
                return tuple(t.view(torch.uint8).index_select(0, pages).flatten(0, 1)[:end - start].view(t.dtype)
                             for t in (key_cache, value_cache))
            return tuple(t.index_select(0, pages).flatten(0, 1)[:end - start] for t in (key_cache, value_cache))

        if batch.first_position == 0:
            # The first chunk already has every key and value in logical order.
            keys, values = k, v
        elif self.prefix_block and batch.first_position + 1 - table.shape[-1] >= self.prefix_block:
            return [hybrid_prefill(q, table, sink, batch.first_position, batch.total_tokens,
                                   self.prefix_block, read_kv)]
        else:
            keys, values = read_kv(0, batch.total_tokens)
        return [attention_prefill_esimd(q, keys, values, batch.first_position, 0, table, sink, qtw=4)]

    def _metadata(self, size, device):
        metadata = get_forward_context().attn_metadata
        if metadata is not None:
            md = metadata[self.kv_layers[0].layer_name]
            if md.seq_lens.numel() != 1 or md.num_actual_tokens != size:
                raise ValueError("Inkling XPU requires one unpadded sequence per forward")
            self.page_tokens = self.kv_layers[0].kv_cache.shape[2]
            total = int(md.seq_lens[0].item())
            return SimpleNamespace(caches=[layer.kv_cache for layer in self.kv_layers], block_table=md.block_table,
                                   seq_lens=md.seq_lens, out_cache_loc=md.slot_mapping[:size].long(),
                                   total_tokens=total, first_position=total - size)

        # Run the model during vLLM's memory profiling before it binds a KV arena.
        num_pages = (size + self.page_tokens - 1) // self.page_tokens
        pages = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
        caches = [torch.empty((num_pages, self.Hkv, self.page_tokens, 2 * self.hd), dtype=self.cache_dtype, device=device)
                  for _ in self.kv_layers]
        return SimpleNamespace(caches=caches, block_table=pages,
                               seq_lens=torch.tensor([size], dtype=torch.int32, device=device),
                               out_cache_loc=torch.arange(size, device=device), total_tokens=size, first_position=0)

    @torch.no_grad()
    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        size = positions.numel()
        positions = positions.long()
        batch = self._metadata(size, positions.device)
        self._static = batch.first_position > 0 and size in (1, 2, 4, 8)
        if self._static:
            self._st_nq = size
            self._st_req_slots = torch.zeros(1, dtype=torch.int32, device=positions.device)
            self._st_slot = torch.zeros(size, dtype=torch.long, device=positions.device)
            self._st_pos = positions
            self.swa_pos[0, positions.remainder(RING)] = positions.int()
            self.conv_pos[0, positions.remainder(CONV_RING)] = positions
            return self._forward_body(input_ids, positions, batch, None, inputs_embeds, return_hidden=True)

        reqs = [(0, 0, size, batch.first_position)]
        self._reqs = reqs
        self._conv_pos_update = []
        self._prepare(batch, reqs, positions, kv_pool=self)
        return self._forward_body(input_ids, positions, batch, reqs, inputs_embeds, return_hidden=True)
