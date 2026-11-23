"""Inkling 2 Mini DSpark for the Intel XPU target and vLLM V2 runner."""

from copy import copy
import json
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

from .split_attention import merge_splits

from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator


def norm(x, weight):
    values = x.float()
    return (values * (values.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * weight.float()).to(
        torch.bfloat16
    )


class InklingDraft:
    def __init__(self, checkpoint, device, max_context, *, grow_global=False):
        config = json.loads((checkpoint / "config.json").read_text())
        self.config = config["mtp_config"]
        self.width = self.config["hidden_size"]
        self.head = self.config["head_dim"]
        self.heads = self.config["num_attention_heads"]
        self.groups = self.config["num_key_value_heads"]
        self.block = self.config["block_size"]
        self.rank = self.config["markov_rank"]
        self.window = self.config["sliding_window_size"]
        self.local = set(self.config["local_layer_ids"])
        self.taps = self.config["target_layer_ids"]
        self.cap = config["text_config"]["final_logit_softcapping"]
        self.vocab = config["text_config"]["unpadded_vocab_size"]
        # The shared projection uses the target's group-32 INT8 vocabulary weights.
        self.shared_head = os.environ.get("INKLING2_DRAFT_Q8_HEAD", "0") == "1"
        self.q8_head = None
        self.shortlist = int(os.environ.get("INKLING2_DSPARK_SHORTLIST", "0"))
        if not 0 <= self.shortlist <= self.vocab:
            raise ValueError("Invalid DSpark candidate count")
        self.w = {k: v.to(device) for k, v in load_file(checkpoint / "mtp.safetensors").items()}
        self.mix = self.w["fuse_weight"].float().softmax(-1)
        self.freq = self.config["rope_theta"] ** (
            -torch.arange(0, self.head, 2, device=device).float() / self.head
        )
        self.offsets = torch.arange(self.block, dtype=torch.long, device=device)
        self.split_attention = os.environ.get("INKLING2_DSPARK_BLOCK", "0") == "1"
        self.fp8_global = self.split_attention and os.environ.get("INKLING2_DRAFT_FP8_GLOBAL", "0") == "1"
        self.graph_enabled = os.environ.get("INKLING2_DSPARK_GRAPH", "0") == "1"
        if self.graph_enabled and not self.split_attention:
            raise ValueError("The DSpark graph requires the fixed-layout split attention kernel")
        self.reference_math = os.environ.get("INKLING2_DRAFT_REFERENCE_MATH", "0") == "1"
        self.debug_replay = os.environ.get("INKLING2_COMPARE_DRAFT_GRAPH", "0") == "1"
        self.sampling = None
        self.graph_checks = 0
        self.graph_state = None
        if self.reference_math:
            self.mix = self.mix.to(self.w["fuse_weight"].dtype).float()
        self.draft_mix_norm = getattr(torch.ops.inkling2_xpu, "draft_mix_norm", None)
        fuse_taps = os.environ.get("INKLING2_DRAFT_FUSED_TAPS", "1") == "1"
        if fuse_taps and self.draft_mix_norm is not None and self.mix.shape == (4, 4) and self.width == 2304:
            norms = [self.w[f"layers.{i}.dflare_norm.weight"] for i in range(3)]
            self.draft_norms = torch.stack([*norms, self.w["markov_head.dflare_norm.weight"]]).float()
        else:
            self.draft_mix_norm = None
        self.warmed = set()
        self.max_context = max_context
        self.grow_global = grow_global
        self.kv = []
        self.capacities = []
        slack = 64 if os.environ.get("INKLING2_RING_PREFIX", "0") == "1" else self.block * 2
        self.f_ring = torch.empty(slack, self.width, device=device, dtype=torch.bfloat16)
        for index in range(self.config["num_hidden_layers"]):
            p = f"layers.{index}."
            for group, stems in (
                ("self_attn.qkv", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
                ("self_attn.context_kv", ["self_attn.k_proj_ctx", "self_attn.v_proj_ctx"]),
                ("mlp.gate_up", ["mlp.gate_proj", "mlp.up_proj"]),
            ):
                self.w[p + group] = torch.cat([self.w.pop(p + stem + ".weight") for stem in stems])
            length = self.window + slack if index in self.local else max_context
            self.capacities.append(length if index in self.local else 0)
            if grow_global and index not in self.local:
                length = min(max_context, 4096)
            rows = (
                (2 * length if index in self.local else length + self.block)
                if self.split_attention
                else length
            )
            dtype = torch.float16 if self.split_attention else torch.bfloat16
            if self.fp8_global and index not in self.local:
                rows = (rows + 15) // 16 * 16
                dtype = torch.float8_e4m3fn
            self.kv.append(torch.empty((2, rows, self.groups, self.head), device=device, dtype=dtype))
        shard = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"][
            "model.llm.unembed.weight"
        ]
        with safe_open(checkpoint / shard, framework="pt", device="cpu") as reader:
            self.lm = reader.get_tensor("model.llm.unembed.weight")[: self.vocab].to(device).contiguous()
        candidate_count = self.lm.shape[0]
        if self.shortlist > candidate_count:
            raise ValueError("DSpark shortlist exceeds the draft projection vocabulary")
        w = self.w.pop("markov_head.hidden_out.weight").float()
        b = self.w.pop("markov_head.hidden_out.bias").float()
        self.fold = torch.empty(candidate_count, self.rank, device=device, dtype=torch.bfloat16)
        self.bias = torch.empty(candidate_count, device=device, dtype=torch.float32)
        for first in range(0, candidate_count, 2048):
            rows = self.lm[first : first + 2048].float()
            self.fold[first : first + 2048] = rows @ w
            self.bias[first : first + 2048] = rows @ b
        if self.shared_head:
            del self.lm
        weight = self.w.pop("markov_head.rnn_proj.weight").float()
        self.recurrent, self.context = (
            v.contiguous() for v in weight.split([2 * self.rank, 2 * self.width], dim=1)
        )

    def fork(self):
        """Share read-only weights. Each request slot owns its KV and recurrent state."""
        other = copy(self)
        other.kv = [torch.empty_like(cache) for cache in self.kv]
        other.f_ring = torch.empty_like(self.f_ring)
        other.warmed = set()
        other.graph_state = None
        return other

    def reserve(self, start, end):
        """Grow global draft KV as a slot advances and release surplus on cold reuse."""
        if not self.grow_global:
            return False
        changed = False
        for i, cache in enumerate(self.kv):
            if i in self.local:
                continue
            pad = self.block if self.split_attention else 0
            pages = max(1, (end + 4095) // 4096)
            capacity = min(self.max_context, 4096 * (1 << (pages - 1).bit_length()))
            rows = capacity + pad
            if self.fp8_global:
                rows = (rows + 15) // 16 * 16
            if cache.size(1) < end + pad or (start == 0 and cache.size(1) > rows * 2):
                new = cache.new_empty((2, rows, self.groups, self.head))
                if start:
                    # Packed FP8 tiles interleave bytes from sixteen positions.
                    keep = (start + 15) // 16 * 16 if self.fp8_global else start
                    new[:, :keep].copy_(cache[:, :keep])
                self.kv[i] = new
                self.graph_state = None
                changed = True
        return changed

    def bind_target_head(self, head):
        if self.shared_head:
            self.q8_head = head

    def rotation(self, positions):
        angle = positions.float()[:, None] * self.freq
        return angle.cos()[:, None, :], angle.sin()[:, None, :]

    def rotate(self, x, rotation):
        a, b = x.float().chunk(2, -1)
        cos, sin = rotation
        return torch.cat([a * cos - b * sin, b * cos + a * sin], -1).to(torch.bfloat16)

    def store(self, features, positions):
        # The local and recurrent rings have slack for a whole verification
        # block. Rejected rows are past the next committed boundary and get overwritten.
        fused = None
        if self.draft_mix_norm is not None and len(positions) <= 8:
            fused = self.draft_mix_norm(*features, self.mix, self.draft_norms)
        else:
            sources = torch.stack(features, dim=1).float()
        rotation = self.rotation(positions)
        for i, kv in enumerate(self.kv):
            p = f"layers.{i}."
            limit = self.capacities[i] or len(positions)
            if fused is not None:
                mixed = fused[i, -limit:]
            else:
                src = sources[-limit:]
                mixed = norm((src * self.mix[i, :, None]).sum(1), self.w[p + "dflare_norm.weight"])
            k, v = (
                F.linear(mixed, self.w[p + "self_attn.context_kv"])
                .view(len(mixed), 2, self.groups, self.head)
                .unbind(1)
            )
            k = self.rotate(
                norm(k, self.w[p + "self_attn.k_norm_ctx.weight"]), tuple(x[-limit:] for x in rotation)
            )
            rows = positions[-limit:]
            self.write_rows(i, rows, k, v)
        ring = self.f_ring.size(0)
        if fused is not None:
            self.f_ring[positions[-ring:].remainder(ring)] = fused[-1, -ring:]
            return
        context = (sources[-ring:] * self.mix[-1, :, None]).sum(1)
        self.f_ring[positions[-ring:].remainder(ring)] = norm(
            context, self.w["markov_head.dflare_norm.weight"]
        )

    def write_rows(self, i, rows, k, v):
        kv = self.kv[i]
        capacity = self.capacities[i]
        k, v = k.to(kv.dtype), v.to(kv.dtype)
        if kv.dtype == torch.float8_e4m3fn:
            kv = kv.view(torch.uint8)
            k, v = k.view(torch.uint8), v.view(torch.uint8)
        if capacity:
            rows = rows.remainder(capacity)
            if self.split_attention:
                kv[0, rows + capacity], kv[1, rows + capacity] = k, v
        kv[0, rows], kv[1, rows] = k, v

    def attend_block(self, i, q, k, v, positions, start, splits):
        self.write_rows(i, positions, k, v)
        kv = self.kv[i]
        capacity = self.capacities[i]
        parts = 16 if capacity else splits
        out = q.new_empty((parts, self.block, self.heads, self.head), dtype=torch.float32)
        lse = torch.empty_like(out[:, :, :, 0], memory_format=torch.contiguous_format)
        query = q.to(torch.float16).contiguous()
        if self.fp8_global and not capacity:
            torch.ops.inkling2_xpu.draft_global_xmx(query, kv[0], kv[1], start, out, lse)
        else:
            torch.ops.inkling2_xpu.dspark_attention(
                query, kv[0], kv[1], start, out, lse, capacity, self.window if capacity else 0
            )
        return merge_splits(out, lse).reshape(self.block, -1).to(torch.bfloat16)

    def propose(self, target, anchor, start, steps, upper):
        if steps >= self.block:
            raise ValueError("DSpark needs an anchor and one noise row per proposed token")
        splits = min(512, max(16, 1 << max(0, (max(upper, 1) - 1).bit_length() - 7)))
        if not self.split_attention:
            return self._block(target, anchor, start, steps, splits)
        position = start
        if not self.graph_enabled or splits not in self.warmed:
            self.warmed.add(splits)
            return self._block(target, anchor, position, steps, splits)
        if self.graph_state is None or self.graph_state[0] != splits:
            buf_id, buf_pos = anchor.clone(), position.clone()
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):
                output = self._block(target, buf_id, buf_pos, steps, splits)
            self.graph_state = (splits, graph, buf_id, buf_pos, output)
        _, graph, buf_id, buf_pos, output = self.graph_state
        buf_id.copy_(anchor)
        buf_pos.copy_(position)
        graph.replay()
        if self.debug_replay and self.graph_checks < 8:
            reference = self._block(target, anchor, position, steps, splits)
            if not torch.equal(output, reference):
                raise RuntimeError("DSpark graph proposals differ from an eager pass at the same position")
            self.graph_checks += 1
        return output

    def _block(self, target, anchor, start, steps, splits, drafts=None, sampling=None):
        batch_size = anchor.numel()
        if drafts is None:
            positions = self.offsets + start
            x = torch.cat([target.get_input_embeddings()(anchor).to(torch.bfloat16),
                           self.w["mask_embed"].expand(self.block - 1, -1)])
        else:
            positions = (start[:, None] + self.offsets).flatten()
            x = torch.cat([target.get_input_embeddings()(anchor).to(torch.bfloat16)[:, None],
                           self.w["mask_embed"].expand(batch_size, self.block - 1, -1)], dim=1).flatten(0, 1)
        rotation = self.rotation(positions)
        for i, kv in enumerate(self.kv):
            p = f"layers.{i}."
            h = norm(x, self.w[p + "input_layernorm.weight"])
            q, k, v = (
                F.linear(h, self.w[p + "self_attn.qkv"])
                .view(batch_size * self.block, -1, self.head)
                .split([self.heads, self.groups, self.groups], dim=1)
            )
            q = self.rotate(norm(q, self.w[p + "self_attn.q_norm.weight"]), rotation)
            k = self.rotate(norm(k, self.w[p + "self_attn.k_norm.weight"]), rotation)
            if drafts is not None:
                outputs = []
                for row, draft in enumerate(drafts):
                    span = slice(row * self.block, (row + 1) * self.block)
                    outputs.append(draft.attend_block(i, q[span], k[span], v[span], positions[span],
                                                      start[row:row + 1], splits))
                h = torch.cat(outputs)
            elif self.split_attention:
                h = self.attend_block(i, q, k, v, positions, start, splits)
            else:
                if i in self.local:
                    rows = torch.arange(max(0, start - self.window), start, device=x.device).remainder(
                        kv.size(1)
                    )
                    before_k, before_v = kv[:, rows].unbind()
                else:
                    before_k, before_v = kv[:, :start].unbind()
                keys = torch.cat([before_k, k]).transpose(0, 1).unsqueeze(1)
                values = torch.cat([before_v, v]).transpose(0, 1).unsqueeze(1)
                q = q.transpose(0, 1).reshape(self.groups, -1, self.block, self.head)
                h = F.scaled_dot_product_attention(q, keys, values)
                h = h.reshape(self.heads, self.block, self.head).transpose(0, 1).reshape(self.block, -1)
            projected = F.linear(h, self.w[p + "self_attn.o_proj.weight"])
            if self.reference_math:
                x = x.to(torch.bfloat16).float() + projected.float()
            else:
                x = x + projected
            h = norm(x, self.w[p + "post_attention_layernorm.weight"])
            gate, up = F.linear(h, self.w[p + "mlp.gate_up"]).chunk(2, -1)
            if self.reference_math:
                activated = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
                x = (
                    x.to(torch.bfloat16).float()
                    + F.linear(activated, self.w[p + "mlp.down_proj.weight"]).float()
                )
            else:
                x = x + F.linear(F.silu(gate) * up, self.w[p + "mlp.down_proj.weight"])
        hidden = norm(x.view(batch_size, self.block, -1)[:, 1:steps + 1].reshape(-1, self.width),
                      self.w["norm.weight"])
        if self.q8_head is None:
            base = F.linear(hidden, self.lm).float()
        else:
            base = self.q8_head(hidden)[:, :self.vocab]
        if drafts is None:
            recurrent = self.f_ring[(start - 1) % self.f_ring.size(0)].reshape(1, self.width)
            sampling = self.sampling
        else:
            recurrent = torch.cat([d.f_ring[(start[r:r + 1] - 1) % d.f_ring.size(0)]
                                   for r, d in enumerate(drafts)])
        f_rnn = recurrent[:, None].expand(-1, steps, -1).reshape_as(hidden)
        context = F.linear(torch.cat([hidden, f_rnn], dim=1).float(), self.context,
                           self.w["markov_head.rnn_proj.bias"].float()).view(batch_size, steps, -1)
        base = base.view(batch_size, steps, -1)
        if self.shortlist:
            base, indices = base.topk(self.shortlist, dim=-1)
            folded = F.embedding(indices, self.fold).float()
            biases = self.bias[indices]
        state = hidden.new_zeros((batch_size, self.rank))
        prev = anchor
        proposals = []
        for i in range(steps):
            token = F.embedding(prev, self.w["markov_head.markov_w1.weight"])
            raw = F.linear(torch.cat([state, token], dim=-1).float(), self.recurrent) + context[:, i]
            gate, candidate, latent = raw.to(torch.bfloat16).chunk(3, -1)
            gate = gate.sigmoid()
            state = gate * state + (1 - gate) * candidate.tanh()
            if self.shortlist:
                if batch_size == 1:
                    correction = (folded[0, i] @ latent.tanh().float().T).T + biases[:, i]
                else:
                    correction = torch.bmm(folded[:, i], latent.tanh().float().unsqueeze(-1)).squeeze(-1) + biases[:, i]
            elif self.reference_math:
                correction = F.linear(latent.tanh().float(), self.fold.float()).float() + self.bias
            else:
                correction = F.linear(latent.tanh(), self.fold).float() + self.bias
            if self.reference_math:
                combined = (base[:, i] + correction.to(torch.bfloat16).float()).to(torch.bfloat16)
                logits = (self.cap * torch.tanh(combined.float() / self.cap)).to(torch.bfloat16)
            else:
                logits = self.cap * torch.tanh((base[:, i] + correction) / self.cap)
            ids = indices[:, i] if self.shortlist else None
            if sampling is not None:
                prev = sampling.sample(logits, ids, start + i, i)
            elif self.shortlist:
                prev = torch.where(logits == logits.amax(-1, keepdim=True), ids, self.vocab).amin(-1)
            else:
                prev = logits.argmax(dim=-1)
            proposals.append(prev)
        if sampling is not None and sampling.adaptive:
            keep = sampling.confidence[0] >= 0.75
            for i in range(1, steps):
                proposals[i] = torch.where(keep, proposals[i], -1)
                keep = keep & (sampling.confidence[i] >= 0.75)
        return torch.stack(proposals, dim=1)


class BatchedDraftFeatures:
    def __init__(self, drafts, requests):
        self.drafts = drafts
        self.requests = requests
        self.taps = drafts[0].taps

    def store(self, features, positions):
        for slot, first, count, _ in self.requests:
            rows = slice(first, first + count)
            self.drafts[slot].store([f[rows] for f in features], positions[rows])


class DraftSampling:
    """Use vLLM's position-keyed draft RNG and its matching rejection sampler."""

    def __init__(self, width, logits, device, fp64):
        self.logits = logits
        vocab = logits.shape[-1]
        self.adaptive = os.environ.get("INKLING2_DSPARK_ADAPTIVE", "0") == "1"
        self.confidence = [None] * width if self.adaptive else []
        self.temperature = torch.zeros(logits.size(0), device=device)
        self.seed = torch.zeros(logits.size(0), dtype=torch.int64, device=device)
        self.slot = torch.zeros_like(self.seed)
        self.columns = [torch.tensor(i, dtype=torch.int32, device=device) for i in range(width)]
        self.fp64 = fp64
        self.topk = int(os.environ.get("INKLING2_DRAFT_SAMPLE_TOPK", "0"))
        if not 0 <= self.topk <= vocab:
            raise ValueError("Invalid proposal top-k")

    def sample(self, values, ids, position, step):
        from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

        if 0 < self.topk < values.shape[-1]:
            values, indices = values.topk(self.topk, dim=-1)
            ids = indices if ids is None else ids.gather(1, indices)
        if self.adaptive:
            scaled = values.float() / self.temperature[self.slot].clamp_min(0.01)[:, None]
            self.confidence[step] = scaled.softmax(-1).amax(-1)
        if ids is not None:
            full = torch.full((values.size(0), self.logits.size(-1)), -float("inf"),
                              dtype=torch.float32, device=values.device)
            full.scatter_(1, ids, values.float())
        else:
            full = values.float()
        return gumbel_sample(
            full,
            self.slot,
            self.temperature,
            self.seed,
            torch.as_tensor(position, device=values.device).reshape(-1),
            apply_temperature=True,
            is_drafting=True,
            logits_cache=self.logits,
            logits_cache_col=self.columns[step],
            use_fp64=self.fp64,
        )


class InklingDSparkSpeculator(BaseSpeculator):
    supports_mm_inputs = False
    draft_logits = None

    def __init__(self, config, device):
        self.width = config.speculative_config.num_speculative_tokens
        if not 1 <= self.width < 8 or not 1 <= config.scheduler_config.max_num_seqs <= 4:
            raise ValueError("Inkling DSpark supports up to four sequences and between one and seven proposals")
        slots = config.scheduler_config.max_num_seqs
        self.empty = torch.zeros(slots, self.width, device=device, dtype=torch.long)
        self.no_proposals = torch.full_like(self.empty, -1)
        self.samplers = []
        self.joint_sampling = None
        self.joint_graphs = {}
        self.joint_warm = set()
        if config.speculative_config.draft_sample_method == "probabilistic":
            vocab = config.model_config.get_vocab_size()
            self.draft_logits = torch.full((slots, self.width, vocab), -float("inf"), device=device, dtype=torch.float32)
            self.joint_sampling = DraftSampling(self.width, self.draft_logits, device,
                                                config.model_config.use_fp64_gumbel)
            self.samplers = [
                DraftSampling(self.width, self.draft_logits[i:i + 1], device, config.model_config.use_fp64_gumbel)
                for i in range(slots)
            ]

    def _propose_batch(self, slots, worker_slots, starts, last_sampled, temperature, seeds):
        # Stable ring order keeps a graph usable after vLLM reorders its batch.
        order = sorted(range(len(slots)), key=slots.__getitem__)
        perm = torch.tensor(order, device=starts.device)
        indices = worker_slots[perm].long()
        starts = starts[perm].contiguous()
        anchor = last_sampled[indices].reshape(-1).long()
        drafts = [self.target.drafts[slots[r]] for r in order]
        upper = max(self.target.draft_lengths[slots[r]] for r in order)
        splits = min(512, max(16, 1 << max(0, (max(upper, 1) - 1).bit_length() - 7)))
        sample = self.joint_sampling
        if sample is not None:
            sample.seed.copy_(seeds[:len(self.samplers)])
            sample.temperature.copy_(temperature[:len(self.samplers)])
        key = tuple(slots[r] for r in order)
        signature = (splits, tuple((k.data_ptr(), k.size(1)) for d in drafts for k in d.kv))
        def forward(ids, pos, mapping):
            if sample is not None:
                sample.slot = mapping
            return drafts[0]._block(self.target, ids, pos, self.width, splits, drafts, sample)
        if not drafts[0].graph_enabled or (key, splits) not in self.joint_warm:
            self.joint_warm.add((key, splits))
            out = forward(anchor, starts, indices)
        else:
            state = self.joint_graphs.get(key)
            if state is None or state[0] != signature:
                buffers = tuple(t.clone() for t in (anchor, starts, indices))
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    out = forward(*buffers)
                state = (signature, graph, buffers, out)
                self.joint_graphs[key] = state
            _, graph, buffers, out = state
            for buf, t in zip(buffers, (anchor, starts, indices), strict=True):
                buf.copy_(t)
            graph.replay()
        return out[perm.argsort()]

    def init_cudagraph_manager(self, cudagraph_mode):
        pass

    def capture(self):
        pass

    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        *,
        dummy_run=False,
        is_profile=False,
        **kwargs,
    ):
        if dummy_run or is_profile or not input_batch.num_reqs:
            return self.empty[: input_batch.num_reqs]
        slots = self.target.inkling_batch[1] if self.target.n_slots > 1 else (0,)
        count = input_batch.num_reqs
        starts = input_batch.seq_lens[:count].int() - num_rejected[:count].int()
        worker_indices = input_batch.idx_mapping_np
        worker_slots = input_batch.idx_mapping
        selected = None
        if input_batch.has_prefill:
            # A non-final chunk has no sampled anchor. Leave that request's
            # proposal rows empty while its neighbours continue drafting.
            complete = (input_batch.num_computed_prefill_tokens_np + input_batch.num_scheduled_tokens
                        >= input_batch.prefill_len_np)
            if not complete.all():
                rows = complete.nonzero()[0]
                if not len(rows):
                    return self.no_proposals[:count]
                selected = torch.as_tensor(rows, device=starts.device)
                slots = [slots[i] for i in rows]
                worker_indices = worker_indices[rows]
                worker_slots = worker_slots[selected]
                starts = starts[selected]
        if len(slots) > 1 and self.target.drafts[0].split_attention:
            result = self._propose_batch(slots, worker_slots, starts, last_sampled, temperature, seeds)
        else:
            result = self._propose_separately(slots, worker_indices, starts, last_sampled, temperature, seeds)
        if selected is None:
            return result
        return self.no_proposals[:count].index_copy(0, selected, result)

    def _propose_separately(self, slots, worker_indices, starts, last_sampled, temperature, seeds):
        proposals = []
        for i, (slot, worker_idx) in enumerate(zip(slots, worker_indices, strict=True)):
            draft = self.target.drafts[slot]
            if self.samplers:
                sampler = self.samplers[worker_idx]
                if draft.sampling is not sampler:
                    draft.graph_state = None
                    draft.sampling = sampler
                sampler.temperature.copy_(temperature[worker_idx:worker_idx + 1])
                sampler.seed.copy_(seeds[worker_idx:worker_idx + 1])
            upper = self.target.draft_lengths[slot]
            start = starts[i:i + 1] if draft.split_attention else int(starts[i].item())
            anchor = last_sampled[worker_idx].reshape(1).long()
            proposals.append(draft.propose(self.target, anchor, start, self.width, upper))
        return proposals[0] if len(proposals) == 1 else torch.cat(proposals)
