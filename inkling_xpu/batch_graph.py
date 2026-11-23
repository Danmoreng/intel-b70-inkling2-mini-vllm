"""Replay native target forwards across request reorderings and different B7 widths."""

from collections import OrderedDict
from types import SimpleNamespace

import torch


class FeatureTaps:
    """Keep target feature tensors so the current slot's drafter can consume a replay."""

    def __init__(self, taps):
        self.taps = taps
        self.features = None

    def store(self, features, positions):
        self.features = tuple(features)


class BatchGraphs:
    def __init__(self, limit=16):
        self.graphs = OrderedDict()
        self.limit = limit
        self.seen = set()

    def run(self, model, ids, positions, batch, tensors, width):
        # Request membership and token boundaries are all tensor inputs. Split
        # counts, allocation addresses and tensor shapes specialize the graph.
        splits = model._batch_splits(batch.total_tokens, width)
        key = (ids.shape[0], tensors[1].numel(), width, splits, tuple(batch.block_table.shape), ids.dtype,
               tuple(cache.data_ptr() for cache in batch.caches))
        state = self.graphs.get(key)
        args = (ids, positions, batch.block_table, batch.seq_lens, batch.out_cache_loc, *tensors)
        if state is None:
            if key not in self.seen:
                self.seen.add(key)
                return model._run_verify(ids, positions, batch, None, tensors, width)
            if len(self.graphs) >= self.limit:
                self.graphs.popitem(last=False)
            buffers = tuple(arg.clone() for arg in args)
            staged = SimpleNamespace(**vars(batch))
            staged.block_table, staged.seq_lens, staged.out_cache_loc = buffers[2:5]
            graph = torch.xpu.XPUGraph()
            draft = getattr(model, "active_draft", None)
            taps = FeatureTaps(draft.taps) if draft is not None else None
            model.active_draft = taps
            try:
                with torch.xpu.graph(graph):
                    out = model._run_verify(*buffers[:2], staged, None, buffers[5:], width)
            finally:
                model.active_draft = draft
            state = (graph, buffers, out, taps)
            self.graphs[key] = state
        else:
            self.graphs.move_to_end(key)
            for buf, src in zip(state[1], args, strict=True):
                buf.copy_(src)
        graph, _, out, taps = state
        graph.replay()
        if taps is not None:
            model.active_draft.store(taps.features, positions)
        return out
