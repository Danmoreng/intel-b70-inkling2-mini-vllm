# Inkling 2 Mini measurements on the Arc Pro B70

The reference runs used one 32 GB B70 at 180 W. The benchmark client generates
a synthetic logbook with an exact token count and a validation code in the
middle. The receipts include the prompt hash, timing, completion length and
retrieval result. All model requests here return 128 output tokens. For vLLM,
prefill and decode times come from the request histogram deltas at `/metrics`.
Decode throughput counts the 127 tokens after the first. The llama.cpp
client uses that server's own prefill and decode timers.

## vLLM EXL3

The A/B configuration uses 32-row XMX expert tiles, a 2,048-token prefill
budget, a 256-token DPAS threshold, FP8 E4M3 KV storage and prefix caching
disabled. The full-ESIMD control uses `INKLING2_ONEDNN_PREFIX_BLOCK=0`.
The hybrid run uses `INKLING2_ONEDNN_PREFIX_BLOCK=8192`.
The 4K/32K runs used 40,960 maximum context and a 768 MiB KV pool.
Each long run used a separate 4K warm-up followed by the measured request.
The long runs used 262,144 maximum context and a 4,311,744,512-byte KV pool.

| Input tokens | Run | Prefill tok/s | Decode tok/s | Evidence |
| ---: | --- | ---: | ---: | --- |
| 4,096 | ESIMD, second 4K request | 1,847.25 | 19.21 | [4K/32K control](vllm-exl3-dpas-fp8-mb32-b2048.jsonl) |
| 4,096 | Hybrid configured, second 4K request | 1,865.84 | 20.06 | [4K/32K hybrid](vllm-onednn-8k.jsonl) |
| 32,768 | ESIMD, two prompts | 1,652.08–1,660.68 | 19.78–19.90 | [4K/32K control](vllm-exl3-dpas-fp8-mb32-b2048.jsonl) |
| 32,768 | Hybrid, same two prompts | 1,717.60–1,755.21 | 19.96–19.97 | [4K/32K hybrid](vllm-onednn-8k.jsonl) |
| 196,608 | ESIMD | 952.87 | 19.51 | [192K control](vllm-exl3-192k.jsonl) |
| 196,608 | Hybrid, same prompt | 1,392.86 | 19.30 | [192K hybrid](vllm-onednn-192k.jsonl) |
| 262,016 | Hybrid, full 256K total context | 1,289.78 | 19.21 | [256K hybrid](vllm-onednn-256k.jsonl) |

For a long test, start the server using the [README command](../../README.md#build-and-serve)
and warm it up with one separate 4K request. Then run the client with
`--lengths 196608` or `--lengths 262016`, `--decode 128` and `--timeout 300`.
For the 32K A/B, use `--lengths 4096 32768 --replicates 2` and restart the
server between configurations.

## llama.cpp GGUF

At 4K, the second request yielded 308.9 prefill and 26.5 decode tok/s
for [ordinary decode](prefill-tuned-ordinary-4k.jsonl), and 303.7 prefill
and 33.3 decode tok/s for [DSpark](prefill-tuned-dspark-4k.jsonl). The
DSpark request accepted 91 draft tokens. The first request in each process
includes SYCL JIT costs.

## Native correctness checks

The [SYCL test output](vllm-onednn-hybrid-parity.log) covers simulated vLLM
block-table layouts, BF16 and FP8 KV storage, and the hybrid split with two
prefix blocks, a nonzero attention sink and relative bias. The hybrid output
had relative L2 error of `3.57e-4` versus the full ESIMD kernel and passed a
sampled FP32 CPU reference check.
