# Third-party notices

- The patches to [Trellis Serve](https://github.com/sybil-solutions/trellis-serve/tree/8f8e2fccf0c82c9715b774812f30d9d52f78a5fb)
  include context from upstream MIT-licensed code. [Upstream MIT license](licenses/trellis-serve.txt).
- The llama.cpp patches target [Unsloth's release](https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8).
  [Upstream MIT license](licenses/llama.cpp.txt).
- The optional oneDNN path loads the native extension from the
  [Qwen B70 recipe](https://github.com/Danmoreng/intel-b70-qwen38-vllm/blob/0356a370da5f73f6b0a4eab613d7103a2356b513/NOTICE.md).
  Its notice records the licenses for components that are built or downloaded
  by that recipe.
- The integration patches for vLLM 0.33 modify its V2 speculator adapter,
  text renderer and single-sequence prefix-cache scheduler. vLLM is
  licensed under Apache 2.0. [vLLM license](licenses/vllm.txt).
- The checkpoint downloads link to the respective publishers' repositories and
  terms. See [setup](docs/SETUP.md) for pinned revisions.
