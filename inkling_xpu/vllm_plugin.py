"""Register the separate Inkling model in each vLLM process."""


def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model("Inkling2B70ForCausalLM", "inkling_xpu.vllm_model:Inkling2B70ForCausalLM")
