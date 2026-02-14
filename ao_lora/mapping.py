from peft import LoraConfig, get_peft_model as hf_get_peft_model
from .layer import GatedAOLoRALinear

def get_ao_peft_model(model, config):
    peft_config = LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        task_type="CAUSAL_LM"
    )
    model = hf_get_peft_model(model, peft_config)

    for name, module in dict(model.named_modules()).items():
        if hasattr(module, "lora_A") and any(proj in name for proj in config.target_modules):
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr = parts[-1]

            setattr(parent, attr, GatedAOLoRALinear(module))

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model.ao_config = config
    return model