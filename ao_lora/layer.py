import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAOLoRALinear(nn.Module):
    def __init__(self, lora_layer):
        super().__init__()
        self.lora_layer = lora_layer

        target_device = lora_layer.lora_A["default"].weight.device
        target_dtype = torch.float32

        self.gate = nn.Parameter(
            torch.tensor(1.0, device=target_device, dtype=target_dtype)
        )
        self.current_cos = None

    def forward(self, x, *args, **kwargs):
        h_base = self.lora_layer.base_layer(x, *args, **kwargs)
        dropout_x = self.lora_layer.lora_dropout["default"](x)
        lora_A_out = self.lora_layer.lora_A["default"](dropout_x)
        h_lora = self.lora_layer.lora_B["default"](lora_A_out) * self.lora_layer.scaling["default"]

        hb = F.normalize(h_base.float(), dim=-1)
        hl = F.normalize(h_lora.float(), dim=-1)
        cos = (hb * hl).sum(-1)

        self.current_cos = cos.mean()
        output = h_base + torch.sigmoid(self.gate) * h_lora.to(h_base.device)

        return output.to(h_base.dtype)