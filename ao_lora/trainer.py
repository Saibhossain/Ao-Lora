import torch
from transformers import Trainer
from .layer import GatedAOLoRALinear


class AOTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        outputs = model(**inputs)
        task_loss = outputs.loss

        ao_loss = 0.0
        count = 0

        tau = getattr(model, "ao_config", None).tau if hasattr(model, "ao_config") else 0.15
        lambda_ao = getattr(model, "ao_config", None).lambda_ao if hasattr(model, "ao_config") else 0.01

        for module in model.modules():
            if isinstance(module, GatedAOLoRALinear) and module.current_cos is not None:
                # L_AO = E[ max(0, |cos| - tau)^2 ]
                penalty = torch.relu(torch.abs(module.current_cos) - tau) ** 2
                ao_loss += penalty
                count += 1

        if count > 0:
            ao_loss = ao_loss / count

        total_loss = task_loss + (lambda_ao * ao_loss)

        return (total_loss, outputs) if return_outputs else total_loss