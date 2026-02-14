import torch
from transformers import Trainer
from .layer import GatedAOLoRALinear


class AOTrainer(Trainer):
    """
    Custom Trainer that applies the Activation Orthogonality (AO) penalty
    to prevent representational collapse and overconfidence.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1. Standard task loss calculation
        outputs = model(**inputs)
        task_loss = outputs.loss

        ao_loss = 0.0
        count = 0

        # Fetch hyperparameters dynamically from the model's attached config
        tau = getattr(model, "ao_config", None).tau if hasattr(model, "ao_config") else 0.15
        lambda_ao = getattr(model, "ao_config", None).lambda_ao if hasattr(model, "ao_config") else 0.01

        # 2. Calculate L_AO penalty
        for module in model.modules():
            if isinstance(module, GatedAOLoRALinear) and module.current_cos is not None:
                # L_AO = E[ max(0, |cos| - tau)^2 ]
                penalty = torch.relu(torch.abs(module.current_cos) - tau) ** 2
                ao_loss += penalty
                count += 1

        if count > 0:
            ao_loss = ao_loss / count

        # 3. Final Objective
        total_loss = task_loss + (lambda_ao * ao_loss)

        return (total_loss, outputs) if return_outputs else total_loss