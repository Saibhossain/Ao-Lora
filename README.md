# AO-LoRA: Activation-Orthogonal Low-Rank Adaptation

Official PyTorch implementation of **AO-LoRA**, a stability-oriented PEFT framework for data-efficient fine-tuning. 

Standard PEFT methods often exhibit high variance and overconfidence in low-resource regimes due to uncontrolled interaction between pretrained representations and adapter updates. AO-LoRA explicitly regulates representational overlap between backbone and adapter pathways using an activation-level orthogonality penalty and a per-layer gated pathway.

This reduces Expected Calibration Error (ECE) and prevents overconfidence, making it highly effective for fine-tuning expert agents on small, specialized datasets (e.g., Clinical/Oncology literature).

## Installation

```bash
git clone [https://github.com/Saibhossain/AO-LORA.git](https://github.com/Saibhossain/AO-LORA.git)
cd AO-LORA
pip install -e .