---
base_model: liuhaotian/llava-v1.5-7b
library_name: peft
tags:
  - llava
  - lora
  - image-restoration
---

# LLaVA-1.5-7B restoration planner (LoRA, checkpoint-28800)

LoRA adapter for `liuhaotian/llava-v1.5-7b`. Given a photo, it answers with the
distortions it finds (JPEG compression, Gaussian noise, exposure shift; zero to
three), their severity, the order in which to undo them, and the restoration
method for each step.

## Files

| File | Size | What |
|---|---|---|
| `adapter_model.bin` | 640 MB | LoRA weights (r=64, alpha=128, all attention and MLP projections) |
| `non_lora_trainables.bin` | 84 MB | fine-tuned multimodal projector |
| `adapter_config.json`, `config.json` | small | configs |

## Loading

The base model is downloaded from Hugging Face separately. With the LLaVA repo
(see the training repo for the exact commit and our changes):

```python
from llava.model.builder import load_pretrained_model

tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path="path/to/this/folder",
    model_base="liuhaotian/llava-v1.5-7b",
    model_name="llava-lora-restoration",
)
```

The model name has to contain `lora`, otherwise LLaVA does not merge the adapter
and you silently get the base model. The same goes for the folder name if the name
is derived from the path.

## Training

Trained on 38400 examples (2400 photos, 16 examples per photo, quarter each with
0/1/2/3 distortions) with the distortions applied on the fly. 3 epochs, lr 2e-4,
batch size 4, one L40S. Training code and data: see the GitHub repo.
Evaluation on 1000 held-out photos: https://github.com/gramatchi/final-test-llava

## Limitations

Trained and tested only on synthetic distortions of three types applied to
ordinary photos. Mild distortions are often missed, and exposure shifts are hard
to tell from a naturally dark or bright scene. Not tested on real forensic or
surveillance material.

## License

The adapter is a derivative of LLaVA-1.5-7B, so the terms of the base model
(and of the Llama 2 model it builds on) apply. Check them before reuse.
