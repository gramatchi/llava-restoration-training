"""Asks the model for a restoration plan for one photo.

Run it from inside the LLaVA repo, with the files from llava_files/ copied in.
The photo is given to the model as it is (nothing is distorted here).
"""
import argparse

import torch
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

DEFAULT_QUESTION = "Analyze this image. Identify any distortions present and give a step-by-step restoration plan."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="folder with the LoRA adapter, its name must contain 'lora'")
    parser.add_argument("--model-base", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    if "lora" not in model_name.lower():
        raise SystemExit(f"'{model_name}' has no 'lora' in it, LLaVA would ignore the adapter. Rename the folder.")
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, args.model_base, model_name)

    qs = args.question
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image = Image.open(args.image).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)[0]
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).to(dtype=model.dtype, device="cuda"),
            image_sizes=[image.size],
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            use_cache=True)
    print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip())


if __name__ == "__main__":
    main()
