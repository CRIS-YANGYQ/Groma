import os
import copy
import torch
import argparse
import requests
from io import BytesIO
from PIL import Image, ImageDraw
from transformers.image_transforms import center_to_corners_format
from transformers import AutoTokenizer, AutoImageProcessor, BitsAndBytesConfig
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from groma.utils import disable_torch_init
from groma.model.groma import GromaModel
from groma.constants import DEFAULT_TOKENS
from groma.data.conversation import conv_templates


def init_distributed():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.barrier()


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def draw_box(box, image, index, output_dir):
    w, h = image.size
    box = [box[0] * w, box[1] * h, box[2] * w, box[3] * h]
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline="red")
    output_file = os.path.join(output_dir, 'r{}.jpg'.format(index))
    image.save(output_file, "JPEG")


def eval_model(model_name, quant_type, image_file, query, args):
    disable_torch_init()
    model_name = os.path.expanduser(model_name)
    vis_processor = AutoImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    kwargs = {}
    if quant_type == 'fp16':
        kwargs['torch_dtype'] = torch.float16
    elif quant_type == '8bit':
        kwargs['load_in_8bit'] = True
    elif quant_type == '4bit':
        int4_quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.uint8,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type='nf4'
        )
        kwargs = {'quantization_config': int4_quant_cfg}

    # Load model on multiple GPUs using DDP
    model = GromaModel.from_pretrained(model_name, **kwargs).to(torch.device("cuda"))
    model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])
    model.init_special_token_id(tokenizer)

    conversations = []
    instruct = "Here is an image with region crops from it. "
    instruct += "Image: {}. ".format(DEFAULT_TOKENS['image'])
    instruct += "Regions: {}.".format(DEFAULT_TOKENS['region'])
    answer = 'Thank you for the image! How can I assist you with it?'
    conversations.append((conv_templates['llava'].roles[0], instruct))
    conversations.append((conv_templates['llava'].roles[1], answer))
    conversations.append((conv_templates['llava'].roles[0], query))
    conversations.append((conv_templates['llava'].roles[1], ''))
    prompt = conv_templates['llava'].get_prompt(conversations)

    inputs = tokenizer([prompt])
    input_ids = torch.as_tensor(inputs.input_ids).to(torch.device("cuda"))

    raw_image = load_image(image_file)
    raw_image = raw_image.resize((448, 448))
    image = vis_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'].to(torch.device("cuda"))

    with torch.inference_mode():
        with torch.autocast(device_type="cuda"):
            outputs = model.module.generate(  # Use model.module for DDP
                input_ids,
                images=image,
                use_cache=True,
                do_sample=False,
                max_new_tokens=1024,
                return_dict_in_generate=True,
                output_hidden_states=True,
                generation_config=model.module.generation_config,
            )
    output_ids = outputs.sequences
    input_token_len = input_ids.shape[1]
    pred_boxes = outputs.hidden_states[0][-1]['pred_boxes'][0].cpu()
    pred_boxes = center_to_corners_format(pred_boxes)

    box_idx_token_ids = model.module.box_idx_token_ids
    selected_box_inds = [box_idx_token_ids.index(id) for id in output_ids[0] if id in box_idx_token_ids]
    selected_box_inds = [x for x in selected_box_inds if x < len(pred_boxes)]

    output_dir = os.path.join(args.output_dir, image_file.split('.')[0].split('/')[-1])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, box in enumerate(pred_boxes[selected_box_inds, :]):
        img_copy = copy.deepcopy(raw_image)
        draw_box(box, img_copy, selected_box_inds[i], output_dir)

    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=False)[0]
    outputs = outputs.strip()
    print(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="checkpoints/groma-finetune/")
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default='output')
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--quant_type", type=str, default='none')
    args = parser.parse_args()

    init_distributed()  # Initialize distributed environment

    model_name = os.path.expanduser(args.model_name)
    if args.image_dir is not None:
        image_files = sorted(os.listdir(args.image_dir))
        for image_file in image_files:
            image_file = os.path.join(args.image_dir, image_file)
            eval_model(model_name, args.quant_type, image_file, args.query, args)
    elif args.image_file is not None:
        eval_model(model_name, args.quant_type, args.image_file, args.query, args)
    else:
        print("Please specify image file or image directory.")

    dist.destroy_process_group()  # Clean up distributed environment
