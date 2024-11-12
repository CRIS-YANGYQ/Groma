import json
import re
import os
import torch
import random
import argparse
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh

from groma.utils import init_distributed_mode
from groma.constants import DEFAULT_TOKENS
from groma.model.groma import GromaModel
from groma.data.datasets.refcoco_cap import RefCOCOCap, INSTRUCTIONS
from groma.data.datasets.det_data import normalize_box_coordinates


class RefCOCOGCapTest(RefCOCOCap):
    def _parse_ann_info(self, img_info, ann_info):
        gt_labels = []
        gt_bboxes = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        # flickr
        for ann in ann_info:
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue

            bbox = [x1, y1, x1 + w, y1 + h]
            gt_bboxes.append(bbox)
            gt_labels.append(ann['caption'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['file_name'].replace('jpg', 'png')

        ann = dict(
            id=img_info['id'],
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)
        return ann

    def preprocess(self, data_item):
        image = data_item['img'].data
        bboxes = data_item['gt_bboxes'].data
        img_id = data_item['img_info']['id']
        img_shape = data_item['img_metas'].data['img_shape']
        bboxes = bbox_xyxy_to_cxcywh(bboxes)
        bboxes = normalize_box_coordinates(bboxes, img_shape)
        assert len(bboxes) == 1, bboxes

        conversations = []
        instruct = "Here is an image with region crops from it. "
        instruct += "Image: {}. ".format(DEFAULT_TOKENS['image'])
        instruct += "Regions: {}.".format(DEFAULT_TOKENS['region'])
        answer = "Thank you for the image! How can I assist you with it?"
        conversations.append((self.conv_temp.roles[0], instruct))
        conversations.append((self.conv_temp.roles[1], answer))

        refer_exp = DEFAULT_TOKENS['bor'] + DEFAULT_TOKENS['rbox'] + DEFAULT_TOKENS['eor']
        refer_exp += DEFAULT_TOKENS['rfeat']
        instruct = random.choice(INSTRUCTIONS).format(refer_exp)
        conversations.append((self.conv_temp.roles[0], instruct))
        conversations.append((self.conv_temp.roles[1], ''))
        prompt = self.conv_temp.get_prompt(conversations)

        # tokenize conversations
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True
        ).input_ids

        data_dict = dict(
            input_ids=input_ids,
            image=image,
            image_id=img_id,
            bboxes=bboxes)
        return data_dict


def custom_collate_fn(batch):
    assert len(batch) == 1
    input_ids = batch[0]['input_ids']
    image = batch[0]['image'].unsqueeze(dim=0)
    image_id = batch[0]['image_id']
    bboxes = batch[0]['bboxes']
    return input_ids, image, image_id, bboxes


def eval_model(args):
    # Model
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = GromaModel.from_pretrained(model_name).cuda()
    model.init_special_token_id(tokenizer)

    dataset = RefCOCOGCapTest(
        ann_file=args.ann_file,
        img_prefix=args.img_prefix,
        tokenizer=tokenizer,
        test_mode=True,
        conv_temp='llava'
    )

    distributed_sampler = DistributedSampler(dataset, rank=args.rank, shuffle=False)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size_per_gpu, num_workers=4,
        sampler=distributed_sampler, collate_fn=custom_collate_fn)

    results = []
    for input_ids, image, image_id, bboxes in dataloader:
        input_ids = input_ids.cuda()
        image = image.cuda()
        bboxes = bboxes.cuda()
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=image,
                refer_boxes=[bboxes],
                use_cache=True,
                do_sample=False,
                max_new_tokens=100,
                return_dict_in_generate=True,
                output_hidden_states=True,
                generation_config=model.generation_config
            )
        output_ids = outputs.sequences
        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.replace("\n", "").replace("  ", " ")
        outputs = re.sub(r'<.*?>', '', outputs)
        outputs = ' '.join(outputs.split()).strip("'")
        outputs = outputs.strip()
        result = {"image_id": image_id, "caption": outputs}
        results.append(result)

    os.makedirs(args.result_dir, exist_ok=True)
    results_path = f"{args.result_dir}/{os.path.basename(args.model_name)}_{args.rank}.json"
    with open(results_path, 'w') as json_file:
        json.dump(results, json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="checkpoints/groma-finetune/")
    parser.add_argument("--ann-file", type=str, default="refcocog_cap_val.json")
    parser.add_argument("--img-prefix", type=str, default="coco/train2017")
    parser.add_argument("--result-dir", type=str, default="refcocog_eval_output")
    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()

    init_distributed_mode(args)
    eval_model(args)