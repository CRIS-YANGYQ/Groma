import os
import copy
import torch
import argparse
import requests
from io import BytesIO
from PIL import Image, ImageDraw
from transformers.image_transforms import center_to_corners_format
from transformers import AutoTokenizer, AutoImageProcessor, BitsAndBytesConfig

from torch.utils.data import Dataset
from PIL import Image
import json
# import sys
# sys.path.append('/comp_robot/yangyuqin/workplace/Multi-model/models/Groma')
# print(1)
from groma.utils import disable_torch_init
# print(2)
from groma.model.groma import GromaModel
# print(3)
from groma.constants import DEFAULT_TOKENS
# print(4)
from groma.data.conversation import conv_templates
# print(5)

class COCO_Data(Dataset):
    def __init__(self, image_path, json_path):
        self.image_path = image_path
        self.json_path = json_path
        self.json_data = json.load(open(json_path, 'r'))
        self.image_data = self.json_data['images']
        self.annotation_data = self.json_data['annotations']
        self.category_data = self.json_data['categories']
        self.img_id_to_filename = {img['id']: img['file_name'] for img in self.image_data}
        self.cato_id_to_name = {cat['id']: cat['name'] for cat in self.category_data}
        self.cato_name_to_id = {cat['name']: cat['id'] for cat in self.category_data}
        self.img_id_to_annos = {}
        for ann in self.annotation_data:
            img_id = ann['image_id']
            if img_id not in self.img_id_to_annos:
                self.img_id_to_annos[img_id] = []
            self.img_id_to_annos[img_id].append(ann)

    def check_image_path(self, index):
        img_info = self.image_data[index]
        img_id = img_info['id']
        img_name = self.img_id_to_filename[img_id]
        img_path = os.path.join(self.image_path, img_name)
        if not os.path.exists(img_path):
            print(f"Image {img_path} not found")
            return False
        return True
    
    def get_file_id2name(self):
        return self.img_id_to_filename
    
    def get_category_id2name(self):
        return self.cato_id_to_name
    
    def get_category_name2id(self):
        return self.cato_name_to_id

    def __len__(self) -> int:
        return len(self.image_data)

    def __getitem__(self, idx):
        img_info = self.image_data[idx]
        img_id = img_info['id']
        img_name = self.img_id_to_filename[img_id]
        img_path = os.path.join(self.image_path, img_name)
        img = Image.open(img_path).convert('RGB')
        annos = self.img_id_to_annos.get(img_id, [])
        target = []
        for ann in annos:
            cat_id = ann['category_id']
            cat_name = self.cato_id_to_name[cat_id]
            bbox = ann['bbox']
            area = ann['area']
            target.append({'id': ann['id'], 'file_path': img_path, 'category_id': cat_id, 'category_name': cat_name, 'bbox': bbox, 'area': area, "iscrowd": ann['iscrowd']})
        return {'image': img, 'target': target, 'id': img_id, 'img_info': img_info}
    
    def __getitems__(self, list_idx):
        results = []
        for idx in list_idx:
            temp_dict = self.__getitem__(idx)
            results.append(temp_dict)
        return results
    
    
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
    return

def find_target_bboxes(input_str, target_str=None):
    """
    在字符串中找到目标子字符串，并返回其后连续的bbox列表。
    
    参数:
        input_str (str): 输入字符串，可能包含bbox。
        target_str (str): 目标子字符串。
        
    返回:
        list: 连续bbox的列表，每个bbox为一个包含4个float数字的列表。
    """
    import re
    # 正则表达式匹配bbox格式：[float, float, float, float]
    bbox_pattern = r'\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\]'
    input_str = input_str.lower()
    if target_str is not None:
        target_str = target_str.lower()

        # 找到目标子字符串的位置
        target_index = input_str.find(target_str)
        if target_index == -1:
            return []  # 若未找到目标字符串，返回空列表
        
        # 从目标字符串开始切片
        sliced_str = input_str[target_index + len(target_str):]

    else:
        sliced_str = input_str
    # 在切片后寻找所有bbox
    matches = re.findall(bbox_pattern, sliced_str)
    
    # 判断连续性，确保bbox是相邻的
    result = set()
    current_index = 0  # 初始从切片的开头
    
    for match in matches:
        # 格式化当前bbox为字符串形式
        bbox_str = f"[{', '.join(match)}]"
        next_index = sliced_str.find(bbox_str, current_index)
        
        # 如果当前bbox和上一个bbox是连续的（中间没有其他字符）
        if next_index == current_index or current_index == 0:
            # print(str(list(map(float, match))))
            result.add(str(list(map(float, match))))
            current_index = next_index + len(bbox_str)
        else:
            for i in range(current_index, next_index):
                if sliced_str[i]!= " ":
                    break
                if i == next_index - 1:  # 若所有字符都为空格，则认为是连续的
                    # print(str(list(map(float, match))))
                    result.add(str(list(map(float, match))))
                    current_index = next_index + len(bbox_str)
    result_lst = [eval(ele) for ele in result]


    return result_lst


def eval_model(model_name, quant_type, image_file, query,):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(model_name)
    vis_processor = AutoImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    # print(1)


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
    # print(2)

    if quant_type == '8bit' or quant_type == '4bit':
        model = GromaModel.from_pretrained(model_name, **kwargs)
    else:
        model = GromaModel.from_pretrained(model_name, **kwargs).cuda()
    # print(3)

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
    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    raw_image = load_image(image_file)
    raw_image = raw_image.resize((448, 448))
    image = vis_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'].to('cuda')
    # print(4)

    with torch.inference_mode():
        with torch.autocast(device_type="cuda"):
            outputs = model.generate(
                input_ids,
                images=image,
                use_cache=True,
                do_sample=False,
                max_new_tokens=1024,
                return_dict_in_generate=True,
                output_hidden_states=True,
                generation_config=model.generation_config,
                # user-specified box input [x, y, w, h] (normalized)
                # refer_boxes=[torch.tensor([0.5874, 0.4748, 0.3462, 0.5059]).cuda().reshape(1, 4)]
            )
    # print(5)

    output_ids = outputs.sequences
    input_token_len = input_ids.shape[1]
    pred_boxes = outputs.hidden_states[0][-1]['pred_boxes'][0].cpu()
    pred_boxes = center_to_corners_format(pred_boxes)
    # print(6)

    box_idx_token_ids = model.box_idx_token_ids
    selected_box_inds = [box_idx_token_ids.index(id) for id in output_ids[0] if id in box_idx_token_ids]
    selected_box_inds = [x for x in selected_box_inds if x < len(pred_boxes)]

    selected_bboxes = pred_boxes[selected_box_inds, :].tolist()  # Convert to list for easier usage

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
    # print(outputs)
    return outputs, selected_bboxes

question_template = "Find out all instances of {} in the image"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="/comp_robot/yangyuqin/workplace/Multi-model/models/Groma/checkpoints/groma-finetune")
    parser.add_argument("--image-dir", type=str, default=None)
    # parser.add_argument("--image-file", type=str, default='/comp_robot/yangyuqin/workplace/Multi-model/datasets/raw/Downstream/Paddle/xml/Falldown_pp8/JPEGImages/people(38).jpg')
    parser.add_argument("--output-dir", type=str, default='output')
    # parser.add_argument("--query", type=str, default='Can you provide a description of the image and include the coordinates [[x0,y0,x1,y1]] for each instance of person?')
    # parser.add_argument("--query", type=str, default='Find out all instances of {} in the image')
    parser.add_argument("--image_path", type=str, default="/comp_robot/liushilong/data/coco/val2017")
    parser.add_argument("--json_path", type=str, default="/comp_robot/liushilong/data/coco/annotations/instances_val2017.json")
    parser.add_argument("--quant_type", type=str, default='fp16') # support ['none', 'fp16', '8bit', '4bit'] for inference
    parser.add_argument("--result_path", type=str, default="/comp_robot/yangyuqin/workplace/Multi-model/result/coco_obj_detection/groma-finetune.json")
    parser.add_argument("--response_max_length", type=int, default=2048, help="Maximum length of the response tokens")

    
    args = parser.parse_args()
    print("Start evaluating Groma model.")

    model_name = os.path.expanduser(args.model_name)

    # COCO dataset
    print("Loading COCO dataset")
    dataset = COCO_Data(json_path=args.json_path, image_path=args.image_path)
    category_id2name = dataset.get_category_id2name()
    category_name2id = dataset.get_category_name2id()
    file_id2name = dataset.get_file_id2name()

    
    # question_template = "In this picture, what object is the region {}?"
    # question_template = "Identify all instances of {} in the photo."
    print("Start inference")
    results = []
    empty_img_ids = []

    for element_idx, element in enumerate(dataset):
        # if element_idx >= 20:
        #     break
        target = element['target']
        img_info = element['img_info']
        img_id = img_info['id']
        img_basename = img_info['file_name']
        image_path = os.path.join(args.image_path, img_basename)
        # if not isinstance(element, dict):
        #     print(f"No object in the picture")
        #     print(f"img_basename: {img_basename}")
        #     # print(f"image_path: {image_path}")
        #     empty_img_ids.append(element)
        #     continue
        
        if len(target) == 0:
            result_element = dict(id=img_id, file_name=img_info['file_name'], file_path=image_path, annotations=[], response="")
            results.append(result_element)
            print(f"image_path: {image_path}")
            print("No object in the picture")
            empty_img_ids.append(img_id)
            continue

        if not os.path.exists(image_path):
            print(f"Image {image_path} not found")
            continue


        category_ids = [target_element['category_id'] for target_element in target]
        category_names = list(set(category_id2name[cato_id] for cato_id in category_ids))
        category_names = [category_name for category_name in category_names if category_name in category_name2id]

        rect_set = set()
        new_annotations = []
        outputs_lst = []

        for category in category_names:
            # Query with category name
            question = question_template.format(category)
            outputs_text, selected_boxes = eval_model(model_name, args.quant_type, image_path, question)
            outputs_lst.append(dict(text=outputs_text, bboxes=selected_boxes))
            # print(f"image_path: {image_path}")

            # print(f"question: {question}")
            # print(f"outputs: {outputs}")

            anno_lst = selected_boxes
            for i, anno in enumerate(anno_lst):
                # Add annotations of unique bboxes
                new_ann = {}
                new_ann['class_id'] = category_name2id[category]
                new_ann['class'] = category
                x1, y1, x2, y2 = anno
                new_ann['rect'] = [int(x1*1000), int(y1*1000), int((x2-x1)*1000), int((y2-y1)*1000)]
                if str(new_ann['rect']) in rect_set:
                    continue
                rect_set.add(str(new_ann['rect']))
                new_annotations.append(new_ann)
        result_element = dict(id=img_id, file_name=os.path.basename(image_path), file_path=image_path, annotations=new_annotations, response=outputs_lst)
        results.append(result_element)
        

    print(f"Inference finished, total {len(results)} images")


    # Write results to file
    print(f"Write results to file {args.result_path}")
    with open(args.result_path, 'w') as f:
        json.dump(results, f, indent=4)

    # White empty images
    empty_file_basename = os.path.basename(args.result_path).replace(".json", "_empty.json")
    empty_file_path = os.path.join(os.path.dirname(args.result_path),"empty", empty_file_basename)
    os.makedirs(os.path.dirname(empty_file_path), exist_ok=True)
    print(f"Write empty images to file {empty_file_path}")
    with open(empty_file_path, 'w') as f:
        json.dump(empty_img_ids, f, indent=4)
