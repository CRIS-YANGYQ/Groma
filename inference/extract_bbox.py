import os
import re
import copy
from numpy import indices
import torch
import argparse
import requests
from io import BytesIO
from collections import defaultdict
from PIL import Image, ImageDraw
from transformers.image_transforms import center_to_corners_format
from transformers import AutoTokenizer, AutoImageProcessor, BitsAndBytesConfig
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
import json
import sys

def extract_categories_with_rois(text):
    """
    Extracts categories and their corresponding ROI placeholders from the given text.
    
    Args:
        text (str): Input text containing categories enclosed in <p> tags and ROIs enclosed in <roi> tags.
        
    Returns:
        dict: A dictionary mapping each category to a list of its corresponding ROI placeholders.
    """
    category_pattern = re.compile(r'<p>\s*(.*?)\s*</p>')
    roi_pattern = re.compile(r'<roi>(.*?)</roi>')
    
    category_to_rois = defaultdict(list)
    
    categories = category_pattern.findall(text)
    rois = roi_pattern.findall(text)
    
    if not categories or not rois:
        return dict(category_to_rois)
    
    roi_index = 0
    for category in categories:
        category = category.lower()
        category = category.replace('_', ' ')
        category = category.replace('-', ' ')
        if roi_index < len(rois):
            # Extract individual ROI placeholders within the current <roi> block
            roi_placeholders = re.findall(r'<r\d+>', rois[roi_index])
            category_to_rois[category].extend(roi_placeholders)
            roi_index += 1
    
    return dict(category_to_rois)


def find_cato2regions(raw_dict, categories):
    category_to_rois = defaultdict(list)
    for category in categories:
        # For each prompt category 
        for element_key, element_value in raw_dict.items():
            # filter out the not related category
            if element_key != category:
                subs = element_key.split(' ')
                if category not in subs:
                    for sub in subs:
                        if category in sub:
                            category_to_rois[category] = category_to_rois[category] + element_value
                            continue
                        elif category == 'person':
                            if 'woman' in sub or 'man' in sub or 'people' in sub or 'boy' in sub or 'girl' in sub:
                                print(f"category: {category}, sub: {sub}\n element_key: {element_key}, element_value{element_value}")
                                category_to_rois[category] = category_to_rois[category] + element_value
                                continue
                    continue
            category_to_rois[category] = category_to_rois[category] + element_value
    return category_to_rois
                      
             
def extract_code_from_placeholder(placeholder):
    """
    Extracts the numeric code from a placeholder like <r72>.
    
    Args:
        placeholder (str): The placeholder string, e.g., <r72>.
        
    Returns:
        str: The extracted numeric code, e.g., "72".
    """
    match = re.search(r'<r(\d+)>', placeholder)
    if match:
        return int(match.group(1))  # Return the numeric part
    return None

def find_cato2boxes(cato2regions_dict, bboxes_lst, selected_indices):
    cato2boxes = defaultdict(list)
    selected_mapping = {}
    for bbox_index, region_index in enumerate(selected_indices):
        # print(bbox_index)
        selected_mapping[region_index] = bbox_index
    # print(f"selected_mapping: {selected_mapping}")
    for category, regions in cato2regions_dict.items():
        for region in regions:
            region_code = extract_code_from_placeholder(region)
            # print(f"region: {region}, region_code: {region_code}")
            if region_code is not None and region_code in selected_mapping:
                bbox_index = selected_mapping[region_code]
                bbox = bboxes_lst[bbox_index]
                cato2boxes[category].append(bbox)
            else:
                print(f"Error in {region}")
    return cato2boxes

def extract_code_from_placeholder(placeholder):
    """
    Extracts the numeric code from a placeholder like <r72>.
    
    Args:
        placeholder (str): The placeholder string, e.g., <r72>.
        
    Returns:
        str: The extracted numeric code, e.g., "72".
    """
    match = re.search(r'<r(\d+)>', placeholder)
    if match:
        return int(match.group(1))  # Return the numeric part
    return None

def convert_raw_to_annotations(input_str, categories, bboxes, indices, category_name2id):
    raw_cato2rois_dict = extract_categories_with_rois(input_str)
    cato2regions = find_cato2regions(raw_cato2rois_dict, categories)
    cato2boxes= find_cato2boxes(cato2regions, bboxes, indices)
    new_annotations = []
    for cato, boxes in cato2boxes.items():
        for box in boxes:
            x1, y1, x2, y2 = box
            new_anno = {}
            new_anno["class_id"] = category_name2id[cato]
            new_anno['class'] = cato
            new_anno['rect'] = [int(x1*1000), int(y1*1000), int((x2-x1)*1000), int((y2-y1)*1000)]
            new_annotations.append(new_anno)

    return new_annotations

if __name__ == '__main__':
    extract_json_path = "/comp_robot/yangyuqin/workplace/Multi-model/result/lvis_obj_detection/groma-finetune_all.json"
    category_name2id_path = "/comp_robot/yangyuqin/workplace/Multi-model/workplace/tools/lmdeploy/category_name2id_lvis.json"
    convert_json_path = "/comp_robot/yangyuqin/workplace/Multi-model/result/lvis_obj_detection/groma-finetune_all_converted_2.json"
    print("Loading json file...")
    with open(extract_json_path, 'r') as f:
        extract_json = json.load(f)
    with open(category_name2id_path, 'r') as f:
        category_name2id = json.load(f)
    convert_json = []
    print("Converting json file...")
    raw_box_counter = 0
    cvt_box_counter = 0
    for json_data in tqdm(extract_json):
        raw_box_counter += len(json_data['annotations'])
        new_data = json_data.copy()
        response = json_data['response']
        if len(response) == 0:
            convert_json.append(new_data)
            continue
        del new_data['annotations']

        outputs_text = response[0]['text']
        selected_boxes = response[0]['bboxes']
        selected_box_inds = response[0]['indices']
        category_names = response[0]['categories']
        # outputs_lst.append(dict(text=outputs_text, bboxes=selected_boxes, indices=selected_box_inds, categories=category_names))
        new_annotations = convert_raw_to_annotations(outputs_text, category_names, selected_boxes, selected_box_inds, category_name2id)
        new_data['annotations'] = new_annotations
        convert_json.append(new_data)
        cvt_box_counter += len(new_annotations)

    print(f"Raw box counter is {raw_box_counter}")
    print(f"Converted box counter is {cvt_box_counter}")
    print(f"Converted data size is {len(convert_json)}")
    print("Saving converted json file...")
    with open(convert_json_path, 'w') as f:
        json.dump(convert_json, f, indent=4)
    print(f"Saved converted json file to {convert_json_path}")
    