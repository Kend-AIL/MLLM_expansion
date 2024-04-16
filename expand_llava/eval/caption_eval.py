import os
os.environ['CUDA_VISIBLE_DEVICES']='6,7'
import argparse
import torch

from expand_llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from expand_llava.conversation import conv_templates, SeparatorStyle
from expand_llava.model.builder import load_pretrained_model
from expand_llava.utils import disable_torch_init
from expand_llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re

import os


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out
def truncate(text):
    # Find the index of the first occurrence of "Question" or "Answer"
    question_index = text.find("Question")
    answer_index = text.find("Answer")
    change_row=text.find("\n")
    # Determine the earliest index, ignoring -1 (which indicates the keyword was not found)
    indices = [idx for idx in [question_index, answer_index,change_row] if idx != -1]
    if not indices:
        # If neither keyword is found, return the original text
        return text
    
    # Truncate the text at the earliest found keyword index and strip any trailing whitespace
    return text[:min(indices)].rstrip()
import json
import torch
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import numpy as np
import re

def initialize_model(args):
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    return tokenizer, model, image_processor, context_len
def eval_model(model, tokenizer, image_processor, args):
    # 适应性地更新函数以使用传入的模型和tokenizer等
    
    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    # input_token_len = input_ids.shape[1]
    # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    # if n_diff_input_output > 0:
    #     print(
    #         f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
    #     )
    outputs = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    
    # 返回生成的文本而不是打印它
    return outputs

def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_metrics(reference, output):
    rouge = Rouge()
    rouge_score = rouge.get_scores(output, reference)[0]['rouge-l']['f']
    reference = reference.split()
    output = output.split()
    bleu_score = sentence_bleu([reference], output, weights=(0.25, 0.25, 0.25, 0.25))
    return rouge_score, bleu_score

def process_dataset(dataset, tokenizer, model, image_processor, image_base,args):
    rouge_scores = []
    bleu_scores = []
    
    # 打开文件以追加数据
    with open('processed_data_8.txt', 'w') as file:
        for item in dataset:
            args.image_file = image_base + "/" + item['image']
            # 假设你已经定义了eval_model来生成模型的输出
            model_output = eval_model(model, tokenizer, image_processor, args)
            model_output=truncate(model_output)
            expected_output = item['conversations'][1]['value']
            rouge_score, bleu_score = calculate_metrics(expected_output, model_output)
            rouge_scores.append(rouge_score)
            bleu_scores.append(bleu_score)
            
            # 准备要保存的数据
            data_to_save = {
                'image': args.image_file,
                'model_output': model_output,
                'expected_output': expected_output,
                'rouge_score': rouge_score,
                'bleu_score': bleu_score,
            }
            
            # 将数据转换为JSON字符串并保存到文件
            file.write(json.dumps(data_to_save,indent=4) + '\n')  # 使用换行符分隔每个条目

    # 计算平均指标
    average_rouge = np.mean(rouge_scores)
    average_bleu = np.mean(bleu_scores)

    print(f"Average ROUGE-L: {average_rouge}, Average BLEU: {average_bleu}")

    # 可以选择将平均分数也写入文件
    with open('average_scores_8.txt', 'w') as file:
        average_scores_data = {
            'average_rouge': average_rouge,
            'average_bleu': average_bleu,
        }
        file.write(json.dumps(average_scores_data))


model_path = "/workspace/new_train/tiny-llava-base-phi-2-siglip-so400m-patch14-384_8_fineall-pretrain"
prompt = "This photo is generated from a portrait of a model. The original content of the photo can be divided into three parts: the clothing worn by the model, the model's appearance, and the scene in which the photo was taken. Now, all elements except for the model's clothing are masked. Please infer the model and background as accurately as possible based on the provided clothing, and provide a natural language paragraph description. Keep it concise and accurate, less than 200 words."
test_dataset="/workspace/data/imaterialist-fashion-2020-fgvc7/test_dataset.json"
image_base="/workspace/data/imaterialist-fashion-2020-fgvc7"
args = argparse.Namespace(
    model_path=model_path,
    model_base="microsoft/phi-2",
    query=prompt,
    conv_mode="phi",
    sep=",",
    temperature=0.3,
    top_p=1,
    num_beams=3,
    max_new_tokens=1024
)
tokenizer, model, image_processor, _ = initialize_model(args)
dataset = load_dataset(test_dataset)
process_dataset(dataset, tokenizer, model, image_processor,image_base,args)