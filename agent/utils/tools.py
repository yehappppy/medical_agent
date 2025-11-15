import os
import io
import sys
import re
import base64
from PIL import Image
from pathlib import Path
from loguru import logger
from datetime import datetime
from typing import Union, List, Dict, Any
from pydantic import SecretStr
from pdf2image import convert_from_path
from langchain_openai import ChatOpenAI

# Initialize ChatOpenAI model
def get_agent(name: str, tags: list[str] | None = None) -> ChatOpenAI:
    base_url = os.getenv("MODEL_URL", "https://api.siliconflow.cn/v1")
    api_key = SecretStr(os.getenv("API_KEY", ""))
    model_name = os.getenv(f"{name}_MODEL", "")
    temperature = float(os.getenv(f"{name}_TEMPERATURE", 0.3))
    max_completion_tokens = int(os.getenv(f"{name}_MAX_TOKENS", 16384))
    model = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        tags=tags,
    )
    return model


def get_logger(module_name: str = "medical_agent"):
    """Setup logger with module-specific log files"""
    logger.remove()

    logs_dir = Path("log")
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = logs_dir / f"{module_name}_{timestamp}.log"

    logger.add(
        sys.stdout,
        format=
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True)

    logger.add(
        log_filename,
        format=
        "{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
        level="INFO",
        rotation="10 MB",
        retention="7 days",
        compression="zip")

    return logger


def image_to_base64(
    image_input: Union[str, Image.Image, List[Image.Image]]
) -> Union[str, List[str]]:
    if isinstance(image_input, str):
        with open(image_input, "rb") as image_file:
            binary_data = image_file.read()
            base64_encoded = base64.b64encode(binary_data).decode('utf-8')
            return base64_encoded

    elif isinstance(image_input, Image.Image):
        buffer = io.BytesIO()
        image_input.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        base64_encoded = base64.b64encode(img_bytes).decode('utf-8')
        return base64_encoded

    elif isinstance(image_input, list):
        base64_list = []
        for img in image_input:
            if not isinstance(img, Image.Image):
                raise TypeError(
                    f"List item must be a PIL Image object, got {type(img)}")
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            base64_str = base64.b64encode(img_bytes).decode('utf-8')
            base64_list.append(base64_str)
        return base64_list

    else:
        raise TypeError(
            f"Input must be str, PIL.Image.Image, or List[PIL.Image.Image], got {type(image_input)}"
        )

def deepseek_ocr_postprocess(ocr_response: str) -> Dict[str, Any]:
    # text & position extraction
    ref_matches = re.findall(r"<\|ref\|>(.*?)<\|/ref\|>", ocr_response, re.DOTALL)
    det_matches = re.findall(r"<\|det\|>\[\[(.*?)\]\]<\|/det\|>", ocr_response, re.DOTALL)
    
    text_blocks = []
    for i, (text, coords) in enumerate(zip(ref_matches, det_matches)):
        cleaned_text = text.strip()
        if cleaned_text:
            coord_list = [int(x.strip()) for x in coords.split(',')]
            text_blocks.append({
                "id": i + 1,
                "text": cleaned_text,
                "bbox": coord_list,   
                "position": {
                    "x": coord_list[0],
                    "y": coord_list[1],
                    "width": coord_list[2] - coord_list[0],
                    "height": coord_list[3] - coord_list[1]
                }
            })
    
    text_blocks.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
    plain_text = generate_reading_order_text(text_blocks)
    
    return {
        "plain_text": plain_text,
        "structured_data": text_blocks,
        "total_blocks": len(text_blocks),
        "source_format": "deepseek_ocr"
    }

def generate_reading_order_text(text_blocks: List[Dict]) -> str:
    if not text_blocks:
        return ""

    lines = []
    current_line = [text_blocks[0]]
    line_threshold = 20  
    
    for block in text_blocks[1:]:
        last_block = current_line[-1]
        y_diff = abs(block["bbox"][1] - last_block["bbox"][1])
        
        if y_diff <= line_threshold:
            current_line.append(block) # new line
        else:
            lines.append(current_line)
            current_line = [block]
    
    lines.append(current_line)
    
    reading_lines = []
    for line in lines:
        sorted_line = sorted(line, key=lambda x: x["bbox"][0])
        line_text = " ".join(block["text"] for block in sorted_line)
        reading_lines.append(line_text)
    
    return "\n".join(reading_lines)

def extract_ocr_text(ocr_response: str) -> str:
    pattern = r"<\|ref\|>(.*?)<\|/ref\|>"
    matches = re.findall(pattern, ocr_response, re.DOTALL)
    
    if not matches:
        return ocr_response.strip()
    
    cleaned_texts = []
    seen_texts = set()
    
    for match in matches:
        text = match.strip()
        if text and text not in seen_texts:
            seen_texts.add(text)
            cleaned_texts.append(text)
    
    return "\n".join(cleaned_texts)
def medical_ocr_postprocess(ocr_response: str) -> str:
    processed = deepseek_ocr_postprocess(ocr_response)
    text_blocks = processed["structured_data"]
    medical_categories = {
        "patient_info": ["患者", "姓名", "性别", "年龄", "病历号", "就诊日期"],
        "symptoms": ["主诉", "症状", "不适", "疼痛", "发热", "头痛", "咳嗽"],
        "diagnosis": ["诊断", "印象", "结论", "考虑", "疑似", "确诊"],
        "examination": ["检查", "检验", "结果", "指标", "CT", "MRI", "超声"],
        "treatment": ["处方", "治疗", "用药", "剂量", "用法", "建议", "手术"]
    }
    categorized = {category: [] for category in medical_categories.keys()}
    categorized["other"] = []
    
    for block in text_blocks:
        text = block["text"]
        categorized_flag = False
        
        for category, keywords in medical_categories.items():
            if any(keyword in text for keyword in keywords):
                categorized[category].append(block)
                categorized_flag = True
                break
        
        if not categorized_flag:
            categorized["other"].append(block)
    
    return format_medical_document(categorized)

def format_medical_document(categorized_blocks: Dict) -> str:
    category_names = {
        "patient_info": "患者信息",
        "symptoms": "症状描述", 
        "diagnosis": "诊断结果",
        "examination": "检查检验",
        "treatment": "治疗方案",
        "other": "其他信息"
    }
    
    sections = []
    
    for category_key, category_name in category_names.items():
        blocks = categorized_blocks.get(category_key, [])
        if blocks:
            sections.append(f"【{category_name}】")
            blocks.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
            for block in blocks:
                sections.append(f"- {block['text']}")
            sections.append("") 
    
    return "\n".join(sections).strip()