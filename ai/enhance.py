import os
import json
import sys
import time
import random
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import dotenv
import argparse

import langchain_core.exceptions
from langchain_google_genai import ChatGoogleGenerativeAI

# Import our API management helpers
try:
    from . import api_manager
except ImportError:
    import api_manager
from langchain.prompts import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
)
from structure import Structure
from google.api_core.exceptions import ResourceExhausted, InternalServerError, ServiceUnavailable
# --- 1. 获取脚本所在的目录 ---
# __file__ 是当前脚本的文件名
# os.path.dirname() 获取该文件所在的目录路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- 2. 构造模板文件的绝对路径 ---
# os.path.join() 会智能地将目录和文件名拼接成一个完整的路径
template_path = os.path.join(script_dir, "template.txt")
system_path = os.path.join(script_dir, "system.txt")

# --- 3. 使用绝对路径打开文件 ---
try:
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    with open(system_path, 'r', encoding='utf-8') as f:
        system = f.read()
except FileNotFoundError as e:
    print(f"❌ Error: Template file not found. Ensure 'template.txt' and 'system.txt' are in the same directory as enhance.py.", file=sys.stderr)
    print(f"Details: {e}", file=sys.stderr)
    sys.exit(1) # 发现错误后立即退出，避免后续错误


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    return parser.parse_args()

def main():
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", 'gemini-2.0-flash')
    language = os.environ.get("LANGUAGE", 'Chinese')
    
    # Configure environment for GitHub Actions if needed
    api_manager.setup_environment()

    data = []
    with open(args.data, "r") as f:
        for line in f:
            data.append(json.loads(line))

    seen_ids = set()
    unique_data = []
    for item in data:
        if item['id'] not in seen_ids:
            seen_ids.add(item['id'])
            unique_data.append(item)

    data = unique_data

    print('Open:', args.data, file=sys.stderr)

    llm = ChatGoogleGenerativeAI(model=model_name).with_structured_output(Structure)
    print('Connect to:', model_name, file=sys.stderr)
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system),
        HumanMessagePromptTemplate.from_template(template=template)
    ])

    chain = prompt_template | llm

    # 定义一个带有重试机制的调用函数
    @retry(
        reraise=True,
        stop=stop_after_attempt(5),  # 最多尝试5次
        wait=wait_exponential(multiplier=1, min=4, max=60),  # 指数退避策略
        retry=retry_if_exception_type((ResourceExhausted, InternalServerError, ServiceUnavailable))
    )
    def invoke_with_retry(content):
        return chain.invoke({
            "language": language,
            "content": content
        })
    
    # 输出文件路径
    output_file = args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')
    
    # 清空输出文件（如果存在）
    with open(output_file, 'w') as f:
        pass
    
    for idx, d in enumerate(data):
        # 使用API管理器添加智能延迟，避免触发频率限制
        api_manager.smart_delay()  # 使用默认参数以最优化API使用率
        print(f"Processing item {idx+1}/{len(data)}: {d['id']}", file=sys.stderr)
        
        try:
            print(f"Processing {idx+1}/{len(data)}: {d['id']}", file=sys.stderr)
            response = invoke_with_retry(d['summary'])
            d['AI'] = response.dict()
            print(f"Successfully processed item {idx+1}", file=sys.stderr)
        except langchain_core.exceptions.OutputParserException as e:
            print(f"{d['id']} has an error: {e}", file=sys.stderr)
            d['AI'] = {
                 "tldr": "Error",
                 "motivation": "Error",
                 "method": "Error",
                 "result": "Error",
                 "conclusion": "Error"
            }
        except Exception as e:
            # 记录其他任何异常
            print(f"Unexpected error processing {d['id']}: {str(e)}", file=sys.stderr)
            d['AI'] = {
                 "tldr": f"Error: {str(e)}",
                 "motivation": "Error",
                 "method": "Error",
                 "result": "Error",
                 "conclusion": "Error"
            }
            
        # 立即写入文件，确保即使中断也能保留已处理的结果
        with open(output_file, "a") as f:
            f.write(json.dumps(d) + "\n")

        print(f"Finished {idx+1}/{len(data)}", file=sys.stderr)

if __name__ == "__main__":
    main()
