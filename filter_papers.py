#!/usr/bin/env python
# filepath: /Users/junle/Code/daily-arXiv-gemini-enhanced/filter_papers.py
import argparse
import json
import os
import sys
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Filter papers based on relevance to trajectory prediction and large models using LLM analysis."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the JSONL data file containing papers.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to the output filtered JSONL file. If not provided, a default name will be generated.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gemini-2.0-flash",
        help="LLM model to use for relevance detection.",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.7,
        help="Confidence threshold for relevance (0.0-1.0).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.data):
        print(f"❌ Error: Input file not found at '{args.data}'", file=sys.stderr)
        return

    # 如果用户没有提供输出文件名，则自动生成一个
    if args.output:
        output_path = args.output
    else:
        # 例如: 'data/2025-06-07_unique.jsonl' -> 'data/2025-06-07_trajectory_llm.jsonl'
        base, ext = os.path.splitext(args.data)
        output_path = f"{base}_trajectory_llm{ext}"

    # 初始化LLM模型
    try:
        llm = ChatGoogleGenerativeAI(model=args.model)
        print(f"ℹ️ Using {args.model} for relevance detection", file=sys.stderr)
    except Exception as e:
        print(f"❌ Error: Could not initialize LLM model: {e}", file=sys.stderr)
        return

    # 创建提示模板
    system_template = """你是一个学术论文过滤器。你的任务是判断一篇论文是否与轨迹预测（trajectory prediction）和大语言模型（Large Language Models）相关。
    
    论文内容可能涉及以下主题：
    1. 轨迹预测：行人轨迹预测、车辆轨迹预测、移动物体路径规划、动作预测等
    2. 大模型：大型语言模型（LLMs）、基础模型（foundation models）、大规模AI模型等
    
    请仔细分析论文标题和摘要，判断其与上述主题的相关性。如果论文同时涉及轨迹预测和大模型，或者将这两个领域结合起来的工作，相关性会更高。"""

    human_template = """分析以下论文，判断其与轨迹预测和大型语言模型的相关性：
    
    标题：{title}
    摘要：{summary}
    
    请输出一个JSON格式的回复，包含以下字段：
    1. relevance_score: 0.0-1.0之间的数字，表示相关性程度（1.0表示高度相关）
    2. explanation: 简短解释为什么这篇论文相关或不相关
    3. keywords: 提取的与轨迹预测或大模型相关的关键词列表
    
    只返回JSON对象，不要有其他文本。"""

    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(template=human_template),
        ]
    )

    filtered_papers = []
    total_papers = 0
    processed_papers = 0

    # 读取论文数据
    papers = []
    with open(args.data, "r", encoding="utf-8") as f_in:
        for line in f_in:
            try:
                papers.append(json.loads(line))
            except json.JSONDecodeError:
                print(
                    f"⚠️ Warning: Skipping malformed JSON: {line.strip()}",
                    file=sys.stderr,
                )

    total_papers = len(papers)
    print(
        f"ℹ️ Processing {total_papers} papers for relevance analysis...", file=sys.stderr
    )

    for paper in papers:
        processed_papers += 1

        # 提取标题和摘要
        title = paper.get("title", "")
        summary = paper.get("summary", "")

        # 如果没有足够的内容分析，则跳过
        if not title or not summary:
            continue

        try:
            # 使用LLM分析论文内容
            response = llm.invoke(prompt_template.format(title=title, summary=summary))

            # 解析响应
            try:
                # 尝试从LLM响应中提取JSON部分
                result = response.content
                if isinstance(result, str):
                    # 删除可能的Markdown代码块标记
                    result = result.replace("```json", "").replace("```", "").strip()

                analysis = json.loads(result)

                # 检查相关性分数是否超过阈值
                if analysis.get("relevance_score", 0) >= args.threshold:
                    # 添加分析结果到论文数据
                    paper["relevance_analysis"] = analysis
                    filtered_papers.append(paper)

                    # 输出进度和结果
                    print(
                        f"✓ [{processed_papers}/{total_papers}] 相关论文: {title} (分数: {analysis.get('relevance_score')})",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"× [{processed_papers}/{total_papers}] 不相关: {title}",
                        file=sys.stderr,
                    )

            except json.JSONDecodeError:
                # 如果无法解析JSON，使用备用方法检查是否包含关键词
                content = (title + " " + summary).lower()
                if any(
                    kw in content
                    for kw in [
                        "trajectory",
                        "predict",
                        "forecasting",
                        "motion",
                        "path",
                        "llm",
                        "large model",
                        "foundation model",
                    ]
                ):
                    paper["relevance_analysis"] = {
                        "relevance_score": 0.6,
                        "explanation": "基于关键词匹配",
                        "keywords": ["backup_method"],
                    }
                    filtered_papers.append(paper)

        except Exception as e:
            print(f"⚠️ Error analyzing paper '{title}': {e}", file=sys.stderr)

    # 按相关性分数排序
    filtered_papers.sort(
        key=lambda p: p.get("relevance_analysis", {}).get("relevance_score", 0),
        reverse=True,
    )

    # 将过滤后的论文写入输出文件
    with open(output_path, "w", encoding="utf-8") as f_out:
        for paper in filtered_papers:
            f_out.write(json.dumps(paper) + "\n")

    print(
        f"ℹ️ 处理完成 {total_papers} 篇论文. 找到 {len(filtered_papers)} 篇与轨迹预测和大模型相关的论文.",
        file=sys.stderr,
    )
    print(f"✅ 成功保存过滤后的论文到: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
