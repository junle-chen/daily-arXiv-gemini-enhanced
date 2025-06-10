#!/usr/bin/env python
# filepath: /Users/junle/Code/daily-arXiv-gemini-enhanced/process_db_papers.py
import argparse
import json
import os
import sys


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Extract database-related papers (cs.DB category) without filtering"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the JSONL data file containing all papers",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to the output filtered JSONL file. If not provided, a default name will be generated.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.data):
        print(f"❌ Error: Input file not found at '{args.data}'", file=sys.stderr)
        return 1

    # 如果用户没有提供输出文件名，则自动生成一个
    if args.output:
        output_path = args.output
    else:
        # 例如: 'data/2025-06-07_unique.jsonl' -> 'data/2025-06-07_database.jsonl'
        base, ext = os.path.splitext(args.data)
        output_path = f"{base}_database{ext}"

    # 读取论文数据
    db_papers = []
    total_papers = 0

    with open(args.data, "r", encoding="utf-8") as f_in:
        for line in f_in:
            try:
                paper = json.loads(line)
                total_papers += 1

                # 检查论文的类别是否包含cs.DB
                categories = paper.get("categories", [])
                # 确保categories是一个列表
                if isinstance(categories, str):
                    categories = categories.split()

                # 在类别列表中查找cs.DB
                if "cs.DB" in categories:
                    db_papers.append(paper)

            except json.JSONDecodeError:
                print(
                    f"⚠️ Warning: Skipping malformed JSON: {line.strip()}",
                    file=sys.stderr,
                )

    # 将数据库相关论文写入输出文件
    with open(output_path, "w", encoding="utf-8") as f_out:
        for paper in db_papers:
            f_out.write(json.dumps(paper) + "\n")

    print(
        f"ℹ️ 处理完成 {total_papers} 篇论文. 找到 {len(db_papers)} 篇数据库相关论文.",
        file=sys.stderr,
    )
    print(f"✅ 成功保存数据库相关论文到: {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
