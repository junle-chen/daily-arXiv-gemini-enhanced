import json
import argparse
import os
from datetime import datetime

def create_paper_markdown(paper: dict) -> str:
    """
    将单个论文的字典数据格式化为 Markdown 字符串。
    """
    # 安全地获取字段，如果字段不存在则返回空值
    title = paper.get("title", "No Title Provided").strip()
    abs_url = paper.get("abs", "#")
    pdf_url = paper.get("pdf", "#")
    authors = ", ".join(paper.get("authors", ["N/A"]))
    categories = ", ".join([f"`{cat}`" for cat in paper.get("categories", [])])
    # 移除摘要中可能存在的换行符，并用 blockquote 格式化
    summary = "> " + paper.get("summary", "No summary available.").replace("\n", " ").strip()

    # 使用 f-string 构建 Markdown 模板
    return f"""### [{title}]({abs_url})

**Authors:** {authors}

**Categories:** {categories}

[**[PDF]**]({pdf_url})

#### Summary

{summary}
"""

def main():
    """
    主函数，负责读取 jsonl 文件并生成 markdown 文件。
    """
    parser = argparse.ArgumentParser(description="Convert an ArXiv JSONL file to a formatted Markdown file.")
    parser.add_argument("input_file", help="Path to the input .jsonl file")
    args = parser.parse_args()

    input_path = args.input_file

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    papers = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                papers.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed line: {line.strip()}")

    if not papers:
        print("No papers found in the input file. Exiting.")
        return

    # 从输入文件名中提取日期
    base_name = os.path.basename(input_path)
    date_str = base_name.split('_')[0].split('.')[0]
    
    # 格式化文档标题
    header = f"# Daily ArXiv Digest: {date_str}\n\n"
    
    # 为每篇论文生成 Markdown 内容
    markdown_parts = [create_paper_markdown(paper) for paper in papers]

    # 将所有部分用分隔线连接起来
    final_markdown = header + "---\n\n".join(markdown_parts)

    # 定义输出文件名
    output_file = input_path.replace('.jsonl', '.md')

    # 写入最终的 Markdown 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_markdown)

    print(f"✅ Successfully converted {len(papers)} papers to {output_file}")


if __name__ == "__main__":
    main()
