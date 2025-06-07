import json
import argparse
import os

# in to_md/convert.py

def format_paper_to_markdown(paper: dict) -> str:
    """
    将单个论文的字典数据，格式化成一段漂亮的 Markdown 文本。
    """
    # ... (提取 title, authors, abs_url, pdf_url, categories 的代码保持不变) ...
    title = paper.get("title", "No Title Provided").strip()
    authors = ", ".join(paper.get("authors", ["N/A"]))
    abs_url = paper.get("abs", "#")
    pdf_url = paper.get("pdf", "#")
    categories = ", ".join([f"`{cat}`" for cat in paper.get("categories", [])])

    ai_data = paper.get("AI", {})

    # --- 新增：提取 TL;DR ---
    # 使用 .get() 安全地获取 tldr，如果不存在则提供一个默认值
    tldr_summary = ai_data.get("tldr", "AI summary not available.")
    
    # --- 智能选择摘要 (这部分代码保持不变) ---
    summary_to_display = ai_data.get("summary_zh")
    summary_title = "中文摘要 (Abstract in Chinese)"
    if not summary_to_display:
        summary_to_display = paper.get("summary", "No summary available.")
        summary_title = "Abstract"
    summary_markdown = "> " + summary_to_display.replace("\n", " ").strip()

    # --- 修改 f-string，加入 TL;DR 的展示部分 ---
    return f"""### [{title}]({abs_url})

**Authors:** {authors}
**Categories:** {categories}

[**[PDF]**]({pdf_url})

#### {summary_title}

{summary_markdown}
"""

def main():
    """
    主函数：读取 JSONL 文件，并将其转换为一个格式化的 Markdown 文件。
    """
    parser = argparse.ArgumentParser(
        description="Convert an ArXiv JSONL file to a formatted Markdown file."
    )
    parser.add_argument(
        "--data",  # 与您现有工作流保持一致
        dest="input_file",
        required=True,
        help="Path to the input .jsonl file (can be raw, unique, or AI-enhanced)."
    )
    args = parser.parse_args()

    input_path = args.input_file

    if not os.path.exists(input_path):
        print(f"❌ Error: Input file not found at '{input_path}'")
        return

    print(f"▶️  Reading from: {input_path}")

    papers = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                papers.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"⚠️ Warning: Skipping malformed JSON line: {line.strip()}")
    
    if not papers:
        print("ℹ️ No papers found in the input file. An empty Markdown file will be created.")
    
    # --- 生成 Markdown 内容 ---
    
    # 从输入文件名中提取日期，例如 '.../data/2025-06-07_unique.jsonl' -> '2025-06-07'
    base_name = os.path.basename(input_path)
    date_str = base_name.split('_')[0].split('.')[0]

    # 创建文档大标题
    header = f"# 每日 ArXiv 摘要速递: {date_str}\n\n"
    
    # 为每一篇论文生成 Markdown 内容
    markdown_parts = [format_paper_to_markdown(paper) for paper in papers]

    # 将所有论文的 Markdown 内容用分隔线 `---` 连接起来
    final_markdown = header + "\n---\n\n".join(markdown_parts)

    # --- 写入文件 ---
    
    # 定义输出文件名，例如 '.../data/2025-06-07_unique.jsonl' -> '.../data/2025-06-07.md'
    # 这个逻辑可以正确处理带 _unique 或 _AI_enhanced 的文件名
    output_filename = f"{date_str}.md"
    output_path = os.path.join(os.path.dirname(input_path), output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_markdown)

    print(f"✅ Successfully converted {len(papers)} papers to: {output_path}")


if __name__ == "__main__":
    main()
