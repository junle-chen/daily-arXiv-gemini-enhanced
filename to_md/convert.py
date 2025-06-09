import json
import argparse
import os
from collections import defaultdict


def format_paper_to_markdown(paper: dict, index: int) -> str:
    """
    将单个论文的字典数据，格式化成一段漂亮的 Markdown 文本。
    按照paper_template.md模板展示：motivation, method, result, conclusion
    """
    # --- 提取基础信息 ---
    title = paper.get("title", "No Title Provided").strip()
    authors = ", ".join(paper.get("authors", ["N/A"]))
    abs_url = paper.get("abs", "#")
    pdf_url = abs_url.replace("abs", "pdf") if abs_url != "#" else "#"
    categories = paper.get("categories", [])
    main_category = categories[0] if categories else "Unknown"

    ai_data = paper.get("AI", {})

    # 提取结构化内容
    tldr = ai_data.get("tldr", "AI one-sentence summary is not available.")
    motivation = ai_data.get("motivation", "Not available")
    method = ai_data.get("method", "Not available")
    result = ai_data.get("result", "Not available")
    conclusion = ai_data.get("conclusion", "Not available")
    summary = paper.get("summary", "No summary available.").replace("\n", " ").strip()
    # 获取中文摘要，如果没有则使用原始摘要
    summary_zh = ai_data.get("summary_zh", summary).replace("\n", " ").strip()

    return f"""### [{index}] [{title}]({abs_url})
*{authors}*

Main category: {main_category}

TL;DR: {tldr}


<details>
  <summary>Details</summary>
Motivation: {motivation}

Method: {method}

Result: {result}

Conclusion: {conclusion}

Abstract: {summary_zh}

</details>

[**[PDF]**]({pdf_url}) | **Categories:** {', '.join(categories)}

---
"""


def main():
    """
    主函数：读取 JSONL 文件，并将其转换为一个格式化的 Markdown 文件。
    专注于轨迹预测和大模型相关论文，并按类别组织内容
    """
    parser = argparse.ArgumentParser(
        description="Convert an ArXiv JSONL file to a formatted Markdown file."
    )
    parser.add_argument(
        "--data",
        dest="input_file",
        required=True,
        help="Path to the input .jsonl file (can be raw, unique, or AI-enhanced).",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output_file",
        help="Path to the output markdown file. If not provided, a default name will be generated.",
    )
    args = parser.parse_args()

    input_path = args.input_file

    if not os.path.exists(input_path):
        print(f"❌ Error: Input file not found at '{input_path}'")
        return

    print(f"▶️  Reading from: {input_path}")

    papers = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                papers.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"⚠️ Warning: Skipping malformed JSON line: {line.strip()}")

    if not papers:
        print(
            "ℹ️ No papers found in the input file. An empty Markdown file will be created."
        )

    # --- 从输入文件名中提取日期 ---
    base_name = os.path.basename(input_path)
    date_str = base_name.split("_")[0].split(".")[0]

    # --- 创建文档大标题 ---
    header = f"# 每日 ArXiv 轨迹预测与大模型摘要速递: {date_str}\n\n"

    # --- 按类别组织论文 ---
    papers_by_category = defaultdict(list)

    # 先统计所有类别及其主类别
    category_counts = defaultdict(int)
    main_categories = {}

    # 注意：不再在此处筛选论文，而是直接使用输入的论文集合
    # 筛选工作已经由filter_papers.py完成

    print(f"📊 处理 {len(papers)} 篇论文")

    # 清空并重建按类别分类的论文
    papers_by_category.clear()
    category_counts.clear()
    main_categories.clear()

    # 按主类别归类
    for paper in papers:
        categories = paper.get("categories", ["Unknown"])
        if categories:
            main_cat = categories[0]
            papers_by_category[main_cat].append(paper)

            # 更新主类别计数
            for cat in categories:
                category_counts[cat] += 1
                if cat not in main_categories:
                    main_categories[cat] = main_cat

    # --- 生成目录 ---
    toc_items = []
    category_display_names = {
        "cs.CV": "计算机视觉 (Computer Vision)",
        "cs.RO": "机器人学 (Robotics)",
        "cs.LG": "机器学习 (Machine Learning)",
        "cs.AI": "人工智能 (Artificial Intelligence)",
        "cs.CL": "计算语言学 (Computation and Language)",
        "stat.ML": "统计机器学习 (Machine Learning Statistics)",
        "cs.HC": "人机交互 (Human-Computer Interaction)",
        "cs.NE": "神经与进化计算 (Neural and Evolutionary Computing)",
    }

    # 按类别生成目录
    toc_items = []
    for cat, cat_papers in sorted(papers_by_category.items()):
        display_name = category_display_names.get(cat, cat)
        toc_items.append(
            f"- [{display_name} ({len(cat_papers)})](#{cat.lower().replace('.', '-')})"
        )

    toc_content = "## 目录\n\n" + "\n".join(toc_items) + "\n\n"

    # --- 生成分类内容 ---
    category_sections = []
    for cat, cat_papers in sorted(papers_by_category.items()):
        display_name = category_display_names.get(cat, cat)
        section = [f"## {display_name} [{cat}]"]

        # 为每个类别的论文生成内容
        for i, paper in enumerate(cat_papers):
            section.append(format_paper_to_markdown(paper, i + 1))

        category_sections.append("\n".join(section))

    # --- 合并所有内容 ---
    final_markdown = header + toc_content + "\n\n".join(category_sections)

    # --- 写入文件 ---
    if args.output_file:
        output_path = args.output_file
    else:
        # 默认路径
        output_filename = f"{date_str}_trajectory_and_large_models.md"
        output_path = os.path.join(os.path.dirname(input_path), output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_markdown)

    print(f"✅ 成功转换 {len(papers)} 篇论文到: {output_path}")


if __name__ == "__main__":
    main()
