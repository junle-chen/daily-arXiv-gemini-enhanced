#!/bin/bash

# --- 脚本设置 ---
# set -e: 任何命令失败，脚本将立即停止执行。这对于调试至关重要！
set -e

echo "🚀 Starting Daily ArXiv Update Workflow..."

# --- 1. 定义变量 ---
# 这样做可以让脚本更易于阅读和维护
today=$(date -u "+%Y-%m-%d")
RAW_JSONL_FILE="data/${today}.jsonl"
UNIQUE_JSONL_FILE="data/${today}_unique.jsonl"
# 假设你的 AI 增强脚本会使用这个名字
ENHANCED_JSONL_FILE="data/${today}_unique_AI_enhanced_Chinese.jsonl" 
# 最终的 Markdown 文件名
FINAL_MD_FILE="data/${today}.md"

# --- 2. 运行 Scrapy 爬虫 ---
# 始终从项目根目录调用，并明确指定路径
echo "--- Step 1: Crawling data from ArXiv ---"
(cd daily_arxiv && scrapy crawl arxiv -o ../${RAW_JSONL_FILE})
echo "✅ Raw data saved to ${RAW_JSONL_FILE}"

# --- 3. 运行去重脚本 ---
# 明确指定输入和输出文件
echo "--- Step 2: Deduplicating data ---"
python deduplicate.py ${RAW_JSONL_FILE} -o ${UNIQUE_JSONL_FILE}
echo "✅ Unique data saved to ${UNIQUE_JSONL_FILE}"

# --- 4. (可选) 运行 AI 增强脚本 ---
# 确保它的输入是去重后的文件
# echo "--- Step 3: Enhancing data with AI ---"
# python ai/enhance.py --data ${UNIQUE_JSONL_FILE}
# echo "✅ AI enhancement complete."
# # 如果你运行了 AI 增强，后续步骤的输入文件就需要改变
# INPUT_FOR_MD=${ENHANCED_JSONL_FILE}

# --- 5. 运行 Markdown 生成脚本 ---
# 如果你没有 AI 增强步骤，就用去重后的文件
INPUT_FOR_MD=${UNIQUE_JSONL_FILE}

echo "--- Step 4: Converting JSONL to Markdown ---"
# 从根目录调用，并使用相对于根目录的路径
python to_md/convert.py --data ${INPUT_FOR_MD}
echo "✅ Markdown report generated at ${FINAL_MD_FILE}"

# --- 6. 更新主 README 文件 ---
echo "--- Step 5: Updating main README.md ---"
python update_readme.py
echo "✅ README.md updated."

echo "🎉 Workflow finished successfully!"
