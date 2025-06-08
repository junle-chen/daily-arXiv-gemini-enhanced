#!/bin/bash
set -e

echo "🚀 Starting Daily ArXiv Update Workflow..."

# --- 1. 定义变量 ---
today=$(date -u "+%Y-%m-%d")
# 新增：获取昨天的日期 (适用于 Linux 和 macOS)
yesterday=$(date -u -d "yesterday" "+%Y-%m-%d")

RAW_JSONL_FILE="data/${today}.jsonl"
UNIQUE_JSONL_FILE="data/${today}_unique.jsonl"
YESTERDAY_UNIQUE_FILE="data/${yesterday}_unique.jsonl"
ENHANCED_JSONL_FILE="data/${today}_unique_AI_enhanced_Chinese.jsonl"
FINAL_MD_FILE="data/${today}.md"

# --- 2. 运行 Scrapy 爬虫 ---
echo "--- Step 1: Crawling data from ArXiv ---"
(cd daily_arxiv && scrapy crawl arxiv -o ../${RAW_JSONL_FILE})
echo "✅ Raw data saved to ${RAW_JSONL_FILE}"

# --- 3. 运行去重脚本 ---
echo "--- Step 2: Deduplicating raw data ---"
python deduplicate.py ${RAW_JSONL_FILE} -o ${UNIQUE_JSONL_FILE}
echo "✅ Unique data saved to ${UNIQUE_JSONL_FILE}"

# --- 4. 新增：检查今天的内容是否与昨天相同 ---
echo "--- Step 3: Checking for new content compared to yesterday ---"
# 首先检查昨天的文件是否存在
if [ -f "$YESTERDAY_UNIQUE_FILE" ]; then
    # 使用 cmp 命令静默比较两个文件。如果相同，cmp 返回 0
    if cmp -s "$UNIQUE_JSONL_FILE" "$YESTERDAY_UNIQUE_FILE"; then
        echo "ℹ️  No new papers found today. Content is the same as yesterday. Exiting workflow."
        # 清理今天生成的临时文件
        rm "$RAW_JSONL_FILE" "$UNIQUE_JSONL_FILE"
        exit 0 # 正常退出，不执行后续步骤
    else
        echo "✅ New content found. Proceeding with the workflow."
    fi
else
    echo "ℹ️  Yesterday's file not found. Assuming first run or fresh start."
fi


# --- 5. 运行 AI 增强脚本 ---
echo "--- Step 4: Enhancing data with AI ---"
python ai/enhance.py --data ${UNIQUE_JSONL_FILE}
python deduplicate.py ${ENHANCED_JSONL_FILE} -o ${ENHANCED_JSONL_FILE}

echo "✅ AI enhancement complete."

# --- 6. 运行 Markdown 生成脚本 ---
echo "--- Step 5: Converting JSONL to Markdown ---"
python to_md/convert.py --data ${ENHANCED_JSONL_FILE}
echo "✅ Markdown report generated."

# --- 7. 更新主 README 文件 ---
echo "--- Step 6: Updating main README.md ---"
python update_readme.py
echo "✅ README.md updated."

echo "🎉 Workflow finished successfully!"
