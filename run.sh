#!/bin/bash
set -e

echo "🚀 Starting Daily ArXiv Update Workflow..."

# --- 1. 使用北京时区获取正确的当天日期 ---
today=$(TZ=Asia/Shanghai date "+%Y-%m-%d")
yesterday=$(TZ=Asia/Shanghai date -d "yesterday" "+%Y-%m-%d")
echo "✅ Workflow date set to: ${today} (Asia/Shanghai)"

# --- 2. 定义所有文件名 ---
RAW_JSONL_FILE="data/${today}.jsonl"
UNIQUE_JSONL_FILE="data/${today}_unique.jsonl"
YESTERDAY_UNIQUE_FILE="data/${yesterday}_unique.jsonl"
ENHANCED_JSONL_FILE="data/${today}_unique_AI_enhanced_Chinese.jsonl"
FINAL_MD_FILE="data/${today}.md"

# --- 3. 运行 Scrapy 爬虫 ---
echo "--- Step 1: Crawling data from ArXiv ---"
(cd daily_arxiv && scrapy crawl arxiv -o ../${RAW_JSONL_FILE})
echo "✅ Raw data saved to ${RAW_JSONL_FILE}"

# --- 4. 运行去重脚本 ---
echo "--- Step 2: Deduplicating raw data ---"
python deduplicate.py ${RAW_JSONL_FILE} -o ${UNIQUE_JSONL_FILE}
echo "✅ Unique data saved to ${UNIQUE_JSONL_FILE}"

# --- 5. 新增：智能比较今天和昨天的内容 (忽略行序) ---
echo "--- Step 3: Checking for new content (ignoring line order) ---"
if [ -f "$YESTERDAY_UNIQUE_FILE" ]; then
    # 提取、排序并比较两个文件的 ID 集合
    # diff <(command1) <(command2) 是一种高级用法，用于比较两个命令的输出
    # grep -o '"id": "[^"]*"' 会只提取出 id 字段
    # sort 会对提取出的 id 排序
    # 如果两个排序后的 id 列表没有差异，diff 命令的输出就是空的
    if [ -z "$(diff <(grep -o '"id": "[^"]*"' "$UNIQUE_JSONL_FILE" | sort) <(grep -o '"id": "[^"]*"' "$YESTERDAY_UNIQUE_FILE" | sort))" ]; then
        echo "ℹ️  No new papers found. The set of papers is the same as yesterday. Exiting workflow."
        rm "$RAW_JSONL_FILE" "$UNIQUE_JSONL_FILE"
        exit 0
    else
        echo "✅ New content found. Proceeding with the workflow."
    fi
else
    echo "ℹ️  Yesterday's file not found. Assuming first run."
fi

# --- 6. 运行 AI 增强脚本 ---
echo "--- Step 4: Enhancing data with AI ---"
# 确保它的输入是去重后的文件
python ai/enhance.py --data ${UNIQUE_JSONL_FILE}
# 我已经移除了你脚本中那个在 enhance.py 之后多余的去重命令
echo "✅ AI enhancement complete. Output is ${ENHANCED_JSONL_FILE}"

# --- 7. 运行 Markdown 生成脚本 ---
echo "--- Step 5: Converting JSONL to Markdown ---"
python to_md/convert.py --data ${ENHANCED_JSONL_FILE}
echo "✅ Markdown report generated at ${FINAL_MD_FILE}"

# --- 8. 更新主 README 文件 ---
echo "--- Step 6: Updating main README.md ---"
python update_readme.py
echo "✅ README.md updated."

echo "🎉 Workflow finished successfully!"
