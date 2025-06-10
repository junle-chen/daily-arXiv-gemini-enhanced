#!/bin/bash
set -e

echo "🚀 Starting Daily ArXiv Update Workflow..."

# --- 测试脚本：使用固定的测试数据 ---
today="test-run"
yesterday="2025-06-07"
echo "✅ Test workflow using fixed date: ${today}"

# --- 2. 定义所有文件名 ---

RAW_JSONL_FILE="data/2025-06-07.jsonl"
UNIQUE_JSONL_FILE="data/2025-06-07_unique.jsonl"
YESTERDAY_UNIQUE_FILE="data/${yesterday}_unique.jsonl"
ENHANCED_JSONL_FILE="data/${today}_unique_AI_enhanced_Chinese.jsonl"
FINAL_MD_FILE="data/${today}.md"

# 去掉爬虫部分，加快测试速度

# --- 4. 运行去重脚本 ---
echo "--- Step 2: Deduplicating raw data ---"
python deduplicate.py ${RAW_JSONL_FILE} -o ${UNIQUE_JSONL_FILE}
echo "✅ Unique data saved to ${UNIQUE_JSONL_FILE}"

# --- 新增: 筛选轨迹预测和大模型相关论文 ---
TRAJECTORY_LLM_FILE="data/${today}_trajectory_llm.jsonl"
echo "--- Step 2.1: Filtering papers related to trajectory prediction and large models using LLM ---"
# 使用与AI增强相同的模型
MODEL_NAME=${MODEL_NAME:-"gemini-2.0-flash"}

# 执行过滤
echo "Filtering papers with ${MODEL_NAME} model"
if python filter_papers.py --data ${UNIQUE_JSONL_FILE} -o ${TRAJECTORY_LLM_FILE} --model ${MODEL_NAME} --threshold 0.6; then
    echo "✅ Filtered data saved to ${TRAJECTORY_LLM_FILE}"
else
    echo "⚠️ Filtering failed. Continuing workflow without filtered papers."
    # 创建一个空的筛选文件，以便后续步骤可以继续
    touch ${TRAJECTORY_LLM_FILE}
fi

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
        # rm "$RAW_JSONL_FILE" "$UNIQUE_JSONL_FILE"
        exit 0
    else
        echo "✅ New content found. Proceeding with the workflow."
    fi
else
    echo "ℹ️  Yesterday's file not found. Assuming first run."
fi

# --- 6. 处理数据库相关论文 ---
DB_FILE="data/${today}_database.jsonl"
echo "--- Step 4: Processing database-related papers ---"

if ./process_db_papers.py --data ${UNIQUE_JSONL_FILE} -o ${DB_FILE}; then
    echo "✅ Database papers extracted to ${DB_FILE}"
else
    echo "⚠️ Database paper extraction failed. Creating empty file to continue workflow."
    touch ${DB_FILE}
fi

# --- 7. 为轨迹预测和大模型以及数据库生成增强内容 ---
TRAJECTORY_LLM_ENHANCED_FILE="data/${today}_trajectory_llm_AI_enhanced_Chinese.jsonl"
echo "--- Step 5: Enhancing trajectory prediction and large model data with AI ---"

DB_ENHANCED_FILE="data/${today}_database_AI_enhanced_Chinese.jsonl"
echo "--- Step 5: Enhancing database-related data with AI ---"
if [ -s "${DB_FILE}" ]; then
    echo "Enhancing database data with ${MODEL_NAME} model"
    if python ai/enhance.py --data ${DB_FILE}; then
        echo "✅ AI enhancement for database papers complete. Output is ${DB_ENHANCED_FILE}"
    else
        echo "⚠️ Database enhancement failed. Continuing without enhanced database data."
        # 创建一个空文件以便后续步骤可以继续
        touch ${DB_ENHANCED_FILE}
    fi
else
    echo "⚠️ Database data file is empty or doesn't exist. Skipping enhancement."
    touch ${DB_ENHANCED_FILE}
fi

# 检查轨迹预测数据文件是否存在且非空
if [ -s "${TRAJECTORY_LLM_FILE}" ]; then
    echo "Enhancing trajectory data with ${MODEL_NAME} model"
    if python ai/enhance.py --data ${TRAJECTORY_LLM_FILE}; then
        echo "✅ AI enhancement for trajectory prediction papers complete. Output is ${TRAJECTORY_LLM_ENHANCED_FILE}"
    else
        echo "⚠️ Trajectory enhancement failed. Continuing without enhanced trajectory data."
        # 创建一个空文件以便后续步骤可以继续
        touch ${TRAJECTORY_LLM_ENHANCED_FILE}
    fi
else
    echo "⚠️ Trajectory prediction data file is empty or doesn't exist. Skipping enhancement."
    touch ${TRAJECTORY_LLM_ENHANCED_FILE}
fi

# --- 8. 为轨迹预测、大模型和数据库相关数据生成Markdown ---
echo "--- Step 6: Converting JSONL to Markdown ---"

# 处理轨迹预测和大模型数据
TRAJECTORY_LLM_MD_FILE="data/${today}_trajectory_and_large_models.md"
if [ -s "${TRAJECTORY_LLM_ENHANCED_FILE}" ]; then
    python to_md/convert.py --data ${TRAJECTORY_LLM_ENHANCED_FILE} --output ${TRAJECTORY_LLM_MD_FILE}
    echo "✅ Markdown report for trajectory and large model papers generated at ${TRAJECTORY_LLM_MD_FILE}"
else
    echo "⚠️ No trajectory and large model papers data available. Creating empty markdown file."
    echo "# No trajectory and large model papers found today" > ${TRAJECTORY_LLM_MD_FILE}
fi

# 处理数据库相关数据
DB_MD_FILE="data/${today}_database.md"
if [ -s "${DB_FILE}" ]; then
    python to_md/convert.py --data ${DB_ENHANCED_FILE} --output ${DB_MD_FILE}
    echo "✅ Markdown report for database papers generated at ${DB_MD_FILE}"
else
    echo "⚠️ No database papers data available. Creating empty markdown file."
    echo "# No database papers found today" > ${DB_MD_FILE}
fi

# --- 9. 更新主 README 文件 ---
echo "--- Step 7: Updating main README.md ---"
python update_readme.py
echo "✅ README.md updated."

echo "🎉 Test workflow finished successfully!"
