#!/bin/bash
set -e

echo "🚀 Starting Modified Daily ArXiv Workflow (Skip Crawling)..."

# 使用已有的jsonl文件，跳过爬虫步骤
INPUT_JSONL_FILE="$1"
if [ -z "$INPUT_JSONL_FILE" ]; then
    echo "❌ Error: Please provide a JSONL file path as argument. Example:"
    echo "    ./test_workflow.sh data/2025-06-08.jsonl"
    exit 1
fi

if [ ! -f "$INPUT_JSONL_FILE" ]; then
    echo "❌ Error: File $INPUT_JSONL_FILE not found."
    exit 1
fi

# 提取基本文件名（不含路径和扩展名）
filename=$(basename -- "$INPUT_JSONL_FILE")
base_name="${filename%.*}"

# --- 1. 定义所有文件名 ---
RAW_JSONL_FILE="$INPUT_JSONL_FILE"
UNIQUE_JSONL_FILE="data/${base_name}_unique.jsonl"
TRAJECTORY_LLM_FILE="data/${base_name}_trajectory_llm.jsonl"
ENHANCED_JSONL_FILE="data/${base_name}_unique_AI_enhanced_Chinese.jsonl"
TRAJECTORY_LLM_ENHANCED_FILE="data/${base_name}_trajectory_llm_AI_enhanced_Chinese.jsonl"
FINAL_MD_FILE="data/${base_name}.md"
TRAJECTORY_LLM_MD_FILE="data/${base_name}_trajectory_and_large_models.md"

echo "✅ Using existing data file: ${RAW_JSONL_FILE}"

# --- 2. 运行去重脚本 ---
# echo "--- Step 1: Deduplicating raw data ---"
# python deduplicate.py ${RAW_JSONL_FILE} -o ${UNIQUE_JSONL_FILE}
# echo "✅ Unique data saved to ${UNIQUE_JSONL_FILE}"

# --- 3. 筛选轨迹预测和大模型相关论文 ---
echo "--- Step 2: Filtering papers related to trajectory prediction and large models using LLM ---"
# 使用与AI增强相同的模型
MODEL_NAME=${MODEL_NAME:-"gemini-2.0-flash"}
python filter_papers.py --data ${UNIQUE_JSONL_FILE} -o ${TRAJECTORY_LLM_FILE} --model ${MODEL_NAME} --threshold 0.6
echo "✅ Filtered data saved to ${TRAJECTORY_LLM_FILE}"

# # --- 4. 运行 AI 增强脚本 ---
# echo "--- Step 3: Enhancing data with AI ---"
# python ai/enhance.py --data ${UNIQUE_JSONL_FILE}
# echo "✅ AI enhancement complete. Output is ${ENHANCED_JSONL_FILE}"

# --- 5. 为轨迹预测和大模型数据生成增强内容 ---
echo "--- Step 3.1: Enhancing trajectory prediction and large model data with AI ---"
python ai/enhance.py --data ${TRAJECTORY_LLM_FILE}
echo "✅ AI enhancement for trajectory prediction papers complete. Output is ${TRAJECTORY_LLM_ENHANCED_FILE}"

# # --- 6. 运行 Markdown 生成脚本 ---
# echo "--- Step 4: Converting JSONL to Markdown ---"
# python to_md/convert.py --data ${ENHANCED_JSONL_FILE}
# echo "✅ Markdown report generated at ${FINAL_MD_FILE}"

# --- 7. 为轨迹预测和大模型数据生成Markdown ---
echo "--- Step 4.1: Converting trajectory prediction and large model JSONL to Markdown ---"
python to_md/convert.py --data ${TRAJECTORY_LLM_ENHANCED_FILE} --output ${TRAJECTORY_LLM_MD_FILE}
echo "✅ Trajectory prediction and large model Markdown report generated at ${TRAJECTORY_LLM_MD_FILE}"

# --- 8. 更新主 README 文件 ---
echo "--- Step 5: Updating main README.md ---"
python update_readme.py
echo "✅ README.md updated."

echo "🎉 Test workflow finished successfully!"
