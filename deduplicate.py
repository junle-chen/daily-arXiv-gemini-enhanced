import json
import argparse
import os

def main():
    """
    读取一个 JSONL 文件，根据 'id' 字段去除重复行，
    并将其写入新的输出文件。
    """
    parser = argparse.ArgumentParser(
        description="Deduplicate a JSONL file based on the 'id' field."
    )
    parser.add_argument(
        "input_file", 
        help="Path to the input .jsonl file with duplicate entries."
    )
    parser.add_argument(
        "-o", "--output", 
        dest="output_file",
        help="Path to the output .jsonl file. If not provided, a default name will be generated."
    )
    args = parser.parse_args()

    input_path = args.input_file
    
    # 如果用户没有提供输出文件名，则自动生成一个
    if args.output_file:
        output_path = args.output_file
    else:
        # 例如: 'data/2025-06-07.jsonl' -> 'data/2025-06-07_unique.jsonl'
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_unique{ext}"

    if not os.path.exists(input_path):
        print(f"❌ Error: Input file not found at '{input_path}'")
        return

    print(f"▶️  Reading from: {input_path}")

    seen_ids = set()
    unique_papers = []
    total_lines = 0
    duplicate_count = 0

    with open(input_path, 'r', encoding='utf-8') as f_in:
        for i, line in enumerate(f_in, 1):
            total_lines += 1
            try:
                paper = json.loads(line)
                paper_id = paper.get('id')

                # 检查 paper_id 是否存在且未被记录
                if paper_id and paper_id not in seen_ids:
                    seen_ids.add(paper_id)
                    unique_papers.append(paper)
                else:
                    duplicate_count += 1

            except json.JSONDecodeError:
                print(f"⚠️ Warning: Skipping malformed JSON on line {i}: {line.strip()}")
            except KeyError:
                print(f"⚠️ Warning: Skipping line {i} due to missing 'id' field: {line.strip()}")

    print(f"ℹ️  Processed {total_lines} lines. Found {len(unique_papers)} unique papers and {duplicate_count} duplicates.")

    # 将去重后的内容写入新文件
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for paper in unique_papers:
            f_out.write(json.dumps(paper) + '\n')

    print(f"✅ Successfully saved unique papers to: {output_path}")

if __name__ == "__main__":
    main()
