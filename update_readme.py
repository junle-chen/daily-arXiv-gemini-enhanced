import os
from os.path import join
from datetime import datetime
# Python 3.9+ 内置模块，用于处理时区
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

def main():
    """
    扫描 data 目录，根据模板生成包含所有 .md 文件链接的 README.md。
    """
    # 定义模板文件路径
    main_template_path = 'template.md'
    content_template_path = 'readme_content_template.md'
    data_dir = 'data'
    output_file = 'README.md'

    # --- 1. 读取 data 目录下的所有 .md 文件并排序 ---
    try:
        md_files = [f for f in os.listdir(data_dir) if f.endswith('.md')]
        md_files = sorted(md_files, reverse=True)
    except FileNotFoundError:
        print(f"❌ Error: The '{data_dir}' directory was not found.")
        return

    # --- 2. 读取两个模板文件 ---
    try:
        with open(main_template_path, 'r', encoding='utf-8') as f:
            main_template = f.read()
        with open(content_template_path, 'r', encoding='utf-8') as f:
            content_template = f.read()
    except FileNotFoundError as e:
        print(f"❌ Error: Template file not found: {e.filename}")
        return

    # --- 3. 根据模板生成动态的链接列表 ---
    content_lines = []
    for file_name in md_files:
        date_str = file_name.replace('.md', '')
        file_url = join(data_dir, file_name).replace('\\', '/')
        line = content_template.format(date=date_str, url=file_url)
        content_lines.append(line)
    
    readme_content = "\n".join(content_lines)

    # --- 4. 获取当前时间并填充到主模板中 (使用 zoneinfo) ---
    try:
        # 使用 zoneinfo 获取中国时区
        tz = ZoneInfo("Asia/Shanghai")
    except ZoneInfoNotFoundError:
        # 如果系统找不到时区数据 (很少见), 则使用 UTC 时间
        print("⚠️ Warning: Timezone 'Asia/Shanghai' not found. Defaulting to UTC.")
        tz = ZoneInfo("UTC")

    current_time = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')

    # 将链接列表和时间填充到主模板中
    final_markdown = main_template.format(readme_content=readme_content)
    final_markdown = final_markdown.replace('{{ time }}', current_time)

    # --- 5. 写入最终的 README.md 文件 ---
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_markdown)

    print(f"✅ Successfully updated {output_file} with {len(md_files)} entries.")


if __name__ == '__main__':
    main()
