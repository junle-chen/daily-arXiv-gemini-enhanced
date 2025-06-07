import os
from os.path import join
from datetime import datetime
import pytz

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
        # os.listdir 获取文件名，然后过滤出 .md 文件
        md_files = [f for f in os.listdir(data_dir) if f.endswith('.md')]
        # reverse=True 确保最新的日期排在最前面
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
        # 从文件名 '2025-06-07.md' 中提取日期 '2025-06-07'
        date_str = file_name.replace('.md', '')
        # 创建文件的相对路径 'data/2025-06-07.md'
        file_url = join(data_dir, file_name).replace('\\', '/') # 确保路径使用 /
        
        # 格式化每一行链接
        line = content_template.format(date=date_str, url=file_url)
        content_lines.append(line)
    
    # 将所有行连接成一个字符串
    readme_content = "\n".join(content_lines)

    # --- 4. 获取当前时间并填充到主模板中 ---
    # 使用 pytz 获取中国时区
    tz = pytz.timezone('Asia/Shanghai')
    current_time = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')

    # 将链接列表和时间填充到主模板中
    final_markdown = main_template.format(readme_content=readme_content)
    final_markdown = final_markdown.replace('{{ time }}', current_time)

    # --- 5. 写入最终的 README.md 文件 ---
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_markdown)

    print(f"✅ Successfully updated {output_file} with {len(md_files)} entries.")


if __name__ == '__main__':
    # 在运行前，确保你安装了 pytz
    # pip install pytz
    main()
