today=`date -u "+%Y-%m-%d"`
cd daily_arxiv
scrapy crawl arxiv -o ../data/${today}.jsonl

cd ..
python deduplicate.py data/${today}.jsonl

cd ../to_md
python convert.py --data ../data/${today}.jsonl

python update_readme.py
