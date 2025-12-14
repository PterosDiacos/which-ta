from collections import defaultdict
from pypinyin import lazy_pinyin, Style
import pandas as pd
import re
import swifter


def corpus_scan(path):
    pattern = r'(\d+):(zh-cn|de):(.*?)(?=(\d+):(zh-cn|de):|$)'
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    matches = re.findall(pattern, content, re.DOTALL)
    d = defaultdict(dict)
    for match in matches:
        idx, lang, text, *_ = match
        d[idx][lang] = text.replace('\n', ' ').strip()
        if len(d[idx]) == 2:
            yield {'id': idx, **d[idx]}
            d.clear()


def txt_to_parquet(path='data/zh-de.txt'):
    df = pd.DataFrame(corpus_scan(path))
    df['zh-py'] = df['zh-cn'].swifter.apply(
        lambda s: ' '.join(lazy_pinyin(s, style=Style.TONE))
    )
    parquet_path = path.replace('.txt', '.parquet')
    df.to_parquet(parquet_path, index=False)
