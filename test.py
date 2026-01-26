#%%
from functools import cache
from client import get_resp
from tqdm import tqdm
import pandas as pd


def get_mask(df, col):
    @cache
    def mask(w):
        return df[col].str.contains(w)
    return mask


def py_to_en(py: str, **kwargs) -> dict:
    template = """
Translate the following Chinese pinyin transcription into English.
Answer only with your translation.
## Chinese pinyin transcription: 
{}

## English translation:
"""
    inp = [{"role": "user", "content": template.format(py)}]
    resp = get_resp(messages=inp, **kwargs)
    resp_text = resp.choices[0].message.content
    return {resp.model: resp_text.strip()}


if __name__ == "__main__":
    inp_file = "data/zh-de.parquet"
    out_file = "data/pred-en-3.parquet"
    
    df = pd.read_parquet(inp_file).astype({"id": int}).set_index("id", drop=True).sort_index()
    mask = get_mask(df, "zh-cn")
    df1 = df[mask("她")]
    sample = df1.sample(1000, random_state=1)

    tqdm.pandas(desc="PY to EN")
    sample["pred_en"] = sample["zh-py"].progress_apply(
        lambda x: py_to_en(x, temperature=0.1)
    )
    sample.to_parquet(out_file)
