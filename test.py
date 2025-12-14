from functools import cache
import pandas as pd


def get_mask(df, col):
    @cache
    def mask(w):
        return df[col].str.contains(w)
    return mask

df = pd.read_parquet('data/zh-de.parquet')\
    .astype({'id': int})\
    .set_index('id', drop=True)
mask = get_mask(df, 'zh-cn')


#%% ['他', '她', '它']
df[mask('她')]

#%%
idx = 6079
df.loc[idx-10:idx+10]
