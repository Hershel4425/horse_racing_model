import pandas as pd
import numpy as np
import re
from tqdm import tqdm

def unify_kana_underscore(name: str) -> str:
    """
    血統名に含まれる「カタカナ_アルファベット(国)」などの表記ゆれを統一する例。
    先頭がカタカナ＋アンダースコア構造になっていたら、カタカナ部分だけ返すようにする。
    """
    pattern_kana_underscore = re.compile(r'^([\u30A0-\u30FFー]+)_.+$')
    if pd.isna(name):
        return name
    m = pattern_kana_underscore.match(name)
    if m:
        return m.group(1)
    else:
        return name


def _aggregate_sire_stats(df: pd.DataFrame, sire_col: str = 'father_name') -> pd.DataFrame:
    """
    特定の列(例: 父馬名)をキーに、芝・ダートの勝率/連対率などを集計して返す。
    """
    df = df.copy()
    df[sire_col] = df[sire_col].fillna("NoData")
    df = df.drop_duplicates(subset=["race_id", "馬番"])

    df['win_flag'] = (df['着順'] == 1).astype(int)
    df['rentai_flag'] = (df['着順'] <= 2).astype(int)

    df['is_turf'] = (df['コース種類'] == '芝').astype(int)
    df['is_dirt'] = (df['コース種類'] == 'ダート').astype(int)

    grouped = df.groupby([sire_col, 'コース種類'], dropna=False).agg(
        starts=('race_id','count'),
        wins=('win_flag','sum'),
        rentai=('rentai_flag','sum'),
        sum_dist=('距離','sum')
    ).reset_index()

    stats_pivot = grouped.pivot_table(
        index=sire_col, 
        columns='コース種類',
        values=['starts','wins','rentai','sum_dist'],
        fill_value=0
    )
    stats_pivot.columns = [f"{c1}_{c2}" for (c1,c2) in stats_pivot.columns]

    # 勝率・連対率・平均距離
    if 'wins_芝' in stats_pivot and 'starts_芝' in stats_pivot:
        stats_pivot['win_rate_芝'] = stats_pivot['wins_芝'] / stats_pivot['starts_芝'].replace(0,np.nan)
        stats_pivot['rentai_rate_芝'] = stats_pivot['rentai_芝'] / stats_pivot['starts_芝'].replace(0,np.nan)
        stats_pivot['mean_dist_芝'] = stats_pivot['sum_dist_芝'] / stats_pivot['starts_芝'].replace(0, np.nan)
    if 'wins_ダート' in stats_pivot and 'starts_ダート' in stats_pivot:
        stats_pivot['win_rate_ダート'] = stats_pivot['wins_ダート'] / stats_pivot['starts_ダート'].replace(0,np.nan)
        stats_pivot['rentai_rate_ダート'] = stats_pivot['rentai_ダート'] / stats_pivot['starts_ダート'].replace(0,np.nan)
        stats_pivot['mean_dist_ダート'] = stats_pivot['sum_dist_ダート'] / stats_pivot['starts_ダート'].replace(0, np.nan)

    stats_pivot = stats_pivot.fillna(0)
    stats_pivot = stats_pivot.reset_index().rename(columns={sire_col: sire_col})

    return stats_pivot


def create_extensive_pedigree_features(
    df: pd.DataFrame,
    pedigree_df: pd.DataFrame,
    top_k: int = 50,
    max_generation: int = 5
) -> pd.DataFrame:
    """
    血統情報を使った拡張特徴量をいろいろ作って返す。
    - 主要血統(top_k件)の登場回数とインブリード血量
    - 父系・母父系などの主要系統判定フラグ
    - 父馬・母父馬などの戦績集計を結合

    戻り値:
        血統特徴量が追加された df
    """
    # 1) pedigree_df の表記ゆれを修正
    ped_cols = [c for c in pedigree_df.columns if 'pedigree_' in c]
    for c in ped_cols:
        pedigree_df[c] = pedigree_df[c].apply(unify_kana_underscore)

    # 2) 主要血統抽出
    all_ancestors = pedigree_df[ped_cols].values.flatten()
    s = pd.Series(all_ancestors).dropna()
    counts = s.value_counts()
    major_ancestors = counts.index[:top_k].tolist()

    # 3) 血量計算
    max_ped_index = 2**max_generation - 1
    def find_generation(idx):
        g = 1
        while True:
            lower = 2**(g-1)
            upper = 2**g - 1
            if lower <= (idx + 1) <= upper:
                return g
            g += 1
            if g > max_generation:
                return max_generation

    blood_fraction = {}
    for i in range(2**max_generation):
        gen_ = find_generation(i)
        blood_fraction[i] = 1.0 / (2**gen_)

    # 4) インブリード度合いを計算
    extended_pedigree_df = pedigree_df.copy()
    def calc_one_ancestor_inbreed(row, anc_name):
        count_, coeff_sum = 0, 0.0
        for i in range(max_ped_index+1):
            col_name = f"pedigree_{i}"
            if col_name not in row or pd.isna(row[col_name]):
                continue
            if row[col_name] == anc_name:
                count_ += 1
                coeff_sum += blood_fraction.get(i, 0.0)
        return count_, coeff_sum

    for anc_name in major_ancestors:
        col_count = f"{anc_name}_登場回数"
        col_inbr  = f"{anc_name}_インブリード血量"
        tmp = extended_pedigree_df.apply(lambda row: calc_one_ancestor_inbreed(row, anc_name), axis=1)
        extended_pedigree_df[col_count] = tmp.apply(lambda x: x[0])
        extended_pedigree_df[col_inbr]  = tmp.apply(lambda x: x[1])

    # 5) 父系/母父系などのフラグ作成
    extended_pedigree_df['father_name'] = extended_pedigree_df['pedigree_0']
    extended_pedigree_df['mother_father_name'] = extended_pedigree_df['pedigree_2'] if 'pedigree_2' in extended_pedigree_df else np.nan

    def check_sunday_line(name):
        if pd.isna(name):
            return 0
        if 'サンデーサイレンス' in name:
            return 1
        return 0

    extended_pedigree_df['father_is_sunday_line'] = extended_pedigree_df['father_name'].apply(check_sunday_line)
    extended_pedigree_df['mf_is_sunday_line'] = extended_pedigree_df['mother_father_name'].apply(check_sunday_line)

    # 6) 父馬・母父馬の戦績集計
    df_merged = pd.merge(
        df,
        extended_pedigree_df[['horse_id','father_name','mother_father_name']],
        on='horse_id',
        how='left'
    )
    father_stats = _aggregate_sire_stats(df_merged, sire_col='father_name')
    mf_stats = _aggregate_sire_stats(df_merged, sire_col='mother_father_name')

    # 父馬
    df_merged = pd.merge(df_merged, father_stats, on='father_name', how='left')
    # 母父馬
    df_merged = pd.merge(df_merged, mf_stats, on='mother_father_name', how='left', suffixes=('','_mf'))
    # 列名を多少整備してもOK

    # 7) インブリードなどをマージ
    use_cols = ['horse_id'] + [c for c in extended_pedigree_df.columns if ('登場回数' in c or 'インブリード血量' in c)]
    df_merged = pd.merge(
        df_merged,
        extended_pedigree_df[use_cols],
        on='horse_id',
        how='left'
    )

    return df_merged
