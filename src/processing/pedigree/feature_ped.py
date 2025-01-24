import pandas as pd
import numpy as np
import re

def unify_kana_underscore(name: str) -> str:
    """
    血統名に含まれる「カタカナ_アルファベット(国)」などの表記ゆれを統一する例。
    先頭がカタカナ＋アンダースコア構造になっていたら、カタカナ部分だけ返す。
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
    特定の列(例: 父馬名)をキーに、芝・ダートの勝率/連対率などを集計して返すサンプル。
    (ご要望とは別の簡易サンプル。 create_extensive_pedigree_features 内部で使用中)
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


def _add_sire_additional_stats_no_leak(
    df: pd.DataFrame,
    group_col: str,
    prefix: str
) -> pd.DataFrame:
    """
    父馬 or 母父馬ごとに、以下の値を「レース実施日時点」で累積しておき、当該レースには反映させない(リーク防止のためshift(1))。
    - 重賞勝利数(芝・ダート別、性別統合/牡馬牝馬別)
    - 平均連帯距離(芝・ダート別、性別統合/牡馬牝馬別)
    - 平均勝率(芝・ダート別、性別統合/牡馬牝馬別)

    さらに「母父が同じ馬」も同じロジックで集計可能。
    ただし、この関数は「group_col」「prefix」を変えて再利用することで対応する。
    
    引数:
        df: 元のDataFrame（date、race_id、馬番、性別、グレード、着順、コース種類、距離 などを含む）
        group_col: 集計対象となる列名（例: 'father_name', 'mother_father_name'）
        prefix: 出力列に付与するプレフィックス（例: 'father', 'mf', 'mf_sib' など）
    戻り値:
        df: 新たに以下の列が追加されたDataFrameを返す
    """
    # 作業用コピー
    df = df.copy()
    df[group_col] = df[group_col].fillna("NoData")

    # 性別をまとめて male(牡,せん) / female(牝) / unknown に分類しておく
    def get_sex_class(x):
        if x in ['牡','せん']:
            return 'male'
        elif x == '牝':
            return 'female'
        else:
            return 'unknown'
    df['sex_class'] = df['性別'].apply(get_sex_class)

    # コース種類(芝,ダート以外は集計しない)
    def get_surface_class(x):
        if x == '芝':
            return '芝'
        elif x == 'ダート':
            return 'ダート'
        else:
            return 'その他'
    df['surface_class'] = df['コース種類'].apply(get_surface_class)

    # 重賞勝利数のカウント用フラグ
    # グレードが "G1","G2","G3" のいずれか含むか単に "G" 含むかはお好みで。ここでは "G" 含む場合に。
    df['is_grade_race_win'] = np.where(
        df['グレード'].fillna('').str.contains('G') & (df['着順'] == 1),
        1, 0
    )

    # スタート数、勝利数、連対(2着以内)数、連対時距離合計
    df['start_flag'] = np.where(df['着順'].notna(), 1, 0)
    df['win_flag'] = np.where(df['着順'] == 1, 1, 0)
    df['top2_flag'] = np.where(df['着順'] <= 2, 1, 0)
    df['dist_for_top2'] = np.where(df['着順'] <= 2, df['距離'].fillna(0), 0)

    # =============
    # 以下で、(group_col, sex_class, surface_class) 単位で累積計算→shift(1) する
    # ただし "surface_class=その他" は集計対象外
    # =============

    # 入れ物(最終的に merge 用)
    # カラム例:
    #   prefix + "_all_芝_重賞勝利数", prefix + "_all_芝_平均連帯距離", prefix + "_all_芝_平均勝率"
    #   prefix + "_male_芝_重賞勝利数", prefix + "_male_芝_平均連帯距離", prefix + "_male_芝_平均勝率"
    #   prefix + "_female_ダート_重賞勝利数", ...
    # など
    new_cols = [
        f"{prefix}_all_芝_重賞勝利数", f"{prefix}_all_芝_平均連帯距離", f"{prefix}_all_芝_平均勝率",
        f"{prefix}_male_芝_重賞勝利数", f"{prefix}_male_芝_平均連帯距離", f"{prefix}_male_芝_平均勝率",
        f"{prefix}_female_芝_重賞勝利数", f"{prefix}_female_芝_平均連帯距離", f"{prefix}_female_芝_平均勝率",

        f"{prefix}_all_ダート_重賞勝利数", f"{prefix}_all_ダート_平均連帯距離", f"{prefix}_all_ダート_平均勝率",
        f"{prefix}_male_ダート_重賞勝利数", f"{prefix}_male_ダート_平均連帯距離", f"{prefix}_male_ダート_平均勝率",
        f"{prefix}_female_ダート_重賞勝利数", f"{prefix}_female_ダート_平均連帯距離", f"{prefix}_female_ダート_平均勝率",
    ]
    # 初期化
    for c in new_cols:
        df[c] = 0.0

    # 処理しやすいように date, race_id, 馬番 でソート
    df = df.sort_values(["date","race_id","馬番"]).reset_index(drop=True)

    # 性別を3パターン: (all, male, female)、コースを2パターン: (芝,ダート) でループ
    sex_list = ['all', 'male', 'female']
    surface_list = ['芝', 'ダート']

    for s in sex_list:
        for sur in surface_list:
            # マスク
            if s == 'all':
                mask = (df['surface_class'] == sur)
            else:
                mask = (df['surface_class'] == sur) & (df['sex_class'] == s)

            # 集計したい列を用意
            # 重賞勝利数 → cumsum('is_grade_race_win')
            # スタート数 → cumsum('start_flag')
            # 勝利数 → cumsum('win_flag')
            # 連対数 → cumsum('top2_flag')
            # 連対距離合計 → cumsum('dist_for_top2')
            # groupbyキーは [group_col], ただしマスク外は計算不要なので 0埋め
            sub = df.loc[mask, [group_col, 'is_grade_race_win','start_flag','win_flag','top2_flag','dist_for_top2']].copy()

            # グループごとに累積→shift(1)
            sub['cum_gwin'] = sub.groupby(group_col)['is_grade_race_win'].cumsum().shift(1).fillna(0)
            sub['cum_starts'] = sub.groupby(group_col)['start_flag'].cumsum().shift(1).fillna(0)
            sub['cum_wins'] = sub.groupby(group_col)['win_flag'].cumsum().shift(1).fillna(0)
            sub['cum_top2'] = sub.groupby(group_col)['top2_flag'].cumsum().shift(1).fillna(0)
            sub['cum_dist_top2'] = sub.groupby(group_col)['dist_for_top2'].cumsum().shift(1).fillna(0)

            # sub を df にマージ
            df = df.merge(
                sub[[group_col,'cum_gwin','cum_starts','cum_wins','cum_top2','cum_dist_top2']].reset_index(drop=True),
                how='left',
                left_index=True,
                right_index=True,
                suffixes=(None, "_dummy")
            )

            # 求めたい列に代入
            # 重賞勝利数
            df_col1 = f"{prefix}_{s}_{sur}_重賞勝利数"
            # 平均連帯距離 = cum_dist_top2 / cum_top2
            df_col2 = f"{prefix}_{s}_{sur}_平均連帯距離"
            # 平均勝率 = cum_wins / cum_starts
            df_col3 = f"{prefix}_{s}_{sur}_平均勝率"

            # マスク該当部だけ更新し、それ以外は 0 のまま
            df.loc[mask, df_col1] = df.loc[mask, 'cum_gwin']
            df.loc[mask & (df['cum_top2']>0), df_col2] = df.loc[mask & (df['cum_top2']>0), 'cum_dist_top2'] / df.loc[mask & (df['cum_top2']>0), 'cum_top2']
            df.loc[mask & (df['cum_starts']>0), df_col3] = df.loc[mask & (df['cum_starts']>0), 'cum_wins'] / df.loc[mask & (df['cum_starts']>0), 'cum_starts']

            # 後処理: 不要なカラムを削除
            df.drop(['cum_gwin','cum_starts','cum_wins','cum_top2','cum_dist_top2'], axis=1, inplace=True)

    # 不要な作業列を削除
    df.drop(['sex_class','surface_class','is_grade_race_win','start_flag','win_flag','top2_flag','dist_for_top2'], axis=1, inplace=True)

    return df


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
    - (追加) 父産駒 / 母父産駒 / 母父が同じ馬 それぞれについて
       - 芝ダート別、かつ 性別(牡馬(せん含む)・牝馬・合計)別に
         「重賞勝利数」「平均連帯距離」「平均勝率」をリークなしで計算し付与

    戻り値:
        血統特徴量が追加された df
    """
    df = df.copy()
    pedigree_df = pedigree_df.copy()

    # 1) pedigree_df の表記ゆれを修正
    ped_cols = [c for c in pedigree_df.columns if 'pedigree_' in c]
    for c in ped_cols:
        pedigree_df[c] = pedigree_df[c].apply(unify_kana_underscore)

    # 2) 主要血統抽出
    all_ancestors = pedigree_df[ped_cols].values.flatten()
    s = pd.Series(all_ancestors).dropna()
    counts = s.value_counts()
    major_ancestors = counts.index[:top_k].tolist()

    # 3) 血量計算に必要な辞書作成
    #   pedigree_0 ~ pedigree_(2**max_generation -1)
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

    # 6) df への父馬・母父馬マージ
    #    (最低限の戦績サンプル: _aggregate_sire_stats を利用して父・母父の集計を結合する例)
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

    # 7) インブリード等をマージ
    use_cols = ['horse_id'] + [c for c in extended_pedigree_df.columns if ('登場回数' in c or 'インブリード血量' in c) or (c.endswith('_is_sunday_line'))]
    df_merged = pd.merge(
        df_merged,
        extended_pedigree_df[use_cols],
        on='horse_id',
        how='left'
    )

    # 8) (追加) 父産駒、母父産駒、母父が同じ馬 の「重賞勝利数/平均連帯距離/平均勝率」を
    #            芝・ダート別、性別(合計・牡馬(せん含)・牝馬)別々にリークなしで計算
    #    - father_name を group_col としてまとめる → prefix = 'father'
    #    - mother_father_name を group_col としてまとめる → prefix = 'mf'
    #    - mother_father_name を group_col としてまとめる → prefix = 'mf_sib'
    #      （実質母父産駒と同じ集計だけど、別列として欲しいとの要望なので二重計算）
    #
    #    ※ 速度を優先するなら1つの関数でまとめて処理してから列名だけ変えるなどしてもOK

    # father
    df_merged = _add_sire_additional_stats_no_leak(
        df_merged,
        group_col='father_name',
        prefix='father'
    )

    # mother_father
    df_merged = _add_sire_additional_stats_no_leak(
        df_merged,
        group_col='mother_father_name',
        prefix='mf'
    )

    # mother_father (同じ馬という扱いで再度)
    df_merged = _add_sire_additional_stats_no_leak(
        df_merged,
        group_col='mother_father_name',
        prefix='mf_sib'
    )

    return df_merged
