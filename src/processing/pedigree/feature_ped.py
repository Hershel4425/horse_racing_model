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
    print("父と母父産駒の成績を集計します。")
    print("処理前DF：", df.shape)
    # 重複行削除
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # ソート
    df = df.sort_values(['date', 'race_id', '着順']).reset_index(drop=True)
    df[group_col] = df[group_col].fillna("NoData")

    # 性別をまとめて male(牡,せん) / female(牝) / unknown に分類しておく
    def get_sex_class(x):
        if x in ['牡']:
            return 'male'
        elif x == '牝':
            return 'female'
        else:
            return 'male' # センバは男扱い
    df['sex_class'] = df['性'].apply(get_sex_class)

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
    # 重複行削除
    df = df.drop_duplicates(subset=["race_id", "馬番"])

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

            # sub 内で必要な列名をつくる
            col_gwin = f"{prefix}_{s}_{sur}_重賞勝利数"
            col_dist = f"{prefix}_{s}_{sur}_平均連帯距離"
            col_wr   = f"{prefix}_{s}_{sur}_平均勝率"

            # ここがポイント: dfに直接代入する
            df.loc[mask, col_gwin] = sub['cum_gwin'].values

            # 平均連帯距離
            # 連対数>0 の場合のみ計算
            df.loc[mask & (sub['cum_top2']>0), col_dist] = (
                sub.loc[sub['cum_top2']>0, 'cum_dist_top2'] / sub.loc[sub['cum_top2']>0, 'cum_top2']
            ).values

            # 平均勝率
            # 出走数>0 の場合のみ計算
            df.loc[mask & (sub['cum_starts']>0), col_wr] = (
                sub.loc[sub['cum_starts']>0, 'cum_wins'] / sub.loc[sub['cum_starts']>0, 'cum_starts']
            ).values

    # 不要な作業列を削除
    df.drop(['sex_class','surface_class','is_grade_race_win','start_flag','win_flag','top2_flag','dist_for_top2'], axis=1, inplace=True)

    return df


def add_sire_age_range_stats_no_leak(
    df: pd.DataFrame,
    group_col: str,    # 'father_name' など
    prefix: str,       # 'father' など
    ages: list = [2,3,4,5,6,7]
) -> pd.DataFrame:
    """
    【同じ父馬】でグループ化し、
    指定したすべての馬齢 (ages) について、
    「過去の累積出走数」「過去の累積勝利数」「勝率」を
    各レース行に付与する。

    ポイント:
      - レースを (date, race_id) で時系列順に並べて cumsum → shift(1) を行うことでリーク防止
      - 1つのレース行が馬齢3の馬でも、父の2歳実績/4歳実績/... 全てが付く
      - 出力列例:
          father_age2_starts, father_age2_wins, father_age2_win_rate,
          father_age3_starts, father_age3_wins, father_age3_win_rate, ...
    """

    # 作業用コピー
    df = df.copy()
    print("父と母父産駒の年齢別成績を集計します。")
    print("処理前DF：", df.shape)
    # 重複行削除
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # ソート
    df = df.sort_values(['date', 'race_id', '着順']).reset_index(drop=True)
    # -------------------------
    # 1) 必要列の確認・準備
    # -------------------------
    if group_col not in df.columns:
        raise ValueError(f"DataFrameに '{group_col}' 列がありません。")

    if "date" not in df.columns or "race_id" not in df.columns:
        raise ValueError("DataFrameに 'date' と 'race_id' 列が必要です。")

    if "齢" not in df.columns:
        raise ValueError("DataFrameに '齢' 列がありません。")

    # 出走フラグ, 勝利フラグ
    df["start_flag"] = df["着順"].notna().astype(int)
    df["win_flag"] = (df["着順"] == 1).astype(int)

    # 父馬名などの欠損補完
    df[group_col] = df[group_col].fillna("NoData")

    # -------------------------
    # 2) race単位で (father_name, date, race_id, ageごとの出走数/勝利数) を集計
    # -------------------------
    # まず行ベースで「(father_name, date, race_id, 齢)」に対し出走/勝利を合計
    # → 同じレースに同じ父馬の複数頭がいれば、そのぶんカウントが増える
    agg_df = (
        df.groupby([group_col, "date", "race_id", "齢"], as_index=False)
          .agg(
              starts_sum = ("start_flag", "sum"),
              wins_sum   = ("win_flag", "sum")
          )
    )
    # これで1行 = 1 (father_name, date, race_id, 齢)

    # pivotして、「father_age2_starts」「father_age3_starts」... のように列を展開
    # (可読性のため、いったん age2, age3... というカテゴリ名を作る)
    agg_df["age_cat"] = agg_df["齢"].apply(lambda x: f"age{x}" if x in ages else "other")

    # 出力に使う馬齢だけをピックアップする
    agg_df_sub = agg_df[agg_df["age_cat"].isin([f"age{a}" for a in ages])].copy()

    # pivot (starts / wins をそれぞれpivot してあとで結合でも良いが、一括のほうが手軽)
    # "starts_sum", "wins_sum" をそれぞれ "age_cat" × {starts_sum, wins_sum} に
    # wide化してやりたいので、ややトリッキーですが stack/unstack か pivot_table を使う
    pivot_df = pd.pivot_table(
        agg_df_sub,
        index=[group_col, "date", "race_id"], 
        columns=["age_cat"], 
        values=["starts_sum","wins_sum"],
        aggfunc="sum",  # groupbyレベルでさらに合計（通常重複はないはずだが念のため）
        fill_value=0
    )
    # MultiIndex になるので整形
    pivot_df.columns = [f"{col[0]}_{col[1]}" for col in pivot_df.columns]

    # pivot_df にない age(a) は全部 0 の列を作っておく（安全策）
    for a in ages:
        col1 = f"starts_sum_age{a}"
        col2 = f"wins_sum_age{a}"
        if col1 not in pivot_df.columns:
            pivot_df[col1] = 0
        if col2 not in pivot_df.columns:
            pivot_df[col2] = 0

    pivot_df = pivot_df.reset_index()  # => (father_name, date, race_id) の一意キー

    # -------------------------
    # 3) 時系列順に cumsum & shift(1) で「過去の累積」へ
    # -------------------------
    # father_nameごとに、(date, race_id) 昇順でソートしつつ cumsum → shift(1)
    # => 当該レースより前の累計値
    pivot_df = pivot_df.sort_values([group_col, "date", "race_id"])

    group_cols_for_cumsum = [f"starts_sum_age{a}" for a in ages] + [f"wins_sum_age{a}" for a in ages]
    
    # グループ単位で cumsum → shift(1)
    pivot_df[group_cols_for_cumsum] = (
        pivot_df.groupby(group_col)[group_cols_for_cumsum]
                .apply(lambda g: g.cumsum().shift(1).fillna(0))
                .reset_index(drop=True)
    )

    # ここで pivot_df の各行は
    #  father_name, date, race_id, starts_sum_age2, ..., wins_sum_age7
    # に「過去レースまでの累積値」が入っている

    # -------------------------
    # 4) pivot_df を元の df に結合 (キーは father_name, date, race_id)
    # -------------------------
    # (注意) 同じ (father_name, date, race_id) に複数頭出走がある場合も、その累計は同じ値
    df_merged = pd.merge(
        df,
        pivot_df,
        on=[group_col, "date", "race_id"],
        how="left"
    )

    # -------------------------
    # 5) 列名を prefix_age{a}_starts / prefix_age{a}_wins / prefix_age{a}_win_rate に変更
    #    & 勝率計算
    # -------------------------
    for a in ages:
        col_starts_in = f"starts_sum_age{a}"
        col_wins_in   = f"wins_sum_age{a}"

        col_starts_out = f"{prefix}_age{a}_starts"
        col_wins_out   = f"{prefix}_age{a}_wins"
        col_wr_out     = f"{prefix}_age{a}_win_rate"

        # リネーム
        df_merged[col_starts_out] = df_merged[col_starts_in].fillna(0)
        df_merged[col_wins_out]   = df_merged[col_wins_in].fillna(0)

        # 勝率 ( starts>0 の場合のみ計算 )
        df_merged[col_wr_out] = 0.0
        mask_starts = (df_merged[col_starts_out] > 0)
        df_merged.loc[mask_starts, col_wr_out] = (
            df_merged.loc[mask_starts, col_wins_out] / df_merged.loc[mask_starts, col_starts_out]
        )

    # 不要な中間列 (starts_sum_ageX, wins_sum_ageX) を削除
    drop_cols = [f"starts_sum_age{a}" for a in ages] + [f"wins_sum_age{a}" for a in ages]
    df_merged.drop(columns=drop_cols, inplace=True)

    # 仕上げ
    df_merged.drop(["start_flag","win_flag"], axis=1, errors="ignore", inplace=True)

    # 必要に応じて元の順序に並べ直す (date, race_id, 馬番 などで)
    # 例:
    # df_merged = df_merged.sort_values(["date","race_id","馬番"]).reset_index(drop=True)

    return df_merged



def _add_sire_first_appear_year(df: pd.DataFrame, group_col: str, prefix: str) -> pd.DataFrame:
    """
    父(または母父)がデータ中で「初めて登場した日付」から、現在のレース日(date)までが何年経過しているかを計算する。
    => (レース日 - 最初に登場した日付).days // 365
    
    例:
      df[f"{prefix}_first_appear_year"] に「初登場からの年数」を格納。
    """
    # 作業用コピー
    df = df.copy()
    print("父と母父産駒が最初に登場してから何年経ったか集計します。")
    print("処理前DF：", df.shape)
    # 重複行削除
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # ソート
    df = df.sort_values(['date', 'race_id', '着順']).reset_index(drop=True)
    df[group_col] = df[group_col].fillna("NoData")

    # group_col(例: father_name)ごとに最初に登場した日付を求める
    first_date_df = (
        df.groupby(group_col, as_index=False)['date']
          .min()
          .rename(columns={'date': f"{prefix}_earliest_date"})
    )
    df = pd.merge(df, first_date_df, on=group_col, how='left')

    # もし日付が欠損している場合は何もできないのでfillna(0)等してもOK
    df[f"{prefix}_first_appear_year"] = (
        (df['date'] - df[f"{prefix}_earliest_date"]).dt.days // 365
    ).fillna(0).clip(lower=0)

    # # 不要なら earliest_date 列を削除
    # df.drop(columns=[f"{prefix}_earliest_date"], inplace=True)

    return df


# -------------------------------
# メイン処理
# -------------------------------
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
    print("血統情報をもとにインブリードを計算します。。")
    print("処理前DF：", df.shape)
    # 重複行削除
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # ソート
    df = df.sort_values(['date', 'race_id', '着順']).reset_index(drop=True)
    # 重複行削除
    pedigree_df = pedigree_df.drop_duplicates(subset=["horse_id"])

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
    extended_pedigree_df['mother_father_name'] = extended_pedigree_df['pedigree_4'] if 'pedigree_4' in extended_pedigree_df else np.nan

    # 6) df への父馬・母父馬マージ
    #    (最低限の戦績サンプル: _aggregate_sire_stats を利用して父・母父の集計を結合する例)
    df_merged = pd.merge(
        df,
        extended_pedigree_df[['horse_id','father_name','mother_father_name']],
        on='horse_id',
        how='left'
    )

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

    # 10) (追加) 父系・母父系 × 馬齢 の年齢影響(成長曲線)
    #    例では「スタート数」「勝利数」「勝率」の3つを作るわ。
    #    father_age_starts, father_age_wins, father_age_win_rate
    #    mf_age_starts,     mf_age_wins,     mf_age_win_rate
    df_merged = add_sire_age_range_stats_no_leak(
        df_merged,
        group_col='father_name',
        prefix='father_age',
        ages=[2,3,4,5,6,7]
    )
    df_merged = add_sire_age_range_stats_no_leak(
        df_merged,
        group_col='mother_father_name',
        prefix='mf_age',
        ages=[2,3,4,5,6,7]
    )

    # === 11) 今回のリクエスト: 父として初登場して何年目？ 母父として初登場して何年目？ ===
    # father / mother_father 別々に計算し、dfに加える
    df_merged = _add_sire_first_appear_year(
        df_merged,
        group_col='father_name',
        prefix='father'
    )  # => father_first_appear_year
    df_merged = _add_sire_first_appear_year(
        df_merged,
        group_col='mother_father_name',
        prefix='mf'
    )  # => mf_first_appear_year


    return df_merged
