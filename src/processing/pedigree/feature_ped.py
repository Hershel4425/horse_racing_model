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
    父馬 or 母父馬ごとに、以下の値を「レース日時点」で累積（リーク防止のため shift(1)）し、
    芝・ダート × 性別(牡(せん含) / 牝 / all) の全組み合わせについて
    それぞれの列(重賞勝利数, 平均連対距離, 平均勝率)を当該レース行に付与する。

    ※ add_sire_age_range_stats_no_leak と同じような「pivotしてから時系列順にcumsum＋shift」方式に変更。
    """

    # 作業用にコピー
    df = df.copy()

    # (race_id, 馬番) が重複している行は削除しておく
    df.drop_duplicates(subset=["race_id", "馬番"], inplace=True)

    # group_col(例: father_name) の欠損補完
    df[group_col] = df[group_col].fillna("NoData")

    # ソート (念のため date, race_id, 着順の昇順に)
    df = df.sort_values(["date", "race_id", "着順"]).reset_index(drop=True)

    # 性別を "male"/"female"/(セン含む場合も"male") に分ける
    def get_sex_class(x):
        if x == "牡":
            return "牡馬"
        elif x == "牝":
            return "牝馬"
        else:
            return "牡馬"  # センは male 扱い
    df["sex_class"] = df["性"].apply(get_sex_class)

    # コース種類を "芝"/"ダート"/"その他" にまとめる
    def get_surface_class(x):
        if x == "芝":
            return "芝"
        elif x == "ダート":
            return "ダート"
        else:
            return "その他"
    df["surface_class"] = df["コース種類"].apply(get_surface_class)

    # 重賞勝利フラグ
    df["重賞勝利フラグ"] = np.where(
        df["グレード"].fillna("").str.contains("G") & (df["着順"] == 1),
        1, 0
    )

    # 出走フラグ、勝利フラグ、連対フラグ、連対距離
    df["出走フラグ"] = np.where(df["着順"].notna(), 1, 0)
    df["勝利フラグ"]   = np.where(df["着順"] == 1, 1, 0)
    df["連対フラグ"]  = np.where(df["着順"] <= 2, 1, 0)
    df["連対距離"] = np.where(df["着順"] <= 2, df["距離"].fillna(0), 0)

    # 芝 or ダート のみ対象 (その他は集計に含めない)
    df = df[df["surface_class"].isin(["芝","ダート"])].copy()

    # sex_class='all' の行を追加
    df_all = df.copy()
    df_all["sex_class"] = "牡牝混合"
    df2 = pd.concat([df, df_all], ignore_index=True)

    # ================
    # (A) レース単位で合計を集計 (groupby)
    # ================
    # group: (group_col, date, race_id, sex_class, surface_class)
    agg_df = (
        df2.groupby([group_col, "date", "race_id", "sex_class", "surface_class"], as_index=False)
           .agg({
               "重賞勝利フラグ":"sum",   # 重賞勝利数
               "出走フラグ":"sum",         # 出走数
               "勝利フラグ":"sum",           # 勝利数
               "連対フラグ":"sum",          # 連対数
               "連対距離":"sum",      # 連対距離
           })
    )

    # ================
    # (B) pivotして wide化
    #    => index=[group_col, date, race_id]
    #    => columns=[(sex_class, surface_class)]
    #       values=[重賞勝利フラグ, 出走フラグ, ...]
    # ================
    # ここで multi-index になるので、flatten後にわかりやすく rename する
    pivot_cols = ["重賞勝利フラグ","出走フラグ","勝利フラグ","連対フラグ","連対距離"]
    agg_df["sex_surface"] = agg_df["sex_class"] + "_" + agg_df["surface_class"]

    pivot_df = pd.pivot_table(
        agg_df,
        index=[group_col, "date", "race_id"],
        columns="sex_surface",
        values=pivot_cols,
        aggfunc="sum",  # 重複があれば合計
        fill_value=0
    )

    # flatten
    # 本来の想定通り、先に pivot_cols を付けてから sex_surface を付ける
    pivot_df.columns = [f"{col}_{val}" for col, val in pivot_df.columns]
    pivot_df.reset_index(inplace=True)

    # ================
    # (C) 時系列順に cumsum -> shift(1) で「当該レースより前の累積」に変換
    # ================
    pivot_df = pivot_df.sort_values([group_col, "date", "race_id"])
    cumsum_cols = list(set(pivot_df.columns) - set([group_col, "date", "race_id"]))

    # group_col(父馬名など) 単位で累積
    pivot_df[cumsum_cols] = (
        pivot_df.groupby(group_col)[cumsum_cols]
                .apply(lambda g: g.cumsum().shift(1).fillna(0))
                .reset_index(drop=True)
    )

    # ================
    # (D) 累積したフラグから「重賞勝利数」「平均連対距離」「平均勝率」を計算
    #     sex_surface ごとに計算したいので列をループ
    # ================
    # ピボット後の列例: 出走フラグ_all_芝, 勝利フラグ_all_芝, 連対フラグ_all_芝, 連対距離_all_芝, ...
    # まず「gwin_cnt_〜」「avg_dist_〜」「win_rate_〜」列を作る
    # sex_surface の一覧を推定 (例: all_芝, male_芝, female_芝, all_ダート, male_ダート, ...)
    # pivot前に .isin() してるので、そこまで多くないはず
    set_sex_surf = set()
    for c in pivot_df.columns:
        if c.endswith("_芝") or c.endswith("_ダート"):
            # 例: 出走フラグ_male_芝 => suffix = "male_芝"
            #     勝利フラグ_all_ダート => suffix = "all_ダート"
            # 接頭語: 重賞勝利フラグ, 出走フラグ, 勝利フラグ, 連対フラグ, 連対距離
            pass_prefix = None
            for pfx in pivot_cols:
                if c.startswith(pfx+"_"):
                    pass_prefix = pfx
                    break
            if pass_prefix:
                suffix = c[len(pass_prefix)+1:]  # 例: c="出走フラグ_male_芝" なら suffix="male_芝"
                set_sex_surf.add(suffix)

    # sex_surfaceごとに、新たに計算する列を作る
    for ss in set_sex_surf:
        # 重賞勝利数: 重賞勝利フラグ_xxxx
        col_gwin_in  = f"重賞勝利フラグ_{ss}"
        col_starts   = f"出走フラグ_{ss}"
        col_wins     = f"勝利フラグ_{ss}"
        col_top2     = f"連対フラグ_{ss}"
        col_dist2    = f"連対距離_{ss}"

        # 出力列
        col_gwin_out = f"gwin_cnt_{ss}"
        col_avg_dist = f"avg_dist_{ss}"
        col_win_rate = f"win_rate_{ss}"

        # ゼロ埋めしてから計算 (万一欠損があれば)
        pivot_df[col_gwin_in] = pivot_df[col_gwin_in].fillna(0)
        pivot_df[col_starts]  = pivot_df[col_starts].fillna(0)
        pivot_df[col_wins]    = pivot_df[col_wins].fillna(0)
        pivot_df[col_top2]    = pivot_df[col_top2].fillna(0)
        pivot_df[col_dist2]   = pivot_df[col_dist2].fillna(0)

        pivot_df[col_gwin_out] = pivot_df[col_gwin_in]
        
        # 平均連対距離
        pivot_df[col_avg_dist] = 0.0
        mask_top2 = (pivot_df[col_top2] > 0)
        pivot_df.loc[mask_top2, col_avg_dist] = (
            pivot_df.loc[mask_top2, col_dist2] / pivot_df.loc[mask_top2, col_top2]
        )

        # 勝率
        pivot_df[col_win_rate] = 0.0
        mask_st = (pivot_df[col_starts] > 0)
        pivot_df.loc[mask_st, col_win_rate] = (
            pivot_df.loc[mask_st, col_wins] / pivot_df.loc[mask_st, col_starts]
        )

    # いらない元フラグ列( 重賞勝利フラグ_***, 出走フラグ_***, など )は落としてOK
    drop_base_cols = []
    for ss in set_sex_surf:
        for pfx in pivot_cols:
            c = f"{pfx}_{ss}"
            if c in pivot_df.columns:
                drop_base_cols.append(c)
    pivot_df.drop(columns=drop_base_cols, errors="ignore", inplace=True)

    # ================
    # (E) 使いやすいようにリネーム (元の実装の列名に合わせる)
    # ================
    # 例: gwin_cnt_all_芝 -> father_all_芝_重賞勝利数  (prefix=father)
    #     avg_dist_all_芝 -> father_all_芝_平均連対距離
    #     win_rate_all_芝 -> father_all_芝_平均勝率
    rename_map = {}
    for ss in set_sex_surf:
        rename_map[f"gwin_cnt_{ss}"]   = f"{prefix}_{ss}_重賞勝利数"       # father_牡馬_芝_重賞勝利数 など
        rename_map[f"avg_dist_{ss}"]  = f"{prefix}_{ss}_平均連対距離"
        rename_map[f"win_rate_{ss}"]  = f"{prefix}_{ss}_平均勝率"


    # pivot_dfの列をリネーム
    pivot_df.rename(columns=rename_map, inplace=True)

    # 万一 "female_ダート" などが一度も存在しなかった場合に備え、必須列を0埋めで用意
    needed_ss = [
                "牡牝混合_芝", "牡馬_芝", "牝馬_芝",
                "牡牝混合_ダート", "牡馬_ダート", "牝馬_ダート"
            ]
    for ss in needed_ss:
        for outcol in ["重賞勝利数","平均連対距離","平均勝率"]:
            colname = f"{prefix}_{ss}_{outcol}"
            if colname not in pivot_df.columns:
                pivot_df[colname] = 0.0

    # ================
    # (F) 元の df に merge して新しい列を付与
    # ================
    df_merged = pd.merge(
        df,
        pivot_df,
        on=[group_col, "date", "race_id"],
        how="left"
    )

    # マージできずに NaN になった列を 0 埋め
    for c in rename_map.values():
        if c in df_merged.columns:
            df_merged[c] = df_merged[c].fillna(0)

    # ================
    # (G) 中間列を片付けて終了
    # ================
    df_merged.drop(
        ["sex_class", "surface_class",
         "重賞勝利フラグ","出走フラグ","勝利フラグ","連対フラグ","連対距離"],
        axis=1, errors="ignore", inplace=True
    )

    # 元の順序に並べなおすなら最後にソートしてもいいわね
    # df_merged = df_merged.sort_values(["date","race_id","馬番"]).reset_index(drop=True)

    return df_merged


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
    df["出走フラグ"] = df["着順"].notna().astype(int)
    df["勝利フラグ"] = (df["着順"] == 1).astype(int)

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
              starts_sum = ("出走フラグ", "sum"),
              wins_sum   = ("勝利フラグ", "sum")
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
    df_merged.drop(["出走フラグ","勝利フラグ"], axis=1, errors="ignore", inplace=True)

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
         「重賞勝利数」「平均連対距離」「平均勝率」をリークなしで計算し付与

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

    # 8) (追加) 父産駒、母父産駒、母父が同じ馬 の「重賞勝利数/平均連対距離/平均勝率」を
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
