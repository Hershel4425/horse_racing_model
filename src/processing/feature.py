import os
import pickle
from IPython.display import display
from tqdm import tqdm

import pandas as pd
import numpy as np

import trueskill

import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder


ROOT_PATH = "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/153_競馬_v3/data"

DF_PATH = os.path.join(ROOT_PATH, "01_processed/60_combined/race_result_info_performance_pace_with_future_df.csv")

COURSE_PATH = os.path.join(ROOT_PATH, "03_input/course.csv") 

OUTPUT_PATH = os.path.join(ROOT_PATH, "02_features/feature.csv")


# -------------------------------
# 日付処理
# -------------------------------
def process_date(df):
    """日付を処理する関数
    Args:
        df (dataframe): データフレーム
    Returns:
        df (dataframe): 日付列を処理したデータフレーム
    """
    # 年月日時分をintにする,nanはそれぞれ対処する
    df["年"] = df["年"].fillna(2999).astype(int)
    df["月"] = df["月"].fillna(12).astype(int)
    df["日"] = df["日"].fillna(31).astype(int)
    df["時"] = df["時"].fillna(23).astype(int)
    df["分"] = df["分"].fillna(59).astype(int)
    # 日付をdatetime型に変換
    df["date"] = pd.to_datetime(
        df["年"].astype(str)
        + "-"
        + df["月"].astype(str)
        + "-"
        + df["日"].astype(str)
        + " "
        + df["時"].astype(str)
        + ":"
        + df["分"].astype(str),
        format="%Y-%m-%d %H:%M",
    )  # 日付をdatetime型に変換

    # 不要列を削除
    df = df.drop(columns=["年", "月", "日", "時", "分"])

    return df


# -------------------------------
# レーティング計算
# -------------------------------
def create_trueskill_ratings(df):
    print("レーティングを計算します。")
    print("処理前DF：", df.shape)

    df = df.copy()
    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # ソート
    df = df.sort_values(['date', 'race_id', '着順']).reset_index(drop=True)

    # === TrueSkill環境の作成 (必要に応じてパラメータを調整) ===
    env = trueskill.TrueSkill(draw_probability=0.0)

    # === horse_id, jockey_id のRatingを管理する辞書を初期化 ===
    horse_ratings = {}
    jockey_ratings = {}

    # === レース前のレーティングを保持する列を追加 ===
    df["馬レーティング"] = None
    df["馬レーティング_sigma"] = None
    df["騎手レーティング"] = None
    df["騎手レーティング_sigma"] = None

    # レース単位で groupby
    for race_id, group in tqdm(df.groupby("race_id", sort=False)):
        idxs = group.index.tolist()
        # ---------------------------------
        # 1) 競走馬のレーティング更新
        # ---------------------------------
        # レースに出た馬を取り出して「レース前」の値を df に記録
        participants_horse = []
        ranks_horse = []
        for i in idxs:
            h_id = df.at[i, "horse_id"]
            rank = df.at[i, "着順"]
            # 欠損がある場合はスキップ
            if pd.isna(h_id) or pd.isna(rank):
                continue

            # 初めての馬なら初期レーティングを作成
            if h_id not in horse_ratings:
                horse_ratings[h_id] = env.create_rating()

            # レース前のRatingをDataFrameに格納
            df.at[i, "馬レーティング"]   = horse_ratings[h_id].mu
            df.at[i, "馬レーティング_sigma"] = horse_ratings[h_id].sigma

            # TrueSkillに渡すためのデータを作成
            # 「馬1頭 = 1チーム」→ [rating_obj] のリストにする
            participants_horse.append([horse_ratings[h_id]])
            # ranks は 1着=0, 2着=1, ... という形で渡すのが基本
            ranks_horse.append(int(rank) - 1)

        # 2頭以上いないと TrueSkill のレーティング更新はできない
        if len(participants_horse) >= 2:
            updated_ratings_horse = env.rate(participants_horse, ranks=ranks_horse)
            # 更新したレーティングを辞書に上書き
            idx_for_update = 0
            for i in idxs:
                h_id = df.at[i, "horse_id"]
                rank = df.at[i, "着順"]
                if pd.isna(h_id) or pd.isna(rank):
                    continue
                horse_ratings[h_id] = updated_ratings_horse[idx_for_update][0]
                idx_for_update += 1
        # ---------------------------------

        # ---------------------------------
        # 2) 騎手のレーティング更新 (同様)
        # ---------------------------------
        participants_jockey = []
        ranks_jockey = []
        for i in idxs:
            j_id = df.at[i, "jockey_id"]
            rank = df.at[i, "着順"]
            if pd.isna(j_id) or pd.isna(rank):
                continue

            if j_id not in jockey_ratings:
                jockey_ratings[j_id] = env.create_rating()

            df.at[i, "騎手レーティング"]   = jockey_ratings[j_id].mu
            df.at[i, "騎手レーティング_sigma"] = jockey_ratings[j_id].sigma

            participants_jockey.append([jockey_ratings[j_id]])
            ranks_jockey.append(int(rank) - 1)

        if len(participants_jockey) >= 2:
            updated_ratings_jockey = env.rate(participants_jockey, ranks=ranks_jockey)
            idx_for_update = 0
            for i in idxs:
                j_id = df.at[i, "jockey_id"]
                rank = df.at[i, "着順"]
                if pd.isna(j_id) or pd.isna(rank):
                    continue
                jockey_ratings[j_id] = updated_ratings_jockey[idx_for_update][0]
                idx_for_update += 1
        # ---------------------------------

    return df


# -------------------------------
# レースごとのレーティング平均を計算する関数
# -------------------------------
def calc_rating_mean(df):
    print("レーティングの平均を計算します。。")
    print("処理前DF：", df.shape)

    df = df.copy()
    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # ソート
    df = df.sort_values(['horse_id', 'date', 'race_id']).reset_index(drop=True)

    # レースごとにhorse_mu_beforeの平均を取る
    df["馬平均レーティング"] = df.groupby("race_id")["馬レーティング"].transform("mean")
    df["騎手平均レーティング"] = df.groupby("race_id")["騎手レーティング"].transform("mean")

    print("処理後DF：", df.shape)

    return df


# -------------------------------
# 騎乗回数、テン乗り成績の計算
# -------------------------------
def add_jockey_change_features(df):
    """
    テン乗りフラグ・継続騎乗回数・騎手テン乗り成績を追加する関数。

    1) テン乗りフラグ: 前走(同一horse_id)から騎手が変わった場合=1、同一騎手=0
       - ただし、初出走(または前走がない)場合は0とする
    2) 継続騎乗回数: 同一馬・同一騎手の継続騎乗回数
       - 初騎乗なら1、次走も同じ騎手なら2、…とカウントアップ
    3) 騎手テン乗り成績: 騎手が「馬との初コンビ(初乗り)」で騎乗したレースの成績を
       騎手単位で集計したもの(過去分のみ)を「初騎乗時の勝率」等として特徴量にする例。

       例: jockey_id=100の騎手が
         - 過去に(馬X,馬Y,馬Z)で '初乗り' をしており、その際に
           1着1回 / 全3回 だった場合、勝率=1/3=0.333...
         - その騎手が新たに馬Wに初騎乗する時、騎手テン乗り成績=0.333... とする(リーク回避のため当該レース結果は含めない)
         - レース終了後に成績を更新し、次の初騎乗時に反映する

    Args:
        df (DataFrame): ['horse_id', 'date', 'race_id', '着順', '騎手'] などを含むデータフレーム

    Returns:
        df (DataFrame): 上記3つの列( 'テン乗りフラグ', '継続騎乗回数', '騎手テン乗り成績' )が追加されたデータフレーム
    """
    df = df.copy()
    # 重複行削除
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # 日付順で並べる(馬単位→日付昇順)
    df = df.sort_values(['horse_id', 'date', 'race_id']).reset_index(drop=True)

    # ============== 1) テン乗りフラグ, 2) 継続騎乗回数 ==============
    # 1) テン乗りフラグ
    def fix_ten_nori_flag(df):
        df = df.copy()
        df = df.drop_duplicates(subset=["race_id", "馬番"])
        df = df.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
        df["テン乗りフラグ"] = 0

        visited_pairs = set()
        print('テン乗りフラグ付与開始')
        for i, row in tqdm(df.iterrows()):
            h_id = row["horse_id"]
            j_id = row["騎手"]
            if pd.isna(h_id) or pd.isna(j_id):
                continue
            pair = (h_id, j_id)
            if pair not in visited_pairs:
                df.at[i, "テン乗りフラグ"] = 1
                visited_pairs.add(pair)

        return df

    df = fix_ten_nori_flag(df)

    # 2) 継続騎乗回数付与
    # 今はまだ

    # ============== 3) 騎手テン乗り成績 ==============
    # ここもまだ

    return df


# -------------------------------
# ジョッキーの変化を特徴量としてついか
# -------------------------------
def create_jockey_rating_change_feature(df):
    """
    前走とのジョッキー変更を検出し、ジョッキーのレーティング変化量を特徴量として付与する関数。
    """
    print("ジョッキーの変化を計算します。。")
    print("処理前DF：", df.shape)

    df = df.copy()
    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # ソート
    df = df.sort_values(['horse_id', 'date', 'race_id']).reset_index(drop=True)

    # 馬ごとにグループ化し、前走のジョッキーID と ジョッキーのレーティング を shift() で取り出す
    df["prev_jockey_id"] = df.groupby('horse_id')["jockey_id"].shift(1)
    df["prev_騎手レーティング"] = df.groupby('horse_id')["騎手レーティング"].shift(1)

    # ジョッキーが変わったかどうかを判定する
    # 変わっていなければ 0、変わっていれば (今のjockey_mu - 前走のjockey_mu)
    # ただし前走が存在しない（NaN）の場合は 0 とする
    def calc_jockey_change(row):
        if pd.isna(row["prev_jockey_id"]):
            return 0
        if row["jockey_id"] == row["prev_jockey_id"]:
            return 0
        else:
            # 変化があった場合
            if pd.isna(row["prev_騎手レーティング"]):
                # 前走の mu が NaN (初回レース等) なら変化量は 0 と定義
                return 0
            else:
                # 今回のジョッキーの "jockey_mu_before" から、前回ジョッキーの "prev_jockey_mu" を引いた差をとる
                return row["騎手レーティング"] - row["prev_騎手レーティング"]
            
    df["騎手レーティング_change"] = df.apply(calc_jockey_change, axis=1)

    df.drop(["prev_jockey_id", "prev_騎手レーティング"], axis=1, inplace=True)

    return df



# -------------------------------
# 過去成績の作成の一覧
# -------------------------------
def calc_cumulative_stats(df, who, whoid, grade_list, dist_list, racetrack_list):

    surfaces = ["芝", "ダート"]
    suffixes = ["通算試合数", "通算勝利数", "通算複勝数"]
    all_columns = []

    # カラム名生成のベクトル化（ループ回数の削減）
    base_cols = []
    for surface in surfaces:
        base_cols.extend([f"{who}{surface}{s}" for s in suffixes])
        for grade in grade_list:
            base_cols.extend([f"{who}{surface}{grade}{s}" for s in suffixes])
        base_cols.extend([f"{who}{surface}重賞{s}" for s in suffixes])
        for dist in dist_list:
            base_cols.extend([f"{who}{surface}{dist}{s}" for s in suffixes])
        for racetrack in racetrack_list:
            base_cols.extend([f"{who}{surface}{racetrack}{s}" for s in suffixes])

    all_columns = base_cols
    # ソート
    df = df.sort_values([whoid, "date"])

    # groupbyシフトを一括適用
    df[all_columns] = df.groupby(whoid, group_keys=False)[all_columns].shift(1)
    df[all_columns] = df[all_columns].fillna(0)
    # groupby cumsumを一括適用
    df[all_columns] = df.groupby(whoid, group_keys=False)[all_columns].cumsum()
    # 最小値を一括で引く
    df[all_columns] = df[all_columns] - df.groupby(whoid)[all_columns].transform("min")

    # 単年成績計算
    df.loc[:, "年"] = df["date"].dt.year
    # 年・whoid単位の最小値を一括計算
    yearly_min = df.groupby(["年", whoid])[all_columns].transform("min")
    # 年単位の成績差分をまとめて計算
    diff_df = df[all_columns].sub(yearly_min, axis=0)
    diff_df.columns = ["単年" + c for c in diff_df.columns]

    # concat で一括結合することで断片化を防ぐ
    df = pd.concat([df, diff_df], axis=1)

    df = df.drop(columns=["年"])

    return df

# 過去成績作成関数
def calc_career_statics(df, who):
    pd.set_option("compute.use_bottleneck", True)
    pd.set_option("compute.use_numexpr", True)

    print("過去の競争成績を集計します。")
    print("処理前DF：", df.shape)

    df = df.copy()
    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # ソート
    df = df.sort_values(['horse_id', 'date', 'race_id']).reset_index(drop=True)

    # 入力から必要な列だけ抽出
    ext_columns = [
        "race_id",
        "horse_id",
        "jockey_id",
        "着順",
        "人気",
        "グレード",
        "距離",
        "賞金",
        "競馬場",
        "date",
        "コース種類"
    ]
    df = df.drop_duplicates(subset=["race_id", "馬番"]).sort_values(["date", "馬番"]).reset_index(drop=True)
    ext_df = df[ext_columns].copy()
    whoid = "horse_id" if who == "競走馬" else "jockey_id"

    grade_list = ["新馬","未勝利","1勝クラス","2勝クラス","3勝クラス","オープン","G3","G2","G1"]
    dist_list = ["短距離","マイル","中距離","クラシック","長距離"]
    dist_dict = {
        "短距離": [1000, 1400],
        "マイル": [1401, 1799],
        "中距離": [1800, 2200],
        "クラシック": [2201, 2600],
        "長距離": [2601, 4000],
    }
    racetrack_list = ["札幌","函館","福島","新潟","東京","中山","中京","京都","阪神","小倉"]

    surfaces = ["芝", "ダート"]

    # 着順・人気をint化
    ext_df["着順"] = ext_df["着順"].astype(int)
    ext_df["人気"] = ext_df["人気"].astype(int)

    # 条件毎にフラグをまとめて作成
    new_cols = {}  # 全ての新列を格納する辞書

    for surface in surfaces:
        s_mask = (df["コース種類"] == surface)
        new_cols[f"{who}{surface}通算試合数"] = s_mask.astype(int)
        new_cols[f"{who}{surface}通算勝利数"] = (s_mask & (ext_df["着順"]==1)).astype(int)
        new_cols[f"{who}{surface}通算複勝数"] = (s_mask & (ext_df["着順"]<=3)).astype(int)

        for grade in grade_list:
            g_mask = s_mask & (ext_df["グレード"] == grade)
            new_cols[f"{who}{surface}{grade}通算試合数"] = g_mask.astype(int)
            new_cols[f"{who}{surface}{grade}通算勝利数"] = (g_mask & (ext_df["着順"]==1)).astype(int)
            new_cols[f"{who}{surface}{grade}通算複勝数"] = (g_mask & (ext_df["着順"]<=3)).astype(int)

        g_mask2 = s_mask & ext_df["グレード"].str.contains("G", na=False)
        new_cols[f"{who}{surface}重賞通算試合数"] = g_mask2.astype(int)
        new_cols[f"{who}{surface}重賞通算勝利数"] = (g_mask2 & (ext_df["着順"]==1)).astype(int)
        new_cols[f"{who}{surface}重賞通算複勝数"] = (g_mask2 & (ext_df["着順"]<=3)).astype(int)

        for dist in dist_list:
            low, up = dist_dict[dist]
            d_mask = s_mask & (ext_df["距離"]>=low) & (ext_df["距離"]<=up)
            new_cols[f"{who}{surface}{dist}通算試合数"] = d_mask.astype(int)
            new_cols[f"{who}{surface}{dist}通算勝利数"] = (d_mask & (ext_df["着順"]==1)).astype(int)
            new_cols[f"{who}{surface}{dist}通算複勝数"] = (d_mask & (ext_df["着順"]<=3)).astype(int)

        for racetrack in racetrack_list:
            r_mask = s_mask & (ext_df["競馬場"] == racetrack)
            new_cols[f"{who}{surface}{racetrack}通算試合数"] = r_mask.astype(int)
            new_cols[f"{who}{surface}{racetrack}通算勝利数"] = (r_mask & (ext_df["着順"]==1)).astype(int)
            new_cols[f"{who}{surface}{racetrack}通算複勝数"] = (r_mask & (ext_df["着順"]<=3)).astype(int)


    # 全ての新規カラムをまとめて結合
    new_cols_df = pd.DataFrame(new_cols, index=ext_df.index)
    ext_df = pd.concat([ext_df, new_cols_df], axis=1)

    ext_df = calc_cumulative_stats(ext_df, who, whoid, grade_list, dist_list, racetrack_list)
    ext_df = ext_df.drop_duplicates(subset=["race_id","horse_id","jockey_id"])
    df = pd.merge(df, ext_df, on=ext_columns, how="left").sort_values(["date", "馬番"])

    return df


# -------------------------------
# コース情報を追加
# -------------------------------
def concat_race_info(df):

    print('コースを結合')
    print("処理前DF：", df.shape)
    course_df = pd.read_csv(COURSE_PATH, encoding='utf-8-sig')

    df = pd.merge(df, course_df, on = ['競馬場','内外'], how = 'left')

    return df



# -------------------------------
# 適正列を追加して計算する
# -------------------------------
def create_flag_features_and_update(df):
    """
    指定のフラグ作成と、フラグの更新（新規値 = 0.2*過去レースの順位点 + 0.8*旧値）を行う関数。
    リークを防ぐため、直前レースの順位点のみを参照し、当該レースの順位点は使わない。
    フラグの初期値は0.5とする。

    Args:
        df (DataFrame): 処理対象のデータフレーム
    
    Returns:
        DataFrame: フラグ生成および更新後のデータフレーム
    """
    print("適正列を追加します。")
    print("処理前DF：", df.shape)

    df = df.copy()
    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # ソート
    df = df.sort_values(['horse_id', 'date', 'race_id']).reset_index(drop=True)

    print('適正列を追加')
    def create_flag_features(df_):
        df_ = df_.copy()

        # -------------------------------
        # 枠によるフラグ作成
        # -------------------------------
        df_["内枠"] = df_["馬番"].apply(lambda x: 1 if 1 <= x <= 6 else 0)
        df_["中枠"] = df_["馬番"].apply(lambda x: 1 if 7 <= x <= 12 else 0)
        df_["外枠"] = df_["馬番"].apply(lambda x: 1 if 13 <= x <= 18 else 0)

        # -------------------------------
        # 距離によるフラグ作成
        # -------------------------------
        df_["短距離"] = df_["距離"].apply(lambda x: 1 if 1000 <= x <= 1400 else 0)
        df_["マイル"] = df_["距離"].apply(lambda x: 1 if 1401 <= x <= 1799 else 0)
        df_["中距離"] = df_["距離"].apply(lambda x: 1 if 1800 <= x <= 2200 else 0)
        df_["クラシック"] = df_["距離"].apply(lambda x: 1 if 2201 <= x <= 2600 else 0)
        df_["長距離"] = df_["距離"].apply(lambda x: 1 if 2601 <= x <= 4000 else 0)

        # -------------------------------
        # 方向フラグ作成
        # -------------------------------
        df_["方向_右"] = (df_["方向"] == "右").astype(int)
        df_["方向_左"] = (df_["方向"] == "左").astype(int)
        df_["方向_直線"] = (df_["方向"] == "直線").astype(int)

        # -------------------------------
        # コース種類フラグ作成
        # -------------------------------
        df_["芝"] = (df_["コース種類"] == "芝").astype(int)
        df_["ダート"] = (df_["コース種類"] == "ダート").astype(int)

        # -------------------------------
        # カーブ種類フラグ作成
        # -------------------------------
        df_["大回り"] = (df_["カーブ"] == "大回り").astype(int)
        df_["小回り"] = (df_["カーブ"] == "小回り").astype(int)
        df_["急"] = (df_["カーブ"] == "小回り").astype(int)

        # -------------------------------
        # 坂フラグ作成
        # -------------------------------
        df_["急坂"] = (df_["ゴール前坂"] == "急坂").astype(int)
        df_["平坦"] = (df_["ゴール前坂"] == "平坦").astype(int)
        df_["緩坂"] = (df_["ゴール前坂"] == "緩坂").astype(int)

        # -------------------------------
        # 芝タイプフラグ作成
        # -------------------------------
        df_["芝軽"] = (df_["芝タイプ"] == "軽").astype(int)
        df_["芝中"] = (df_["芝タイプ"] == "中").astype(int)
        df_["芝重"] = (df_["芝タイプ"] == "重").astype(int)

        # -------------------------------
        # ダートタイプフラグ作成
        # -------------------------------
        df_["ダート軽"] = (df_["ダートタイプ"] == "軽").astype(int)
        df_["ダート中"] = (df_["ダートタイプ"] == "中").astype(int)
        df_["ダート重"] = (df_["ダートタイプ"] == "重").astype(int)

        # -------------------------------
        # スパートタイプフラグ作成
        # -------------------------------
        df_["ロンスパ"] = (df_["スパートタイプ"] == "ロンスパ").astype(int)
        df_["瞬発力"] = (df_["スパートタイプ"] == "瞬発力").astype(int)

        # -------------------------------
        # スパート速度フラグ作成
        # -------------------------------
        df_["高速"] = (df_["スパート速度"] == "高速").astype(int)
        df_["中速"] = (df_["スパート速度"] == "中速").astype(int)
        df_["低速"] = (df_["スパート速度"] == "低速").astype(int)

        # -------------------------------
        # 天気の統合 & フラグ作成
        # -------------------------------
        df_["天気"] = df_["天気"].replace({"小雪": "雪", "小雨": "雨"})
        df_["天気_晴"] = (df_["天気"] == "晴").astype(int)
        df_["天気_雨"] = (df_["天気"] == "雨").astype(int)
        df_["天気_曇"] = (df_["天気"] == "曇").astype(int)
        df_["天気_雪"] = (df_["天気"] == "雪").astype(int)

        # -------------------------------
        # 馬場フラグ作成
        # -------------------------------
        df_["馬場_良"] = (df_["馬場"] == "良").astype(int)
        df_["馬場_不"] = (df_["馬場"] == "不").astype(int)
        df_["馬場_重"] = (df_["馬場"] == "重").astype(int)
        df_["馬場_稍"] = (df_["馬場"] == "稍").astype(int)

        # -------------------------------
        # 重賞or平場フラグ作成
        # -------------------------------
        df_["平場"] = ((df_["グレード"]!="G1") & (df_["グレード"]!="G2") & (df_["グレード"]!="G3")).astype(int)
        df_["重賞"] = ((df_["グレード"]=="G1") | (df_["グレード"]=="G2") | (df_["グレード"]=="G3")).astype(int)

        # -------------------------------
        # 順位点の作成
        # ※ 当該行のレース情報として「最終的に付与される順位点」は残すが、
        #   更新時には"直前レース"の順位点だけを参照するようにする
        # -------------------------------
        df_["順位点"] = (df_["立て数"] - df_["着順"] + 1) / df_["立て数"]
        return df_

    # ----------------------------------------
    # 元のデータを複製してフラグ列を作成
    # ----------------------------------------
    df = df.copy()
    df.drop_duplicates(subset=["race_id", "馬番"], inplace=True)
    df = df.sort_values(["date", "race_id", "着順"]).reset_index(drop=True)
    df = create_flag_features(df)

    # ----------------------------------------
    # ここで、フラグ列の一覧を作成
    # ----------------------------------------
    flag_cols = [
        "内枠", "中枠", "外枠",
        "短距離", "マイル", "中距離", "クラシック", "長距離",
        "方向_右", "方向_左", "方向_直線",
        "芝", "ダート",
        "大回り", "小回り", "急",
        "急坂", "平坦", "緩坂",
        "芝軽", "芝中", "芝重",
        "ダート軽", "ダート中", "ダート重",
        "ロンスパ", "瞬発力",
        "高速", "中速", "低速",
        "天気_晴", "天気_雨", "天気_曇", "天気_雪",
        "馬場_良", "馬場_不", "馬場_重", "馬場_稍",
        "平場", "重賞"
    ]

    # ----------------------------------------
    # フラグ列ごとに、馬 & 騎手のTrueSkillレーティングを更新
    #   - 各フラグ用のレーティング辞書を作成
    #   - レースごとに「フラグ=1」のもののみレーティングを更新
    #   - フラグ=0のレースはレーティング更新せず、直前のレーティングを保持する
    #   - 結果は各レース行に horse_mu_{flag}, horse_sigma_{flag}, 
    #     jockey_mu_{flag}, jockey_sigma_{flag} として格納
    # ----------------------------------------

    # TrueSkill 環境の作成
    env = trueskill.TrueSkill(draw_probability=0.0)

    # フラグ列×馬(jockey)ごとのレーティングを管理する辞書
    #   horse_ratings[flag][horse_id] -> Rating
    #   jockey_ratings[flag][jockey_id] -> Rating
    horse_ratings = {flag: {} for flag in flag_cols}
    jockey_ratings = {flag: {} for flag in flag_cols}

    # 各フラグ列のレーティング(mu, sigma)を格納するための列を df に追加
    for flag in flag_cols:
        df[f"horse_mu_{flag}"]   = None
        df[f"horse_sigma_{flag}"] = None
        df[f"jockey_mu_{flag}"]  = None
        df[f"jockey_sigma_{flag}"] = None

    # レース単位でgroupbyし、時系列順に TrueSkill を更新
    # レース前の rating を各行に格納し、フラグ=1なら更新
    for race_id, group in tqdm(df.groupby("race_id", sort=False)):
        # 同一レース内の行インデックスを取得
        idxs = group.index.tolist()

        # ----------------------------
        # 1) 馬のレーティング更新
        #    フラグごとに処理を行う
        # ----------------------------
        for flag in flag_cols:
            # 更新に必要なデータを用意
            participants_for_update = []  # [[Rating], [Rating], ...]
            ranks_for_update = []
            indices_for_update = []       # レース行のindexを覚えておく

            for i in idxs:
                h_id = df.at[i, "horse_id"]
                rank = df.at[i, "着順"]
                # 初出の馬ならRatingを初期化
                if h_id not in horse_ratings[flag]:
                    horse_ratings[flag][h_id] = env.create_rating()

                # 現時点のレーティングを格納（更新前の値を入れる）
                df.at[i, f"horse_mu_{flag}"]   = horse_ratings[flag][h_id].mu
                df.at[i, f"horse_sigma_{flag}"] = horse_ratings[flag][h_id].sigma

            # 今レースのうち「flag=1」の行だけ更新対象とする
            # ただし、rank が NaN ならスキップ
            group_flag_1 = group[group[flag] == 1]
            for i2 in group_flag_1.index:
                h_id = df.at[i2, "horse_id"]
                rank = df.at[i2, "着順"]
                if pd.isna(rank):
                    continue
                participants_for_update.append([horse_ratings[flag][h_id]])
                ranks_for_update.append(int(rank) - 1)
                indices_for_update.append(i2)

            # 実際に更新処理を走らせる (2頭以上揃っている場合のみ)
            if len(participants_for_update) >= 2:
                updated_ratings = env.rate(participants_for_update, ranks=ranks_for_update)
                for ix, i2 in enumerate(indices_for_update):
                    h_id = df.at[i2, "horse_id"]
                    horse_ratings[flag][h_id] = updated_ratings[ix][0]

            # 更新後のレーティングを格納（今回レース時点の値として扱う）
            # フラグ=0の行も最新値を格納（更新は行われていないが、値は保持）
            for i in idxs:
                h_id = df.at[i, "horse_id"]
                df.at[i, f"horse_mu_{flag}"]   = horse_ratings[flag][h_id].mu
                df.at[i, f"horse_sigma_{flag}"] = horse_ratings[flag][h_id].sigma

        # ----------------------------
        # 2) 騎手のレーティング更新
        #    馬の場合と同様
        # ----------------------------
        for flag in flag_cols:
            participants_for_update = []
            ranks_for_update = []
            indices_for_update = []

            for i in idxs:
                j_id = df.at[i, "jockey_id"]
                rank = df.at[i, "着順"]
                if j_id not in jockey_ratings[flag]:
                    jockey_ratings[flag][j_id] = env.create_rating()

                df.at[i, f"jockey_mu_{flag}"]   = jockey_ratings[flag][j_id].mu
                df.at[i, f"jockey_sigma_{flag}"] = jockey_ratings[flag][j_id].sigma

            group_flag_1 = group[group[flag] == 1]
            for i2 in group_flag_1.index:
                j_id = df.at[i2, "jockey_id"]
                rank = df.at[i2, "着順"]
                if pd.isna(rank):
                    continue
                participants_for_update.append([jockey_ratings[flag][j_id]])
                ranks_for_update.append(int(rank) - 1)
                indices_for_update.append(i2)

            if len(participants_for_update) >= 2:
                updated_ratings = env.rate(participants_for_update, ranks=ranks_for_update)
                for ix, i2 in enumerate(indices_for_update):
                    j_id = df.at[i2, "jockey_id"]
                    jockey_ratings[flag][j_id] = updated_ratings[ix][0]

            # 更新後のレーティングを再度書き込み
            for i in idxs:
                j_id = df.at[i, "jockey_id"]
                df.at[i, f"jockey_mu_{flag}"]   = jockey_ratings[flag][j_id].mu
                df.at[i, f"jockey_sigma_{flag}"] = jockey_ratings[flag][j_id].sigma

    return df


# -------------------------------
# 累計獲得賞金列の追加
# -------------------------------
def calucrate_add_money(df):
    """賞金の合計額を計算する関数

    Args:
        df (dataframe): 処理対象データフレーム

    Returns:
        df (dataframe): 処理後のデータフレーム
    """
    print("賞金の合計額を計算します。")
    print("処理前DF：", df.shape)
    df = df.copy()
    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # ソート
    df = df.sort_values(['horse_id', 'date', 'race_id']).reset_index(drop=True)

    print("賞金合計額を計算します。")

    df['累計獲得賞金'] = df.groupby('horse_id')['賞金'].transform(lambda s: s.shift(1).expanding().sum())

    return df


# -------------------------------
# 上がり3F順位の追加
# -------------------------------
def calculate_3f_features(df):
    """上がり3F系特徴量を計算する関数

    Args:
        df (dataframe): 処理対象データフレーム

    Returns:
        df (dataframe): 処理後のデータフレーム
    """
    df = df.copy()
    # 重複行削除
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # ソート
    df = df.sort_values(['date', 'race_id', 'horse_id']).reset_index(drop=True)

    print('上がり3F列を追加')

    # 上がり3F順位の計算（同率考慮）
    # 「上がり3F」が小さいほど順位が高いことを想定し、昇順でrankする
    df['上がり3F順位'] = df.groupby('race_id')['上がり3F'].rank(method='min', ascending=True)

    # 過去平均上がり3F順位の計算（リーク防止のためシフトしてexpandingで平均）
    df['過去平均上がり3F順位'] = df.groupby('horse_id')['上がり3F順位'].transform(lambda s: s.shift(1).expanding().mean())

    return df



# -------------------------------
# ここから上がり3Fを用いた指数作成
# -------------------------------
# ===== 設定値 =====
LOOKBACK_YEARS = 5
ROOT_PATH = "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/153_競馬_v3"
MODEL_SAVE_DIR = ROOT_PATH + "/models/上がり3Fモデル_固定効果のみ"
OUTPUT_COEF_PATH = ROOT_PATH + "/result/intercept_fixed_effects_fixed_only.csv"
MODEL_FREQ = '3MS'  # 3ヶ月おき

if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

def generate_model_dates(df, start_date=None, end_date=None, freq=MODEL_FREQ):
    if start_date is None:
        start_date = df['date'].min()
    if end_date is None:
        end_date = df['date'].max()
    model_dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    three_years_after_start = start_date + pd.DateOffset(years=LOOKBACK_YEARS)
    model_dates = model_dates[model_dates >= three_years_after_start]
    return model_dates

def fit_model_for_date(df, model_date, lookback_years=LOOKBACK_YEARS, cat_cols=None):
    model_filename = os.path.join(MODEL_SAVE_DIR, f"model_{model_date.date()}.pkl")
    print("="*50)
    print(f"【モデル作成日: {model_date.date()}】")

    # 既存モデルがあればロード
    if os.path.exists(model_filename):
        print(f"  既存モデル検出: {model_filename}")
        try:
            with open(model_filename, 'rb') as f:
                model_info = pickle.load(f)
            print("  モデルロード成功")
            return model_info
        except Exception as e:
            print(f"  モデル読み込み失敗: {e}")
            print("  モデル再学習を実施")

    cutoff_start = model_date - pd.DateOffset(years=lookback_years)
    train_data = df[(df['date'] <= model_date) & (df['date'] > cutoff_start)].copy()
    # データ抽出（上がり3F: 30〜40）
    train_data = train_data[(train_data['上がり3F'] >= 30) & (train_data['上がり3F'] <= 40)]
    print(f"  訓練データ件数: {len(train_data)}")
    if len(train_data) < 50:
        print("  訓練データが50件未満のため学習スキップ")
        return None

    # カテゴリ変数OneHotエンコード
    enc = OneHotEncoder(sparse_output=False)
    cat_data = enc.fit_transform(train_data[cat_cols])
    X = cat_data
    y = train_data['上がり3F'].values
    feature_names = enc.get_feature_names_out(cat_cols)

    print("  モデルフィット開始 (Ridge)")
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    print("  モデルフィット完了")
    score = model.score(X, y)
    print(f"  R-squared(近似): {score:.4f}")
    print("  係数概要:")
    for fname, cval in zip(feature_names, model.coef_):
        print(f"    {fname}: {cval:.4f}")
    print(f"  Intercept: {model.intercept_:.4f}")

    # モデル情報を辞書で保存
    model_info = {
        'enc': enc,
        'model': model,
        'feature_names': feature_names
    }

    # モデル保存
    try:
        with open(model_filename, 'wb') as f:
            pickle.dump(model_info, f)
        print(f"  モデル保存完了: {model_filename}")
    except Exception as e:
        print(f"  モデル保存失敗: {e}")

    return model_info

def assign_horse_ability(df, model_dict, cat_cols):
    print("="*50)
    print("【馬能力指数付与開始】")
    df = df.copy()
    df['horse_ability'] = np.nan
    sorted_model_dates = sorted(model_dict.keys())

    for idx, row in df.iterrows():
        race_date = row['date']
        use_model_date = None
        for md in reversed(sorted_model_dates):
            if md <= race_date:
                use_model_date = md
                break

        if use_model_date is None or model_dict[use_model_date] is None:
            # 利用可能なモデルがなければ1とする
            df.at[idx, 'horse_ability'] = 1
            continue

        model_info = model_dict[use_model_date]
        enc = model_info['enc']
        model = model_info['model']

        # カラム名付きデータフレームを作成することでwarningを回避
        row_cat_df = pd.DataFrame([row[cat_cols].values], columns=cat_cols)
        row_ohe = enc.transform(row_cat_df)
        pred = model.predict(row_ohe)[0]
        ability = row['上がり3F'] / pred
        # 異常値は1にする
        if ability <= 0.1:
            ability = 1
        df.at[idx, 'horse_ability'] = ability

    print("【馬能力指数付与完了】")
    return df

def extract_coefficients(model_dict):
    print("="*50)
    print("【係数抽出開始】")
    records = []
    for md, model_info in model_dict.items():
        if model_info is None:
            continue
        row_dict = {'model_date': md, 'Intercept': model_info['model'].intercept_}
        for fname, cval in zip(model_info['feature_names'], model_info['model'].coef_):
            row_dict[fname] = cval
        records.append(row_dict)
    coef_df = pd.DataFrame(records)
    print("【係数抽出完了】")
    return coef_df

def calc_ability(train_df):
    print("【能力指数計算開始】")
    train_df = train_df.copy()
    train_df = train_df.drop_duplicates(subset=["race_id", "馬番"])
    train_df = train_df.sort_values(['horse_id', 'date', 'race_id']).reset_index(drop=True)

    cat_cols = ['馬場', '競馬場', 'コース種類']

    model_dates = generate_model_dates(train_df)
    print(f"モデル学習基準日一覧: {model_dates}")

    model_dict = {}
    for md in model_dates:
        model_info = fit_model_for_date(train_df, md, cat_cols=cat_cols)
        model_dict[md] = model_info

    train_df = assign_horse_ability(train_df, model_dict, cat_cols)
    coef_df = extract_coefficients(model_dict)
    coef_df.to_csv(OUTPUT_COEF_PATH, index=False)
    print(f"係数を{OUTPUT_COEF_PATH}に保存しました")
    print("【能力指数計算完了】")
    return train_df

def calculate_horse_ability_mean(df):
    """horse_abilityの平均を馬ごとに、リークを防ぐためshift(過去情報のみ)を用いて計算する関数
    
    Args:
        df (DataFrame): 処理対象データフレーム。horse_ability列があることを前提。

    Returns:
        df (DataFrame): horse_abilityの平均列(horse_ability_mean)が追加されたデータフレーム
    """
    print("horse_abilityの平均を計算します。")
    print("処理前DF：", df.shape)
    df = df.copy()
    # 重複行の削除（レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # ソート
    df = df.sort_values(['horse_id', 'date', 'race_id']).reset_index(drop=True)

    print("horse_ability平均(過去分)を計算します。")
    # shift(1)で現在行を除外、expanding().mean()で過去データのみの平均を算出
    df['horse_ability_mean'] = df.groupby('horse_id')['horse_ability'].transform(lambda s: s.shift(1).expanding().mean())

    print("処理後DF：", df.shape)
    return df


# -------------------------------
# 平均回収率計算
# -------------------------------
def calculate_return_rate(df, key_col):
    """key_col単位で過去の平均回収率を計算する関数
    1着の場合のみ「単勝」列の値、それ以外は0とし、
    shift(1)で現在行を除外した過去データからexpanding().mean()で平均回収率を算出する。

    Args:
        df (DataFrame): 処理対象データフレーム
        key_col (str): 集計単位のカラム名（例: 'horse_id', '騎手'）

    Returns:
        Series: key_col単位で計算された平均回収率を持つSeries (dfと同じIndexを持つ)
    """
    returns = ((df['着順'] == 1).astype(int) * df['単勝']).fillna(0)
    returns.name = '回収'
    return returns.groupby(df[key_col]).transform(lambda s: s.shift(1).expanding().mean())


# -------------------------------
# 平均回収率計算+1~n走前の特徴量追加
# -------------------------------
def add_past_features_and_stats(df):
    """各馬ごとにn走前の特徴量や平均連帯距離、馬・騎手の平均回収率を計算してdfに追加する関数

    Args:
        df (DataFrame): 処理対象データフレーム
            必須列：
            - horse_id, race_id, 馬番, date
            - 着順, 距離, グレード, コース種類, 着差, horse_ability
            - 単勝, 騎手

    Returns:
        df (DataFrame)
    """
    df = df.copy()
    # 重複行削除
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # ソート
    df = df.sort_values(['horse_id', 'date', 'race_id']).reset_index(drop=True)

    # n走前特徴量追加
    print('過去特徴量を追加します。')
    group = df.groupby('horse_id')
    for col in ['着順', '距離', 'グレード', 'コース種類', '着差', '上がり3F順位', 'horse_ability',
                '1着タイム差', '1C通過順位', '2C通過順位', '3C通過順位', '4C通過順位', 'タイム',
                '単勝', '人気',
                '馬平均レーティング', '騎手平均レーティング']:
        for n in range(1, 6):
            df[f'{n}走前{col}'] = group[col].shift(n)

    # 平均連帯距離(過去)
    df['連帯対象'] = (df['着順'] <= 2).astype(int)
    df['連帯距離累積'] = group.apply(lambda g: g['距離'].where(g['連帯対象'] == 1, 0).cumsum().shift(1)).values
    df['連帯回数累積'] = group['連帯対象'].apply(lambda s: s.cumsum().shift(1)).values
    df['平均連帯距離'] = df['連帯距離累積'] / df['連帯回数累積']
    df.drop(['連帯対象', '連帯距離累積', '連帯回数累積'], axis=1, inplace=True)

    # 馬・騎手ごとの平均回収率(過去)
    df['馬平均回収率'] = calculate_return_rate(df, 'horse_id')
    df['騎手平均回収率'] = calculate_return_rate(df, '騎手')

    return df


# -------------------------------
# 脚質計算
# -------------------------------
def calc_running_style(df):
    """脚質を判断する関数"""
    df = df.copy()
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    df = df.sort_values(['horse_id', 'date', 'race_id']).reset_index(drop=True)

    print('脚質列を追加')

    df['平均4C通過順位'] = df.groupby('horse_id')['4C通過順位'].transform(lambda s: s.shift(1).expanding().mean())
    conditions = [
        (df['平均4C通過順位'] <= 3),
        (df['平均4C通過順位'] > 3) & (df['平均4C通過順位'] <= 8),
        (df['平均4C通過順位'] > 8) & (df['平均4C通過順位'] <= 13),
        (df['平均4C通過順位'] > 13)
    ]
    choices = ['逃げ', '先行', '差し', '追込']
    df['脚質'] = np.select(conditions, choices, default='先行')
    df.drop(columns=['平均4C通過順位'], inplace=True)
    return df


# -------------------------------
# ペースの計算
# -------------------------------
def calc_pace(df):
    """ペースを判断する関数"""
    df = df.copy()
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    df = df.sort_values(['horse_id', 'date', 'race_id']).reset_index(drop=True)

    print('ペース列を追加')

    def get_pace(row):
        d = row['距離']
        p = row['前半ペース']
        if pd.isna(p):
            return np.nan
        if ((d // 100) % 2) != 0:  
            if p <= 29.5:
                return 'ハイ'
            elif p >= 31:
                return 'スロー'
            else:
                return 'ミドル'
        else:
            if p <= 34.5:
                return 'ハイ'
            elif p >= 37:
                return 'スロー'
            else:
                return 'ミドル'

    df['ペース'] = df.apply(get_pace, axis=1)
    return df


# -------------------------------
# 過去の対戦成績を作る
# -------------------------------
def create_distance_category(distance):
    if 1000 <= distance <= 1400:
        return "短距離"
    elif 1401 <= distance <= 1799:
        return "マイル"
    elif 1800 <= distance <= 2200:
        return "中距離"
    elif 2201 <= distance <= 2600:
        return "クラシック"
    else:
        return "長距離"

def create_competitor_features_horse_id_and_bangou(df):
    """
    対戦履歴は horse_id で管理しつつ、特徴量の列は「対馬番X_...」形式にする例。
    さらに「全距離カテゴリを参照」するように修正し、
    中距離レース中でも短距離・クラシック等の過去対戦情報を取り出せるようにする。
    """
    df = df.copy()
    df = df.sort_values(["date", "race_id", "着順"]).reset_index(drop=True)

    print('対戦戦績を追加')

    # 対戦履歴 (horse_id 同士 × 距離カテゴリ) を蓄積
    competitor_dist_history = {}

    # 馬番最大18想定
    max_bangou = 18
    dist_cats = ["短距離","マイル","中距離","クラシック","長距離"]

    # ---- 多数列をまとめて作る (フラグメンテーション回避) ----
    new_cols_data = {}
    for bn in range(1, max_bangou+1):
        for dc in dist_cats:
            new_cols_data[f"対馬番{bn}_{dc}勝ち数"] = np.zeros(len(df), dtype=int)
            new_cols_data[f"対馬番{bn}_{dc}負け数"] = np.zeros(len(df), dtype=int)
    new_cols_df = pd.DataFrame(new_cols_data, index=df.index)
    df = pd.concat([df, new_cols_df], axis=1)

    # レースごとに処理
    for race_id, group in tqdm(df.groupby("race_id", sort=False)):
        # 当該レースのカテゴリー(更新には使うが、読み出し時には限定しない)
        if group["距離"].nunique() == 1:
            this_cat = create_distance_category(group["距離"].iloc[0])
        else:
            this_cat = "短距離"

        # レースに出走する馬 [(df_idx, horse_id, 馬番, 着順), ...]
        group_info = group[["horse_id","馬番","着順"]].reset_index()
        group_list = list(group_info.to_records(index=False))

        # ========== レース前: 過去対戦成績を参照(全カテゴリ) ==========
        for (df_idx, hA, bnA, rankA) in group_list:
            for (df_idx2, hB, bnB, rankB) in group_list:
                if hA == hB:
                    continue
                # ★ ここが変更点: dist_cats 全部を読む
                for dc in dist_cats:
                    keyAB = (hA, hB, dc)
                    if keyAB in competitor_dist_history:
                        faced_count, wins_of_A = competitor_dist_history[keyAB]
                        df.at[df_idx, f"対馬番{bnB}_{dc}勝ち数"] = wins_of_A
                        df.at[df_idx, f"対馬番{bnB}_{dc}負け数"] = faced_count - wins_of_A
                    else:
                        # 対戦なし → 0のまま
                        pass

        # ========== レース後: 今回の結果で this_cat の辞書を更新 ==========
        #   (短距離,マイル...すべてのカテゴリを更新してしまうと誤記録になるため)
        for i in range(len(group_list)):
            for j in range(i+1, len(group_list)):
                df_idx_i, hA, bnA, rankA = group_list[i]
                df_idx_j, hB, bnB, rankB = group_list[j]

                # A→B
                keyAB = (hA, hB, this_cat)
                if keyAB not in competitor_dist_history:
                    competitor_dist_history[keyAB] = [0, 0]
                competitor_dist_history[keyAB][0] += 1
                if rankA < rankB:
                    competitor_dist_history[keyAB][1] += 1

                # B→A
                keyBA = (hB, hA, this_cat)
                if keyBA not in competitor_dist_history:
                    competitor_dist_history[keyBA] = [0, 0]
                competitor_dist_history[keyBA][0] += 1
                if rankB < rankA:
                    competitor_dist_history[keyBA][1] += 1

    return df


# -------------------------------
# 欠損値処理
# -------------------------------
def edit_missing(df):
    """欠損値を処理する関数
    """
    df = df.copy()
    # 重複行削除
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # ソート
    df = df.sort_values(['horse_id', 'date', 'race_id']).reset_index(drop=True)

    print('欠損値を処理します。')
    # 欠損処理例
    # カテゴリ列: レース名
    df['レース名_missing'] = df['レース名'].isnull().astype(int)
    df['レース名'] = df['レース名'].fillna('Unknown')

    # 数値列: 出走間隔（過去走なしの場合0、フラグ付与）
    df['出走間隔_missing'] = df['出走間隔'].isnull().astype(int)
    df['出走間隔'] = df['出走間隔'].fillna(0)

    # 累計獲得賞金（デビュー前0で、欠損フラグ）
    df['累計獲得賞金_missing'] = df['累計獲得賞金'].isnull().astype(int)
    df['累計獲得賞金'] = df['累計獲得賞金'].fillna(0)

    # horse_ability（0埋め＋フラグ）
    df['horse_ability_missing'] = df['horse_ability'].isnull().astype(int)
    df['horse_ability'] = df['horse_ability'].fillna(0)

    # horse_ability_mean（0埋め＋フラグ）
    df['horse_ability_mean_missing'] = df['horse_ability_mean'].isnull().astype(int)
    df['horse_ability_mean'] = df['horse_ability_mean'].fillna(0)

    # 過去走情報系（例：1走前着順、2走前着順など）
    for col in ['1走前着順','2走前着順','3走前着順','4走前着順','5走前着順']:
        df[col+'_missing'] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(0)
    for col in ['1走前距離','2走前距離','3走前距離','4走前距離','5走前距離']:
        df[col+'_missing'] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(0)
    for col in ['1走前グレード','2走前グレード','3走前グレード','4走前グレード','5走前グレード']:
        df[col+'_missing'] = df[col].isnull().astype(int)
        df[col] = df[col].fillna('NoRace')
    for col in ['1走前コース種類','2走前コース種類','3走前コース種類','4走前コース種類','5走前コース種類']:
        df[col+'_missing'] = df[col].isnull().astype(int)
        df[col] = df[col].fillna('NoRace')
    for col in ['1走前着差','2走前着差','3走前着差','4走前着差','5走前着差']:
        df[col+'_missing'] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(0)
    for col in ['1走前上がり3F順位','2走前上がり3F順位','3走前上がり3F順位','4走前上がり3F順位','5走前上がり3F順位']:
        df[col+'_missing'] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(0)
    for col in ['1走前horse_ability','2走前horse_ability','3走前horse_ability','4走前horse_ability','5走前horse_ability']:
        df[col+'_missing'] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(0)
    for col in ['1走前タイム','2走前タイム','3走前タイム','4走前タイム','5走前タイム']:
        df[col+'_missing'] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(0)
    for col in ['1走前1着タイム差','2走前1着タイム差','3走前1着タイム差','4走前1着タイム差','5走前1着タイム差']:
        df[col+'_missing'] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(0)
    for col in ['1走前1C通過順位','2走前1C通過順位','3走前1C通過順位','4走前1C通過順位','5走前1C通過順位']:
        df[col+'_missing'] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(0)
    for col in ['1走前2C通過順位','2走前2C通過順位','3走前2C通過順位','4走前2C通過順位','5走前2C通過順位']:
        df[col+'_missing'] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(0)
    for col in ['1走前3C通過順位','2走前3C通過順位','3走前3C通過順位','4走前3C通過順位','5走前3C通過順位']:
        df[col+'_missing'] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(0)
    for col in ['1走前4C通過順位','2走前4C通過順位','3走前4C通過順位','4走前4C通過順位','5走前4C通過順位']:
        df[col+'_missing'] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(0)
    for col in ['1C通過順位','2C通過順位','3C通過順位','4C通過順位']:
        df[col+'_missing'] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(0)
    for col in ['1走前単勝','2走前単勝','3走前単勝','4走前単勝','5走前単勝']:
        df[col+'_missing'] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(0)
    for col in ['1走前人気','2走前人気','3走前人気','4走前人気','5走前人気']:
        df[col+'_missing'] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(0)
    for col in ['1走前馬平均レーティング','2走前馬平均レーティング','3走前馬平均レーティング','4走前馬平均レーティング','5走前馬平均レーティング']:
        df[col+'_missing'] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(0)
    for col in ['1走前騎手平均レーティング','2走前騎手平均レーティング','3走前騎手平均レーティング','4走前騎手平均レーティング','5走前騎手平均レーティング']:
        df[col+'_missing'] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(0)

    # 平均連帯距離（0埋め＋フラグ）
    df['平均連帯距離_missing'] = df['平均連帯距離'].isnull().astype(int)
    df['平均連帯距離'] = df['平均連帯距離'].fillna(0)

    # 馬平均回収率（0埋め＋フラグ）
    df['馬平均回収率_missing'] = df['馬平均回収率'].isnull().astype(int)
    df['馬平均回収率'] = df['馬平均回収率'].fillna(0)

    # 騎手平均回収率（0埋め＋フラグ）
    df['騎手平均回収率_missing'] = df['騎手平均回収率'].isnull().astype(int)
    df['騎手平均回収率'] = df['騎手平均回収率'].fillna(0)

    # 上がり3F順位（0埋め＋フラグ）
    df['上がり3F順位_missing'] = df['上がり3F順位'].isnull().astype(int)
    df['上がり3F順位'] = df['上がり3F順位'].fillna(0)

    # 過去平均上がり3F順位（0埋め＋フラグ）
    df['過去平均上がり3F順位_missing'] = df['過去平均上がり3F順位'].isnull().astype(int)
    df['過去平均上がり3F順位'] = df['過去平均上がり3F順位'].fillna(0)
    
    # タイム順位（0埋め＋フラグ）
    df['タイム_missing'] = df['タイム'].isnull().astype(int)
    df['タイム'] = df['タイム'].fillna(0)

    print("処理後DF：", df.shape)
    return df



# 処理を行う関数
def run_feature():
    df = pd.read_csv(DF_PATH, encoding="utf_8_sig")


    df = df.loc[df['コース種類'] != '障害']

    # 日付列の作成
    df = process_date(df)
    # レーティング計算+列追加
    df = create_trueskill_ratings(df)
    # レースごとのレーティング平均値を追加
    df = calc_rating_mean(df)
    # テン乗りフラグの付与
    df = add_jockey_change_features(df)
    # 乗り替わり騎手のレーティング差列を作成
    df = create_jockey_rating_change_feature(df)
    # 追加したカラムは要らないという場合は削除してもOK
    # df.drop(["prev_jockey_id", "prev_jockey_mu"], axis=1, inplace=True)
    # 過去成績計算
    df = calc_career_statics(df, "競走馬")
    df = calc_career_statics(df, "騎手")
    # 競馬場情報付与
    df = concat_race_info(df)
    # 適正を付与し、計算
    df = create_flag_features_and_update(df)
    # 収得賞金の計算
    df = calucrate_add_money(df)
    # 上がり3F関連の特徴量計算
    df = calculate_3f_features(df)
    # 能力指数の計算
    df = calc_ability(df)
    df = calculate_horse_ability_mean(df)
    # 1~5レース前までの特徴量追加
    df = add_past_features_and_stats(df)
    # 脚質追加
    df = calc_running_style(df)
    # ペースを判断する関数
    df = calc_pace(df)
    # 過去の対戦成績を計算
    df = create_competitor_features_horse_id_and_bangou(df)
    # 欠損値処理
    df = edit_missing(df)

    # horse_abilityのヒストグラム表示
    # horse_abilityとhorse_ability_meanのヒストグラムを重ねて表示
    plt.hist(df.loc[df['horse_ability'] <= 30]['horse_ability'].dropna(), bins=50, alpha=0.7, label='Horse Ability')
    plt.hist(df.loc[df['horse_ability_mean'] <= 30]['horse_ability_mean'].dropna(), bins=50, alpha=0.7, label='Horse Ability Mean')

    plt.title('Distribution of Horse Ability and Horse Ability Mean')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # display(df.sort_values(['date']).reset_index(drop=True).tail(5))
    display(df.loc[df['馬名'] == 'イクイノックス']) # 確認用

    print(df.isnull().sum())

    df.to_csv(OUTPUT_PATH, encoding="utf_8_sig")