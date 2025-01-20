import os
import pickle
from IPython.display import display
from tqdm import tqdm

import pandas as pd
import numpy as np

from joblib import Parallel, delayed

import trueskill

import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder


ROOT_PATH = "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/153_競馬_v3/data"

DF_PATH = os.path.join(ROOT_PATH, "01_processed/60_combined/race_result_info_performance_pace_with_future_df.csv")

# コース情報
COURSE_PATH = os.path.join(ROOT_PATH, "03_input/course.csv")
# 血統情報
PEDIGREE_ID_PATH = ROOT_PATH + "/01_processed/40_pedigree/pedigree_horse_id.csv"


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

    # レースごとに馬レーティング_beforeの平均を取る
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
    # 変わっていなければ 0、変わっていれば (今の騎手レーティング - 前走の騎手レーティング)
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
                # 今回のジョッキーの "騎手レーティング_before" から、前回ジョッキーの "prev_騎手レーティング" を引いた差をとる
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

    def create_flag_features(df_):
        df_ = df_.copy()

        # -------------------------------
        # 枠によるフラグ作成
        # -------------------------------
        df_["内枠"] = (((1 <= df_["馬番"]) & (df_["馬番"] <= 6))).astype(int)
        df_["中枠"] = (((7 <= df_["馬番"]) & (df_["馬番"] <= 12))).astype(int)
        df_["外枠"] = (((13 <= df_["馬番"]) & (df_["馬番"] <= 18))).astype(int)

        # -------------------------------
        # 距離によるフラグ作成
        # -------------------------------
        df_["短距離"] = (((1000 <= df_["距離"]) & (df_["距離"] <= 1400))).astype(int)
        df_["マイル"] = (((1401 <= df_["距離"]) & (df_["距離"] <= 1799))).astype(int)
        df_["中距離"] = (((1800 <= df_["距離"]) & (df_["距離"] <= 2200))).astype(int)
        df_["クラシック"] = (((2201 <= df_["距離"]) & (df_["距離"] <= 2600))).astype(int)
        df_["長距離"] = (((2601 <= df_["距離"]) & (df_["距離"] <= 4000))).astype(int)

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

        return df_
    
    def update_ratings_for_single_flag(df, flag, env):
        """
        同一レース内の全馬を TrueSkill で対戦させる。
        ただし、実際にレーティングを更新するのは "flag == 1" の馬だけ。
        """
        # 辞書: horse_id → Rating
        horse_ratings = {}
        jockey_ratings = {}

        n = len(df)
        horse_mu_array = np.zeros(n, dtype=np.float32)
        horse_sigma_array = np.zeros(n, dtype=np.float32)
        jockey_mu_array = np.zeros(n, dtype=np.float32)
        jockey_sigma_array = np.zeros(n, dtype=np.float32)

        race_groups = df.groupby("race_id", sort=False)
        race_ids = list(race_groups.groups.keys())

        for race_id in race_ids:
            idxs = race_groups.groups[race_id]
            idxs_sorted = idxs.sort_values()

            # -- レース前のレーティングを df に保存 --
            for i in idxs_sorted:
                h_id = df.at[i, "horse_id"]
                j_id = df.at[i, "jockey_id"]
                if h_id not in horse_ratings:
                    horse_ratings[h_id] = env.create_rating()
                if j_id not in jockey_ratings:
                    jockey_ratings[j_id] = env.create_rating()

                horse_mu_array[i]   = horse_ratings[h_id].mu
                horse_sigma_array[i] = horse_ratings[h_id].sigma
                jockey_mu_array[i]  = jockey_ratings[j_id].mu
                jockey_sigma_array[i] = jockey_ratings[j_id].sigma

            # -- レース参加馬 (全馬) を TrueSkill の対戦チームとして作成 --
            participants_horse  = []
            participants_jockey = []
            ranks_horse  = []
            ranks_jockey = []

            idx_list_horse  = []
            idx_list_jockey = []

            for i2 in idxs_sorted:
                rank = df.at[i2, "着順"]
                if pd.isna(rank):
                    # 着順欠損ならスキップ
                    participants_horse.append(None)  # ダミー
                    ranks_horse.append(None)
                    idx_list_horse.append(i2)
                    participants_jockey.append(None)
                    ranks_jockey.append(None)
                    idx_list_jockey.append(i2)
                    continue

                rank_int = int(rank) - 1
                h_id = df.at[i2, "horse_id"]
                j_id = df.at[i2, "jockey_id"]

                participants_horse.append([horse_ratings[h_id]])
                ranks_horse.append(rank_int)
                idx_list_horse.append(i2)

                participants_jockey.append([jockey_ratings[j_id]])
                ranks_jockey.append(rank_int)
                idx_list_jockey.append(i2)

            # -- Horse の対戦結果で更新 (全馬を含む) --
            #    → 更新結果を一時的に updated_ratings_horse に保持
            if len(participants_horse) >= 2:
                # TrueSkillでは、リスト内にNoneがあるとエラーになるので除去
                valid_entries = [(team, rnk, idxv) for (team, rnk, idxv)
                                in zip(participants_horse, ranks_horse, idx_list_horse)
                                if team is not None and rnk is not None]
                if len(valid_entries) >= 2:
                    teams_horse, ranks_horse_, idxs_horse_ = zip(*valid_entries)
                    updated_ratings_horse = env.rate(teams_horse, ranks=ranks_horse_)

                    # 「フラグがある馬だけ」更新を反映
                    for x, row_idx in enumerate(idxs_horse_):
                        h_id = df.at[row_idx, "horse_id"]
                        if df.at[row_idx, flag] == 1:
                            # フラグ馬のみ更新を適用
                            horse_ratings[h_id] = updated_ratings_horse[x][0]

            # -- Jockey の対戦結果で更新 (全馬を含む) --
            if len(participants_jockey) >= 2:
                valid_entries_j = [(team, rnk, idxv) for (team, rnk, idxv)
                                in zip(participants_jockey, ranks_jockey, idx_list_jockey)
                                if team is not None and rnk is not None]
                if len(valid_entries_j) >= 2:
                    teams_jockey, ranks_jockey_, idxs_jockey_ = zip(*valid_entries_j)
                    updated_ratings_jockey = env.rate(teams_jockey, ranks=ranks_jockey_)

                    for x, row_idx in enumerate(idxs_jockey_):
                        j_id = df.at[row_idx, "jockey_id"]
                        if df.at[row_idx, flag] == 1:
                            jockey_ratings[j_id] = updated_ratings_jockey[x][0]

        # 結果列として df_result をまとめて返す
        col_mu_horse     = f"競走馬レーティング_{flag}"
        col_sigma_horse  = f"競走馬レーティング_sigma_{flag}"
        col_mu_jockey    = f"騎手レーティング_{flag}"
        col_sigma_jockey = f"騎手レーティング_sigma_{flag}"

        df_result = pd.DataFrame({
            col_mu_horse: horse_mu_array,
            col_sigma_horse: horse_sigma_array,
            col_mu_jockey: jockey_mu_array,
            col_sigma_jockey: jockey_sigma_array
        }, index=df.index)

        return df_result

    def create_flag_features_and_update_parallel(df):
        df = df.copy()
        df = df.drop_duplicates(subset=["race_id", "馬番"])
        df = df.sort_values(["date", "race_id", "着順"]).reset_index(drop=True)

        df = create_flag_features(df)

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

        env = trueskill.TrueSkill(draw_probability=0.0)

        results = Parallel(n_jobs=-1)(
            delayed(update_ratings_for_single_flag)(
                df[["date", "race_id", "着順", "horse_id", "jockey_id", f]],
                f,
                env
            )
            for f in flag_cols
        )

        df_final = df.copy()
        for res_df in results:
            df_final = df_final.join(res_df)

        return df_final
    
    df = create_flag_features_and_update_parallel(df)

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
# 血統特徴量の処理
# -------------------------------
def create_pedigree_feature(df):
    df = df.copy()
    # 重複行削除
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # ソート
    df = df.sort_values(['horse_id', 'date', 'race_id']).reset_index(drop=True)
    
    print('血統特徴量を処理します。')

    pedigree_df = pd.read_csv(PEDIGREE_ID_PATH, encoding='utf-8-sig')

    # 表記揺れ処理
    import re
    # 例: 「サンデーサイレンス_Sunday_Silence(米)」 -> 「サンデーサイレンス」
    #     「ノーザンテースト_Northern_Taste(加)」 -> 「ノーザンテースト」
    #     ただし、「Kingmambo_(米)」などは先頭がカナじゃないからスルーされる想定。
    #     （もしアルファベットも一律で分割したいなら、別の正規表現パターンを使う。）

    # カナ文字（全角カタカナや長音符）を定義:
    #   [\u30A0-\u30FFー]+
    # 「^ ～ $」で行頭～行末を縛る。
    # グループ1に「カナ部分」をキャプチャし、そこに "_何か" が続くかどうかをチェックする。
    pattern = re.compile(r'^([\u30A0-\u30FFー]+)_.+$')

    def unify_kana_underscore(name: str):
        """
        先頭から「全角カタカナ(＋長音符)の並び → '_' → 何か...」という形式にマッチしたら、
        カナ部分だけを返す。
        それ以外はそのまま返す。
        """
        if pd.isna(name):
            return name
        m = pattern.match(name)
        if m:
            return m.group(1)  # カナだけ取り出し
        else:
            return name

    # (2) カナ+_ を検出して分割
    ped_cols = [c for c in pedigree_df.columns if 'pedigree' in c]

    for col in ped_cols:
        pedigree_df[col] = pedigree_df[col].apply(unify_kana_underscore)

    ###############################
    # (1) 血統名の登場頻度ランキングを作る
    ###############################
    ped_cols = [col for col in pedigree_df.columns if 'pedigree' in col]

    # 全血統名を 1 次元にまとめる (NaN除去)
    all_ancestors = pedigree_df[ped_cols].values.flatten()
    s = pd.Series(all_ancestors).dropna()

    # 頻度カウント
    counts = s.value_counts()

    # 上位50件だけ抜き出し (実際にはご自由に数を設定)
    print("=== 上位50血統のランキング ===")
    print(counts.head(50))

    ###############################
    # (4) 主要血統を選ぶ
    ###############################
    major_ancestors = counts.index[:50].tolist()
    print("主要血統候補:", major_ancestors)

    ###############################
    # (5) インブリード度合いの計算
    ###############################
    # 5代血統(32列)を想定して、世代ごとの血量を割り当てる例。
    # pedigree0 ~ pedigree31 の各列にどの程度の血が含まれるか。
    # ざっくり「世代が増えるごとに1/2, 1/4, 1/8, 1/16, 1/32」と決める。
    # 世代計算は手軽に: 1代目=ped0~1, 2代目=ped2~5, 3代目=ped6~13, 4代目=ped14~29, 5代目=ped30~31 など。

    # 下記は簡単な方法で indices の範囲から決め打ち
    gen_fraction_map = {}
    # 1代目 (pedigree0,1) → 1/2
    for i in [0,1]:
        gen_fraction_map[i] = 1/2
    # 2代目 (pedigree2~5) → 1/4
    for i in range(2,6):
        gen_fraction_map[i] = 1/4
    # 3代目 (pedigree6~13) → 1/8
    for i in range(6,14):
        gen_fraction_map[i] = 1/8
    # 4代目 (pedigree14~29) → 1/16
    for i in range(14,30):
        gen_fraction_map[i] = 1/16
    # 5代目 (pedigree30,61) → 1/32
    for i in range(30,62):
        gen_fraction_map[i] = 1/32

    # まだ ped列が31超ある場合は適宜延長。


    ###############################
    # (6) データフレームに新たな列を追加:
    #     「主要血統X_登場回数」 「主要血統X_インブリード血量」
    ###############################

    # まず、既存の列に馬IDなどがあるとして、これに計算結果を join したいから
    # 新しいDFを作っておくといい
    df_result = pedigree_df.copy()

    # 計算用の関数を用意
    def calc_ancestor_inbreed(row, major_name):
        """
        row: 1レコード (pedigree0..31 を含む)
        major_name: 主要血統の標準化済名 (例: "サンデーサイレンス")
        
        戻り値:
        (登場回数, 血量合計)
        """
        count_ = 0
        coeff_sum = 0.0
        for i in range(62):  # 5代と仮定
            col_name = f"pedigree_{i}"
            if pd.isna(row[col_name]):
                continue
            if row[col_name] == major_name:
                count_ += 1
                coeff_sum += gen_fraction_map.get(i, 0.0)
        return count_, coeff_sum

    # 各主要血統について列を作る
    for anc_name in major_ancestors:
        # 列名 (例) "サンデーサイレンス_登場回数", "サンデーサイレンス_インブリード"
        col_count = f"{anc_name}_登場回数"
        col_inbr  = f"{anc_name}_インブリード血量"

        # applyで各行に対して計算
        tmp = df_result.apply(lambda row: calc_ancestor_inbreed(row, anc_name), axis=1)
        # 戻りが (count, coeff_sum) のタプルだから、それぞれ展開
        df_result[col_count] = tmp.apply(lambda x: x[0])
        df_result[col_inbr]  = tmp.apply(lambda x: x[1])

    # これで各主要血統の「登場回数」「血量」が列として追加される

    ###############################
    # (7) 確認
    ###############################
    print("サンプル出力:")
    display(df_result.head())

    df = pd.merge(df, df_result, on = "horse_id", how = "left")

    def get_sire_stats(all_races_df):
        print('父の戦績を集計')
        df = all_races_df.copy()
        
        # 馬ごと(行)に勝ちフラグや連対フラグを付ける
        df['win_flag'] = (df['着順'] == 1).astype(int)
        df['rentaiflag'] = (df['着順'] <= 2).astype(int)

        # 芝・ダートのフラグ
        df['is_turf'] = (df['コース種類'] == '芝').astype(int)
        df['is_dirt'] = (df['コース種類'] == 'ダート').astype(int)

        # 父馬単位で groupby
        grouped = df.groupby(['pedigree_0', 'コース種類'], dropna=True)

        # 集計: 出走回数, 勝利数, 連対数
        stat = grouped.agg(
            starts=('race_id', 'count'),
            wins=('win_flag', 'sum'),
            rentai=('rentaiflag', 'sum')
        ).reset_index()

        # pivot して "芝" と "ダート" で列を分ける (父馬IDをインデックス)
        stat_pivot = stat.pivot_table(
            index='pedigree_0',
            columns='コース種類',
            values=['starts','wins','rentai'],
            fill_value=0
        )
        # マルチインデックスをフラットに
        stat_pivot.columns = [f'{col[0]}_{col[1]}' for col in stat_pivot.columns]

        # "芝" "ダート" で勝率/連対率を計算
        # 例: 'wins_芝' / 'starts_芝'
        stat_pivot['win_rate_芝']    = stat_pivot['wins_芝'] / stat_pivot['starts_芝'].replace(0, np.nan)
        stat_pivot['rentai_rate_芝'] = stat_pivot['rentai_芝'] / stat_pivot['starts_芝'].replace(0, np.nan)
        stat_pivot['win_rate_ダート']    = stat_pivot['wins_ダート'] / stat_pivot['starts_ダート'].replace(0, np.nan)
        stat_pivot['rentai_rate_ダート'] = stat_pivot['rentai_ダート'] / stat_pivot['starts_ダート'].replace(0, np.nan)

        # 欠損を0埋め
        stat_pivot = stat_pivot.fillna(0)

        # インデックス名を列名に
        stat_pivot = stat_pivot.reset_index().rename(columns={'pedigree_0': 'father_name'})
        
        return stat_pivot

    sire_stats_df = get_sire_stats(df)

    # これで father_id(=pedigree_0) をキーに、
    # [ 'starts_芝', 'wins_芝', 'rentai_芝', 'win_rate_芝', 'rentai_rate_芝', 
    #   'starts_ダート', ...,
    #   'win_rate_ダート', 'rentai_rate_ダート', ... ] 
    # といった列を持つデータフレームが得られる。

    def merge_sire_stats_features(df, sire_stats_df):
        df = df.copy()
        # 'pedigree_0' 列に父馬の名前(ID)が入っている想定。
        # もし既に "father_id" みたいな列を作ってるならそのままでもOK。
        df['father_name'] = df['pedigree_0']  # 例

        # sire_stats_df のキーは 'father_id'
        df_merged = pd.merge(
            df,
            sire_stats_df,
            on='father_name',
            how='left'
        )
        df_merged = df_merged.drop(columns = 'father_name')

        return df_merged
    
    df = merge_sire_stats_features(df, sire_stats_df)

    def get_sire_distance_stats(all_races_df):
        # 種牡馬の平均連対距離を集計
        df = all_races_df.copy()
        df['win_flag'] = (df['着順'] == 1).astype(int)

        # 父ID × (勝利したレースの距離) の平均を取る
        # → "平均勝利距離" が推定距離適性の一つの指標になる
        #   (もちろん「勝ち馬しか見てないからバイアスある」など注意は必要)

        # 勝ってないレースは距離=NaNにして除外する方法
        df.loc[df['win_flag'] == 0, '距離'] = np.nan

        sire_dist = df.groupby('pedigree_0')['距離'].mean().reset_index()
        sire_dist.columns = ['father_id','mean_win_dist']

        return sire_dist
    
    sire_dist = get_sire_distance_stats(df)
    df = pd.merge(df, sire_dist, on='father_id', how='left')

    df = df.drop(columns = ['win_flag', 'rentaiflag', 'is_turf', 'is_dirt'])

    print(f'処理後のデータフレーム：{df.shape}')
    display(df.head(1))

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

    # 最後に重複削除
    # 重複行削除
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # ソート
    df = df.sort_values(['date', 'race_id', '馬番']).reset_index(drop=True)


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
    # df.drop(["prev_jockey_id", "prev_騎手レーティング"], axis=1, inplace=True)
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
    display(df.loc[(df['race_id']==202405021212)])

    print(df.isnull().sum())

    df.to_csv(OUTPUT_PATH, encoding="utf_8_sig")