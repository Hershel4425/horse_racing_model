import polars as pl
import pandas as pd
from glicko2 import glicko2
import copy
import pickle
import numpy as np
import gensim
from IPython.display import display
from sklearn.decomposition import PCA
from xfeat import SelectCategorical, LabelEncoder, Pipeline
import time
import gc

import logging

# ログの設定
logging.basicConfig(filename="processing.log", level=logging.ERROR)


# 実行時間を記録するためのディクショナリ
execution_times = {}


# 実行時間を計測するデコレータ
def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_times[func.__name__] = (end_time - start_time) / 60  # 分単位で記録
        return result

    return wrapper


def save_execution_times_to_csv():
    """関数の実行時間をCSVファイルに保存する"""
    execution_times_df = pd.DataFrame(
        list(execution_times.items()),
        columns=["Function Name", "Execution Time (Minutes)"],
    )
    path = "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2/20_data_processing/関数ごと実行時間/execution_times.csv"
    execution_times_df.to_csv(path, index=False)


@measure_execution_time
def process_date(df):
    """日付を処理する関数
    Args:
        df (dataframe): データフレーム
    Returns:
        df (dataframe): 日付列を処理したデータフレーム
    """
    # 年月日時分をintにする,nanはそれぞれ対処する
    df = df.with_columns(
        pl.col("年").fill_null(2999).cast(pl.Int32),
        pl.col("月").fill_null(12).cast(pl.Int32),
        pl.col("日").fill_null(31).cast(pl.Int32),
        pl.col("時").fill_null(23).cast(pl.Int32),
        pl.col("分").fill_null(59).cast(pl.Int32),
    )
    # 日付をdatetime型に変換
    df = df.with_column(
        pl.concat_str(
            [
                pl.col("年").cast(pl.Utf8),
                pl.lit("-"),
                pl.col("月").cast(pl.Utf8),
                pl.lit("-"),
                pl.col("日").cast(pl.Utf8),
                pl.lit(" "),
                pl.col("時").cast(pl.Utf8),
                pl.lit(":"),
                pl.col("分").cast(pl.Utf8),
            ],
            sep="",
        ).str.strptime(pl.Datetime, fmt="%Y-%m-%d %H:%M", strict=False).alias("date")
    )

    # 不要列を削除
    df = df.drop(["年", "月", "日", "時", "分"])

    return df


@measure_execution_time
def calc_cumulative_stats(df, who, whoid, grade_list, dist_list, racetrack_list):
    """通算成績を計算する関数のうち、通算試合数や勝利数を計算する関数

    Args:
        df (_type_): データフレーム
        who (str): 騎手か競走馬か
        whoid (str): jockey_idかhorse_idか
        grade_list (_type_): _description_
        dist_list (_type_): _description_
        racetrack_list (_type_): 競馬場名のリスト

    Returns:
        _type_: _description_
    """
    import warnings
    from pandas.core.common import SettingWithCopyWarning

    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
    # 計算する列名を設定
    all_columns = []

    # 芝ダート振り分け
    for surface in ["芝", "ダート"]:
        all_columns.extend(
            [
                f"{who}{surface}{suffix}"
                for suffix in ["通算試合数", "通算勝利数", "通算複勝数"]
            ]
        )

        for grade in grade_list:
            columns = [
                f"{who}{surface}{grade}{suffix}"
                for suffix in ["通算試合数", "通算勝利数", "通算複勝数"]
            ]
            all_columns.extend(columns)
        all_columns.extend(
            [
                f"{who}{surface}重賞{suffix}"
                for suffix in ["通算試合数", "通算勝利数", "通算複勝数"]
            ]
        )

        for dist in dist_list:
            columns = [
                f"{who}{surface}{dist}{suffix}"
                for suffix in ["通算試合数", "通算勝利数", "通算複勝数"]
            ]
            all_columns.extend(columns)

        for racetrack in racetrack_list:
            columns = [
                f"{who}{surface}{racetrack}{suffix}"
                for suffix in ["通算試合数", "通算勝利数", "通算複勝数"]
            ]
            all_columns.extend(columns)

        if who == "騎手":
            for ninki in range(1, 19):
                columns = [
                    f"{who}{surface}{ninki}{suffix}"
                    for suffix in ["人気通算試合数", "人気通算勝利数", "人気通算複勝数"]
                ]
                all_columns.extend(columns)

    # 通算成績を計算する
    # 1. IDごとに並び替える
    df = df.sort(by=[whoid, "date"])

    # 2. shiftで一行ずらす
    # 各グループ内で1行ずつシフトし、NaNを後ろの値で埋める操作をtransformを用いて適用
    df = df.with_columns(
        [
            pl.col(col).shift(1).over(whoid).fill_null(pl.col(col)).cast(pl.Float64).alias(col)
            for col in all_columns
        ]
    )

    # 3. 全データフレームに対してcumsumをする
    df = df.with_columns(
        [pl.col(col).fill_null(0).cumsum().over(whoid).alias(col) for col in all_columns]
    )

    # 4. 各IDの各列の最小値を引く
    for col in all_columns:
        min_vals = df.groupby(whoid).agg(pl.min(col))
        df = df.join(min_vals, on=whoid, how="left")
        df = df.with_column((pl.col(col) - pl.col(f"{col}_right")).alias(col))
        df = df.drop(f"{col}_right")

    # 5. 単年成績を計算
    df = df.with_column(pl.col("date").dt.year().cast(pl.Int32).alias("年"))
    for col in all_columns:
        temp_df = df.select(["年", whoid, col]).collect()  # 処理高速化のために一時的にデータフレームを作成
        grouped = temp_df.groupby(["年", whoid]).agg(pl.min(col).cast(pl.Float64))
        df = df.join(grouped, on=["年", whoid], how="left")
        df = df.with_column(
            (pl.col(col) - pl.col(f"{col}_right")).alias(f"単年{col}")
        )
        df = df.drop(f"{col}_right")

    df = df.drop("年")

    return df


@measure_execution_time
def calc_career_statics(df, who):
    """通算成績を計算する関数

    Args:
        df (_type_): _description_
        who (str): 騎手か競走馬か

    Returns:
        _type_: _description_
    """
    # いらない警告を非表示にする
    import warnings

    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    # 通算試合数や勝利数を計算する
    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.unique(subset=["race_id", "馬番"])
    # データをソート
    df = df.sort(["date", "馬番"])

    print("------通算成績処理開始------")
    print("処理前DF：", df.shape)
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
    ]
    ext_df = df.select(ext_columns).collect()
    grade_list = [
        "新馬",
        "未勝利",
        "1勝クラス",
        "2勝クラス",
        "3勝クラス",
        "オープン",
        "G3",
        "G2",
        "G1",
    ]
    dist_list = ["短距離", "マイル", "中距離", "クラシック", "長距離"]
    dist_dict = {
        "短距離": [1000, 1300],
        "マイル": [1301, 1899],
        "中距離": [1900, 2100],
        "クラシック": [2101, 2700],
        "長距離": [2700, 4000],
    }
    racetrack_list = [
        "札幌",
        "函館",
        "福島",
        "新潟",
        "東京",
        "中山",
        "中京",
        "京都",
        "阪神",
        "小倉",
    ]

    # 芝ダート振り分け
    for surface in ["芝", "ダート"]:
        surface_mask = ext_df["コース種類"] == surface

        # まずは条件を満たすレースにフラグ付け
        ext_df.loc[surface_mask, who + surface + "通算試合数"] = 1
        ext_df.loc[surface_mask, who + surface + "通算勝利数"] = np.where(
            ext_df.loc[surface_mask, "着順"].astype(int) == 1, 1, 0
        )
        ext_df.loc[surface_mask, who + surface + "通算複勝数"] = np.where(
            ext_df.loc[surface_mask, "着順"].astype(int) <= 3, 1, 0
        )

        for grade in grade_list:
            grade_mask = (ext_df["グレード"] == grade) & surface_mask
            ext_df.loc[grade_mask, who + surface + grade + "通算試合数"] = 1
            ext_df.loc[grade_mask, who + surface + grade + "通算勝利数"] = np.where(
                ext_df.loc[grade_mask, "着順"].astype(int) == 1, 1, 0
            )
            ext_df.loc[grade_mask, who + surface + grade + "通算複勝数"] = np.where(
                ext_df.loc[grade_mask, "着順"].astype(int) <= 3, 1, 0
            )

        grade_mask = (ext_df["グレード"].str.contains("G")) & surface_mask
        ext_df.loc[grade_mask, who + surface + "重賞通算試合数"] = 1
        ext_df.loc[grade_mask, who + surface + "重賞通算勝利数"] = np.where(
            ext_df.loc[grade_mask, "着順"].astype(int) == 1, 1, 0
        )
        ext_df.loc[grade_mask, who + surface + "重賞通算複勝数"] = np.where(
            ext_df.loc[grade_mask, "着順"].astype(int) <= 3, 1, 0
        )

        for dist in dist_list:
            low, upper = dist_dict[dist]
            dist_mask = (
                (ext_df["距離"] >= low) & (ext_df["距離"] <= upper) & surface_mask
            )
            ext_df.loc[dist_mask, who + surface + dist + "通算試合数"] = 1
            ext_df.loc[dist_mask, who + surface + dist + "通算勝利数"] = np.where(
                ext_df.loc[dist_mask, "着順"].astype(int) == 1, 1, 0
            )
            ext_df.loc[dist_mask, who + surface + dist + "通算複勝数"] = np.where(
                ext_df.loc[dist_mask, "着順"].astype(int) <= 3, 1, 0
            )

        for racetrack in racetrack_list:
            racetrack_mask = (ext_df["競馬場"] == racetrack) & surface_mask
            ext_df.loc[racetrack_mask, who + surface + racetrack + "通算試合数"] = 1
            ext_df.loc[racetrack_mask, who + surface + racetrack + "通算勝利数"] = (
                np.where(ext_df.loc[racetrack_mask, "着順"].astype(int) == 1, 1, 0)
            )
            ext_df.loc[racetrack_mask, who + surface + racetrack + "通算複勝数"] = (
                np.where(ext_df.loc[racetrack_mask, "着順"].astype(int) <= 3, 1, 0)
            )

        if who == "騎手":
            for ninki in range(1, 19):
                ninki_mask = (ext_df["人気"] == ninki) & surface_mask
                ext_df.loc[
                    ninki_mask, who + surface + str(ninki) + "人気" + "通算試合数"
                ] = 1
                ext_df.loc[
                    ninki_mask, who + surface + str(ninki) + "人気" + "通算勝利数"
                ] = np.where(ext_df.loc[ninki_mask, "着順"].astype(int) == 1, 1, 0)
                ext_df.loc[
                    ninki_mask, who + surface + str(ninki) + "人気" + "通算複勝数"
                ] = np.where(ext_df.loc[ninki_mask, "着順"].astype(int) <= 3, 1, 0)

    whoid = "horse_id" if who == "競走馬" else "jockey_id"

    print("通算成績を処理...")
    ext_df = calc_cumulative_stats(
       pl.DataFrame(ext_df), who, whoid, grade_list, dist_list, racetrack_list
   )

    # 重複が生まれている可能性があるため削除する
    ext_df = ext_df.unique(subset=["race_id", "horse_id", "jockey_id"])
    df = df.join(ext_df, on=ext_columns, how="left")
    df = df.sort(["date", "馬番"])

    print("処理後DF：", df.shape)

    return df

@measure_execution_time
def calc_win_rate(df):
   """勝率を計算する関数

   Args:
       df (_type_): _description_

   Returns:
       _type_: _description_
   """
   # いらない警告を非表示にする
   import warnings

   warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

   # 通算試合数や勝利数を計算する
   # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
   df = df.unique(subset=["race_id", "馬番"])
   # データをソート
   df = df.sort(["date", "馬番"])

   print("------通算勝率処理開始------")
   print("処理前DF：", df.shape)
   grade_list = [
       "新馬",
       "未勝利",
       "1勝クラス",
       "2勝クラス",
       "3勝クラス",
       "オープン",
       "G3",
       "G2",
       "G1",
   ]
   dist_list = ["短距離", "マイル", "中距離", "クラシック", "長距離"]
   racetrack_list = [
       "札幌",
       "函館",
       "福島",
       "新潟",
       "東京",
       "中山",
       "中京",
       "京都",
       "阪神",
       "小倉",
   ]

   # 勝率を計算
   for who in ["競走馬", "騎手"]:
       for surface in ["芝", "ダート"]:
           df = df.with_column(
               (pl.col(f"{who}{surface}通算勝利数") / (pl.col(f"{who}{surface}通算試合数") + 1e-6)).alias(f"{who}{surface}通算勝率")
           )
           df = df.with_column(
               (pl.col(f"{who}{surface}通算複勝数") / (pl.col(f"{who}{surface}通算試合数") + 1e-6)).alias(f"{who}{surface}通算複勝率")
           )
           df = df.with_column(
               (pl.col(f"単年{who}{surface}通算勝利数") / (pl.col(f"単年{who}{surface}通算試合数") + 1e-6)).alias(f"単年{who}{surface}通算勝率")
           )
           df = df.with_column(
               (pl.col(f"単年{who}{surface}通算複勝数") / (pl.col(f"単年{who}{surface}通算試合数") + 1e-6)).alias(f"単年{who}{surface}通算複勝率")
           )
           for grade in grade_list:
               df = df.with_column(
                   (pl.col(f"{who}{surface}{grade}通算勝利数") / (pl.col(f"{who}{surface}{grade}通算試合数") + 1e-6)).alias(f"{who}{surface}{grade}通算勝率")
               )
               df = df.with_column(
                   (pl.col(f"{who}{surface}{grade}通算複勝数") / (pl.col(f"{who}{surface}{grade}通算試合数") + 1e-6)).alias(f"{who}{surface}{grade}通算複勝率")
               )
               df = df.with_column(
                   (pl.col(f"単年{who}{surface}{grade}通算勝利数") / (pl.col(f"単年{who}{surface}{grade}通算試合数") + 1e-6)).alias(f"単年{who}{surface}{grade}通算勝率")
               )
               df = df.with_column(
                   (pl.col(f"単年{who}{surface}{grade}通算複勝数") / (pl.col(f"単年{who}{surface}{grade}通算試合数") + 1e-6)).alias(f"単年{who}{surface}{grade}通算複勝率")
               )
           df = df.with_column(
               (pl.col(f"{who}{surface}重賞通算勝利数") / (pl.col(f"{who}{surface}重賞通算試合数") + 1e-6)).alias(f"{who}{surface}重賞通算勝率")
           )
           df = df.with_column(
               (pl.col(f"{who}{surface}重賞通算複勝数") / (pl.col(f"{who}{surface}重賞通算試合数") + 1e-6)).alias(f"{who}{surface}重賞通算複勝率")
           )
           df = df.with_column(
               (pl.col(f"単年{who}{surface}重賞通算勝利数") / (pl.col(f"単年{who}{surface}重賞通算試合数") + 1e-6)).alias(f"単年{who}{surface}重賞通算勝率")
           )
           df = df.with_column(
               (pl.col(f"単年{who}{surface}重賞通算複勝数") / (pl.col(f"単年{who}{surface}重賞通算試合数") + 1e-6)).alias(f"単年{who}{surface}重賞通算複勝率")
           )
           for dist in dist_list:
               df = df.with_column(
                   (pl.col(f"{who}{surface}{dist}通算勝利数") / (pl.col(f"{who}{surface}{dist}通算試合数") + 1e-6)).alias(f"{who}{surface}{dist}通算勝率")
               )
               df = df.with_column(
                   (pl.col(f"{who}{surface}{dist}通算複勝数") / (pl.col(f"{who}{surface}{dist}通算試合数") + 1e-6)).alias(f"{who}{surface}{dist}通算複勝率")
               )
               df = df.with_column(
                   (pl.col(f"単年{who}{surface}{dist}通算勝利数") / (pl.col(f"単年{who}{surface}{dist}通算試合数") + 1e-6)).alias(f"単年{who}{surface}{dist}通算勝率")
               )
               df = df.with_column(
                   (pl.col(f"単年{who}{surface}{dist}通算複勝数") / (pl.col(f"単年{who}{surface}{dist}通算試合数") + 1e-6)).alias(f"単年{who}{surface}{dist}通算複勝率")
               )
           for racetrack in racetrack_list:
               df = df.with_column(
                   (pl.col(f"{who}{surface}{racetrack}通算勝利数") / (pl.col(f"{who}{surface}{racetrack}通算試合数") + 1e-6)).alias(f"{who}{surface}{racetrack}通算勝率")
               )
               df = df.with_column(
                   (pl.col(f"{who}{surface}{racetrack}通算複勝数") / (pl.col(f"{who}{surface}{racetrack}通算試合数") + 1e-6)).alias(f"{who}{surface}{racetrack}通算複勝率")
               )
               df = df.with_column(
                   (pl.col(f"単年{who}{surface}{racetrack}通算勝利数") / (pl.col(f"単年{who}{surface}{racetrack}通算試合数") + 1e-6)).alias(f"単年{who}{surface}{racetrack}通算勝率")
               )
               df = df.with_column(
                   (pl.col(f"単年{who}{surface}{racetrack}通算複勝数") / (pl.col(f"単年{who}{surface}{racetrack}通算試合数") + 1e-6)).alias(f"単年{who}{surface}{racetrack}通算複勝率")
               )
           # 騎手の時は人気別の通算成績を計算
           if who == "騎手":
               for ninki in range(1, 19):
                   df = df.with_column(
                       (pl.col(f"{who}{surface}{ninki}人気通算勝利数") / (pl.col(f"{who}{surface}{ninki}人気通算試合数") + 1e-6)).alias(f"{who}{surface}{ninki}人気通算勝率")
                   )
                   df = df.with_column(
                       (pl.col(f"{who}{surface}{ninki}人気通算複勝数") / (pl.col(f"{who}{surface}{ninki}人気通算試合数") + 1e-6)).alias(f"{who}{surface}{ninki}人気通算複勝率")
                   )
                   df = df.with_column(
                       (pl.col(f"単年{who}{surface}{ninki}人気通算勝利数") / (pl.col(f"単年{who}{surface}{ninki}人気通算試合数") + 1e-6)).alias(f"単年{who}{surface}{ninki}人気通算勝率")
                   )
                   df = df.with_column(
                       (pl.col(f"単年{who}{surface}{ninki}人気通算複勝数") / (pl.col(f"単年{who}{surface}{ninki}人気通算試合数") + 1e-6)).alias(f"単年{who}{surface}{ninki}人気通算複勝率")
                   )
               # 一番人気になった確率を計算
               df = df.with_column(
                   (pl.col("騎手芝1人気通算試合数") / pl.col("騎手芝通算試合数")).alias("騎手一番人気確率")
               )
               df = df.with_column(pl.col("騎手一番人気確率").fill_null(0))

   print("処理後DF：", df.shape)

   return df


@measure_execution_time
def calc_pair_win_rate(df):
   """騎手と調教師のコンビ別の勝率を計算する関数

   Args:
       df (_type_): _description_

   Returns:
       _type_: _description_
   """
   # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
   df = df.unique(subset=["race_id", "馬番"])
   # データをソート
   df = df.sort(["date", "馬番"])

   print("------騎手調教師コンビ成績処理開始------")
   print("処理前DF：", df.shape)

   # 騎手と調教師のコンビ別の勝率を計算
   df = df.with_column(pl.concat_str([pl.col("jockey_id"), pl.lit("_"), pl.col("trainer_id")]).alias("jockey_trainer_combo"))
   
   # コンビ別の通算試合数と勝利数を計算
   df = df.with_column(
       pl.when(pl.col("着順") == 1).then(1).otherwise(0).alias("win_flag")
   )
   df = df.with_column(pl.col("win_flag").fill_null(0))
   combo_win = df.groupby("jockey_trainer_combo").agg(pl.sum("win_flag"))
   combo_races = df.groupby("jockey_trainer_combo").agg(pl.count())
   combo_win_rate = combo_win["win_flag"] / combo_races["count"]
   
   # 勝率をデータフレームに結合
   df = df.join(combo_win_rate, on="jockey_trainer_combo", how="left")
   df = df.rename({"win_flag": "騎手調教師コンビ勝率"})
   
   # shiftしてリークを防ぐ
   df = df.with_column(
       pl.col("騎手調教師コンビ勝率").shift(1).over("jockey_trainer_combo").fill_null(0).alias("騎手調教師コンビ勝率")
   )
   # 不要な行をdrop
   df = df.drop(["win_flag", "jockey_trainer_combo"])

   print("処理後DF：", df.shape)

   return df