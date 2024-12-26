# 特徴量作成を行うプログラム

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
import bottleneck as bn
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
    path = "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2/20_data_processing/関数ごと実行時間/execution_times.csv"
    execution_times_df.to_csv(path, index=False)


def reduce_memory_usage(df):
    """
    Automatically reduce memory usage of a dataframe.
    Convert int64 columns to int32 or int16 and float64 columns to float32 or float16,
    depending on their minimum and maximum values.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print("Initial memory usage: {:.2f} MB".format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if pd.api.types.is_datetime64_any_dtype(df[col]):
                pass
            elif col == "賞金":
                pass
            else:
                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif (
                        c_min > np.iinfo(np.int16).min
                        and c_max < np.iinfo(np.int16).max
                    ):
                        df[col] = df[col].astype(np.int16)
                    elif (
                        c_min > np.iinfo(np.int32).min
                        and c_max < np.iinfo(np.int32).max
                    ):
                        df[col] = df[col].astype(np.int32)
                else:
                    if (
                        c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max
                    ):
                        df[col] = df[col].astype(np.float16)
                    elif (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                    ):
                        df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**2
    print(
        "Reduced memory usage: {:.2f} MB ({:.1f}% reduction)".format(
            end_mem, 100 * (start_mem - end_mem) / start_mem
        )
    )

    return df


@measure_execution_time
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
    df = df.sort_values([whoid, "date"])

    # 2. shiftで一行ずらす
    # 各グループ内で1行ずつシフトし、NaNを後ろの値で埋める操作をtransformを用いて適用
    df[all_columns] = df.groupby(whoid)[all_columns].transform(lambda x: x.shift(1))

    # 3. 全データフレームに対してcumsumをする
    df[all_columns] = df[all_columns].fillna(0)
    df[all_columns] = df[all_columns].cumsum(0)

    # 4. 各IDの各列の最小値を引く
    for col in all_columns:
        min_vals = df.groupby(whoid)[col].transform("min")
        df[col] -= min_vals

    # 5. 単年成績を計算
    df["年"] = df["date"].dt.year
    for col in all_columns:
        temp_df = df[
            ["年", whoid, col]
        ].copy()  # 処理高速化のために一時的にデータフレームを作成
        grouped = temp_df.groupby(["年", whoid])[col]
        df["単年" + col] = grouped.transform("min").astype("float")
        df["単年" + col] = df[col] - df["単年" + col]

    df = df.drop(columns=["年"])

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

    pd.set_option("compute.use_bottleneck", True)
    pd.set_option("compute.use_numexpr", True)

    # 通算試合数や勝利数を計算する
    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

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
    ext_df = df[ext_columns].copy()
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
        surface_mask = df["コース種類"] == surface

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
        ext_df, who, whoid, grade_list, dist_list, racetrack_list
    )
    ext_df = ext_df.reset_index(drop=True)

    # 重複が生まれている可能性があるため削除する
    ext_df = ext_df.drop_duplicates(subset=["race_id", "horse_id", "jockey_id"])
    df = pd.merge(df, ext_df, on=ext_columns, how="left")
    df = df.sort_values(["date", "馬番"])

    print("処理後DF：", df.shape)
    # df = reduce_memory_usage(df)
    # display(df[df['馬名'] == 'イクイノックス'])

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
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

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
            df.loc[:, who + surface + "通算勝率"] = df.loc[
                :, who + surface + "通算勝利数"
            ] / (df.loc[:, who + surface + "通算試合数"] + 1e-6)
            df.loc[:, who + surface + "通算複勝率"] = df.loc[
                :, who + surface + "通算複勝数"
            ] / (df.loc[:, who + surface + "通算試合数"] + 1e-6)
            df.loc[:, "単年" + who + surface + "通算勝率"] = df.loc[
                :, "単年" + who + surface + "通算勝利数"
            ] / (df.loc[:, "単年" + who + surface + "通算試合数"] + 1e-6)
            df.loc[:, "単年" + who + surface + "通算複勝率"] = df.loc[
                :, "単年" + who + surface + "通算複勝数"
            ] / (df.loc[:, "単年" + who + surface + "通算試合数"] + 1e-6)
            for grade in grade_list:
                df.loc[:, who + surface + grade + "通算勝率"] = df.loc[
                    :, who + surface + grade + "通算勝利数"
                ] / (df.loc[:, who + surface + grade + "通算試合数"] + 1e-6)
                df.loc[:, who + surface + grade + "通算複勝率"] = df.loc[
                    :, who + surface + grade + "通算複勝数"
                ] / (df.loc[:, who + surface + grade + "通算試合数"] + 1e-6)
                df.loc[:, "単年" + who + surface + grade + "通算勝率"] = df.loc[
                    :, "単年" + who + surface + grade + "通算勝利数"
                ] / (df.loc[:, "単年" + who + surface + grade + "通算試合数"] + 1e-6)
                df.loc[:, "単年" + who + surface + grade + "通算複勝率"] = df.loc[
                    :, "単年" + who + surface + grade + "通算複勝数"
                ] / (df.loc[:, "単年" + who + surface + grade + "通算試合数"] + 1e-6)
            df.loc[:, who + surface + "重賞通算勝率"] = df.loc[
                :, who + surface + "重賞通算勝利数"
            ] / (df.loc[:, who + surface + "重賞通算試合数"] + 1e-6)
            df.loc[:, who + surface + "重賞通算複勝率"] = df.loc[
                :, who + surface + "重賞通算複勝数"
            ] / (df.loc[:, who + surface + "重賞通算試合数"] + 1e-6)
            df.loc[:, "単年" + who + surface + "重賞通算勝率"] = df.loc[
                :, "単年" + who + surface + "重賞通算勝利数"
            ] / (df.loc[:, "単年" + who + surface + "重賞通算試合数"] + 1e-6)
            df.loc[:, "単年" + who + surface + "重賞通算複勝率"] = df.loc[
                :, "単年" + who + surface + "重賞通算複勝数"
            ] / (df.loc[:, "単年" + who + surface + "重賞通算試合数"] + 1e-6)
            for dist in dist_list:
                df.loc[:, who + surface + dist + "通算勝率"] = df.loc[
                    :, who + surface + dist + "通算勝利数"
                ] / (df.loc[:, who + surface + dist + "通算試合数"] + 1e-6)
                df.loc[:, who + surface + dist + "通算複勝率"] = df.loc[
                    :, who + surface + dist + "通算複勝数"
                ] / (df.loc[:, who + surface + dist + "通算試合数"] + 1e-6)
                df.loc[:, "単年" + who + surface + dist + "通算勝率"] = df.loc[
                    :, "単年" + who + surface + dist + "通算勝利数"
                ] / (df.loc[:, "単年" + who + surface + dist + "通算試合数"] + 1e-6)
                df.loc[:, "単年" + who + surface + dist + "通算複勝率"] = df.loc[
                    :, "単年" + who + surface + dist + "通算複勝数"
                ] / (df.loc[:, "単年" + who + surface + dist + "通算試合数"] + 1e-6)
            for racetrack in racetrack_list:
                df.loc[:, who + surface + racetrack + "通算勝率"] = df.loc[
                    :, who + surface + racetrack + "通算勝利数"
                ] / (df.loc[:, who + surface + racetrack + "通算試合数"] + 1e-6)
                df.loc[:, who + surface + racetrack + "通算複勝率"] = df.loc[
                    :, who + surface + racetrack + "通算複勝数"
                ] / (df.loc[:, who + surface + racetrack + "通算試合数"] + 1e-6)
                df.loc[:, "単年" + who + surface + racetrack + "通算勝率"] = df.loc[
                    :, "単年" + who + surface + racetrack + "通算勝利数"
                ] / (
                    df.loc[:, "単年" + who + surface + racetrack + "通算試合数"] + 1e-6
                )
                df.loc[:, "単年" + who + surface + racetrack + "通算複勝率"] = df.loc[
                    :, "単年" + who + surface + racetrack + "通算複勝数"
                ] / (
                    df.loc[:, "単年" + who + surface + racetrack + "通算試合数"] + 1e-6
                )
            # 騎手の時は人気別の通算成績を計算
            if who == "騎手":
                for ninki in range(1, 19):
                    df.loc[:, who + surface + str(ninki) + "人気" + "通算勝率"] = (
                        df.loc[:, who + surface + str(ninki) + "人気" + "通算勝利数"]
                        / (
                            df.loc[
                                :, who + surface + str(ninki) + "人気" + "通算試合数"
                            ]
                            + 1e-6
                        )
                    )
                    df.loc[:, who + surface + str(ninki) + "人気" + "通算複勝率"] = (
                        df.loc[:, who + surface + str(ninki) + "人気" + "通算複勝数"]
                        / (
                            df.loc[
                                :, who + surface + str(ninki) + "人気" + "通算試合数"
                            ]
                            + 1e-6
                        )
                    )
                    df.loc[
                        :, "単年" + who + surface + str(ninki) + "人気" + "通算勝率"
                    ] = df.loc[
                        :, "単年" + who + surface + str(ninki) + "人気" + "通算勝利数"
                    ] / (
                        df.loc[
                            :,
                            "単年" + who + surface + str(ninki) + "人気" + "通算試合数",
                        ]
                        + 1e-6
                    )
                    df.loc[
                        :, "単年" + who + surface + str(ninki) + "人気" + "通算複勝率"
                    ] = df.loc[
                        :, "単年" + who + surface + str(ninki) + "人気" + "通算複勝数"
                    ] / (
                        df.loc[
                            :,
                            "単年" + who + surface + str(ninki) + "人気" + "通算試合数",
                        ]
                        + 1e-6
                    )
                # 一番人気になった確率を計算
                df["騎手一番人気確率"] = (
                    df["騎手芝1人気通算試合数"] / df["騎手芝通算試合数"]
                )
                df["騎手一番人気確率"] = df["騎手一番人気確率"].fillna(0)

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
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    print("------騎手調教師コンビ成績処理開始------")
    print("処理前DF：", df.shape)

    # 騎手と調教師のコンビ別の勝率を計算
    df["jockey_trainer_combo"] = (
        df["jockey_id"].astype(str) + "_" + df["trainer_id"].astype(str)
    )
    # コンビ別の通算試合数と勝利数を計算
    df.loc[df["着順"] == 1, "win_flag"] = 1
    df["win_flag"] = df["win_flag"].fillna(0)
    combo_win = df.groupby("jockey_trainer_combo")["win_flag"].sum()
    combo_races = df.groupby("jockey_trainer_combo")["win_flag"].count()
    combo_win_rate = combo_win / combo_races
    # 勝率をデータフレームに結合
    df["騎手調教師コンビ勝率"] = df["jockey_trainer_combo"].map(combo_win_rate)
    # shiftしてリークを防ぐ
    df["騎手調教師コンビ勝率"] = df.groupby("jockey_trainer_combo")[
        "騎手調教師コンビ勝率"
    ].transform(lambda x: x.shift(1).fillna(0))
    # 不要な行をdrop
    df = df.drop(columns=["win_flag", "jockey_trainer_combo"])

    print("処理後DF：", df.shape)

    return df


def apply_pca_for_stats(df, who, n_components=50):
    """通算成績に対してPCAを適用する関数

    Args:
        df (_type_): _description_
        who (_type_): 競走馬か騎手か
        n_components (int, optional): _description_. Defaults to 100.
    """
    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    print("------通算成績PCA処理開始------")
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

        print("グレードの通算勝利数を計算中...")
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

        print("距離の通算勝利数を計算中...")
        for dist in dist_list:
            columns = [
                f"{who}{surface}{dist}{suffix}"
                for suffix in ["通算試合数", "通算勝利数", "通算複勝数"]
            ]
            all_columns.extend(columns)

        print("競馬場の通算勝利数を計算中...")
        for racetrack in racetrack_list:
            columns = [
                f"{who}{surface}{racetrack}{suffix}"
                for suffix in ["通算試合数", "通算勝利数", "通算複勝数"]
            ]
            all_columns.extend(columns)

    # 2022年のデータで学習し、2023年以降のデータは学習に使わない
    train_df = df[df["年"] < 2022].copy().reset_index(drop=True)
    test_df = df[df["年"] >= 2022].copy().reset_index(drop=True)

    # pcaの学習
    pca = PCA(n_components=n_components)
    pca.fit(train_df[all_columns])

    # 圧縮した特徴量のデータフレームを作成
    train_pca_df = pd.DataFrame(
        pca.transform(train_df[all_columns]),
        columns=[f"{who}通算成績PCA_{i}" for i in range(n_components)],
    )
    test_pca_df = pd.DataFrame(
        pca.transform(test_df[all_columns]),
        columns=[f"{who}通算成績PCA_{i}" for i in range(n_components)],
    )

    # データフレームに反映
    train_df = pd.concat([train_df, train_pca_df], axis=1)
    test_df = pd.concat([test_df, test_pca_df], axis=1)
    df = pd.concat([train_df, test_df], axis=0)
    df = df.reset_index(drop=True)

    print("中間処理中DF：", df.shape)

    # 圧縮した特徴量を削除
    df = df.drop(columns=all_columns)

    # 重複を削除
    df = df.drop_duplicates(subset=["race_id", "horse_id", "馬番"])
    df = df.sort_values(["date", "馬番"])

    print("処理後DF：", df.shape)
    # df = reduce_memory_usage(df)
    # display(df[df['馬名'] == 'イクイノックス'])

    return df


@measure_execution_time
def calculate_average_top2_distance(df):
    """平均連対距離を計算する関数

    Args:
        df (_type_): _description_
    Returns:
        _type_: _description_
    """

    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)
    print("------平均連対距離計算処理開始------")
    # 各レース時点での累積連対距離と連対出走数を計算
    top2_races = df[df["着順"] <= 2].copy()
    top2_races["累積連対距離"] = top2_races.groupby("horse_id")["距離"].cumsum()
    top2_races["連対出走数"] = top2_races.groupby("horse_id").cumcount() + 1

    # レース時点での平均連対距離を計算
    top2_races["平均連対距離"] = top2_races["累積連対距離"] / top2_races["連対出走数"]

    # 必要な列のみ選択
    top2_races = top2_races[["race_id", "馬番", "平均連対距離"]]

    # 元のデータフレームと結合
    df = df.merge(top2_races, on=["race_id", "馬番"], how="left")

    # 1レース前の平均連対距離を取得
    df["平均連対距離"] = df.groupby("horse_id")["平均連対距離"].shift(1)
    df["平均連対距離"] = df["平均連対距離"].fillna(method="ffill")

    # 欠損値を0で補完
    df["平均連対距離"] = df["平均連対距離"].fillna(0)

    print("処理後DF：", df.shape)
    return df


@measure_execution_time
def calculate_moving_average_distance(df, window=5):
    """馬別の距離移動平均を計算する関数"""
    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])

    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    print(f"------馬別の距離{window}レース移動平均を計算します。------")
    print("処理前DF：", df.shape)

    # 馬別の距離移動平均を計算
    def moving_average(x):
        if len(x) < window:
            return bn.move_mean(x, window=len(x), min_count=1)
        else:
            return bn.move_mean(x, window=window, min_count=1)

    df["距離移動平均"] = df.groupby("horse_id")["距離"].transform(moving_average)

    print("処理後DF：", df.shape)
    return df


@measure_execution_time
def calculate_moving_average_grade(df, window=5):
    """馬別のグレード移動平均を計算する関数"""
    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    print(f"------馬別の距離{window}レース移動平均を計算します。------")
    print("処理前DF：", df.shape)

    grade_mapping = {
        "未勝利": 0,
        "新馬": 1,
        "1勝クラス": 2,
        "2勝クラス": 3,
        "3勝クラス": 4,
        "オープン": 5,
        "G3": 6,
        "G2": 7,
        "G1": 8,
    }

    # グレード数値の割り当て
    df["グレード数値"] = df["グレード"].map(grade_mapping)

    # 馬別の距離移動平均を計算
    def moving_average(x):
        if len(x) < window:
            return bn.move_mean(x, window=len(x), min_count=1)
        else:
            return bn.move_mean(x, window=window, min_count=1)

    df["グレード移動平均"] = df.groupby("horse_id")["グレード数値"].transform(
        moving_average
    )
    print("処理後DF：", df.shape)

    return df


@measure_execution_time
def calc_past_avg_time_by_distance(df):
    """競走馬ごとの条件を満たす過去の「上がり3F」「タイム」の平均値を計算する関数
    タイムは距離別のみ計算

    Args:
        df (pd.DataFrame): 入力データフレーム

    Returns:
        pd.DataFrame: 競走馬ごとの条件を満たす過去の「上がり3F」の平均値を追加したデータフレーム
    """
    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)
    print("------距離別平均タイム計算処理開始------")
    print("処理前DF：", df.shape)
    dist_list = [
        1000,
        1150,
        1200,
        1300,
        1400,
        1500,
        1600,
        1700,
        1800,
        1900,
        2000,
        2100,
        2200,
        2300,
        2400,
        2500,
        2600,
        3000,
        3200,
        3400,
        3600,
    ]
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

    # shiftに用いるリスト
    all_columns = []

    # 単純な上がり3Fの平均値を計算
    race = df.copy()
    race["上がり3F累積和"] = race.groupby("horse_id")["上がり3F"].cumsum()
    race["出走数"] = race.groupby("horse_id").cumcount() + 1
    race["平均上がり3F"] = race["上がり3F累積和"] / race["出走数"]
    # 不要な列を削除
    race = race.drop(columns=["上がり3F累積和", "出走数"])
    # 元のデータフレームと結合
    df = df.merge(
        race[["race_id", "馬番", "平均上がり3F"]], on=["race_id", "馬番"], how="left"
    )
    df["平均上がり3F"] = df.groupby("horse_id")["平均上がり3F"].shift()
    all_columns.append("平均上がり3F")

    # メモリ解放
    del race

    # 距離別の平均上がり3Fタイムを計算
    for dist in dist_list:
        # 距離条件を満たすレースを抽出
        dist_races = df[df["距離"] == dist].copy()
        # 上がり3Fの累積和を計算
        dist_races["上がり3F累積和"] = dist_races.groupby("horse_id")[
            "上がり3F"
        ].cumsum()
        dist_races["タイム累積和"] = dist_races.groupby("horse_id")["タイム"].cumsum()
        # 出走数を計算
        dist_races["出走数"] = dist_races.groupby("horse_id").cumcount() + 1
        # 平均上がり3Fタイムを計算
        dist_races[f"{dist}平均上がり3F"] = (
            dist_races["上がり3F累積和"] / dist_races["出走数"]
        )
        dist_races[f"{dist}平均タイム"] = (
            dist_races["タイム累積和"] / dist_races["出走数"]
        )
        # 不要な列を削除
        dist_races = dist_races.drop(
            columns=["上がり3F累積和", "タイム累積和", "出走数"]
        )
        # 元のデータフレームと結合
        df = df.merge(
            dist_races[["race_id", "馬番", f"{dist}平均上がり3F", f"{dist}平均タイム"]],
            on=["race_id", "馬番"],
            how="left",
        )
        df[f"{dist}平均上がり3F"] = df.groupby("horse_id")[
            f"{dist}平均上がり3F"
        ].shift()
        df[f"{dist}平均タイム"] = df.groupby("horse_id")[f"{dist}平均タイム"].shift()
        all_columns.append(f"{dist}平均上がり3F")
        all_columns.append(f"{dist}平均タイム")

    # メモリ解放
    del dist_races

    # 競馬場別の平均上がり3Fタイムを計算
    for racetrack in racetrack_list:
        # 競馬場条件を満たすレースを抽出
        racetrack_races = df[df["競馬場"] == racetrack].copy()
        # 上がり3Fの累積和を計算
        racetrack_races["上がり3F累積和"] = racetrack_races.groupby("horse_id")[
            "上がり3F"
        ].cumsum()
        # 出走数を計算
        racetrack_races["出走数"] = racetrack_races.groupby("horse_id").cumcount() + 1
        # 平均上がり3Fタイムを計算
        racetrack_races[f"{racetrack}平均上がり3F"] = (
            racetrack_races["上がり3F累積和"] / racetrack_races["出走数"]
        )
        # 不要な列を削除
        racetrack_races = racetrack_races.drop(columns=["上がり3F累積和", "出走数"])
        # 元のデータフレームと結合
        df = df.merge(
            racetrack_races[["race_id", "馬番", f"{racetrack}平均上がり3F"]],
            on=["race_id", "馬番"],
            how="left",
        )
        df[f"{racetrack}平均上がり3F"] = df.groupby("horse_id")[
            f"{racetrack}平均上がり3F"
        ].shift()
        all_columns.append(f"{racetrack}平均上がり3F")

    # メモリ解放
    del racetrack_races

    # 競馬場と距離の組み合わせ別の平均上がり3Fタイムを計算
    for racetrack in racetrack_list:
        for dist in dist_list:
            # 競馬場と距離の条件を満たすレースを抽出
            comb_races = df[(df["競馬場"] == racetrack) & (df["距離"] == dist)].copy()
            # 上がり3Fの累積和を計算
            comb_races["上がり3F累積和"] = comb_races.groupby("horse_id")[
                "上がり3F"
            ].cumsum()
            comb_races["タイム累積和"] = comb_races.groupby("horse_id")[
                "タイム"
            ].cumsum()
            # 出走数を計算
            comb_races["出走数"] = comb_races.groupby("horse_id").cumcount() + 1
            # 平均上がり3Fタイムを計算
            comb_races[f"{racetrack}{dist}平均上がり3F"] = (
                comb_races["上がり3F累積和"] / comb_races["出走数"]
            )
            comb_races[f"{racetrack}{dist}平均タイム"] = (
                comb_races["タイム累積和"] / comb_races["出走数"]
            )
            # 不要な列を削除
            comb_races = comb_races.drop(
                columns=["上がり3F累積和", "タイム累積和", "出走数"]
            )
            # 元のデータフレームと結合
            df = df.merge(
                comb_races[
                    [
                        "race_id",
                        "馬番",
                        f"{racetrack}{dist}平均上がり3F",
                        f"{racetrack}{dist}平均タイム",
                    ]
                ],
                on=["race_id", "馬番"],
                how="left",
            )
            df[f"{racetrack}{dist}平均上がり3F"] = df.groupby("horse_id")[
                f"{racetrack}{dist}平均上がり3F"
            ].shift()
            df[f"{racetrack}{dist}平均タイム"] = df.groupby("horse_id")[
                f"{racetrack}{dist}平均タイム"
            ].shift()
            all_columns.append(f"{racetrack}{dist}平均上がり3F")
            all_columns.append(f"{racetrack}{dist}平均タイム")

    # メモリ解放
    del comb_races

    # 次のレースの行に適用するためにシフト
    temp_df = df[["horse_id"] + all_columns].copy()
    df[all_columns] = temp_df.groupby("horse_id")[all_columns].ffill()
    df[all_columns] = df[all_columns].fillna(0)

    gc.collect()

    print("処理後DF：", df.shape)

    return df


@measure_execution_time
def calc_fastest_time_by_distance(df):
    """距離別の最高タイムを計算する関数

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    print("------距離別最高タイム計算処理開始------")
    print("処理前DF：", df.shape)
    dist_list = [
        1000,
        1150,
        1200,
        1300,
        1400,
        1500,
        1600,
        1700,
        1800,
        1900,
        2000,
        2100,
        2200,
        2300,
        2400,
        2500,
        2600,
        3000,
        3200,
        3400,
        3600,
    ]
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

    # shiftに用いるリスト
    all_columns = []

    # 単純な最高あがり3Fのタイムを計算
    race = df.copy()
    race["最高上がり3F"] = race.groupby("horse_id")["上がり3F"].cummin()
    # 元のデータフレームと結合
    df = df.merge(
        race[["race_id", "馬番", "最高上がり3F"]], on=["race_id", "馬番"], how="left"
    )
    df["最高上がり3F"] = df.groupby("horse_id")["最高上がり3F"].shift()
    all_columns.append("最高上がり3F")

    # メモリ解放
    del race

    # 距離別の最高上がり3Fタイムを計算
    for dist in dist_list:
        # 距離条件を満たすレースを抽出
        dist_races = df[df["距離"] == dist].copy()
        # 最高上がり3Fタイムを計算
        dist_races[f"{dist}最高上がり3F"] = dist_races.groupby("horse_id")[
            "上がり3F"
        ].cummin()
        dist_races[f"{dist}最高タイム"] = dist_races.groupby("horse_id")[
            "タイム"
        ].cummin()
        # 元のデータフレームと結合
        df = df.merge(
            dist_races[["race_id", "馬番", f"{dist}最高上がり3F", f"{dist}最高タイム"]],
            on=["race_id", "馬番"],
            how="left",
        )
        # 次のレースの行に適用するためにシフト
        df[f"{dist}最高上がり3F"] = df.groupby("horse_id")[
            f"{dist}最高上がり3F"
        ].shift()
        df[f"{dist}最高タイム"] = df.groupby("horse_id")[f"{dist}最高タイム"].shift()
        all_columns.append(f"{dist}最高上がり3F")
        all_columns.append(f"{dist}最高タイム")

    # メモリ解放
    del dist_races

    # 競馬場別の最高上がり3Fタイムを計算
    for racetrack in racetrack_list:
        # 競馬場条件を満たすレースを抽出
        racetrack_races = df[df["競馬場"] == racetrack].copy()
        # 最高上がり3Fタイムを計算
        racetrack_races[f"{racetrack}最高上がり3F"] = racetrack_races.groupby(
            "horse_id"
        )["上がり3F"].cummin()
        # 元のデータフレームと結合
        df = df.merge(
            racetrack_races[["race_id", "馬番", f"{racetrack}最高上がり3F"]],
            on=["race_id", "馬番"],
            how="left",
        )
        # 次のレースの行に適用するためにシフト
        df[f"{racetrack}最高上がり3F"] = df.groupby("horse_id")[
            f"{racetrack}最高上がり3F"
        ].shift()
        all_columns.append(f"{racetrack}最高上がり3F")

    # メモリ解放
    del racetrack_races

    # 競馬場と距離の組み合わせ別の最高上がり3Fタイムを計算
    for racetrack in racetrack_list:
        for dist in dist_list:
            # 競馬場と距離の条件を満たすレースを抽出
            comb_races = df[(df["競馬場"] == racetrack) & (df["距離"] == dist)].copy()
            # 最高上がり3Fタイムを計算
            comb_races[f"{racetrack}{dist}最高上がり3F"] = comb_races.groupby(
                "horse_id"
            )["上がり3F"].cummin()
            comb_races[f"{racetrack}{dist}最高タイム"] = comb_races.groupby("horse_id")[
                "タイム"
            ].cummin()
            # 元のデータフレームと結合
            df = df.merge(
                comb_races[
                    [
                        "race_id",
                        "馬番",
                        f"{racetrack}{dist}最高上がり3F",
                        f"{racetrack}{dist}最高タイム",
                    ]
                ],
                on=["race_id", "馬番"],
                how="left",
            )
            # 次のレースの行に適用するためにシフト
            df[f"{racetrack}{dist}最高上がり3F"] = df.groupby("horse_id")[
                f"{racetrack}{dist}最高上がり3F"
            ].shift()
            df[f"{racetrack}{dist}最高タイム"] = df.groupby("horse_id")[
                f"{racetrack}{dist}最高タイム"
            ].shift()
            all_columns.append(f"{racetrack}{dist}最高上がり3F")
            all_columns.append(f"{racetrack}{dist}最高タイム")

    # メモリ解放
    del comb_races

    # 次のレースの行に適用するためにffill
    temp_df = df[["horse_id"] + all_columns].copy()
    df[all_columns] = temp_df.groupby("horse_id")[all_columns].ffill()
    df[all_columns] = df[all_columns].fillna(0)

    gc.collect()

    print("処理後DF：", df.shape)

    return df


@measure_execution_time
def apply_pca_for_time(df, n_components=20):
    """タイム特徴量に対してPCAを適用する関数

    Args:
        df (_type_): _description_
        n_components (int, optional): _description_. Defaults to 50.
    """
    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    print("------タイム特徴量PCA処理開始------")
    print("処理前DF：", df.shape)

    dist_list = [
        1000,
        1150,
        1200,
        1300,
        1400,
        1500,
        1600,
        1700,
        1800,
        1900,
        2000,
        2100,
        2200,
        2300,
        2400,
        2500,
        2600,
        3000,
        3200,
        3400,
        3600,
    ]
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

    # 計算する列名を設定
    all_columns = []
    for dist in dist_list:
        columns = [f"{dist}m平均上がり3F", f"{dist}m最高上がり3F", f"{dist}m最高タイム"]
        all_columns.extend(columns)
        for track in racetrack_list:
            columns = [
                f"{track}{dist}m平均上がり3F",
                f"{track}{dist}m最高上がり3F",
                f"{track}{dist}m最高タイム",
            ]
            all_columns.extend(columns)

    # 2022年のデータで学習し、2023年以降のデータは学習に使わない
    train_df = df[df["年"] < 2022].copy().reset_index(drop=True)
    test_df = df[df["年"] >= 2022].copy().reset_index(drop=True)

    # pcaの学習
    pca = PCA(n_components=n_components)
    pca.fit(train_df[all_columns])

    # 圧縮した特徴量のデータフレームを作成
    train_pca_df = pd.DataFrame(
        pca.transform(train_df[all_columns]),
        columns=[f"タイム特徴量PCA_{i}" for i in range(n_components)],
    )
    test_pca_df = pd.DataFrame(
        pca.transform(test_df[all_columns]),
        columns=[f"タイム特徴量PCA_{i}" for i in range(n_components)],
    )

    # データフレームに反映
    train_df = pd.concat([train_df, train_pca_df], axis=1)
    test_df = pd.concat([test_df, test_pca_df], axis=1)
    df = pd.concat([train_df, test_df], axis=0)
    df = df.reset_index(drop=True)

    print("中間処理中DF：", df.shape)

    # 圧縮した特徴量を削除
    df = df.drop(columns=all_columns)

    # 重複を削除
    df = df.drop_duplicates(subset=["race_id", "horse_id", "馬番"])
    df = df.sort_values(["date", "馬番"])

    print("処理後DF：", df.shape)
    # df = reduce_memory_usage(df)
    # display(df[df['馬名'] == 'イクイノックス'])

    return df


def setPlayer(rating=1500, rd=10, vol=0.06):
    """レーティング計算でプレイヤーのパラメータを設定する関数

    Args:
        rating (int, optional): レーティング. Defaults to 1500.
        rd (int, optional): レーティング信頼区間. Defaults to 10.
        vol (float, optional): レーティング偏差. Defaults to 0.06.

    Returns:
        player: パラメータが格納された変数
    """
    player = glicko2.Player()
    player.rating = rating
    player.rd = rd
    player.vol = vol

    return player


def calcRatings(players, ranks):
    """プレイヤーごとの順位から試合後のレーティングを計算する関数

    Args:
        players (player): パラメータが格納された変数
        ranks (int)): 順位

    Returns:
        player: パラメータが格納された変数
    """
    newPlayers = []
    for i, (target_player, target_rank) in enumerate(zip(players, ranks)):
        new_target_player = copy.deepcopy(target_player)
        ratings = []
        rds = []
        outcomes = []
        for j, (player, rank) in enumerate(zip(players, ranks)):
            if not i == j:
                ratings.append(player.rating)
                rds.append(player.rd)
                if rank > target_rank:
                    outcomes.append(1)
                elif rank < target_rank:
                    outcomes.append(0)
                elif rank == target_rank:
                    outcomes.append(0.5)

        new_target_player.update_player(ratings, rds, outcomes)
        newPlayers.append(new_target_player)

    return newPlayers


@measure_execution_time
def calc_rating(df, who):
    """レーティングを計算する関数"""
    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    print(f"{who}のレーティングを計算します。")
    print("処理前DF：", df.shape)
    if who == "競走馬":
        target = "horse_id"
        unique_ids = df[target].unique()
        rating_dict = {id: [1500] for id in unique_ids}
        rd_dict = {id: [350] for id in unique_ids}
        vol_dict = {id: [0.06] for id in unique_ids}
    elif who == "騎手":
        target = "jockey_id"
        unique_ids = df[target].unique()
        rating_dict = {id: [1500] for id in unique_ids}
        rd_dict = {id: [10] for id in unique_ids}
        vol_dict = {id: [0.0001] for id in unique_ids}

    else:
        raise Exception("whoには「競走馬」か「騎手」を指定してください。")

    if who == "競走馬":
        rating_key = "競走馬レーティング"
        rd_key = "競走馬レーティング偏差"
        vol_key = "競走馬レーティング変動率"
    elif who == "騎手":
        rating_key = "騎手レーティング"
        rd_key = "騎手レーテング偏差"
        vol_key = "騎手レーティング変動率"

    # 列を追加
    if rating_key not in df.columns:
        df[rating_key] = np.nan
    if rd_key not in df.columns:
        df[rd_key] = np.nan
    if vol_key not in df.columns:
        df[vol_key] = np.nan

    # レーティングを計算
    unique_race_ids = df["race_id"].unique()
    print("レーティングを計算中...")
    for race_id in unique_race_ids:
        ext_df = df.loc[df["race_id"] == race_id].copy()
        player_list, rank_list = [], []
        for id in ext_df[target].unique():
            rating, rd, vol = rating_dict[id][-1], rd_dict[id][-1], vol_dict[id][-1]
            player = setPlayer(rating, rd, vol)
            player_list.append(player)
            rank_list.append(ext_df.loc[ext_df[target] == id]["着順"].iat[0])
        try:
            new_player_list = calcRatings(player_list, rank_list)

            for i, id in enumerate(ext_df[target].unique()):
                rating_dict[id].append(new_player_list[i].rating)
                rd_dict[id].append(new_player_list[i].rd)
                vol_dict[id].append(new_player_list[i].vol)
        except Exception:
            print(race_id, rank_list)
            print(player_list)
            for i, id in enumerate(ext_df[target].unique()):
                rating_dict[id].append(1500)
                rd_dict[id].append(350)
                vol_dict[id].append(0.06)

    # レーティングをDFに反映
    print("レーティングをDFに反映中...")
    # レーティングをDFに反映する前の検証とデバッグ情報の出力
    for id in df[target].unique():
        df_id_length = len(df.loc[df[target] == id])
        rating_length = len(rating_dict[id][:-1])
        if df_id_length != rating_length:
            print(f"ID: {id} で問題が発生しました。")
            print(
                f"レーティングリストの長さ: {rating_length}, DFでのIDの出現回数: {df_id_length}"
            )
            # レースIDやその他関連情報をログに出力
            race_ids = df.loc[df[target] == id, "race_id"].unique()
            print(f"関連するレースID: {race_ids}")
            # ここでさらに詳細な調査や修正処理を実施
        else:
            # レーティングリストが期待通りの場合は、DFに反映
            df.loc[df[target] == id, rating_key] = rating_dict[id][:-1]
            df.loc[df[target] == id, rd_key] = rd_dict[id][:-1]
            df.loc[df[target] == id, vol_key] = vol_dict[id][:-1]

    print("処理後DF：", df.shape)
    # df = reduce_memory_usage(df)
    # display(df[df['馬名'] == 'イクイノックス'])

    return df


@measure_execution_time
def calc_conditional_rating(df, who):
    """レーティングを計算する関数"""
    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    # レーティングを計算する条件を設定
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

    # 各条件別のデータを保存し結合するためのlist
    ext_df_list = []
    # 最後に作成した列をffillするためのlist
    all_columns = []

    print("処理前DF：", df.shape)

    # 芝ダート振り分け
    for surface in ["芝", "ダート"]:
        for dist in dist_list:
            const = (
                (df["コース種類"] == surface)
                & (df["距離"] >= dist_dict[dist][0])
                & (df["距離"] <= dist_dict[dist][1])
            )
            # 条件に合致するデータを抽出
            ext_df = df[const].copy()

            if who == "競走馬":
                target = "horse_id"
                unique_ids = ext_df[target].unique()
                rating_dict = {id: [1500] for id in unique_ids}
                rd_dict = {id: [150] for id in unique_ids}
                vol_dict = {id: [0.06] for id in unique_ids}
            elif who == "騎手":
                target = "jockey_id"
                unique_ids = ext_df[target].unique()
                rating_dict = {id: [1500] for id in unique_ids}
                rd_dict = {id: [10] for id in unique_ids}
                vol_dict = {id: [0.0001] for id in unique_ids}

            else:
                raise Exception("whoには「競走馬」か「騎手」を指定してください。")

            if who == "競走馬":
                rating_key = "競走馬" + surface + dist + "レーティング"
                rd_key = "競走馬" + surface + dist + "レーティング偏差"
                vol_key = "競走馬" + surface + dist + "レーティング変動率"
            elif who == "騎手":
                rating_key = "騎手" + surface + dist + "レーティング"
                rd_key = "騎手" + surface + dist + "レーティング偏差"
                vol_key = "騎手" + surface + dist + "レーティング変動率"

            # 列を追加
            if rating_key not in ext_df.columns:
                ext_df[rating_key] = np.nan
            if rd_key not in df.columns:
                ext_df[rd_key] = np.nan
            if vol_key not in df.columns:
                ext_df[vol_key] = np.nan

            all_columns.extend([rating_key, rd_key, vol_key])

            # レーティングを計算
            unique_race_ids = ext_df["race_id"].unique()
            for race_id in unique_race_ids:
                ext_ext_df = ext_df.loc[ext_df["race_id"] == race_id].copy()
                player_list, rank_list = [], []
                for id in ext_ext_df[target].unique():
                    rating, rd, vol = (
                        rating_dict[id][-1],
                        rd_dict[id][-1],
                        vol_dict[id][-1],
                    )
                    player = setPlayer(rating, rd, vol)
                    player_list.append(player)
                    rank_list.append(
                        ext_ext_df.loc[ext_ext_df[target] == id]["着順"].iat[0]
                    )

                try:
                    new_player_list = calcRatings(player_list, rank_list)

                    for i, id in enumerate(ext_ext_df[target].unique()):
                        rating_dict[id].append(new_player_list[i].rating)
                        rd_dict[id].append(new_player_list[i].rd)
                        vol_dict[id].append(new_player_list[i].vol)
                except Exception:
                    print(race_id, rank_list)
                    print(player_list)
                    for i, id in enumerate(ext_ext_df[target].unique()):
                        rating_dict[id].append(1500)
                        rd_dict[id].append(350)
                        vol_dict[id].append(0.06)

            # レーティングをDFに反映
            # dfのshapeが0でない時のみ下記の処理を実行
            if ext_df.shape[0] > 0:
                # IDと行番号でインデックスを作成するための準備
                ext_df["row_num"] = ext_df.groupby(target).cumcount()
                # 新しいデータフレームを作成し、リストを行に展開
                update_rows = []
                for id, ratings, rds, vols in zip(
                    rating_dict.keys(),
                    rating_dict.values(),
                    rd_dict.values(),
                    vol_dict.values(),
                ):
                    for idx, (rating, rd, vol) in enumerate(
                        zip(ratings[:-1], rds[:-1], vols[:-1])
                    ):
                        update_rows.append(
                            {
                                target: id,
                                "row_num": idx,
                                rating_key: rating,
                                rd_key: rd,
                                vol_key: vol,
                            }
                        )
                updates_df = pd.DataFrame(update_rows)
                # IDとrow_numをインデックスとして設定
                ext_df.set_index([target, "row_num"], inplace=True)
                updates_df.set_index([target, "row_num"], inplace=True)
                # 更新
                ext_df.update(updates_df)
                # インデックスのリセット
                ext_df.reset_index(inplace=True)
                ext_df.drop(columns="row_num", inplace=True)

            ext_df_list.append(ext_df)

    # 結合
    df = pd.concat(ext_df_list, axis=0)
    # 各条件別のデータを保存し結合するためのlist
    ext_df_list = []
    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    # 芝ダート振り分け
    for surface in ["芝", "ダート"]:
        for race_track in racetrack_list:
            const = (df["コース種類"] == surface) & (df["競馬場"] == race_track)
            # 条件に合致するデータを抽出
            ext_df = df[const].copy()

            if who == "競走馬":
                target = "horse_id"
                unique_ids = ext_df[target].unique()
                rating_dict = {id: [1500] for id in unique_ids}
                rd_dict = {id: [30] for id in unique_ids}
                vol_dict = {id: [0.06] for id in unique_ids}
            elif who == "騎手":
                target = "jockey_id"
                unique_ids = ext_df[target].unique()
                rating_dict = {id: [1500] for id in unique_ids}
                rd_dict = {id: [10] for id in unique_ids}
                vol_dict = {id: [0.0001] for id in unique_ids}

            else:
                raise Exception("whoには「競走馬」か「騎手」を指定してください。")

            if who == "競走馬":
                rating_key = "競走馬" + surface + race_track + "レーティング"
                rd_key = "競走馬" + surface + race_track + "レーティング偏差"
                vol_key = "競走馬" + surface + race_track + "レーティング変動率"
            elif who == "騎手":
                rating_key = "騎手" + surface + race_track + "レーティング"
                rd_key = "騎手" + surface + race_track + "レーティング偏差"
                vol_key = "騎手" + surface + race_track + "レーティング変動率"

            # 列を追加
            if rating_key not in ext_df.columns:
                ext_df[rating_key] = np.nan
            if rd_key not in df.columns:
                ext_df[rd_key] = np.nan
            if vol_key not in df.columns:
                ext_df[vol_key] = np.nan

            all_columns.extend([rating_key, rd_key, vol_key])

            # レーティングを計算
            unique_race_ids = ext_df["race_id"].unique()
            for race_id in unique_race_ids:
                ext_ext_df = ext_df.loc[ext_df["race_id"] == race_id].copy()
                player_list, rank_list = [], []
                for id in ext_ext_df[target].unique():
                    rating, rd, vol = (
                        rating_dict[id][-1],
                        rd_dict[id][-1],
                        vol_dict[id][-1],
                    )
                    player = setPlayer(rating, rd, vol)
                    player_list.append(player)
                    rank_list.append(
                        ext_ext_df.loc[ext_ext_df[target] == id]["着順"].iat[0]
                    )

                try:
                    new_player_list = calcRatings(player_list, rank_list)

                    for i, id in enumerate(ext_ext_df[target].unique()):
                        rating_dict[id].append(new_player_list[i].rating)
                        rd_dict[id].append(new_player_list[i].rd)
                        vol_dict[id].append(new_player_list[i].vol)
                except Exception:
                    print(race_id, rank_list)
                    print(player_list)
                    for i, id in enumerate(ext_ext_df[target].unique()):
                        rating_dict[id].append(1500)
                        rd_dict[id].append(350)
                        vol_dict[id].append(0.06)

            # レーティングをDFに反映
            # dfのshapeが0でない時のみ下記の処理を実行
            if ext_df.shape[0] > 0:
                # IDと行番号でインデックスを作成するための準備
                ext_df["row_num"] = ext_df.groupby(target).cumcount()
                # 新しいデータフレームを作成し、リストを行に展開
                update_rows = []
                for id, ratings, rds, vols in zip(
                    rating_dict.keys(),
                    rating_dict.values(),
                    rd_dict.values(),
                    vol_dict.values(),
                ):
                    for idx, (rating, rd, vol) in enumerate(
                        zip(ratings[:-1], rds[:-1], vols[:-1])
                    ):
                        update_rows.append(
                            {
                                target: id,
                                "row_num": idx,
                                rating_key: rating,
                                rd_key: rd,
                                vol_key: vol,
                            }
                        )
                updates_df = pd.DataFrame(update_rows)
                # IDとrow_numをインデックスとして設定
                ext_df.set_index([target, "row_num"], inplace=True)
                updates_df.set_index([target, "row_num"], inplace=True)
                # 更新
                ext_df.update(updates_df)
                # インデックスのリセット
                ext_df.reset_index(inplace=True)
                ext_df.drop(columns="row_num", inplace=True)

            ext_df_list.append(ext_df)

    # 結合
    df = pd.concat(ext_df_list, axis=0)
    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    # idごとにグループ化してレーティングをffill
    if who == "競走馬":
        all_columns = list(set(all_columns))
        temp_df = df[["horse_id"] + all_columns].copy()
        df[all_columns] = temp_df.groupby("horse_id")[all_columns].ffill()
    elif who == "騎手":
        all_columns = list(set(all_columns))
        temp_df = df[["jockey_id"] + all_columns].copy()
        df[all_columns] = temp_df.groupby("jockey_id")[all_columns].ffill()

    print("処理後DF：", df.shape)
    # df = reduce_memory_usage(df)
    # display(df[df['馬名'] == 'イクイノックス'])

    return df


@measure_execution_time
def calc_jockey_experience(df):
    """騎手の経験年数を計算する関数"""
    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    print("------騎手の経験年数を計算します。------")
    print("処理前DF：", df.shape)

    # データの最古年を取得
    df["year"] = df["date"].dt.year
    min_year = df["year"].min()

    # 騎手ごとに最初に出走した年を取得
    jockey_first_year = df.groupby("jockey_id")["year"].min()

    # 経験年数を計算
    df["騎手経験年数"] = df["year"] - df["jockey_id"].map(jockey_first_year)

    # 最古年の騎手の経験年数を0に設定
    df.loc[df["year"] == min_year, "騎手経験年数"] = 0

    df = df.drop(columns="year")

    print("処理後DF：", df.shape)
    return df


@measure_execution_time
def extract_race_features_by_conditions(df):
    """条件ごとのレース特徴量を抽出する関数

    Args:
        df (_type_): _description_
    """
    print("------条件ごとのレース特徴量を抽出中------")
    print("処理前DF：", df.shape)
    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    # 条件処理の為のリストを作成
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
    racecourse_list = [
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
    dist_list = [
        1000,
        1150,
        1200,
        1300,
        1400,
        1500,
        1600,
        1700,
        1800,
        1900,
        2000,
        2100,
        2200,
        2300,
        2400,
        2500,
        2600,
        3000,
        3200,
        3400,
        3600,
    ]

    # 次の処理のために距離分類を列に追加
    dist_dict = {
        "短距離": [1000, 1300],
        "マイル": [1301, 1899],
        "中距離": [1900, 2100],
        "クラシック": [2101, 2700],
        "長距離": [2700, 4000],
    }

    # 距離列の値から該当する距離分類を取り出して格納する関数
    def classify_distance(dist):
        for key, (min_dist, max_dist) in dist_dict.items():
            if min_dist <= dist <= max_dist:
                return key
        return None

    # 距離分類列の情報を取り出し格納
    df["距離分類"] = df["距離"].apply(classify_distance)

    # 前後の距離分類成績も加味するために距離分類の順序リストを作成
    dist_order = ["短距離", "マイル", "中距離", "クラシック", "長距離"]
    next_order = ["マイル", "中距離", "クラシック", "長距離", "長距離"]
    prev_order = ["短距離", "短距離", "マイル", "中距離", "クラシック"]

    # 距離分類より一つ長い、または短い分類を決定する関数
    def find_next_prev_dist_class(dist_class, dist_order):
        if dist_class not in dist_order:
            return None, None
        idx = dist_order.index(dist_class)
        next_dist = dist_order[idx + 1] if idx + 1 < len(dist_order) else None
        prev_dist = dist_order[idx - 1] if idx - 1 >= 0 else None
        return next_dist, prev_dist

    # 距離分類に基づいて次の長い、前後の距離分類の特徴量を追加
    df["次の距離分類"] = df["距離分類"].apply(
        lambda x: find_next_prev_dist_class(x, dist_order)[0]
    )
    df["前の距離分類"] = df["距離分類"].apply(
        lambda x: find_next_prev_dist_class(x, dist_order)[1]
    )

    # 競走馬について
    # 競走馬成績を反映
    print("競走馬成績を反映中...")

    # コース種類に基づく条件リスト
    conditions = [df["コース種類"] == "芝", df["コース種類"] == "ダート"]
    # 各計算に対する選択肢リストを定義
    choice_race_count = [df["競走馬芝通算試合数"], df["競走馬ダート通算試合数"]]
    choices_win_rate = [df["競走馬芝通算勝率"], df["競走馬ダート通算勝率"]]
    choices_place_rate = [df["競走馬芝通算複勝率"], df["競走馬ダート通算複勝率"]]
    choice_single_race_count = [
        df["単年競走馬芝通算試合数"],
        df["単年競走馬ダート通算試合数"],
    ]
    choices_single_year_win_rate = [
        df["単年競走馬芝通算勝率"],
        df["単年競走馬ダート通算勝率"],
    ]
    choices_single_year_place_rate = [
        df["単年競走馬芝通算複勝率"],
        df["単年競走馬ダート通算複勝率"],
    ]
    df["当該コース競走馬通算試合数"] = np.select(
        conditions, choice_race_count, default=np.nan
    )
    df["当該コース競走馬通算勝率"] = np.select(
        conditions, choices_win_rate, default=np.nan
    )
    df["当該コース競走馬通算複勝率"] = np.select(
        conditions, choices_place_rate, default=np.nan
    )
    df["当該コース単年競走馬通算試合数"] = np.select(
        conditions, choice_single_race_count, default=np.nan
    )
    df["当該コース単年競走馬通算勝率"] = np.select(
        conditions, choices_single_year_win_rate, default=np.nan
    )
    df["当該コース単年競走馬通算複勝率"] = np.select(
        conditions, choices_single_year_place_rate, default=np.nan
    )

    # クラス成績を反映
    print("クラス成績を反映中...")
    # 条件リストを生成
    conditions = [
        (df["コース種類"] + df["グレード"] == course_type + grade)
        for course_type in ["芝", "ダート"]
        for grade in grade_list
    ]
    # 各成績データに対する選択肢リストを生成
    choice_race_count = [
        df["競走馬" + course_type + grade + "通算試合数"]
        for course_type in ["芝", "ダート"]
        for grade in grade_list
    ]
    choices_win_rate = [
        df["競走馬" + course_type + grade + "通算勝率"]
        for course_type in ["芝", "ダート"]
        for grade in grade_list
    ]
    choices_place_rate = [
        df["競走馬" + course_type + grade + "通算複勝率"]
        for course_type in ["芝", "ダート"]
        for grade in grade_list
    ]
    choice_single_race_count = [
        df["単年競走馬" + course_type + grade + "通算試合数"]
        for course_type in ["芝", "ダート"]
        for grade in grade_list
    ]
    choices_single_year_win_rate = [
        df["単年競走馬" + course_type + grade + "通算勝率"]
        for course_type in ["芝", "ダート"]
        for grade in grade_list
    ]
    choices_single_year_place_rate = [
        df["単年競走馬" + course_type + grade + "通算複勝率"]
        for course_type in ["芝", "ダート"]
        for grade in grade_list
    ]
    df["当該グレード競走馬通算試合数"] = np.select(
        conditions, choice_race_count, default=np.nan
    )
    df["当該グレード競走馬通算勝率"] = np.select(
        conditions, choices_win_rate, default=np.nan
    )
    df["当該グレード競走馬通算複勝率"] = np.select(
        conditions, choices_place_rate, default=np.nan
    )
    df["当該グレード単年競走馬通算試合数"] = np.select(
        conditions, choice_single_race_count, default=np.nan
    )
    df["当該グレード単年競走馬通算勝率"] = np.select(
        conditions, choices_single_year_win_rate, default=np.nan
    )
    df["当該グレード単年競走馬通算複勝率"] = np.select(
        conditions, choices_single_year_place_rate, default=np.nan
    )

    # 重賞成績を反映
    print("重賞成績を反映中...")
    # 条件リストを生成
    conditions = [df["コース種類"] == "芝", df["コース種類"] == "ダート"]
    # 各成績データに対する選択肢リストを生成
    choice_race_count = [df["競走馬芝重賞通算試合数"], df["競走馬ダート重賞通算試合数"]]
    choices_win_rate = [df["競走馬芝重賞通算勝率"], df["競走馬ダート重賞通算勝率"]]
    choices_place_rate = [
        df["競走馬芝重賞通算複勝率"],
        df["競走馬ダート重賞通算複勝率"],
    ]
    choice_single_race_count = [
        df["単年競走馬芝重賞通算試合数"],
        df["単年競走馬ダート重賞通算試合数"],
    ]
    choices_single_year_win_rate = [
        df["単年競走馬芝重賞通算勝率"],
        df["単年競走馬ダート重賞通算勝率"],
    ]
    choices_single_year_place_rate = [
        df["単年競走馬芝重賞通算複勝率"],
        df["単年競走馬ダート重賞通算複勝率"],
    ]
    df["当該コース重賞競走馬通算試合数"] = np.select(
        conditions, choice_race_count, default=np.nan
    )
    df["当該コース重賞競走馬通算勝率"] = np.select(
        conditions, choices_win_rate, default=np.nan
    )
    df["当該コース重賞競走馬通算複勝率"] = np.select(
        conditions, choices_place_rate, default=np.nan
    )
    df["当該コース重賞単年競走馬通算試合数"] = np.select(
        conditions, choice_single_race_count, default=np.nan
    )
    df["当該コース重賞単年競走馬通算勝率"] = np.select(
        conditions, choices_single_year_win_rate, default=np.nan
    )
    df["当該コース重賞単年競走馬通算複勝率"] = np.select(
        conditions, choices_single_year_place_rate, default=np.nan
    )

    # 距離成績を反映
    print("距離成績を反映中...")
    # 条件リストを生成
    conditions = [
        (df["コース種類"] + df["距離分類"] == course_type + dist)
        for course_type in ["芝", "ダート"]
        for dist in dist_order
    ]
    # 各成績データに対する選択肢リストを生成
    choice_race_count = [
        df["競走馬" + course_type + dist + "通算試合数"]
        for course_type in ["芝", "ダート"]
        for dist in dist_order
    ]
    choices_win_rate = [
        df["競走馬" + course_type + dist + "通算勝率"]
        for course_type in ["芝", "ダート"]
        for dist in dist_order
    ]
    choices_dist_rate = [
        df["競走馬" + course_type + dist + "通算複勝率"]
        for course_type in ["芝", "ダート"]
        for dist in dist_order
    ]
    choice_single_race_count = [
        df["単年競走馬" + course_type + dist + "通算試合数"]
        for course_type in ["芝", "ダート"]
        for dist in dist_order
    ]
    choices_single_year_win_rate = [
        df["単年競走馬" + course_type + dist + "通算勝率"]
        for course_type in ["芝", "ダート"]
        for dist in dist_order
    ]
    choices_single_year_dist_rate = [
        df["単年競走馬" + course_type + dist + "通算複勝率"]
        for course_type in ["芝", "ダート"]
        for dist in dist_order
    ]
    choices_rating = [
        df["競走馬" + course_type + dist + "レーティング"]
        for course_type in ["芝", "ダート"]
        for dist in dist_order
    ]
    choices_rating_sigma = [
        df["競走馬" + course_type + dist + "レーティング偏差"]
        for course_type in ["芝", "ダート"]
        for dist in dist_order
    ]
    df["当該距離分類競走馬通算試合数"] = np.select(
        conditions, choice_race_count, default=np.nan
    )
    df["当該距離分類競走馬通算勝率"] = np.select(
        conditions, choices_win_rate, default=np.nan
    )
    df["当該距離分類競走馬通算複勝率"] = np.select(
        conditions, choices_dist_rate, default=np.nan
    )
    df["当該距離分類単年競走馬通算試合数"] = np.select(
        conditions, choice_single_race_count, default=np.nan
    )
    df["当該距離分類単年競走馬通算勝率"] = np.select(
        conditions, choices_single_year_win_rate, default=np.nan
    )
    df["当該距離分類単年競走馬通算複勝率"] = np.select(
        conditions, choices_single_year_dist_rate, default=np.nan
    )
    df["当該距離分類競走馬レーティング"] = np.select(
        conditions, choices_rating, default=np.nan
    )
    df["当該距離分類競走馬レーティング偏差"] = np.select(
        conditions, choices_rating_sigma, default=np.nan
    )

    # 次の距離分類、前の距離分類に対応する特徴量を追加
    print("次の距離分類、前の距離分類に対応する特徴量を追加中...")
    # 条件リストを生成
    conditions = [
        (df["コース種類"] + df["距離分類"] == course_type + dist)
        for course_type in ["芝", "ダート"]
        for dist in dist_order
    ]
    # 各成績データに対する選択肢リストを生成
    choices_next_win_rate = [
        df["競走馬" + course_type + dist + "通算勝率"]
        for course_type in ["芝", "ダート"]
        for dist in next_order
    ]
    choices_prev_win_rate = [
        df["競走馬" + course_type + dist + "通算勝率"]
        for course_type in ["芝", "ダート"]
        for dist in prev_order
    ]
    choices_next_dist_rate = [
        df["競走馬" + course_type + dist + "通算複勝率"]
        for course_type in ["芝", "ダート"]
        for dist in next_order
    ]
    choices_prev_dist_rate = [
        df["競走馬" + course_type + dist + "通算複勝率"]
        for course_type in ["芝", "ダート"]
        for dist in prev_order
    ]
    choices_next_rating = [
        df["競走馬" + course_type + dist + "レーティング"]
        for course_type in ["芝", "ダート"]
        for dist in next_order
    ]
    choices_prev_rating = [
        df["競走馬" + course_type + dist + "レーティング"]
        for course_type in ["芝", "ダート"]
        for dist in prev_order
    ]
    df["次の距離分類競走馬通算勝率"] = np.select(
        conditions, choices_next_win_rate, default=np.nan
    )
    df["前の距離分類競走馬通算勝率"] = np.select(
        conditions, choices_prev_win_rate, default=np.nan
    )
    df["次の距離分類競走馬通算複勝率"] = np.select(
        conditions, choices_next_dist_rate, default=np.nan
    )
    df["前の距離分類競走馬通算複勝率"] = np.select(
        conditions, choices_prev_dist_rate, default=np.nan
    )
    df["次の距離分類競走馬レーティング"] = np.select(
        conditions, choices_next_rating, default=np.nan
    )
    df["前の距離分類競走馬レーティング"] = np.select(
        conditions, choices_prev_rating, default=np.nan
    )

    # 競馬場成績を反映
    print("競馬場成績を反映中...")
    # 条件リストを生成
    conditions = [
        (df["コース種類"] + df["競馬場"] == course_type + racecourse)
        for course_type in ["芝", "ダート"]
        for racecourse in racecourse_list
    ]
    # 各成績データに対する選択肢リストを生成
    choice_race_count = [
        df["競走馬" + course_type + racecourse + "通算試合数"]
        for course_type in ["芝", "ダート"]
        for racecourse in racecourse_list
    ]
    choices_win_rate = [
        df["競走馬" + course_type + racecourse + "通算勝率"]
        for course_type in ["芝", "ダート"]
        for racecourse in racecourse_list
    ]
    choices_racecourse_rate = [
        df["競走馬" + course_type + racecourse + "通算複勝率"]
        for course_type in ["芝", "ダート"]
        for racecourse in racecourse_list
    ]
    choice_single_race_count = [
        df["単年競走馬" + course_type + racecourse + "通算試合数"]
        for course_type in ["芝", "ダート"]
        for racecourse in racecourse_list
    ]
    choices_single_year_win_rate = [
        df["単年競走馬" + course_type + racecourse + "通算勝率"]
        for course_type in ["芝", "ダート"]
        for racecourse in racecourse_list
    ]
    choices_single_year_racecourse_rate = [
        df["単年競走馬" + course_type + racecourse + "通算複勝率"]
        for course_type in ["芝", "ダート"]
        for racecourse in racecourse_list
    ]
    choices_rating = [
        df["競走馬" + course_type + racecourse + "レーティング"]
        for course_type in ["芝", "ダート"]
        for racecourse in racecourse_list
    ]
    choices_rating_sigma = [
        df["競走馬" + course_type + racecourse + "レーティング偏差"]
        for course_type in ["芝", "ダート"]
        for racecourse in racecourse_list
    ]
    df["当該競馬場競走馬通算試合数"] = np.select(
        conditions, choice_race_count, default=np.nan
    )
    df["当該競馬場競走馬通算勝率"] = np.select(
        conditions, choices_win_rate, default=np.nan
    )
    df["当該競馬場競走馬通算複勝率"] = np.select(
        conditions, choices_racecourse_rate, default=np.nan
    )
    df["当該競馬場単年競走馬通算試合数"] = np.select(
        conditions, choice_single_race_count, default=np.nan
    )
    df["当該競馬場単年競走馬通算勝率"] = np.select(
        conditions, choices_single_year_win_rate, default=np.nan
    )
    df["当該競馬場単年競走馬通算複勝率"] = np.select(
        conditions, choices_single_year_racecourse_rate, default=np.nan
    )
    df["当該競馬場競走馬レーティング"] = np.select(
        conditions, choices_rating, default=np.nan
    )
    df["当該競馬場競走馬レーティング偏差"] = np.select(
        conditions, choices_rating_sigma, default=np.nan
    )

    # 速度系成績を反映
    print("速度系成績を反映中...")
    # 条件リストを生成
    conditions = [df["競馬場"] == racecourse for racecourse in racecourse_list]
    # 各成績データに対する選択肢リストを生成
    avg_3f = [df[racecourse + "平均上がり3F"] for racecourse in racecourse_list]
    min_3f = [df[racecourse + "最高上がり3F"] for racecourse in racecourse_list]
    df["当該競馬場平均上がり3F"] = np.select(conditions, avg_3f, default=np.nan)
    df["当該競馬場最高上がり3F"] = np.select(conditions, min_3f, default=np.nan)
    # 条件リストを作成
    conditions = [df["距離"] == dist for dist in dist_list]
    # 各成績データに対する選択肢リストを生成
    avg_3f = [df[str(dist) + "平均上がり3F"] for dist in dist_list]
    min_3f = [df[str(dist) + "最高上がり3F"] for dist in dist_list]
    avg_time = [df[str(dist) + "平均タイム"] for dist in dist_list]
    min_time = [df[str(dist) + "最高タイム"] for dist in dist_list]
    df["当該距離平均上がり3F"] = np.select(conditions, avg_3f, default=np.nan)
    df["当該距離最高上がり3F"] = np.select(conditions, min_3f, default=np.nan)
    df["当該距離平均タイム"] = np.select(conditions, avg_time, default=np.nan)
    df["当該距離最高タイム"] = np.select(conditions, min_time, default=np.nan)
    # 条件リストを作成
    conditions = [
        df["競馬場"] + df["距離"].astype(int).astype(str) == racecourse + str(dist)
        for racecourse in racecourse_list
        for dist in dist_list
    ]
    # 各成績データに対する選択肢リストを生成
    avg_3f = [
        df[racecourse + str(dist) + "平均上がり3F"]
        for racecourse in racecourse_list
        for dist in dist_list
    ]
    min_3f = [
        df[racecourse + str(dist) + "最高上がり3F"]
        for racecourse in racecourse_list
        for dist in dist_list
    ]
    avg_time = [
        df[racecourse + str(dist) + "平均タイム"]
        for racecourse in racecourse_list
        for dist in dist_list
    ]
    min_time = [
        df[racecourse + str(dist) + "最高タイム"]
        for racecourse in racecourse_list
        for dist in dist_list
    ]
    df["当該競馬場距離平均上がり3F"] = np.select(conditions, avg_3f, default=np.nan)
    df["当該競馬場距離最高上がり3F"] = np.select(conditions, min_3f, default=np.nan)
    df["当該競馬場距離平均タイム"] = np.select(conditions, avg_time, default=np.nan)
    df["当該競馬場距離最高タイム"] = np.select(conditions, min_time, default=np.nan)

    # 騎手について
    # 騎手成績を反映
    print("騎手成績を反映中...")
    # コース種類に基づく条件リスト
    conditions = [df["コース種類"] == "芝", df["コース種類"] == "ダート"]
    # 各計算に対する選択肢リストを定義
    choice_race_count = [df["騎手芝通算試合数"], df["騎手ダート通算試合数"]]
    choices_win_rate = [df["騎手芝通算勝率"], df["騎手ダート通算勝率"]]
    choices_place_rate = [df["騎手芝通算複勝率"], df["騎手ダート通算複勝率"]]
    choice_single_race_count = [
        df["単年騎手芝通算試合数"],
        df["単年騎手ダート通算試合数"],
    ]
    choices_single_year_win_rate = [
        df["単年騎手芝通算勝率"],
        df["単年騎手ダート通算勝率"],
    ]
    choices_single_year_place_rate = [
        df["単年騎手芝通算複勝率"],
        df["単年騎手ダート通算複勝率"],
    ]
    df["当該コース騎手通算試合数"] = np.select(
        conditions, choice_race_count, default=np.nan
    )
    df["当該コース騎手通算勝率"] = np.select(
        conditions, choices_win_rate, default=np.nan
    )
    df["当該コース騎手通算複勝率"] = np.select(
        conditions, choices_place_rate, default=np.nan
    )
    df["当該コース単年騎手通算試合数"] = np.select(
        conditions, choice_single_race_count, default=np.nan
    )
    df["当該コース単年騎手通算勝率"] = np.select(
        conditions, choices_single_year_win_rate, default=np.nan
    )
    df["当該コース単年騎手通算複勝率"] = np.select(
        conditions, choices_single_year_place_rate, default=np.nan
    )

    # クラス成績を反映
    print("クラス成績を反映中...")
    # 条件リストを生成
    conditions = [
        (df["コース種類"] + df["グレード"] == course_type + grade)
        for course_type in ["芝", "ダート"]
        for grade in grade_list
    ]
    # 各成績データに対する選択肢リストを生成
    choice_race_count = [
        df["騎手" + course_type + grade + "通算試合数"]
        for course_type in ["芝", "ダート"]
        for grade in grade_list
    ]
    choices_win_rate = [
        df["騎手" + course_type + grade + "通算勝率"]
        for course_type in ["芝", "ダート"]
        for grade in grade_list
    ]
    choices_place_rate = [
        df["騎手" + course_type + grade + "通算複勝率"]
        for course_type in ["芝", "ダート"]
        for grade in grade_list
    ]
    choice_single_race_count = [
        df["単年騎手" + course_type + grade + "通算試合数"]
        for course_type in ["芝", "ダート"]
        for grade in grade_list
    ]
    choices_single_year_win_rate = [
        df["単年騎手" + course_type + grade + "通算勝率"]
        for course_type in ["芝", "ダート"]
        for grade in grade_list
    ]
    choices_single_year_place_rate = [
        df["単年騎手" + course_type + grade + "通算複勝率"]
        for course_type in ["芝", "ダート"]
        for grade in grade_list
    ]
    df["当該グレード騎手通算試合数"] = np.select(
        conditions, choice_race_count, default=np.nan
    )
    df["当該グレード騎手通算勝率"] = np.select(
        conditions, choices_win_rate, default=np.nan
    )
    df["当該グレード騎手通算複勝率"] = np.select(
        conditions, choices_place_rate, default=np.nan
    )
    df["当該グレード単年騎手通算試合数"] = np.select(
        conditions, choice_single_race_count, default=np.nan
    )
    df["当該グレード単年騎手通算勝率"] = np.select(
        conditions, choices_single_year_win_rate, default=np.nan
    )
    df["当該グレード単年騎手通算複勝率"] = np.select(
        conditions, choices_single_year_place_rate, default=np.nan
    )

    # 重賞成績を反映
    print("重賞成績を反映中...")
    # 条件リストを生成
    conditions = [df["コース種類"] == "芝", df["コース種類"] == "ダート"]
    # 各成績データに対する選択肢リストを生成
    choice_race_count = [df["騎手芝重賞通算試合数"], df["騎手ダート重賞通算試合数"]]
    choices_win_rate = [df["騎手芝重賞通算勝率"], df["騎手ダート重賞通算勝率"]]
    choices_place_rate = [df["騎手芝重賞通算複勝率"], df["騎手ダート重賞通算複勝率"]]
    choice_single_race_count = [
        df["単年騎手芝重賞通算試合数"],
        df["単年騎手ダート重賞通算試合数"],
    ]
    choices_single_year_win_rate = [
        df["単年騎手芝重賞通算勝率"],
        df["単年騎手ダート重賞通算勝率"],
    ]
    choices_single_year_place_rate = [
        df["単年騎手芝重賞通算複勝率"],
        df["単年騎手ダート重賞通算複勝率"],
    ]
    df["当該重賞騎手通算試合数"] = np.select(
        conditions, choice_race_count, default=np.nan
    )
    df["当該重賞騎手通算勝率"] = np.select(conditions, choices_win_rate, default=np.nan)
    df["当該重賞騎手通算複勝率"] = np.select(
        conditions, choices_place_rate, default=np.nan
    )
    df["当該重賞単年騎手通算試合数"] = np.select(
        conditions, choice_single_race_count, default=np.nan
    )
    df["当該重賞単年騎手通算勝率"] = np.select(
        conditions, choices_single_year_win_rate, default=np.nan
    )
    df["当該重賞単年騎手通算複勝率"] = np.select(
        conditions, choices_single_year_place_rate, default=np.nan
    )

    # 距離成績を反映
    print("距離成績を反映中...")
    # 条件リストを生成
    conditions = [
        (df["コース種類"] + df["距離分類"] == course_type + dist)
        for course_type in ["芝", "ダート"]
        for dist in dist_order
    ]
    # 各成績データに対する選択肢リストを生成
    choice_race_count = [
        df["騎手" + course_type + dist + "通算試合数"]
        for course_type in ["芝", "ダート"]
        for dist in dist_order
    ]
    choices_win_rate = [
        df["騎手" + course_type + dist + "通算勝率"]
        for course_type in ["芝", "ダート"]
        for dist in dist_order
    ]
    choices_dist_rate = [
        df["騎手" + course_type + dist + "通算複勝率"]
        for course_type in ["芝", "ダート"]
        for dist in dist_order
    ]
    choice_single_race_count = [
        df["単年騎手" + course_type + dist + "通算試合数"]
        for course_type in ["芝", "ダート"]
        for dist in dist_order
    ]
    choices_single_year_win_rate = [
        df["単年騎手" + course_type + dist + "通算勝率"]
        for course_type in ["芝", "ダート"]
        for dist in dist_order
    ]
    choices_single_year_dist_rate = [
        df["単年騎手" + course_type + dist + "通算複勝率"]
        for course_type in ["芝", "ダート"]
        for dist in dist_order
    ]
    choices_rating = [
        df["騎手" + course_type + dist + "レーティング"]
        for course_type in ["芝", "ダート"]
        for dist in dist_order
    ]
    choices_rating_sigma = [
        df["騎手" + course_type + dist + "レーティング偏差"]
        for course_type in ["芝", "ダート"]
        for dist in dist_order
    ]
    df["当該距離分類騎手通算試合数"] = np.select(
        conditions, choice_race_count, default=np.nan
    )
    df["当該距離分類騎手通算勝率"] = np.select(
        conditions, choices_win_rate, default=np.nan
    )
    df["当該距離分類騎手通算複勝率"] = np.select(
        conditions, choices_dist_rate, default=np.nan
    )
    df["当該距離分類単年騎手通算試合数"] = np.select(
        conditions, choice_single_race_count, default=np.nan
    )
    df["当該距離分類単年騎手通算勝率"] = np.select(
        conditions, choices_single_year_win_rate, default=np.nan
    )
    df["当該距離分類単年騎手通算複勝率"] = np.select(
        conditions, choices_single_year_dist_rate, default=np.nan
    )
    df["当該距離分類騎手レーティング"] = np.select(
        conditions, choices_rating, default=np.nan
    )
    df["当該距離分類騎手レーティング偏差"] = np.select(
        conditions, choices_rating_sigma, default=np.nan
    )

    # 次の距離分類、前の距離分類に対応する特徴量を追加
    print("次の距離分類、前の距離分類に対応する特徴量を追加中...")
    # 条件リストを生成
    conditions = [
        (df["コース種類"] + df["距離分類"] == course_type + dist)
        for course_type in ["芝", "ダート"]
        for dist in dist_order
    ]
    # 各成績データに対する選択肢リストを生成
    choices_next_win_rate = [
        df["騎手" + course_type + dist + "通算勝率"]
        for course_type in ["芝", "ダート"]
        for dist in next_order
    ]
    choices_prev_win_rate = [
        df["騎手" + course_type + dist + "通算勝率"]
        for course_type in ["芝", "ダート"]
        for dist in prev_order
    ]
    choices_next_dist_rate = [
        df["騎手" + course_type + dist + "通算複勝率"]
        for course_type in ["芝", "ダート"]
        for dist in next_order
    ]
    choices_prev_dist_rate = [
        df["騎手" + course_type + dist + "通算複勝率"]
        for course_type in ["芝", "ダート"]
        for dist in prev_order
    ]
    choices_next_rating = [
        df["騎手" + course_type + dist + "レーティング"]
        for course_type in ["芝", "ダート"]
        for dist in next_order
    ]
    choices_prev_rating = [
        df["騎手" + course_type + dist + "レーティング"]
        for course_type in ["芝", "ダート"]
        for dist in prev_order
    ]
    df["次の距離分類騎手通算勝率"] = np.select(
        conditions, choices_next_win_rate, default=np.nan
    )
    df["前の距離分類騎手通算勝率"] = np.select(
        conditions, choices_prev_win_rate, default=np.nan
    )
    df["次の距離分類騎手通算複勝率"] = np.select(
        conditions, choices_next_dist_rate, default=np.nan
    )
    df["前の距離分類騎手通算複勝率"] = np.select(
        conditions, choices_prev_dist_rate, default=np.nan
    )
    df["次の距離分類騎手レーティング"] = np.select(
        conditions, choices_next_rating, default=np.nan
    )
    df["前の距離分類騎手レーティング"] = np.select(
        conditions, choices_prev_rating, default=np.nan
    )

    # 競馬場成績を反映
    print("競馬場成績を反映中...")
    # 条件リストを生成
    conditions = [
        (df["コース種類"] + df["競馬場"] == course_type + racecourse)
        for course_type in ["芝", "ダート"]
        for racecourse in racecourse_list
    ]
    # 各成績データに対する選択肢リストを生成
    choice_race_count = [
        df["騎手" + course_type + racecourse + "通算試合数"]
        for course_type in ["芝", "ダート"]
        for racecourse in racecourse_list
    ]
    choices_win_rate = [
        df["騎手" + course_type + racecourse + "通算勝率"]
        for course_type in ["芝", "ダート"]
        for racecourse in racecourse_list
    ]
    choices_racecourse_rate = [
        df["騎手" + course_type + racecourse + "通算複勝率"]
        for course_type in ["芝", "ダート"]
        for racecourse in racecourse_list
    ]
    choice_single_race_count = [
        df["単年騎手" + course_type + racecourse + "通算試合数"]
        for course_type in ["芝", "ダート"]
        for racecourse in racecourse_list
    ]
    choices_single_year_win_rate = [
        df["単年騎手" + course_type + racecourse + "通算勝率"]
        for course_type in ["芝", "ダート"]
        for racecourse in racecourse_list
    ]
    choices_single_year_racecourse_rate = [
        df["単年騎手" + course_type + racecourse + "通算複勝率"]
        for course_type in ["芝", "ダート"]
        for racecourse in racecourse_list
    ]
    choices_rating = [
        df["騎手" + course_type + racecourse + "レーティング"]
        for course_type in ["芝", "ダート"]
        for racecourse in racecourse_list
    ]
    choices_rating_sigma = [
        df["騎手" + course_type + racecourse + "レーティング偏差"]
        for course_type in ["芝", "ダート"]
        for racecourse in racecourse_list
    ]
    df["当該競馬場騎手通算試合数"] = np.select(
        conditions, choice_race_count, default=np.nan
    )
    df["当該競馬場騎手通算勝率"] = np.select(
        conditions, choices_win_rate, default=np.nan
    )
    df["当該競馬場騎手通算複勝率"] = np.select(
        conditions, choices_racecourse_rate, default=np.nan
    )
    df["当該競馬場単年騎手通算試合数"] = np.select(
        conditions, choice_single_race_count, default=np.nan
    )
    df["当該競馬場単年騎手通算勝率"] = np.select(
        conditions, choices_single_year_win_rate, default=np.nan
    )
    df["当該競馬場単年騎手通算複勝率"] = np.select(
        conditions, choices_single_year_racecourse_rate, default=np.nan
    )
    df["当該競馬場騎手レーティング"] = np.select(
        conditions, choices_rating, default=np.nan
    )
    df["当該競馬場騎手レーティング偏差"] = np.select(
        conditions, choices_rating_sigma, default=np.nan
    )

    df = df.drop(columns=["距離分類", "次の距離分類", "前の距離分類"])
    # display(df[df['馬名'] == 'イクイノックス'])

    print("処理後DF：", df.shape)
    return df


@measure_execution_time
def set_past_form(df):
    """過去レース情報を記録する関数

    Args:
        df (dataframe): 過去成績が保存されたDF

    Returns:
        df (dataframe): 過去成績が保存されたDF
    """
    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    print("過去レース情報を記録します。")
    print("処理前DF：", df.shape)
    columns = [
        "着順",
        "タイム",
        "1着タイム差",
        "斤量",
        "競馬場",
        "天気",
        "コース種類",
        "馬場",
        "距離",
        "グレード",
        "年齢条件",
        "性別条件",
    ]

    # 必要な列のみを選択
    df_subset = df[["horse_id"] + columns]

    # シフト操作を適用
    for i in range(1, 6):
        shifted_df = df_subset.groupby("horse_id")[columns].shift(i)
        shifted_df.columns = [f"{i}走前{col}" for col in columns]
        df = pd.concat([df, shifted_df], axis=1)

    print("処理後DF：", df.shape)

    # df = reduce_memory_usage(df)
    # display(df[df['馬名'] == 'イクイノックス'])

    return df


@measure_execution_time
def apply_pca_for_past_form(df, n_components=20):
    """過去レース情報に対してPCAを適用する関数

    Args:
        df (_type_): _description_
        n_components (int, optional): _description_. Defaults to 20.

    Returns:
        _type_: _description_
    """

    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    print("------過去レース情報PCA処理開始------")
    print("処理前DF：", df.shape)

    # まずは過去レース情報の列名を取得
    past_form_columns = []
    for i in range(1, 6):
        for col in [
            "着順",
            "タイム",
            "1着タイム差",
            "斤量",
            "競馬場",
            "天気",
            "コース種類",
            "馬場",
            "距離",
            "グレード",
            "年齢条件",
            "性別条件",
        ]:
            past_form_columns.append(f"{i}走前{col}")

    # 次にPCAにかけるためにカテゴリエンコーディング
    # xfeatを使って型変換
    print("カテゴリ変数化前", df.shape)

    # 変換する列を取り出したデータフレームを作成
    sub_df = df[past_form_columns].copy()

    # カテゴリカルエンコーディング
    encoder = Pipeline(
        [
            SelectCategorical(),
            LabelEncoder(unseen="n_unique", output_suffix=""),
        ]
    )

    # エンコード実行
    encoded_df = encoder.fit_transform(sub_df)
    categorical_features = list(encoded_df.columns)

    # エンコードしたデータフレームを元のデータフレームから削除し、エンコードしたデータフレームを結合
    df = df.drop(columns=encoded_df.columns)
    df = pd.concat([df, encoded_df], axis=1, join="outer")
    df[past_form_columns] = df[past_form_columns].fillna(0)

    print("カテゴリ変数化後", df.shape)
    print("カテゴリ変数一覧", categorical_features)

    # PCAを実行する
    # 2022年のデータで学習し、2023年以降のデータは学習に使わない
    train_df = df[df["年"] < 2022].copy().reset_index(drop=True)
    test_df = df[df["年"] >= 2022].copy().reset_index(drop=True)

    # pcaの学習
    pca = PCA(n_components=n_components)
    pca.fit(train_df[past_form_columns])

    # 圧縮した特徴量のデータフレームを作成
    train_pca_df = pd.DataFrame(
        pca.transform(train_df[past_form_columns]),
        columns=[f"過去レース情報PCA_{i}" for i in range(n_components)],
    )
    test_pca_df = pd.DataFrame(
        pca.transform(test_df[past_form_columns]),
        columns=[f"過去レース情報PCA_{i}" for i in range(n_components)],
    )

    # データフレームに反映
    train_df = pd.concat([train_df, train_pca_df], axis=1)
    test_df = pd.concat([test_df, test_pca_df], axis=1)
    df = pd.concat([train_df, test_df], axis=0)
    df = df.reset_index(drop=True)

    print("中間処理中DF：", df.shape)

    # 圧縮した特徴量を削除
    df = df.drop(columns=past_form_columns)

    # 重複を削除
    df = df.drop_duplicates(subset=["race_id", "horse_id", "馬番"])
    df = df.sort_values(["date", "馬番"])

    print("処理後DF：", df.shape)
    # df = reduce_memory_usage(df)
    # display(df[df['馬名'] == 'イクイノックス'])

    return df


@measure_execution_time
def calc_grade_change(df):
    """グレード変動を計算する関数

    Args:
        df (_type_): _description_
    """
    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    print("------グレード変化処理開始------")
    print("処理前DF：", df.shape)

    # horse_idごとにグレード変化を計算
    temp_df = df[["horse_id", "グレード数値"]].copy()
    df["グレード変化"] = temp_df.groupby("horse_id")["グレード数値"].diff().fillna(0)

    print("処理後DF：", df.shape)

    return df


@measure_execution_time
def calc_changes_from_last_race(df):
    """前走からの斤量の変化、距離変化、休養日数を計算する関数"""
    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    print("------前走からの変化を計算します。------")
    print("処理前DF：", df.shape)

    # 前走の斤量、距離、日付を取得
    last_race_info = df.groupby("horse_id")[["斤量", "距離", "date"]].shift(1)
    last_race_info.columns = ["前走斤量", "前走距離", "前走日付"]

    # 前走からの変化量を計算
    df["斤量変化"] = df["斤量"] - last_race_info["前走斤量"]
    df["距離変化"] = df["距離"] - last_race_info["前走距離"]
    df["日数"] = (df["date"] - last_race_info["前走日付"]).dt.days

    print("処理後DF：", df.shape)

    return df


@measure_execution_time
def determine_running_style(df):
    """
    競走馬の脚質を判断する関数

    Args:
        df (dataframe): 過去のレースデータが含まれるデータフレーム

    Returns:
        df (dataframe): 脚質が追加されたデータフレーム
    """
    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    print("脚質を判断します。")
    print("処理前DF：", df.shape)

    # 新しい列「逃げ」、「先行」、「差し」、「追込」を追加
    df["逃げ"] = 0
    df["先行"] = 0
    df["差し"] = 0
    df["追込"] = 0

    # 各C（コーナー）での処理
    for n in range(1, 4):
        col = f"{n}C通過順位"

        # 通過順位が1なら「逃げ」列を+2にする
        df.loc[df[col] == 1, "逃げ"] += 1

        # 通過順位が1ではなく、「立て数」列の数値の0.4倍未満なら「先行」列を+1にする
        df.loc[(df[col] != 1) & (df[col] < df["立て数"] * 0.4), "先行"] += 1

        # 通過順位が1ではなく、「立て数」列の数値の0.4倍以上0.8倍未満なら「差し」列を+1にする
        df.loc[
            (df[col] != 1)
            & (df[col] >= df["立て数"] * 0.4)
            & (df[col] < df["立て数"] * 0.8),
            "差し",
        ] += 1

        # 通過順位が1ではなく、「立て数」列の数値の0.8倍以下なら「追込」列を+1にする
        df.loc[(df[col] != 1) & (df[col] >= df["立て数"] * 0.8), "追込"] += 1

    # horse_idごとにcumsumで足し合わせて一列ずらす
    df = df.sort_values(["date", "馬番"])
    temp_df = df[["horse_id", "逃げ", "先行", "差し", "追込"]].copy()
    temp_df[["逃げ", "先行", "差し", "追込"]] = temp_df.groupby("horse_id")[
        ["逃げ", "先行", "差し", "追込"]
    ].apply(lambda x: x.shift(1).fillna(0))
    df[["逃げ", "先行", "差し", "追込"]] = temp_df.groupby("horse_id")[
        ["逃げ", "先行", "差し", "追込"]
    ].cumsum()

    # 割合に変換する
    df["脚質合計"] = df["逃げ"] + df["先行"] + df["差し"] + df["追込"]
    df["逃げ"] = df["逃げ"] / df["脚質合計"]
    df["先行"] = df["先行"] / df["脚質合計"]
    df["差し"] = df["差し"] / df["脚質合計"]
    df["追込"] = df["追込"] / df["脚質合計"]

    df = df.drop(columns=["脚質合計"])

    print("処理後DF：", df.shape)
    # df = reduce_memory_usage(df)
    # display(df[df['馬名'] == 'イクイノックス'])

    return df


def create_pedigree_vector(df, model):
    """
    馬の血統ベクトルを作成し、データフレームに結合する関数。

    Args:
    df (pd.DataFrame): 元のデータフレーム。
    model: 馬の名前からベクトルを取得するモデル。

    Returns:
    pd.DataFrame: 血統ベクトルが結合されたデータフレーム。
    """
    # 馬名とベクトルの辞書を作成
    pedigree_vectors = {}

    # 各馬の血統ベクトルを計算
    for horse_name in df["馬名"].unique():
        vector = get_pedigree_vector(horse_name, model)
        pedigree_vectors[horse_name] = vector

    # データフレームにベクトルを結合
    df_vec = pd.DataFrame.from_dict(pedigree_vectors, orient="index")
    df_vec.reset_index(inplace=True)
    df_vec.rename(columns={"index": "馬名"}, inplace=True)

    # 元のデータフレームと結合
    df = pd.merge(df, df_vec, on="馬名", how="left")

    return df


def get_pedigree_vector(horse_name, model):
    """
    特定の馬の血統ベクトルを計算する関数。

    Args:
    horse_name (str): 対象の馬の名前。
    ped_df (pd.DataFrame): 血統情報が含まれるデータフレーム。
    model: 馬の名前からベクトルを取得するモデル。

    Returns:
    np.array: 計算された血統ベクトル。
    """
    try:
        vector = model[horse_name]
    except Exception:
        # nanを返す
        vector = [np.nan for i in range(15)]
        print(f"{horse_name}の血統ベクトルを取得できませんでした。")

    return vector


@measure_execution_time
def calc_pedigrees_vector(df):
    """血統ベクトルを作成する関数
    Args:
        df (dataframe): 血統データが含まれないデータフレーム

    Returns:
        df (dataframe): 血統ベクトルが追加されたデータフレーム
    """

    root_path = (
        "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2"
    )
    pedigree_directory = root_path + "/20_data_processing/processed_data/pedigree"

    # 実行結果を確認
    print("血統をベクトル化して結合します。")
    print("血統結合前:", df.shape)

    model = gensim.models.KeyedVectors.load_word2vec_format(
        pedigree_directory + "/fastText/pedigree_model.vec", binary=False
    )

    # 血統ベクトルをデータフレームにする
    df = create_pedigree_vector(df, model)

    print("血統結合後:", df.shape)
    # df = reduce_memory_usage(df)
    # display(df[df['馬名'] == 'イクイノックス'])

    return df


def calc_same_ped(df, ped):
    """同じ親の競走馬の成績を集計する関数"""
    # 特徴量のリスト
    feature_columns = [
        "賞金",
    ]  # ここに集計したい特徴量を追加

    # 空のDataFrameを用意
    aggregated_features = pd.DataFrame()

    # 計算用のdfをコピー
    ped_df = df[
        ["race_id", "date", "父", "母", "母父", "コース種類", "齢", "賞金"]
    ].copy()

    # 変更が必要な列のリスト
    feature_columns = ["賞金"]  # 適宜、変更が必要な列名に置き換えてください

    # 分類ごとにループ
    print("集計中...")
    if ped != "母":
        for (_, _, _), group in ped_df.groupby(["コース種類", "齢", ped]):
            group = group.sort_values("date")  # 日付でソート
            for feature in feature_columns:
                # 同じrace_idに同じ父親が複数頭出走している場合、その賞金を除外
                group[f"{feature}_exclude"] = (
                    group.duplicated(subset=["race_id", ped], keep=False)
                    * group[feature]
                )
                group[f"{feature}_include"] = (
                    group[feature] - group[f"{feature}_exclude"]
                )

                # race_idでグループ化し、グループ内の累積和を計算
                group[f"{ped}{feature}総額"] = group.groupby("race_id")[
                    f"{feature}_include"
                ].cumsum()

                # race_idが変わる箇所だけ、1つ前のrace_idグループの最終行の値を取得
                last_values = (
                    group.groupby("race_id")[f"{ped}{feature}総額"].last().shift()
                )

                # 累積和をシフトし、1つ前のrace_idグループの最終行の値で欠損値を埋める
                group[f"{ped}{feature}総額"] = (
                    group[f"{ped}{feature}総額"].shift().fillna(last_values)
                )
                group[f"{ped}{feature}総額"] = group[f"{ped}{feature}総額"].cumsum()

                # race_idでグループ化し、グループ内の累積和を計算（同じrace_idの賞金は除外）
                group[f"{ped}{feature}総額2"] = group.groupby("race_id")[
                    feature
                ].cumsum()
                group[f"{ped}{feature}総額2"] = (
                    group[f"{ped}{feature}総額2"] - group[feature]
                )

                # race_idが変わる箇所だけ、1つ前のrace_idグループの最終行の値を取得
                last_values = (
                    group.groupby("race_id")[f"{ped}{feature}総額2"].last().shift()
                )

                # 累積和をシフトし、1つ前のrace_idグループの最終行の値で欠損値を埋める
                group[f"{ped}{feature}総額2"] = (
                    group[f"{ped}{feature}総額2"].shift().fillna(last_values)
                )
                group[f"{ped}{feature}総額2"] = group[f"{ped}{feature}総額2"].cumsum()

                group[f"{ped}{feature}総額"] += group[f"{ped}{feature}総額2"]

                group = group[["race_id", "父", f"{ped}{feature}総額"]]
                group = group.drop_duplicates(subset=["race_id", "父"])

        aggregated_features = pd.concat([aggregated_features, group])

    else:
        for (_), group in ped_df.groupby([ped]):
            group = group.sort_values("date")  # 日付でソート
            for feature in feature_columns:
                # 同じrace_idに同じ父親が複数頭出走している場合、その賞金を除外
                group[f"{feature}_exclude"] = (
                    group.duplicated(subset=["race_id", ped], keep=False)
                    * group[feature]
                )
                group[f"{feature}_include"] = (
                    group[feature] - group[f"{feature}_exclude"]
                )

                # race_idでグループ化し、グループ内の累積和を計算
                group[f"{ped}{feature}総額"] = group.groupby("race_id")[
                    f"{feature}_include"
                ].cumsum()

                # race_idが変わる箇所だけ、1つ前のrace_idグループの最終行の値を取得
                last_values = (
                    group.groupby("race_id")[f"{ped}{feature}総額"].last().shift()
                )

                # 累積和をシフトし、1つ前のrace_idグループの最終行の値で欠損値を埋める
                group[f"{ped}{feature}総額"] = (
                    group[f"{ped}{feature}総額"].shift().fillna(last_values)
                )
                group[f"{ped}{feature}総額"] = group[f"{ped}{feature}総額"].cumsum()

                # race_idでグループ化し、グループ内の累積和を計算（同じrace_idの賞金は除外）
                group[f"{ped}{feature}総額2"] = group.groupby("race_id")[
                    feature
                ].cumsum()
                group[f"{ped}{feature}総額2"] = (
                    group[f"{ped}{feature}総額2"] - group[feature]
                )

                # race_idが変わる箇所だけ、1つ前のrace_idグループの最終行の値を取得
                last_values = (
                    group.groupby("race_id")[f"{ped}{feature}総額2"].last().shift()
                )

                # 累積和をシフトし、1つ前のrace_idグループの最終行の値で欠損値を埋める
                group[f"{ped}{feature}総額2"] = (
                    group[f"{ped}{feature}総額2"].shift().fillna(last_values)
                )
                group[f"{ped}{feature}総額2"] = group[f"{ped}{feature}総額2"].cumsum()

                group[f"{ped}{feature}総額"] += group[f"{ped}{feature}総額2"]

                group = group[["race_id", "父", f"{ped}{feature}総額"]]
                group = group.drop_duplicates(subset=["race_id", "父"])

        aggregated_features = pd.concat([aggregated_features, group])

    # 追加する特徴量の列名をリスト化
    added_feature_columns = []
    for feature in feature_columns:
        added_feature_columns.append(f"{ped}{feature}総額")

    # 集計した特徴量を元のデータフレームにマージ
    print("集計結果をマージします。")
    df = pd.merge(
        df,
        aggregated_features[["race_id"] + added_feature_columns],
        on="race_id",
        how="left",
    )

    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])

    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    return df


@measure_execution_time
def merge_pedigrees_infomation(df):
    """血統情報を結合する関数
    Args:
        df (dataframe): 血統データが含まれないデータフレーム
    Returns:
        df (dataframe): 血統情報が追加されたデータフレーム
    """
    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])

    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    print("------血統情報を結合します。------")
    print("処理前DF：", df.shape)

    # 血統データフレームを読み込む
    root_path = (
        "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2"
    )
    pedigree_directory = (
        root_path + "/20_data_processing/processed_data/pedigree/pedigree_horse_id.csv"
    )
    pedigree_df = pd.read_csv(pedigree_directory, encoding="utf_8_sig")

    # horse_id列の重複を削除
    pedigree_df = pedigree_df.drop_duplicates(subset=["horse_id"])
    pedigree_df = pedigree_df.reset_index(drop=True)

    # 必要な列のみを選択
    pedigree_df = pedigree_df[["horse_id", "pedigree_0", "pedigree_1", "pedigree_4"]]
    pedigree_df = pedigree_df.rename(
        columns={"pedigree_0": "父", "pedigree_1": "母", "pedigree_4": "母父"}
    )

    # horse_idをキーに血統情報を結合
    df = pd.merge(df, pedigree_df, on="horse_id", how="left")

    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])

    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    df = calc_same_ped(df, "父")
    df = calc_same_ped(df, "母")
    df = calc_same_ped(df, "母父")

    print("処理後DF：", df.shape)
    return df


@measure_execution_time
def calculate_pace(df):
    """前半と後半のタイムを計算し、ペースを判定する関数

    Args:
        df (dataframe): レースデータが含まれるデータフレーム

    Returns:
        df (dataframe): ペースが追加されたデータフレーム
    """
    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])

    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    print("------ペースを計算します。------")
    print("処理前DF：", df.shape)

    df["タイム差"] = df["前半ペース"] - df["後半ペース"]
    df.loc[df["タイム差"] >= 1, "ペース"] = 0
    df.loc[(df["タイム差"] < 1) & (df["タイム差"] > -1), "ペース"] = 1
    df.loc[df["タイム差"] <= -1, "ペース"] = 2

    # # ペース列をカテゴリ変数化
    # df['ペース'] = df['ペース'].astype('category')
    # 不要列を削除
    df = df.drop(columns=["タイム差"])

    print("処理後DF：", df.shape)

    return df


@measure_execution_time
def measure_race_upset(df):
    """レースの波乱度を計算する関数

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])

    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    print("------レースの波乱度を計算します。------")
    print("処理前DF：", df.shape)

    # レースIDでグループ化し、着順が1の行のみを抽出
    winner_odds = df[df["着順"] == 1].groupby("race_id")["人気"].first()

    # 結果をDataFrameに変換し、人気の列を波乱度にリネーム
    upset_degree_df = pd.DataFrame(winner_odds).reset_index()
    upset_degree_df = upset_degree_df.rename(columns={"人気": "波乱度"})
    upset_degree_df = upset_degree_df[["race_id", "波乱度"]]

    # レースIDをキーに波乱度を結合
    df = pd.merge(df, upset_degree_df, on="race_id", how="left")

    # 波乱度を最高100最小0に正規化
    df["波乱度"] = (df["波乱度"] - 1) / (df["立て数"] - 1) * 100

    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])

    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    print("処理後DF：", df.shape)

    return df


@measure_execution_time
def calculate_win_horse_features(df):
    """勝利馬の特徴量を計算する関数zzz

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """

    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    print("------勝利馬の特徴量を計算します。------")
    print("処理前DF：", df.shape)

    df["quarter"] = df["date"].dt.quarter

    # 特徴量のリスト
    feature_columns = [
        "5着着差",
        "当該コース競走馬通算試合数",
        "当該コース競走馬通算勝率",
        "当該コース競走馬通算複勝率",
        "当該コース単年競走馬通算勝率",
        "当該コース単年競走馬通算複勝率",
        "当該グレード競走馬通算試合数",
        "当該グレード競走馬通算勝率",
        "当該グレード競走馬通算複勝率",
        "当該グレード単年競走馬通算勝率",
        "当該グレード単年競走馬通算複勝率",
        "当該距離分類競走馬通算試合数",
        "当該距離分類競走馬通算勝率",
        "当該距離分類競走馬通算複勝率",
        "当該距離分類単年競走馬通算勝率",
        "当該距離分類単年競走馬通算複勝率",
        "当該距離分類競走馬レーティング",
        "当該距離分類競走馬レーティング偏差",
        "次の距離分類競走馬通算勝率",
        "前の距離分類競走馬通算勝率",
        "次の距離分類競走馬通算複勝率",
        "前の距離分類競走馬通算複勝率",
        "当該競馬場競走馬通算試合数",
        "当該競馬場競走馬通算勝率",
        "当該競馬場競走馬通算複勝率",
        "当該競馬場単年競走馬通算勝率",
        "当該競馬場単年競走馬通算複勝率",
        "当該競馬場競走馬レーティング",
        "当該競馬場競走馬レーティング偏差",
        "当該コース重賞競走馬通算試合数",
        "当該コース重賞競走馬通算勝率",
        "当該コース重賞競走馬通算複勝率",
        "当該競馬場平均上がり3F",
        "当該競馬場最高上がり3F",
        "当該距離平均上がり3F",
        "当該距離最高上がり3F",
        "当該距離平均タイム",
        "当該距離最高タイム",
        "当該競馬場距離平均上がり3F",
        "当該競馬場距離最高上がり3F",
        "当該競馬場距離平均タイム",
        "当該競馬場距離最高タイム",
        "1走前着順",
        "2走前着順",
        "3走前着順",
        "4走前着順",
        "5走前着順",
        "1走前1着タイム差",
        "2走前1着タイム差",
        "3走前1着タイム差",
        "4走前1着タイム差",
        "5走前1着タイム差",
        "1走前距離",
        "2走前距離",
        "3走前距離",
        "4走前距離",
        "5走前距離",
        "平均連対距離",
        "平均上がり3F",
        "最高上がり3F",
        "グレード移動平均",
        "距離移動平均",
    ]  # ここに集計したい特徴量を追加

    # 着順が1のデータのみを使用して集計
    winning_horses = df[df["着順"] == 1]

    # 空のDataFrameを用意
    aggregated_features = pd.DataFrame()

    # 分類ごとにループ
    print("集計中...")
    for (_, _, _, _, _, _), group in winning_horses.groupby(
        ["コース種類", "性別条件", "年齢条件", "距離", "グレード", "quarter"]
    ):
        group = group.sort_values("date")  # 日付でソート
        for feature in feature_columns:
            group[f"勝利馬{feature}_mean"] = group[feature].expanding().mean().shift(1)
            group[f"勝利馬{feature}_std"] = group[feature].expanding().std().shift(1)
            group[f"勝利馬{feature}_max"] = group[feature].expanding().max().shift(1)
            group[f"勝利馬{feature}_min"] = group[feature].expanding().min().shift(1)
        aggregated_features = pd.concat([aggregated_features, group])
    # 追加する特徴量の列名をリスト化
    added_feature_columns = []
    for feature in feature_columns:
        for stat in ["mean", "std", "max", "min"]:
            added_feature_columns.append(f"勝利馬{feature}_{stat}")

    # 後のmake_input処理で使うために特徴量のリストを保存
    root_path = (
        "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2"
    )
    feature_columns = (
        root_path + "/20_data_processing/feature_data/aggregated_features.pickle"
    )
    with open(feature_columns, "wb") as f:
        pickle.dump(added_feature_columns, f)

    # 集計した特徴量を元のデータフレームにマージ
    print("集計結果をマージします。")
    df = pd.merge(
        df,
        aggregated_features[["race_id"] + added_feature_columns],
        on="race_id",
        how="left",
    )

    # 重複を削除
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    print("処理後DF：", df.shape)

    return df


@measure_execution_time
def calculate_post_position_stats(df):
    """馬番別の着順の平均値や勝率を計算する関数"""

    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    print("------枠番別の特徴量を計算します。------")
    print("処理前DF：", df.shape)

    # 1着の場合は1、それ以外は0とするwin列を作成
    df["win"] = df["着順"].apply(lambda x: 1 if x == 1 else 0)

    # 空のDataFrameを用意
    post_position_stats = pd.DataFrame()

    # 処理高速化のために必要なデータのみ取り出す
    ext_df = df[
        [
            "race_id",
            "馬番",
            "立て数",
            "コース種類",
            "競馬場",
            "距離",
            "着順",
            "win",
            "date",
        ]
    ].copy()

    # 分類ごとにループ
    print("集計中...")
    for (_, _, _, _, _), group in ext_df.groupby(
        ["馬番", "立て数", "コース種類", "競馬場", "距離"]
    ):
        group = group.sort_values("date")  # 日付でソート
        for feature in ["着順", "win"]:
            group[f"{feature}_mean"] = group[feature].expanding().mean().shift(1)
        post_position_stats = pd.concat([post_position_stats, group])

    # 特徴量の名前を変更
    post_position_stats = post_position_stats.rename(
        columns={"着順_mean": "馬番別平均着順", "win_mean": "馬番別勝率"}
    )

    # 集計した特徴量を元のデータフレームにマージ
    print("集計結果をマージします。")
    df = pd.merge(
        df,
        post_position_stats[
            ["race_id", "馬番", "立て数", "馬番別平均着順", "馬番別勝率"]
        ],
        on=["race_id", "馬番", "立て数"],
        how="left",
    )

    # 不要データを削除
    df = df.drop(columns=["win"])
    # 重複を削除
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    print("処理後DF：", df.shape)

    return df


@measure_execution_time
def make_target(df):
    """目的変数、期待値を作成する関数

    Args:
        df (dataframe): 目的変数を作成するデータフレーム

    Returns:
        df (dataframe): 目的変数が追加されたデータフレーム
    """
    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    print("------勝利馬の特徴量を計算します。------")
    print("処理前DF：", df.shape)

    print("目的変数を作成します。")

    df["期待値"] = df["単勝"]
    # 2着と3着は期待値を半分にする
    df.loc[df["着順"] == 2, "期待値"] = df["期待値"] / 2
    df.loc[df["着順"] == 3, "期待値"] = df["期待値"] / 2
    # 4着以下は期待値を0にする
    df.loc[df["着順"] > 3, "期待値"] = 0

    print("処理後DF：", df.shape)

    return df


@measure_execution_time
def make_place(df):
    """複勝列を作成する関数

    Args:
        df (_type_): _description_
    """
    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    print("------複勝列を作成します。------")
    print("処理前DF：", df.shape)

    # 複勝オッズデータを読み込む
    path = "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2/20_data_processing/processed_data/odds/odds_df.csv"
    odds_df = pd.read_csv(path, encoding="utf_8_sig")

    # 複勝部分を取り出す
    place_df = odds_df.loc[odds_df["券種"] == "複勝"]
    place_df = place_df[["馬番1", "払戻金額", "race_id"]]
    place_df["複勝"] = place_df["払戻金額"] / 100
    place_df = place_df.rename(columns={"馬番1": "馬番"})
    place_df = place_df.drop(columns=["払戻金額"])

    # レースIDをキーに複勝を結合
    df = pd.merge(df, place_df, on=["race_id", "馬番"], how="left")

    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    # 欠損値を0で埋める
    df["複勝"] = df["複勝"].fillna(0)

    print("処理後DF：", df.shape)

    return df


def run_feature():
    """特徴量を作成する関数"""

    # データ読み込み
    root_path = (
        "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2"
    )
    predict_df_path = (
        root_path
        + "/20_data_processing/all_data_including_future/race_result_info_performance_pace_df.csv"
    )
    predict_df = pd.read_csv(predict_df_path, encoding="utf_8_sig")

    # 日付処理
    predict_df = process_date(predict_df)

    # # 後ろ27行を削除
    # predict_df = predict_df.drop(predict_df.index[-30:])

    # predict_df = predict_df[(predict_df['馬名'] == 'イクイノックス') | (predict_df['馬名'] == 'オニャンコポン')].copy()

    # # 主戦騎手のフラグを追加
    # predict_df = add_main_jockey_flag(predict_df)

    # 通算成績を計算
    predict_df = calc_career_statics(predict_df, "競走馬")
    save_execution_times_to_csv()
    # データ量を削減する
    predict_df = reduce_memory_usage(predict_df)
    save_execution_times_to_csv()
    # predict_df = apply_pca_for_stats(predict_df, '競走馬', n_components=100)
    predict_df = calc_career_statics(predict_df, "騎手")
    save_execution_times_to_csv()
    # データ量を削減する
    predict_df = reduce_memory_usage(predict_df)
    save_execution_times_to_csv()
    # predict_df = apply_pca_for_stats(predict_df, '騎手', n_components=100)

    # 勝率計算
    predict_df = calc_win_rate(predict_df)
    save_execution_times_to_csv()
    # 騎手調教師コンビ成績を記録
    predict_df = calc_pair_win_rate(predict_df)
    save_execution_times_to_csv()

    # 平均連対距離を計算
    predict_df = calculate_average_top2_distance(predict_df)
    # 距離移動平均を計算
    predict_df = calculate_moving_average_distance(predict_df, window=5)
    # グレード移動平均を計算
    predict_df = calculate_moving_average_grade(predict_df, window=5)

    # 上がり3Fの平均タイムを計算
    predict_df = calc_past_avg_time_by_distance(predict_df)
    # 距離別最高タイムを計算
    predict_df = calc_fastest_time_by_distance(predict_df)
    # タイム特徴量に対してPCAを適用
    # predict_df = apply_pca_for_time(predict_df, n_components=25)
    predict_df = reduce_memory_usage(predict_df)

    # レーティングを計算
    predict_df = calc_rating(predict_df, "騎手")
    save_execution_times_to_csv()
    predict_df = calc_conditional_rating(predict_df, "騎手")
    predict_df = calc_conditional_rating(predict_df, "競走馬")
    save_execution_times_to_csv()

    # 騎手の経験年数を計算
    predict_df = calc_jockey_experience(predict_df)

    # 当該条件の情報を取得
    predict_df = extract_race_features_by_conditions(predict_df)
    save_execution_times_to_csv()

    # 過去成績を記録
    predict_df = set_past_form(predict_df)
    # 過去成績に対してPCAを適用
    # predict_df = apply_pca_for_past_form(predict_df, n_components=10)

    # グレード変動を計算
    predict_df = calc_grade_change(predict_df)
    # 前走からの斤量の変化、距離変化、休養日数を計算
    predict_df = calc_changes_from_last_race(predict_df)
    # 脚質を判断
    predict_df = determine_running_style(predict_df)
    save_execution_times_to_csv()

    # 血統データを結合
    predict_df = calc_pedigrees_vector(predict_df)

    # 血統情報を結合
    predict_df = merge_pedigrees_infomation(predict_df)

    # ペースを計算
    predict_df = calculate_pace(predict_df)
    # レースの荒れ具合を計算
    predict_df = measure_race_upset(predict_df)

    # 重複を削除する
    predict_df = predict_df.drop_duplicates(subset=["race_id", "馬番"], keep="last")

    # 勝利馬の特徴量を計算
    predict_df = calculate_win_horse_features(predict_df)
    predict_df = reduce_memory_usage(predict_df)

    # 馬番別の着順の平均値や勝率を計算
    predict_df = calculate_post_position_stats(predict_df)

    # 目的変数を作成
    predict_df = make_target(predict_df)

    # 複勝列を作成
    predict_df = make_place(predict_df)

    # カラム名を文字列に変換
    predict_df.columns = predict_df.columns.astype(str)

    # pickleで保存
    feature_df_pickle_path = (
        root_path + "/20_data_processing/feature_data/feature_df.pickle"
    )
    with open(feature_df_pickle_path, "wb") as f:
        pickle.dump(predict_df, f)

    display(predict_df[predict_df["馬名"] == "イクイノックス"])

    # null量を削減
    predict_df.isnull().sum()

    # 関数ごとにかかった時間を保存
    save_execution_times_to_csv()
