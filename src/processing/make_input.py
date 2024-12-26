# lightgbmのinputファイルを作成する

# データのインポート
import pandas as pd
import numpy as np
from xfeat import SelectCategorical, LabelEncoder, Pipeline
from tqdm import tqdm
import pickle
from IPython.display import display


# リークデータを切り離す
def separate_leak_data(df):
    """リークデータを切り離す"""
    drop_df = df[
        [
            "race_id",
            "枠番",
            "馬番",
            "着順",
            "馬名",
            "騎手",
            "騎手レーティング",
            "人気",
            "単勝",
            "複勝",
            "5着着差",
            "1C通過順位",
            "2C通過順位",
            "3C通過順位",
            "4C通過順位",
            "前半ペース",
            "後半ペース",
            "ペース",
            "タイム",
        ]
    ]
    df = df.drop(
        columns=[
            "馬名",
            "騎手",
            "着差",
            "上がり3F",
            "調教師",
            "馬体重",
            "1着タイム差",
            "先位タイム差",
            "増減",
            "レース名",
            "賞金",
            "入線",
            "horse_id",
            "jockey_id",
            "trainer_id",
        ]
    )
    # 距離別タイムを削除
    dist_columns = [f"{i}m" for i in range(100, 3601, 100)]
    for m in [2700, 2900, 3100, 3300, 3500]:
        dist_columns.remove(f"{m}m")
    df = df.drop(columns=dist_columns)

    return df, drop_df


def make_aggregated_data(df):
    """同じrace_idのレースを一行に集約"""
    # 重複行の削除（仮定として、レースIDと馬番で重複を判断）
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # データをソート
    df = df.sort_values(["date", "馬番"])
    df = df.reset_index(drop=True)

    # リークデータを切り離す
    df, drop_df = separate_leak_data(df)

    # pickle化した特徴量を読み込む
    root_path = (
        "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2"
    )
    feature_columns = (
        root_path + "/20_data_processing/feature_data/aggregated_features.pickle"
    )
    with open(feature_columns, mode="rb") as f:
        feature_columns = pickle.load(f)

    # レース単位の特徴量を作成
    race_info_columns = [
        "race_id",
        "ラウンド",
        "天気",
        "馬場",
        "競馬場",
        "日数",
        "グレード",
        "立て数",
        "date",
        "方向",
        "内外",
        "コース種類",
        "距離",
        "distance_category",
        "回",
        "年齢条件",
        "性別条件",
        "1着賞金",
        "ペース",
        "波乱度",
    ]
    race_info_columns = race_info_columns + feature_columns

    # レース情報を一行に集約
    race_list = []
    print("race_id数", df["race_id"].nunique())
    for race_id in tqdm(list(df["race_id"].unique())):
        full_result = list()
        ext_df = df.loc[df["race_id"] == race_id].copy()
        if ext_df.shape[0] > 18:
            ext_df = ext_df.iloc[:18, :]
        result_list = ext_df.to_numpy().tolist()

        # 出走頭数が18頭未満の場合は、欠損値を入れる
        if len(result_list) < 18:
            for i in range(18 - len(result_list)):
                result_list.append([np.nan for k in range(len(result_list[0]))])

        # 馬の情報を結合
        horse_columns = [col for col in df.columns if col not in race_info_columns]
        for result in result_list:
            horse_info = [result[df.columns.get_loc(col)] for col in horse_columns]
            full_result += horse_info

        # レース情報を結合
        race_info_columns = race_info_columns
        race_info = [
            result_list[0][df.columns.get_loc(col)] for col in race_info_columns
        ]
        full_result += race_info

        # 1着-3着の馬番を取得
        for place in [1, 2, 3]:
            # 指定された着順に該当する馬番を取得し、列に追加
            horse_number = drop_df.loc[
                (drop_df["race_id"] == race_id) & (drop_df["着順"] == place), "馬番"
            ]
            if not horse_number.empty:
                ext_number = horse_number.min()
            else:
                ext_number = 1
            full_result.append(ext_number)

        # 前結果をリストに追加
        race_list.append(full_result)

    # データフレームに変換するために、列名を作成
    horse_columns = [col for col in df.columns if col not in race_info_columns]
    columns_list = [f"{i + 1}_{col}" for i in range(18) for col in horse_columns]
    columns_list.extend(race_info_columns)
    columns_list.extend(["1着馬番", "2着馬番", "3着馬番"])

    print("新データフレーム列数", len(columns_list))
    input_df = pd.DataFrame(race_list, columns=columns_list)

    # 着順について、nanがある場合は18着に変換
    for i in range(18):
        input_df[str(i + 1) + "_着順"] = input_df[str(i + 1) + "_着順"].fillna(18)

    # 単勝について、nanがある場合は999に変換
    for i in range(18):
        input_df[str(i + 1) + "_単勝"] = input_df[str(i + 1) + "_単勝"].fillna(999)

    print("新データフレームのサイズ: ", input_df.shape)
    display(input_df.head(3))

    # drop_dfの保存
    root_path = (
        "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2"
    )
    drop_df.to_csv(
        root_path + "/20_data_processing/input_data/label_df.csv",
        index=False,
        encoding="utf_8_sig",
    )

    return input_df


def convert_feature_types(df):
    """特徴量の型を変換"""
    # 各id列をcategory変数に変換
    # ids = ['horse_id', 'jockey_id', 'trainer_id']
    # ids_list = [f'{i + 1}_{col}' for i in range(18) for col in ids]
    # df[ids_list] = df[ids_list].astype('category')

    # xfeatを使って型変換

    print("カテゴリ変数化前", df.shape)
    # カテゴリカルエンコーディング
    encoder = Pipeline(
        [
            SelectCategorical(),
            LabelEncoder(unseen="n_unique", output_suffix=""),
        ]
    )
    encoded_df = encoder.fit_transform(df)
    categorical_features = list(encoded_df.columns)
    df = df.drop(columns=encoded_df.columns)
    df = pd.concat([df, encoded_df], axis=1, join="outer")
    # nanが-1になっており学習時にwarningを吐き出すため、nanの部分を最大値+1に変換する
    for col in list(encoded_df.columns):
        df.loc[df[col] == -1, col] = df[col].max() + 1
    print("カテゴリ変数化後", df.shape)
    print("カテゴリ変数一覧", categorical_features)

    return df, categorical_features


def merge_pace(df):
    """pace_dfを結合する"""

    # pace_dfの読み込み
    root_path = (
        "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2"
    )
    pace_df_path = root_path + "/20_data_processing/processed_data/pace/pace_df.csv"
    pace_df = pd.read_csv(pace_df_path, encoding="utf_8_sig")

    # nullを全て0で埋めない(学習時に0が原因で学習がうまくいかない可能性があるため)
    # pace_df = pace_df.fillna(0)

    # pace_dfと結合
    print("結合前サイズ", df.shape)
    df = pd.merge(df, pace_df, on="race_id", how="left")
    print("結合後サイズ", df.shape)
    display(df.head())

    return df


def make_rank_input_df(df):
    """ランクモデル用のデータを作成する"""
    root_path = (
        "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2"
    )
    # リークデータを切り離す
    df, _ = separate_leak_data(df)

    # 必要な列を選択
    rank_columns = ["race_id", "枠番", "馬番", "着順"] + [
        col for col in df.columns if col not in ["race_id", "枠番", "馬番", "着順"]
    ]
    rank_df = df[rank_columns].copy()

    # 着順について、nanがある場合は18着に変換
    rank_df["着順"] = rank_df["着順"].fillna(18)

    # 単勝について、nanがある場合は999に変換
    rank_df["単勝"] = rank_df["単勝"].fillna(999)

    # 特徴量の型を変換
    rank_df, rank_categorical_features = convert_feature_types(rank_df)

    # race_idをカテゴリ変数に変換
    rank_df["race_id"] = rank_df["race_id"].astype("category")

    # グループ（レース）ごとのサンプル数を取得
    group_counts = rank_df.groupby("race_id").size().values
    print("グループごとのサンプル数", group_counts)

    print("ランクモデル用データのサイズ: ", rank_df.shape)
    display(rank_df.head(3))

    # pickleでデータ保存
    with open(
        root_path + "/20_data_processing/input_data/rank_input_df.pickle", mode="wb"
    ) as f:
        pickle.dump(rank_df, f)
    with open(
        root_path + "/20_data_processing/input_data/rank_input_df_test.pickle",
        mode="wb",
    ) as f:
        pickle.dump(rank_df.tail(10000), f)
    with open(
        root_path + "/20_data_processing/input_data/rank_categorical_features.pickle",
        mode="wb",
    ) as f:
        pickle.dump(rank_categorical_features, f)


def make_input_df():
    # データの読み込み
    root_path = (
        "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2"
    )
    # pickle化した特徴量を読み込む
    feature_df_pickle_path = (
        root_path + "/20_data_processing/feature_data/feature_df.pickle"
    )
    with open(feature_df_pickle_path, mode="rb") as f:
        df = pickle.load(f)

    # データをソート
    df = df.sort_values(["date", "race_id", "馬番"])
    # indexを振り直す
    df = df.reset_index(drop=True)

    # ランクモデル用のデータを作成
    make_rank_input_df(df)

    # 一行に集約したデータを作成
    df = make_aggregated_data(df)

    # pace_dfを結合
    df = merge_pace(df)

    # 特徴量の型を変換
    df, categorical_features = convert_feature_types(df)

    # pickleでデータ保存
    with open(
        root_path + "/20_data_processing/input_data/input_df.pickle", mode="wb"
    ) as f:
        pickle.dump(df, f)
    with open(
        root_path + "/20_data_processing/input_data/input_df_test.pickle", mode="wb"
    ) as f:
        pickle.dump(df.tail(10000), f)
    with open(
        root_path + "/20_data_processing/input_data/categorical_features.pickle",
        mode="wb",
    ) as f:
        pickle.dump(categorical_features, f)
