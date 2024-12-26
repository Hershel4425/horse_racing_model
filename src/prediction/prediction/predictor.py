import pickle

import numpy as np
import pandas as pd

import logging
import traceback


def load_rank_data():
    root_path = (
        "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2"
    )
    with open(
        root_path + "/20_data_processing/input_data/rank_input_df.pickle", "rb"
    ) as f:
        rank_df = pickle.load(f)

    return rank_df


def drop_data(df):
    """不要なデータを削除する

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    # 不要な列を削除する
    drop_columns = [
        "人気",
        "1C通過順位",
        "2C通過順位",
        "3C通過順位",
        "4C通過順位",
        "前半ペース",
        "後半ペース",
        "タイム",
        "ペース",
        "複勝",
        "5着着差",
        "波乱度",
        "期待値",
    ]
    df = df.drop(columns=drop_columns)

    return df


def process_odds(df, race_id):
    """オッズの処理をする

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    # 単勝オッズファイルを読み込む
    root_path = (
        "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2"
    )
    odds_df = pd.read_csv(root_path + f"/40_automation/{race_id}_tan_odds.csv")

    # dfの単勝列をodds_dfの単勝列で置き換える
    # dfを馬番でソートし、odds_dfも馬番でソート
    print('デバッグ')
    df = df.sort_values("馬番")
    odds_df = odds_df.sort_values("馬番")
    # 置き換え
    df["単勝"] = odds_df["単勝オッズ"].values
    # 単勝が---.-の時、0に変換
    df["単勝"] = df["単勝"].replace("---.-", 0)
    # float型に変換
    df["単勝"] = df["単勝"].astype(float)
    df = df.drop(columns = ['単勝'])

    return df


def predict(test_df, model_list, accuracy_list):
    """予測を行う

    Args:
        test_df (_type_): _description_
        model_list (_type_): _description_
        accuracy_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    # test_dfを用いて予測をし、予測結果をアンサンブル
    # test_dfのtargetを削除
    X_test = test_df.drop(["着順"], axis=1)
    _ = test_df["着順"]

    # 予測
    pred_list = []
    for model in model_list:
        pred = model.predict(X_test)
        pred_list.append(pred)

    # アンサンブル
    # accuracyを重みとして平均をとる
    pred = np.average(pred_list, axis=0, weights=accuracy_list)

    test_df.loc[:, "pred"] = pred

    return test_df


def calc_win_probs_by_race(df, race_id):
    """レースごとに勝率を計算する"""
    print("勝率予測中")
    scores = df["pred"].values
    print("OKK")
    win_probs_df = pd.DataFrame(
        {
            "race_id": race_id,
            "date": df["date"].values,
            "馬番": df["馬番"].values,
            "score": scores,
            "着順": df["着順"].values,
        }
    )
    print("OK")

    return win_probs_df


def run_prediction(race_id):
    try:
        # ファイルの読み込み
        rank_df = load_rank_data()

        # race_idのみ抽出
        df = rank_df.loc[rank_df["race_id"] == race_id]
        # 不要データのdrop
        df = drop_data(df)
        df = process_odds(df, race_id)

        # modelを読み込む
        with open(
            "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2/50_machine_learning/output_data/model/20241027010706/20241027010706_rank_model.pickle",
            "rb",
        ) as f:
            model_list = pickle.load(f)
        with open(
            "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2/50_machine_learning/output_data/model/20241027010706/20241027010706_rank_accuracy.pickle",
            "rb",
        ) as f:
            accuracy_list = pickle.load(f)

        result_df = predict(df, model_list, accuracy_list)
        win_probs_df = calc_win_probs_by_race(result_df, race_id)
        print('デバッgっy2')
        # 重複を削除
        win_probs_df = win_probs_df.drop_duplicates(subset=["race_id", "馬番"])

        # データの書き込み
        root_path = (
            "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2"
        )
        win_probs_df_path = root_path + f"/40_automation/{race_id}_win_probs_df.csv"
        win_probs_df.to_csv(win_probs_df_path, encoding="utf_8_sig", index=False)
    except Exception as e:
        logging.error(
            f"race_id: {race_id} の予測処理中にエラーが発生しました: {str(e)}"
        )
        logging.error(traceback.format_exc())
