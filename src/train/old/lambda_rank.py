import pickle
import datetime
import os
from tqdm import tqdm

import numpy as np
import pandas as pd

import lightgbm as lgb
import optuna.integration.lightgbm as optuna_lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import ndcg_score

import matplotlib.pyplot as plt

# 今日の日付をYYYYMMDDhhmmss形式で取得
DATA_STRING = datetime.datetime.now().strftime("%Y%m%d%H%M%S")


def load_rank_data(test_flag=False):
    """input dataを読み込む

    Args:
        test_flag (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    root_path = (
        "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2"
    )
    if test_flag:
        with open(
            root_path + "/20_data_processing/input_data/rank_input_df_test.pickle", "rb"
        ) as f:
            rank_df = pickle.load(f)
    else:
        with open(
            root_path + "/20_data_processing/input_data/rank_input_df.pickle", "rb"
        ) as f:
            rank_df = pickle.load(f)

    with open(
        root_path + "/20_data_processing/input_data/rank_categorical_features.pickle",
        "rb",
    ) as f:
        rank_categorical_features = pickle.load(f)

    return rank_df, rank_categorical_features


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
        "単勝",
        "複勝",
        "5着着差",
        "波乱度",
        "期待値",
    ]
    df = df.drop(columns=drop_columns)

    return df


def process_target(df):
    """着順の処理をする"""

    df["着順"] = 18 - df["着順"]
    return df


def save_model(model_list, accuracy_list):
    """モデルを保存する

    Args:
        model_list (_type_): _description_
        accuracy_list (_type_): _description_
    """
    root_path = (
        "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2"
    )

    # pathを作成
    model_path = (
        root_path
        + f"/50_machine_learning/output_data/model/{DATA_STRING}/{DATA_STRING}_rank_model.pickle"
    )
    accuracy_path = (
        root_path
        + f"/50_machine_learning/output_data/model/{DATA_STRING}/{DATA_STRING}_rank_accuracy.pickle"
    )

    # モデルを保存
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model_list, f)

    os.makedirs(os.path.dirname(accuracy_path), exist_ok=True)
    with open(accuracy_path, "wb") as f:
        pickle.dump(accuracy_list, f)


def train_rank_model(df, categorical_features, opt_flag):
    """ランキングモデルを学習する

    Args:
        df (_type_): _description_
        categorical_features (_type_): _description_
        opt_flag (_type_): _description_

    Returns:
        _type_: _description_
    """
    import warnings

    warnings.filterwarnings("ignore")

    model_list = []
    accuracy_list = []
    # 学習結果を保存するための辞書
    evals_result = {}
    # ランキングモデルの学習
    # TimeSeriesSplitを使用して訓練データと検証データに分割
    tscv = TimeSeriesSplit(n_splits=7)

    model_param = {
        "task": "train",
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5],
        "boosting_type": "gbdt",
        "random_state": 42,
        "verbosity": -1,  # ワーニングメッセージを非表示にする
    }

    for train_index, valid_index in tscv.split(df):
        train_df = df.iloc[train_index]
        valid_df = df.iloc[valid_index]

        # クエリグループの作成
        train_query_groups = train_df["race_id"].value_counts().sort_index().tolist()
        valid_query_groups = valid_df["race_id"].value_counts().sort_index().tolist()

        # 特徴量とターゲットの分割
        X_train = train_df.drop(["着順"], axis=1)
        y_train = train_df["着順"]
        X_valid = valid_df.drop(["着順"], axis=1)
        y_valid = valid_df["着順"]

        lgb_train = lgb.Dataset(
            X_train,
            y_train,
            group=train_query_groups,
            categorical_feature=categorical_features,
        )
        lgb_eval = lgb.Dataset(
            X_valid,
            y_valid,
            group=valid_query_groups,
            categorical_feature=categorical_features,
        )

        if opt_flag:
            model = optuna_lgb.train(
                model_param,
                lgb_train,
                valid_sets=[lgb_train, lgb_eval],
                num_boost_round=10000,
                verbose_eval=False,
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=100, verbose=False
                    ),  # early_stopping用コールバック関数
                    # lgb.record_evaluation(evals_result)  # 学習結果を保存するコールバック関数
                ],
            )
            model_param = model.params
            print("Best Params:", model.params)
        else:
            print("Optunaを使用しない")

        model = lgb.train(
            model_param,
            lgb_train,
            valid_sets=[lgb_train, lgb_eval],
            valid_names=["train", "valid"],
            num_boost_round=10000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0),  # ログ出力を抑制
                lgb.record_evaluation(
                    evals_result
                ),  # 学習結果を保存するコールバック関数
            ],
        )

        # 学習曲線をプロット
        train_results = evals_result["train"]["ndcg@5"]
        valid_results = evals_result["valid"]["ndcg@5"]
        plt.plot(train_results, label="train")
        plt.plot(valid_results, label="valid")
        plt.xlabel("Iteration")
        plt.ylabel("RMSE")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True)
        plt.show()

        # モデルの評価
        y_pred = model.predict(X_valid)
        score = ndcg_score([y_valid], [y_pred], k=5)
        print(f"NDCG@5: {score}")

        model_list.append(model)
        accuracy_list.append(score)

    save_model(model_list, accuracy_list)

    return model_list, accuracy_list


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

    # feature dfを読み込む
    with open(
        "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2/20_data_processing/feature_data/feature_df.pickle",
        "rb",
    ) as f:
        feature_df = pickle.load(f)

    # feature_dfから、test_dfにあるrace_id, 馬番の組み合わせに対応する行を取り出す
    result_df = feature_df.merge(
        test_df[["race_id", "馬番", "pred"]], how="inner", on=["race_id", "馬番"]
    )

    return result_df


def calc_win_probs_by_race(df):
    """レースごとに勝率を計算する"""
    win_probs_list = []
    print("勝率予測中")
    for race_id, race_df in tqdm(df.groupby("race_id")):
        scores = race_df["pred"].values
        win_probs_df = pd.DataFrame(
            {
                "race_id": race_id,
                "date": race_df["date"].values,
                "馬番": race_df["馬番"].values,
                "馬名": race_df["馬名"].values,
                "score": scores,
                "着順": race_df["着順"].values,
            }
        )
        win_probs_list.append(win_probs_df)

    result_df = pd.concat(win_probs_list, ignore_index=True)
    return result_df


def lambda_rank(test_flag=False, pred_flag=False, opt_flag=False):
    """ランキングモデルの学習と予測を行う"""
    rank_df, rank_categorical_features = load_rank_data(test_flag)
    rank_df = drop_data(rank_df)
    rank_df = process_target(rank_df)
    if pred_flag:
        # modelを読み込む
        with open(
            "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2/50_machine_learning/output_data/model/20240425224050/20240425224050_rank_model.pickle",
            "rb",
        ) as f:
            model_list = pickle.load(f)
        with open(
            "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2/50_machine_learning/output_data/model/20240425224050/20240425224050_rank_accuracy.pickle",
            "rb",
        ) as f:
            accuracy_list = pickle.load(f)
    else:
        model_list, accuracy_list = train_rank_model(
            rank_df, rank_categorical_features, opt_flag
        )
    result_df = predict(rank_df, model_list, accuracy_list)
    win_probs_df = calc_win_probs_by_race(result_df)

    # 重複を削除
    win_probs_df = win_probs_df.drop_duplicates(subset=["race_id", "馬番"])
    # データをソート
    win_probs_df = win_probs_df.sort_values(["date", "馬番"])

    # 予測結果を保存
    root_path = (
        "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2"
    )
    win_probs_df.to_csv(
        root_path
        + f"/50_machine_learning/output_data/pred/{DATA_STRING}_result_df.csv",
        index=False,
    )
