import yaml

import pandas as pd
import pickle

import datetime

import sys

sys.path.append(
    "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2/91_program/train"
)

from .preprocessing import (
    drop_data,
    process_target,
    extract_training_configs,
)  # , make_pca_features
from .model_training import loop_train
from .prediction import predicts, make_pred_target_list
from .result_saving import make_url_from_date


def load_config(config_path):
    """configファイルを読み込む

    Args:
        config_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    with open(config_path, "r") as yml:
        config = yaml.safe_load(yml)
    return config


def load_data(config, test_flag=False):
    """データを読み込む

    Args:
        config (_type_): _description_
        test_flag (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if test_flag:
        # pickleを使って読み込む
        with open(config["test_input_df_path"], "rb") as f:
            df = pickle.load(f)
    else:
        with open(config["input_df_path"], "rb") as f:
            df = pickle.load(f)
    categorical_features = pickle.load(
        open(config["categorical_feature_path"], "rb")
    )  # モデル学習時に指定するカテゴリカル変数のリスト
    label_df = pd.read_csv(
        config["template_df_path"], encoding="utf-8-sig"
    )  # 予測結果を保存するためのテンプレートデータフレーム
    return df, categorical_features, label_df


def main(test_flag=False):
    """学習と予測を行う
    Args:
        test_flag (bool): 関数実行テスト用のフラグ
    """
    # configファイルの読み込み
    config_path = "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2/91_program/train/config.yaml"
    config = load_config(config_path)

    # 学習記録用の日付を取得
    data_string = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    config["data_string"] = data_string

    # データの読み込み
    df, categorical_features, label_df = load_data(config, test_flag)

    # 不要なデータを削除
    df = drop_data(df, config)
    # 目的変数を加工
    df = process_target(df, config)

    # 学習対象を抽出
    training_configs = extract_training_configs(config)
    target_list = []
    for tc in training_configs:
        target_list.extend(tc.generate_target_list())

    print("学習対象一覧:", target_list)

    # 学習と予測
    model_list_dict, accuracy_list_dict, feature_list_dict = loop_train(
        df, categorical_features, config
    )

    # フラグを元に予測するモデルを取り出す
    pred_target_list, model_list_dict, accuracy_list_dict = make_pred_target_list(
        model_list_dict, accuracy_list_dict, feature_list_dict, config
    )

    # 予測結果を結果確認用データフレームに保存
    with open(config["feature_df_path"], "rb") as f:
        feature_df = pickle.load(f)
    # 重複を削除
    feature_df = feature_df.drop_duplicates(subset=["race_id", "馬番"])
    # データをソート
    feature_df = feature_df.sort_values(["date", "馬番"])
    feature_df = feature_df.reset_index(drop=True)
    df_test = predicts(
        df,
        feature_df,
        label_df,
        model_list_dict,
        accuracy_list_dict,
        feature_list_dict,
        pred_target_list,
        config,
    )

    # df_testからpca特徴量を全て削除
    pca_columns = [col for col in df_test.columns if "PCA" in col]
    df_test = df_test.drop(columns=pca_columns)

    # 予測結果を保存
    df_test.to_csv(
        make_url_from_date(data_string, config, "pred"),
        index=False,
        encoding="utf_8_sig",
    )
