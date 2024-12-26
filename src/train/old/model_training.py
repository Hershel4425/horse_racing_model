import time
import pickle
import os
import warnings


import lightgbm as lgb
import optuna.integration.lightgbm as optuna_lgb
from sklearn.model_selection import TimeSeriesSplit

import torch
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

from .preprocessing import (
    split_input_output,
    under_sampling,
    process_null,
    extract_training_configs,
)
from .result_saving import make_url_from_date, record_training_time
from .visualization import (
    confirm_accuracy,
    feature_importance,
    plot_lgb_learning_curve,
    plot_tabnet_learning_curve,
)
from .stacking import stacking
from .prediction import load_pretrained_models


def extract_feature_from_feature_importance(
    x_tr,
    y_tr,
    x_val,
    y_val,
    target_col,
    model_params_list,
    categorical_features,
    config,
    config_feature,
    idx,
):
    """特徴量重要度を用いて重要なところだけを抽出する関数
    Args:
        x_tr (dataframe): 学習データの入力データフレーム
        y_tr (series): 学習データの目的変数のデータフレーム
        x_val (dataframe): テストデータの入力データフレーム
        y_val (series): テストデータの目的変数のデータフレーム
        target_col (str): 学習対象の列名
        model_params_list (list): パラメータのリスト
        categorical_features (list): カテゴリカル変数のリスト
        config (dict): configファイル
        config_feature (dict): 学習方法を記載した辞書
        idx (int): kfoldのインデックス
    Returns:
        top_features (list): 特徴量重要度の高い特徴量のリスト

    """
    # 特徴量重要度を計算するために、最初の学習を行う
    print("特徴量重要度を計算するために、最初の学習を行います...")
    # 初期学習を行い、特徴量重要度を計算
    model, _ = train_lgb(
        x_tr,
        y_tr,
        x_val,
        y_val,
        target_col,
        model_params_list,
        categorical_features,
        idx,
        config,
        config_feature,
        optuna_flag=False,
    )
    top_features = feature_importance(model, target_col, threshold=1)[
        "Feature"
    ].tolist()

    return top_features


def preprocess_data(df, target_list, target_col, config):
    """データの前処理を行う

    Args:
        df (_type_): _description_
        target_list (_type_): _description_
        target_col (_type_): _description_
        config (_type_): _description_

    Returns:
        _type_: _description_
    """

    # 後のif文処理のため、target_colに対応したconfigのfeatureを取り出す
    if "_" in target_col:
        target_name = target_col.split("_")[1]
    else:
        target_name = target_col
    config_feature = [
        feature for feature in config["feature"] if feature["name"] == target_name
    ][0]

    # target_colに5着着差が含まれている時は、アンダーサンプリングを実行する
    if "5着着差" in target_col:
        print("アンダーサンプリングを行います...")
        undersampled_df = under_sampling(df, target_col)
    else:
        undersampled_df = df

    # 期待値が目的変数であるとき、stackingした勝率予測値から期待値を計算する
    if "期待値" in target_col:
        # 勝率がstackingされていない時はエラーを吐く
        if "stack_1_1着馬番" not in df.columns:
            raise ValueError("stack_1_1着馬番が存在しません")
        # 期待値を計算する
        # 馬番を取り出す
        target_count = int(target_col.split("_")[0])
        undersampled_df[target_col] = (
            undersampled_df[target_col]
            * undersampled_df[f"stack_{target_count}_1着馬番"]
            * 100
        )

    # 目的変数が欠損値になっているデータを削除
    undersampled_df = undersampled_df.dropna(subset=[target_col])
    # インデックスを振り直す
    undersampled_df = undersampled_df.reset_index(drop=True)

    # tabnetで学習する時は、欠損値を処理する
    if config_feature["train_method"] == "tabnet":
        undersampled_df = process_null(undersampled_df, config)

    # 特徴量と目的変数に分ける
    feature_list = [col for col in undersampled_df.columns if col not in target_list]
    target_data = undersampled_df[target_col]
    train_data = undersampled_df[feature_list]

    return train_data, target_data, feature_list, config_feature


def get_model_params(target_col, config_feature, config):
    """モデルのパラメータを取得する

    Args:
        target_col (_type_): _description_
        config_feature (_type_): 学習方法を記載した辞書
        config (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        model_param_list: モデルのパラメータのリスト
    """

    model_params_list = []
    # 学習済みモデルを用いる場合は、モデルを読み込む
    model_url_dict = {}
    training_configs = extract_training_configs(config)
    for tc in training_configs:
        if target_col in tc.feature_name:
            if tc.use_model:
                # モデルurl_dictを取得
                model_url_dict = tc.generate_model_url_dict(config)
                for i in range(config["split_num"]):
                    if config_feature["train_method"] == "lightgbm":
                        model_path = (
                            model_url_dict[target_col]
                            + "/"
                            + target_col
                            + "_"
                            + str(i)
                            + "_lightgbm_model.pkl"
                        )
                    elif config_feature["train_method"] == "tabnet":
                        model_path = (
                            model_url_dict[target_col]
                            + "/"
                            + target_col
                            + "_"
                            + str(i)
                            + "_tabnet_model.pkl"
                        )
                    with open(model_path, "rb") as f:
                        model = pickle.load(f)

                model_params_list.append(model.params)
                print("学習済みモデルを用いて学習を行います...")

                return model_params_list

    # 回帰か分類かで分岐
    target_objective = config_feature["train_type"]

    if config_feature["train_method"] == "lightgbm":
        # 学習済みモデルを用いないとき
        if target_objective == "multiclass":
            model_params = {
                "task": "train",  # 学習、トレーニング ⇔　予測predict
                "boosting_type": "gbdt",  # 勾配ブースティング
                "random_state": 42,  # 乱数シード
                "objective": "multiclass",  # 目的関数：多値分類、マルチクラス分類
                "metric": "multi_logloss",  # 分類モデルの性能を測る指標
                "num_class": 18,  # 目的変数のクラス数
                "num_leaves": 65,  # ノードの数
                "learning_rate": 0.1,  # 学習率
                "bagging_freq": 7,  # 何回目のイテレーションでbaggingを行うか
                "bagging_fraction": 0.9897025613644793,  # baggingの割合
                "min_data_in_leaf": 50,  # ノードの最小データ数
                "max_depth": -1,  # 木の深さ
                "verbose": -1,
                "feature_pre_filter": False,
                "lambda_l1": 9.578403387371191,
                "lambda_l2": 7.78406769746287e-08,
                "feature_fraction": 0.552,
                "num_iterations": 10000,
            }
        elif target_objective == "binary":
            model_params = {
                "task": "train",  # 学習、トレーニング ⇔　予測predict
                "boosting_type": "gbdt",  # 勾配ブースティング
                "random_state": 42,  # 乱数シード
                "objective": "binary",  # 目的関数：2値分類
                "metric": "binary_logloss",  # 分類モデルの性能を測る指標
                "num_leaves": 65,  # ノードの数
                "learning_rate": 0.1,  # 学習率
                "bagging_freq": 7,  # 何回目のイテレーションでbaggingを行うか
                "bagging_fraction": 0.9897025613644793,  # baggingの割合
                "min_data_in_leaf": 50,  # ノードの最小データ数
                "max_depth": -1,  # 木の深さ
                "verbose": -1,
                "feature_pre_filter": False,
                "lambda_l1": 9.578403387371191,
                "lambda_l2": 7.78406769746287e-08,
                "feature_fraction": 0.552,
                "num_iterations": 10000,
            }
        elif target_objective == "regression":
            model_params = {
                "task": "train",  # 学習、トレーニング ⇔　予測predict
                "boosting_type": "gbdt",  # 勾配ブースティング
                "random_state": 42,  # 乱数シード
                "objective": "regression",  # 目的関数：回帰
                "metric": "rmse",  # 分類モデルの性能を測る指標
                "num_leaves": 65,  # ノードの数
                "learning_rate": 0.1,  # 学習率
                "bagging_freq": 7,  # 何回目のイテレーションでbaggingを行うか
                "bagging_fraction": 0.9897025613644793,  # baggingの割合
                "min_data_in_leaf": 50,  # ノードの最小データ数
                "max_depth": -1,  # 木の深さ
                "verbose": -1,
                "feature_pre_filter": False,
                "lambda_l1": 9.578403387371191,
                "lambda_l2": 7.78406769746287e-08,
                "feature_fraction": 0.552,
                "num_iterations": 10000,
            }
        # それ以外の時はエラーを出す
        else:
            raise ValueError(f"{target_col}のtrain_typeが不正です")
        for i in range(config["split_num"]):
            model_params_list.append(model_params)

    return model_params_list


def train_lgb(
    x_train,
    y_train,
    x_valid,
    y_valid,
    target_col,
    model_params_list,
    categorical_features,
    idx,
    config,
    config_feature,
    optuna_flag=False,
):
    """LightGBMを用いて学習を行う関数、optuna_flagがTrueの時はoptunaを行う

    Args:
        x_train (dataframe): 学習データの入力データフレーム
        y_train (series): 学習データの目的変数のデータフレーム
        x_valid (dataframe): テストデータの入力データフレーム
        y_valid (series): テストデータの目的変数のデータフレーム
        target_col (str): 学習対象の列名
        model_params_list (list): パラメータのリスト
        categorical_features (list): カテゴリカル変数のリスト
        idx (int): kfoldのインデックス、学習結果を保存するために使用
        config (dict): configファイル
        config_feature (dict): 学習方法を記載した辞書
        optuna_flag (bool): optunaを行うかどうかのフラグ
    Returns:
        model (model): 学習したモデル
        final_accuracy (float): 最終精度
    """
    lgb_train = lgb.Dataset(x_train, y_train, categorical_feature=categorical_features)
    lgb_eval = lgb.Dataset(x_valid, y_valid, categorical_feature=categorical_features)

    # 学習開始時刻
    start_time = time.time()
    # 学習結果を保存するための辞書
    evals_result = {}

    # パラメータを読み込む
    model_param = model_params_list[idx]

    # optunaを行う時
    if optuna_flag:
        print("optunaを行います")
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

    model = lgb.train(
        model_param,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        valid_names=["train", "valid"],  # 学習評価用に名前を設定
        num_boost_round=10000,
        verbose_eval=False,  # 学習過程を表示しない
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=100, verbose=False
            ),  # early_stopping用コールバック関数
            lgb.record_evaluation(evals_result),  # 学習結果を保存するコールバック関数
        ],
    )

    # 学習終了時刻
    end_time = time.time()
    # 学習過程をplot
    plot_lgb_learning_curve(evals_result, target_col, idx, config_feature)
    # モデル精度を確認
    final_accuracy = confirm_accuracy(model, x_valid, y_valid, config_feature)
    # 最終精度を取得
    final_metric_name = model_param["metric"]
    print(f"最終精度の計算結果: {final_accuracy}")
    final_accuracy = min(evals_result["valid"][final_metric_name])
    print(f"最終精度: {final_accuracy}")
    # 学習時間を記録
    study_record_time_path = make_url_from_date(
        config["data_string"], config, "study_record"
    )
    params = {
        "model_name": "LightGBM",
        "k_fold": "TimeSeriesSplit",
        "target_col": target_col,
        "fold_number": idx,
        "metric_name": final_metric_name,
        "accuracy": final_accuracy,
        "start_time": start_time,
        "end_time": end_time,
        "output_path": study_record_time_path,
    }
    record_training_time(params)

    # モデルを保存
    model_url_head = make_url_from_date(config["data_string"], config, "model")
    os.makedirs(model_url_head, exist_ok=True)
    with open(f"{model_url_head}/{target_col}_{idx}_lightgbm_model.pkl", "wb") as f:
        pickle.dump(model, f)
    # accuracy_list_listを保存
    with open(f"{model_url_head}/{target_col}_{idx}_accuracy.pickle", "wb") as f:
        pickle.dump(final_accuracy, f)
    with open(f"{model_url_head}/{target_col}_{idx}_feature.pickle", "wb") as f:
        pickle.dump(model.feature_name(), f)

    return model, final_accuracy


def train_tabnet(
    x_train, y_train, x_valid, y_valid, target_col, idx, config, config_feature
):
    """TabNetを用いて学習を行う関数

    Args:
        x_train (dataframe): 学習データの入力データフレーム
        y_train (series): 学習データの目的変数のデータフレーム
        x_valid (dataframe): テストデータの入力データフレーム
        y_valid (series): テストデータの目的変数のデータフレーム
        target_col (str): 学習対象の列名
        idx (int): kfoldのインデックス
        config (dict): configファイル
        config_feature (dict): 学習方法を記載した辞書
    Returns:
        model (model): 学習したモデル
        final_accuracy (float): 最終精度
    """

    # 学習開始時刻
    start_time = time.time()

    # 回帰か分類かで分岐
    if config_feature["train_type"] == "multiclass":
        tab_model = TabNetClassifier(
            n_d=64,
            n_a=64,
            n_steps=7,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
            mask_type="entmax",
            scheduler_params=dict(
                mode="min",
                patience=5,
                min_lr=1e-5,
                factor=0.9,
            ),
            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
            verbose=0,  # 学習過程を表示しない
        )
        tab_model.fit(
            x_train.values,
            y_train.values,
            eval_set=[(x_valid.values, y_valid.values)],
            eval_metric=["logloss"],
            patience=20,
            batch_size=2048,
        )

    elif config_feature["train_type"] == "regression":
        tab_model = TabNetRegressor(
            n_d=64,
            n_a=64,
            n_steps=7,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
            mask_type="entmax",
            scheduler_params=dict(
                mode="min",
                patience=5,
                min_lr=1e-5,
                factor=0.9,
            ),
            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
            verbose=0,  # 学習過程を表示しない
        )
        tab_model.fit(
            x_train.values,
            y_train.values.reshape(-1, 1),
            eval_set=[(x_valid.values, y_valid.values.reshape(-1, 1))],
            eval_metric=["rmse"],
            patience=20,
            batch_size=2048,
        )

    # 学習結果をプロット
    plot_tabnet_learning_curve(tab_model, target_col, idx, config_feature)
    final_accuracy = confirm_accuracy(tab_model, x_valid, y_valid, config_feature)
    end_time = time.time()
    # 学習時間を記録
    study_record_time_path = make_url_from_date(
        config["data_string"], config, "study_record"
    )
    params = {
        "model_name": "TabNet",
        "k_fold": "TimeSeriesSplit",
        "target_col": target_col,
        "fold_number": idx,
        "metric_name": "tab",
        "accuracy": final_accuracy,
        "start_time": start_time,
        "end_time": end_time,
        "output_path": study_record_time_path,
    }
    record_training_time(params)

    # モデルを保存
    model_url_head = make_url_from_date(config["data_string"], config, "model")
    os.makedirs(model_url_head, exist_ok=True)
    with open(f"{model_url_head}/{target_col}_{idx}_tabnet_model.pkl", "wb") as f:
        pickle.dump(tab_model, f)
    # accuracy_list_listを保存
    with open(f"{model_url_head}/{target_col}_{idx}_accuracy.pickle", "wb") as f:
        pickle.dump(final_accuracy, f)
    with open(f"{model_url_head}/{target_col}_{idx}_feature.pickle", "wb") as f:
        pickle.dump(x_train.columns, f)

    return tab_model, final_accuracy


def execute_training(df, target_col, target_list, categorical_features, config):
    """特定一列の学習を行う

    Args:
        df (_type_): _description_
        target_col (_type_): _description_
        target_list (_type_): _description_
        categorical_features (_type_): _description_
        config (_type_): _description_

    Returns:
        model_list (_type_): _description_
        accuracy_list (_type_): _description_
        feature_list (_type_): _description_
    """
    # データ前処理...
    train_data, target_data, feature_list, config_feature = preprocess_data(
        df, target_list, target_col, config
    )

    # モデルパラメータの設定...
    model_params_list = get_model_params(target_col, config_feature, config)

    # モデルの学習...
    if config_feature["train_flag"]:
        print(f"{target_col}の学習を行います...")
        model_list, accuracy_list, feature_list = train_model(
            train_data,
            target_data,
            categorical_features,
            target_col,
            model_params_list,
            config_feature,
            config,
        )
    # 学習フラグが立っておらず、stackingフラグのみが立っている場合は、学習をスキップする
    else:
        print(f"{target_col}の学習を行いません...")
        # モデルを読み込む
        date = config_feature["use_model_pred"]["date"]
        model_list = load_pretrained_models(config, target_col, date, "model")
        accuracy_list = load_pretrained_models(config, target_col, date, "accuracy")
        feature_list = load_pretrained_models(config, target_col, date, "feature")

    return model_list, accuracy_list, feature_list


def train_model(
    train_data,
    target_data,
    categorical_features,
    target_col,
    model_params_list,
    config_feature,
    config,
):
    """特定一列の学習を行う

    Args:
        train_data (dataframe): 学習に使う特徴量部分のデータ
        target_data (series): 学習に使うターゲット部分のデータ
        target_col (str): 学習対象の列名
        model_params_list (list): パラメータのリスト
        feature (dict): 学習方法を記載した辞書
        config (dict): configファイル

    Returns:
        model_list (list): 学習したモデルのリスト
        accuracy_list (list): 学習したモデルの精度のリスト
        feature_list (list): 学習したモデルの特徴量のリスト
    """
    # warningを発生させない
    warnings.filterwarnings("ignore")

    x_train, _, y_train, _ = split_input_output(train_data, target_data, config)
    cv = TimeSeriesSplit(n_splits=config["split_num"])

    model_list = []
    accuracy_list = []
    feature_list = []

    for idx, (train_index, valid_index) in enumerate(cv.split(x_train, y_train)):
        print(f"fold: {idx + 1}を行います")
        x_tr, x_val = x_train.loc[train_index, :], x_train.loc[valid_index, :]
        y_tr, y_val = y_train.loc[train_index], y_train.loc[valid_index]

        # 特徴量削減を行う場合は、特徴量重要度を用いてモデルを訓練する
        if config_feature["feature_selection"]:
            top_features = extract_feature_from_feature_importance(
                x_tr,
                y_tr,
                x_val,
                y_val,
                target_col,
                model_params_list,
                categorical_features,
                config,
                config_feature,
                idx,
            )
            print("抽出した特徴量の数:", len(top_features))
            # categorical_featuresからtop_featuresに含まれていないものを削除
            top_categorical_features = []
            for ft in categorical_features:
                if ft in top_features:
                    top_categorical_features.append(ft)
            categorical_features = top_categorical_features
            x_tr = x_tr[top_features]
            x_val = x_val[top_features]
            # 特徴量重要度の高い特徴量のみを用いて学習を行う
            print("特徴量重要度の高い特徴量のみを用いて学習を行います...")
        else:
            top_features = x_tr.columns.tolist()
            top_categorical_features = categorical_features

        # 使用モデルを判定
        if config_feature["train_method"] == "lightgbm":
            model, final_accuracy = train_lgb(
                x_tr,
                y_tr,
                x_val,
                y_val,
                target_col,
                model_params_list,
                top_categorical_features,
                idx,
                config,
                config_feature,
                optuna_flag=False,
            )

        elif config_feature["train_method"] == "tabnet":
            model, final_accuracy = train_tabnet(
                x_tr, y_tr, x_val, y_val, target_col, idx, config, config_feature
            )

        # 出力結果をまとめる
        model_list.append(model)
        accuracy_list.append(final_accuracy)
        feature_list.append(top_features)

    return model_list, accuracy_list, feature_list


def loop_train(df, categorical_features, config):
    """特定一列の学習をループする

    Args:
        df (): 入力データフレーム
        categorical_features (_type_): カテゴリカル変数のリスト
        config (_type_): _description_

    Returns:
        model_list_dict (_type_): _description_
        accuracy_list_dict (_type_): _description_
    """
    model_list_dict = {}
    accuracy_list_dict = {}
    feature_list_dict = {}

    # スタッキングを行うためdfをトレーニング用にコピーする
    loop_df = df.copy()

    # 学習対象を抽出
    training_configs = extract_training_configs(config)
    target_list = []
    for tc in training_configs:
        target_list.extend(tc.generate_target_list())

    # tager_listの数だけ学習を行う
    for target_col in target_list:
        if "_" in target_col:
            target_name = target_col.split("_")[1]
        else:
            target_name = target_col
        config_feature = [
            feature for feature in config["feature"] if feature["name"] == target_name
        ][0]

        model_list, accuracy_list, feature_list = execute_training(
            loop_df, target_col, target_list, categorical_features, config
        )
        model_list_dict[target_col] = model_list
        accuracy_list_dict[target_col] = accuracy_list
        feature_list_dict[target_col] = feature_list

        # スタッキングを行う
        if config_feature["stacking_flag"]:
            loop_df = stacking(
                loop_df,
                target_col,
                model_list,
                accuracy_list,
                feature_list,
                config_feature,
                config,
            )

    return model_list_dict, accuracy_list_dict, feature_list_dict
