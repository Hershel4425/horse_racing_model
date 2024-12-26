import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score


def confirm_accuracy(model, x_valid, y_valid, config_feature):
    """モデルの精度を確認し出力する関数

    Args:
        model (model): 学習したモデル
        x_valid (dataframe): テストデータの入力データフレーム
        y_valid (series): テストデータの目的変数のデータフレーム
        config_feature (dict): 学習方法を記載した辞書

    Returns:
        accuracy (float): 精度
    """

    if config_feature["train_method"] == "lightgbm":
        # 学習タイプが回帰か分類かで分岐
        if config_feature["train_type"] == "multiclass":
            # クラス予測の時、最も確率の高いクラスを予測結果とする
            y_pred = model.predict(x_valid, num_iteration=model.best_iteration)
            y_pred = [list(pred).index(max(list(pred))) + 1 for pred in y_pred]
            y_pred = np.array(y_pred)
            y_valid = np.array(y_valid) + 1

            # クラス1の正解率を計算する
            y_valid_filtered = [y for y, valid in zip(y_pred, y_valid) if valid == 1]
            y_ones = [1] * len(y_valid_filtered)
            accuracy = accuracy_score(y_ones, y_valid_filtered)
            print(f"クラス1の正解率: {accuracy}")

        elif config_feature["train_type"] == "regression":
            y_pred = model.predict(x_valid, num_iteration=model.best_iteration)
            y_pred = np.array(y_pred)
            y_valid = np.array(y_valid)

            # 予測値と正解値の差の絶対値の平均を計算する
            accuracy = np.mean(np.abs(y_pred - y_valid))
            print(f"平均絶対誤差: {accuracy}")

    elif config_feature["train_method"] == "tabnet":
        # 学習タイプが回帰か分類かで分岐
        if config_feature["train_type"] == "multiclass":
            accuracy = model.best_cost
            print(f"best valid score: {accuracy}")

        elif config_feature["train_type"] == "regression":
            accuracy = model.best_cost
            print(f"best vaild score: {accuracy}")

    return accuracy


def feature_importance(model, target_col, threshold=1):
    """特徴量の重要度を計算し、重要度の高い特徴量をプロット、さらに閾値以上の関数を取得する関数
    Args:
        model (model): 学習したモデル
        target_col (str): 学習対象の列名
        threshold (int): 重要度の閾値, default=1
    Returns:
        filtered_df (dataframe): 重要度が閾値以上の特徴量を格納したデータフレーム

    """
    feature_importances = model.feature_importance(importance_type="gain")
    feature_names = model.feature_name()
    feature_importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": feature_importances}
    )
    sorted_df = feature_importance_df.sort_values(by="Importance", ascending=False)
    sorted_df.to_csv(
        f"/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2/50_machine_learning/output_data/importance/{target_col}_importance.csv",
        index=False,
    )
    filtered_df = sorted_df[sorted_df["Importance"] >= threshold]

    # プロット
    print(f"{target_col}の特徴量の重要度をプロットします")
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_df["Feature"][:30], sorted_df["Importance"][:30])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 30 Most Important Features")
    plt.show()

    # # Plot the bottom 20 least important features
    # plt.figure(figsize=(10, 6))
    # plt.barh(sorted_df['Feature'][-30:], sorted_df['Importance'][-30:])
    # plt.xlabel('Importance')
    # plt.ylabel('Feature')
    # plt.title('Bottom 30 Least Important Features')
    # plt.show()

    # # 重要度のヒストグラムを表示
    # plt.figure(figsize=(10, 6))
    # plt.hist(sorted_df['Importance'], bins=50)
    # plt.xlabel('Importance')
    # plt.ylabel('Frequency')
    # plt.title('Importance Distribution')
    # plt.show()

    return filtered_df


def plot_lgb_learning_curve(evals_result, target_col, idx, config_feature):
    """学習曲線をプロットする関数

    Args:
        evals_result (dict): 学習過程を保存したdict
        target_col (str): 学習対象の列名
        idx (int): fold数
        config_feature (dict): 学習方法を記載した辞書
    """

    # metricを取り出す
    if config_feature["train_type"] == "multiclass":
        metric = "multi_logloss"
    else:
        metric = "rmse"

    train_results = evals_result["train"][metric]
    valid_results = evals_result["valid"][metric]

    # 学習曲線をプロット
    plt.plot(train_results, label="train")
    plt.plot(valid_results, label="valid")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.title(f"{target_col}_{idx} Training Progress")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_tabnet_learning_curve(tab_model, target_col, idx, config_feature):
    """学習曲線をプロットする関数

    Args:
        tab_model (model): 学習したモデル
        target_col (str): 学習対象の列名
        idx (int): fold数
        config_feature (dict): 学習方法を記載した辞書

    """

    # metricを取り出す
    if config_feature["train_type"] == "multiclass":
        metric = "logloss"
    else:
        metric = "rmse"

    train_results = tab_model.history["loss"]
    valid_results = tab_model.history["val_0_{}".format(metric)]

    # 学習曲線をプロット
    plt.plot(train_results, label="train")
    plt.plot(valid_results, label="valid")
    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.title(f"{target_col}_{idx} Training Progress")
    plt.legend()
    plt.grid(True)
    plt.show()
