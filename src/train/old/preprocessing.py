from IPython.display import display

import numpy as np


from .result_saving import make_url_from_date


def drop_data(df, config, feature_key="train_flag", drop_value=False):
    """不要なデータを削除する

    Args:
        df (dataframe): inputのデータフレーム
        config (dict): configファイル

    Returns:
        df (dataframe): 不要なデータを削除したデータフレーム
    """
    # 人気は状況に関わらず削除する
    ids_list = [f"{i + 1}_人気" for i in range(18)]
    df = df.drop(columns=ids_list)
    # 距離別タイムも状況に関わらず削除する
    dist_columns = [f"{i}m" for i in range(100, 3601, 100)]
    for m in [2700, 2900, 3100, 3300, 3500]:
        dist_columns.remove(f"{m}m")
    df = df.drop(columns=dist_columns)

    # config情報で削除するデータを指定する
    for feature in config["feature"]:
        if (feature[feature_key] == drop_value) & (not feature["stacking_flag"]):
            column_pattern = (
                f'{feature["name"]}'
                if feature["column_num"] == 1
                else f'{{}}_{feature["name"]}'
            )
            columns_to_drop = (
                [column_pattern.format(i + 1) for i in range(feature["column_num"])]
                if feature["column_num"] > 0
                else [feature["name"]]
            )
            df = df.drop(columns=columns_to_drop, errors="ignore")

    return df


def feature_specific_operations(feature):
    """特徴量ごとの特定の操作を返す"""
    if feature["name"] == "単勝":
        return lambda x: 80 / x
    elif "通過順位" in feature["name"]:
        return lambda x: x
    elif "着順" in feature["name"]:
        return lambda x: x / x.sum()
    else:
        return lambda x: x


def process_target(df, config):
    """学習対象の列を処理する

    Args:
        df (dataframe): inputのデータフレーム
        config (dict): configファイル

    Returns:
        df: 処理したデータフレーム
    """
    for feature in config["feature"]:
        if feature["train_flag"]:
            column_pattern = (
                f'{feature["name"]}'
                if feature["column_num"] == 1
                else f'{{}}_{feature["name"]}'
            )
            if feature["train_type"] == "multiclass":
                def operation(x):
                    return x - 1
            elif feature["train_type"] == "regression":
                operation = feature_specific_operations(feature)
            else:
                raise ValueError(f'{feature["name"]}のtrain_typeが不正です')

            if feature["column_num"] > 1:
                for i in range(feature["column_num"]):
                    df[column_pattern.format(i + 1)] = df[
                        column_pattern.format(i + 1)
                    ].apply(operation)
            else:
                df[feature["name"]] = df[feature["name"]].apply(operation)
    return df


def process_null(df, config):
    """nullを処理する

    Args:
        df (_type_): データフレーム
        config (_type_): configファイル
    Returns:
        df: 処理したデータフレーム
    """
    # 全ての列に対して、nullをその列の平均値で埋める
    print("nullを処理します...")
    df = df.fillna(0)
    print("nullを処理しました")

    return df


class TrainingConfig:
    """学習対象の設定を保持するクラス"""

    def __init__(self, feature_name, column_num, use_model, opt_flag, model_date):
        self.feature_name = feature_name
        self.column_num = column_num
        self.use_model = use_model
        self.opt_flag = opt_flag
        self.model_date = model_date

    def generate_target_list(self):
        if self.column_num == 18:
            return [f"{i + 1}_{self.feature_name}" for i in range(18)]
        elif self.column_num == 1:
            return [self.feature_name]

    def generate_opt_dict(self):
        if self.column_num == 18:
            return {f"{i + 1}_{self.feature_name}": self.opt_flag for i in range(18)}
        elif self.column_num == 1:
            return {self.feature_name: self.opt_flag}

    def generate_model_url_dict(self, config):
        if self.column_num == 18:
            return {
                f"{i + 1}_{self.feature_name}": make_url_from_date(
                    self.model_date, config, "model"
                )
                for i in range(18)
            }
        elif self.column_num == 1:
            return {
                self.feature_name: make_url_from_date(self.model_date, config, "model")
            }


def extract_training_configs(config):
    """学習対象を抽出する

    Args:
        config (dict): configファイル
    Returns:
        training_configs (list): 学習対象のリスト
    """
    training_configs = []
    for feature in config["feature"]:
        if feature["train_flag"] | feature["stacking_flag"]:
            tc = TrainingConfig(
                feature_name=feature["name"],
                column_num=feature["column_num"],
                use_model=feature["use_model_train"]["flag"],
                opt_flag=feature["opt_flag"],
                model_date=feature["use_model_train"]["date"],
            )
            training_configs.append(tc)
    return training_configs


def split_input_output(train_data, target_data, config):
    """学習データとテストデータに分割する

    Args:
        train_data (datagrame): 入力データのデータフレーム
        target_data (series): 目的変数のデータフレーム
        config (dict): configファイル

    Returns:
        x_train (dataframe): 学習データの入力データフレーム
        x_test (dataframe): テストデータの入力データフレーム
        t_train (series): 学習データの目的変数のデータフレーム
        t_test (series): テストデータの目的変数のデータフレーム
    """
    # trainとtestに分ける
    test_num = config["test_num"]
    x_train = train_data.head(train_data.shape[0] - test_num)
    x_test = train_data.tail(test_num)

    # targetを取り出す
    t_train = target_data.head(target_data.shape[0] - test_num)
    t_test = target_data.tail(test_num)

    display(f"x_train.shape:{x_train.shape}, x_test.shape:{x_test.shape}")
    display(f"t_train.shape:{t_train.shape}, t_test.shape:{t_test.shape}")

    return x_train, x_test, t_train, t_test


# ヘルパー関数
def under_sampling(df, target_col):
    """目的変数が0のデータをアンダーサンプリングする

    Args:
        df (dataframe): アンダーサンプリングするデータフレーム
        target_col (str): 目的変数の列名

    Returns:
        undersampled_df (dataframe): アンダーサンプリング後のデータフレーム
    """
    # 目的変数が0のインデックスとそれ以外のインデックス
    zero_indices = df[df[target_col] == 0].index
    non_zero_indices = df[df[target_col] != 0].index
    nan_indices = df[df[target_col].isnull()].index

    print(
        f"0の数:{len(zero_indices)}, 0以外の数:{len(non_zero_indices)}, nanの数:{len(nan_indices)}"
    )

    # 10%をアンダーサンプリングする
    undersampled_zero_indices = np.random.choice(
        zero_indices, int((len(zero_indices) * 0.1)), replace=False
    )

    # 新しいインデックスを合成
    undersampled_indices = np.concatenate([undersampled_zero_indices, non_zero_indices])

    # アンダーサンプリング後のデータセット
    undersampled_df = df.loc[undersampled_indices]
    # インデックスを振り直す
    undersampled_df = undersampled_df.reset_index(drop=True)

    return undersampled_df
