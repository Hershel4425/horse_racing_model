from pathlib import Path
import datetime

import pandas as pd


# ヘルパー関数
def format_time(timestamp):
    """タイムスタンプをフォーマットする"""
    return datetime.datetime.fromtimestamp(timestamp).strftime("%Y/%m/%d %H:%M:%S")


# ヘルパー関数
def create_result_df(params):
    """学習結果をデータフレームに保存する"""
    return pd.DataFrame(
        {
            "model_name": [params["model_name"]],
            "k_fold": [params["k_fold"]],
            "target_col": [params["target_col"]],
            "fold_number": [params["fold_number"]],
            "metric_name": [params["metric_name"]],
            "accuracy": [round(params["accuracy"], 3)],
            "start_time": [format_time(params["start_time"])],
            "end_time": [format_time(params["end_time"])],
            "training_time": [round(params["end_time"] - params["start_time"], 1)],
        }
    )


# ヘルパー関数
def write_df_to_csv(df, output_path):
    """データフレームをcsvに保存する"""
    mode = "a" if Path(output_path).is_file() else "w"
    header = not Path(output_path).is_file()
    df.to_csv(output_path, mode=mode, header=header, index=False)


def record_training_time(params):
    """
    学習時間を記録する関数

    Args:
        params (dict): A dictionary containing all the required parameters.
    """
    result_df = create_result_df(params)
    write_df_to_csv(result_df, params["output_path"])


def make_url_from_date(date, config, mode):
    """日付からurlを作成する

    Args:
        date (str): 日付
        config (dict): configファイル
        mode (str): 作成するurlの種類
    Returns:
        url (str): 作成したurl
    """

    mode_to_path = {
        "model": f"{config['root_path']}/50_machine_learning/output_data/model/{date}",
        "pred": f"{config['root_path']}/50_machine_learning/output_data/pred/{date}_pred.csv",
        "other": f"{config['root_path']}/50_machine_learning/output_data/学習記録/{date}_learning_time.csv",
    }
    return mode_to_path.get(mode, mode_to_path["other"])
