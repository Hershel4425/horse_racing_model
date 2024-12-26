import pandas as pd
import os
from IPython.display import display

# グローバル変数としてパスを定義
ROOT_PATH = "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/153_競馬_v3/data"
DATA_PATHS = {
    "race_result": os.path.join(ROOT_PATH, "01_processed/11_race_result/race_result_df.csv"),
    "odds": os.path.join(ROOT_PATH, "01_processed/50_odds/odds_df.csv"),
    "pace": os.path.join(ROOT_PATH, "01_processed/30_pace/pace_df.csv"),
    "race_info": os.path.join(ROOT_PATH, "01_processed/10_race_info/race_info_df.csv"),
    "horse_past_performance": os.path.join(ROOT_PATH, "01_processed/20_horse_past_performance/horse_past_performance_df.csv"),
}
# 出力
RACE_RESULT_INFO_DF_PATH = os.path.join(ROOT_PATH, "01_processed/60_combined/race_result_info_df.csv")
RACE_RESULT_INFO_PERFORMANCE_DF_PATH = os.path.join(ROOT_PATH, "01_processed/60_combined/race_result_info_performance_df.csv")
RACE_RESULT_INFO_PERFORMANCE_PACE_DF_PATH = os.path.join(ROOT_PATH, "01_processed/60_combined/race_result_info_performance_pace_df.csv")
# 未来込みデータ
ALL_DATA_PATHS = {
    "race_result": os.path.join(ROOT_PATH, "01_processed/12_all_data_including_future/race_result_df.csv"),
    "race_info": os.path.join(ROOT_PATH, "01_processed/12_all_data_including_future/race_info_df.csv"),
    "pace": os.path.join(ROOT_PATH, "01_processed/30_pace/pace_df.csv"),
    "horse_past_performance": os.path.join(ROOT_PATH, "01_processed/12_all_data_including_future/horse_past_performance_df.csv"),
}
# 未来込みデータの出力先
ALL_RACE_RESULT_INFO_DF_PATH = os.path.join(ROOT_PATH, "01_processed/60_combined/race_result_info_with_future_df.csv")
ALL_RACE_RESULT_INFO_PERFORMANCE_DF_PATH = os.path.join(ROOT_PATH, "01_processed/60_combined/race_result_info_performance_with_future_df.csv")
ALL_RACE_RESULT_INFO_PERFORMANCE_PACE_DF_PATH = os.path.join(ROOT_PATH, "01_processed/60_combined/race_result_info_performance_pace_with_future_df.csv")


def run_combined():
    # データの読み込み
    dfs = {
        name: pd.read_csv(path, encoding="utf_8_sig")
        for name, path in DATA_PATHS.items()
    }

    # データの読み込み
    dfs = {
        name: pd.read_csv(path, encoding="utf_8_sig")
        for name, path in DATA_PATHS.items()
    }

    race_result_df = dfs["race_result"]
    odds_df = dfs["odds"]
    pace_df = dfs["pace"]
    race_info_df = dfs["race_info"]
    horse_past_performance_df = dfs["horse_past_performance"]
    horse_past_performance_df = horse_past_performance_df.drop(columns="レース名")

    # データshapeの確認
    print("race_result:", race_result_df.shape)
    print(race_result_df.isnull().sum())
    print("odds:", odds_df.shape)
    print(odds_df.isnull().sum())
    print("pace:", pace_df.shape)
    print(pace_df.isnull().sum())
    print("race_info:", race_info_df.shape)
    print(race_info_df.isnull().sum())
    print("horse_past_performance:", horse_past_performance_df.shape)
    print(horse_past_performance_df.isnull().sum())

    # データの結合
    race_result_info_df = pd.merge(
        race_result_df, race_info_df, on=["race_id"], how="left"
    )
    race_result_info_df.sort_values(
        ["年", "月", "日", "競馬場", "ラウンド"], inplace=True
    )

    # データshapeの確認
    print("race_result_info：", race_result_info_df.shape)
    display(race_result_info_df.head())
    print(race_result_info_df.isnull().sum())

    # データの書き込み
    race_result_info_df.to_csv(
        RACE_RESULT_INFO_DF_PATH, encoding="utf_8_sig", index=False
    )

    # データの結合
    race_result_info_performance_df = pd.merge(
        race_result_info_df,
        horse_past_performance_df,
        on=["horse_id", "年", "月", "日", "競馬場", "ラウンド", "日数", "回"],
        how="left",
    )
    race_result_info_performance_df.sort_values(
        ["年", "月", "日", "競馬場", "ラウンド"], inplace=True
    )
    # グレードがないレースの馬のデータを削除
    race_result_info_performance_df = race_result_info_performance_df.loc[
        ~race_result_info_performance_df["グレード"].isnull()
    ]

    # データshapeの確認
    print("race_result_info_performance_df:", race_result_info_performance_df.shape)
    display(race_result_info_performance_df.head())
    print(race_result_info_performance_df.isnull().sum())

    # データの書き込み
    race_result_info_performance_df.to_csv(
        RACE_RESULT_INFO_PERFORMANCE_DF_PATH, encoding="utf_8_sig", index=False
    )

    # データの結合
    race_result_info_performance_pace_df = pd.merge(
        race_result_info_performance_df, pace_df, on="race_id", how="left"
    )
    race_result_info_performance_pace_df.sort_values(
        ["年", "月", "日", "競馬場", "ラウンド"], inplace=True
    )

    # データshapeの確認
    print(
        "race_result_info_performance_pace_df:",
        race_result_info_performance_pace_df.shape,
    )
    display(race_result_info_performance_pace_df.head())
    print(race_result_info_performance_pace_df.isnull().sum())

    # データの書き込み
    race_result_info_performance_pace_df.to_csv(
        RACE_RESULT_INFO_PERFORMANCE_PACE_DF_PATH, encoding="utf_8_sig", index=False
    )

    # 全データの読み込み
    dfs = {
        name: pd.read_csv(path, encoding="utf_8_sig")
        for name, path in ALL_DATA_PATHS.items()
    }

    race_result_df = dfs["race_result"]
    race_info_df = dfs["race_info"]
    horse_past_performance_df = dfs["horse_past_performance"]
    horse_past_performance_df = horse_past_performance_df.drop(columns="レース名")


    race_result_df = dfs["race_result"]
    race_info_df = dfs["race_info"]
    horse_past_performance_df = dfs["horse_past_performance"]
    horse_past_performance_df = horse_past_performance_df.drop(columns="レース名")

    # データshapeの確認
    print("race_result:", race_result_df.shape)
    print(race_result_df.isnull().sum())
    print("race_info:", race_info_df.shape)
    print(race_info_df.isnull().sum())
    print("pace:", pace_df.shape)
    print(pace_df.isnull().sum())
    print("horse_past_performance:", horse_past_performance_df.shape)
    print(horse_past_performance_df.isnull().sum())

    # データの結合
    race_result_info_df = pd.merge(
        race_result_df, race_info_df, on=["race_id"], how="left"
    )
    race_result_info_df.sort_values(
        ["年", "月", "日", "競馬場", "ラウンド"], inplace=True
    )

    # データshapeの確認
    print("race_result_info：", race_result_info_df.shape)
    display(race_result_info_df.head())
    print(race_result_info_df.isnull().sum())

    # データの書き込み
    race_result_info_df.to_csv(
        ALL_RACE_RESULT_INFO_DF_PATH, encoding="utf_8_sig", index=False
    )

    # データの結合
    race_result_info_performance_df = pd.merge(
        race_result_info_df,
        horse_past_performance_df,
        on=["horse_id", "年", "月", "日", "競馬場", "ラウンド", "日数", "回"],
        how="left",
    )
    race_result_info_performance_df.sort_values(
        ["年", "月", "日", "競馬場", "ラウンド"], inplace=True
    )
    # グレードがないレースの馬のデータを削除
    race_result_info_performance_df = race_result_info_performance_df.loc[
        ~race_result_info_performance_df["グレード"].isnull()
    ]

    # データshapeの確認
    print("race_result_info_performance_df:", race_result_info_performance_df.shape)
    display(race_result_info_performance_df.head())
    print(race_result_info_performance_df.isnull().sum())

    # データの書き込み
    race_result_info_performance_df.to_csv(
        ALL_RACE_RESULT_INFO_PERFORMANCE_DF_PATH, encoding="utf_8_sig", index=False
    )

    # データの結合
    race_result_info_performance_pace_df = pd.merge(
        race_result_info_performance_df, pace_df, on="race_id", how="left"
    )
    race_result_info_performance_pace_df.sort_values(
        ["年", "月", "日", "競馬場", "ラウンド"], inplace=True
    )

    # データshapeの確認
    print(
        "race_result_info_performance_pace_df:",
        race_result_info_performance_pace_df.shape,
    )
    display(race_result_info_performance_pace_df.head())
    print(race_result_info_performance_pace_df.isnull().sum())

    # データの書き込み
    race_result_info_performance_pace_df.to_csv(
        ALL_RACE_RESULT_INFO_PERFORMANCE_PACE_DF_PATH, encoding="utf_8_sig", index=False
    )
