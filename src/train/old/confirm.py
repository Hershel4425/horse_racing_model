from IPython.display import display
import warnings

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from scipy.stats import norm
from matplotlib import rcParams
import matplotlib.pyplot as plt

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = [
    "Hiragino Maru Gothic Pro",
    "Yu Gothic",
    "Meirio",
    "Takao",
    "IPAexGothic",
    "IPAPGothic",
    "VL PGothic",
    "Noto Sans CJK JP",
]


def process_prediction_data(pred_data_path):
    """予測データを加工する関数

    Args:
        pred_data_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    pred_df = pd.read_csv(pred_data_path, encoding="utf_8_sig")
    pred_df = pred_df.sort_values(["date", "race_id", "馬番"]).reset_index(drop=True)

    if "予測_1_単勝" in pred_df.columns:
        for n in range(1, 19):
            pred_df.loc[pred_df[f"予測_{n}_単勝"] < 0.004, f"予測_{n}_単勝"] = 0.004
        for m in range(1, 19):
            pred_df[f"ソフトマックス_予測_{m}_単勝"] = np.exp(
                pred_df[f"予測_{m}_単勝"] / 5
            )
        softmax_totals = pred_df[
            [f"ソフトマックス_予測_{i+1}_単勝" for i in range(18)]
        ].sum(axis=1)
        for m in range(1, 19):
            pred_df.loc[pred_df["馬番"] == m, "予測単勝"] = pred_df[f"予測_{m}_単勝"]
            pred_df.loc[pred_df["馬番"] == m, "修正予測単勝"] = (
                pred_df[f"ソフトマックス_予測_{m}_単勝"] / softmax_totals
            )

        pred_df["予測単勝"] = 80 / pred_df["予測単勝"]
        pred_df["修正予測単勝"] = 0.8 / pred_df["修正予測単勝"]
        pred_df = pred_df.drop(
            columns=[f"ソフトマックス_予測_{i+1}_単勝" for i in range(18)]
            + [f"予測_{i+1}_単勝" for i in range(18)]
        )

    if "予測_1_期待値" in pred_df.columns:
        for m in range(1, 19):
            pred_df.loc[pred_df["馬番"] == m, "期待値"] = pred_df[f"予測_{m}_期待値"]

    if "1着馬番確率" in pred_df.columns:
        for race_id in pred_df["race_id"].unique():
            pred_df.loc[pred_df["race_id"] == race_id, "1着馬番確率"] = (
                pred_df.loc[pred_df["race_id"] == race_id, "1着馬番確率"]
                / pred_df.loc[pred_df["race_id"] == race_id, "1着馬番確率"].sum()
            )

    if "予測_1_5着着差" in pred_df.columns:
        for i in range(18):
            pred_df.loc[pred_df["馬番"] == i, "予測5着着差"] = pred_df[
                f"予測_{i+1}_5着着差"
            ]
        pred_df = pred_df.drop(columns=[f"予測_{i+1}_5着着差" for i in range(18)])

    for cols in [
        ("1C通過順位", "1C"),
        ("2C通過順位", "2C"),
        ("3C通過順位", "3C"),
        ("4C通過順位", "4C"),
    ]:
        if f"予測_1_{cols[0]}" in pred_df.columns:
            for i in range(18):
                pred_df.loc[pred_df["馬番"] == i, f"予測{cols[1]}通過順位"] = pred_df[
                    f"予測_{i+1}_{cols[0]}"
                ].astype(int)
            pred_df = pred_df.drop(columns=[f"予測_{i+1}_{cols[0]}" for i in range(18)])

    if "予測_1_前半ペース" in pred_df.columns:
        for i in range(18):
            pred_df.loc[pred_df["馬番"] == i, "予測前半ペース"] = pred_df[
                f"予測_{i+1}_前半ペース"
            ]
        pred_df = pred_df.drop(columns=[f"予測_{i+1}_前半ペース" for i in range(18)])

    if "予測_1_後半ペース" in pred_df.columns:
        for i in range(18):
            pred_df.loc[pred_df["馬番"] == i, "予測後半ペース"] = pred_df[
                f"予測_{i+1}_後半ペース"
            ]
        pred_df = pred_df.drop(columns=[f"予測_{i+1}_後半ペース" for i in range(18)])

    if "予測_1_タイム" in pred_df.columns:
        for i in range(18):
            pred_df.loc[pred_df["馬番"] == i, "予測タイム"] = pred_df[
                f"予測_{i+1}_タイム"
            ]
        pred_df = pred_df.drop(columns=[f"予測_{i+1}_タイム" for i in range(18)])

    return pred_df


def probability_calibration_plot(
    y_true,
    y_preds,
    y_cali=None,
    n_bins=30,
    yerr_c=1,
    xylim=1,
    tick=0.1,
    calib_method="",
):
    """予測確率のキャリブレーションプロットを作成する関数"""
    palette = [
        "#302c36",
        "#037d97",
        "#E4591E",
        "#C09741",
        "#EC5B6D",
        "#90A6B1",
        "#6ca957",
        "#D8E3E2",
    ]
    prob_true, prob_pred = calibration_curve(y_true, y_preds, n_bins=n_bins)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), dpi=120)
    ax = ax.flatten()
    ax[0].errorbar(
        x=prob_pred,
        y=prob_true,
        yerr=abs(prob_true - prob_pred) * yerr_c,
        fmt=".k",
        label="Actual",
        color=palette[1],
        capthick=0.5,
        capsize=3,
        elinewidth=0.7,
        ecolor=palette[1],
    )
    sns.lineplot(
        x=np.linspace(0, xylim, 11),
        y=np.linspace(0, xylim, 11),
        color=palette[-3],
        label="Perfectly calibrated",
        ax=ax[0],
        linestyle="dashed",
    )

    if isinstance(y_cali, np.ndarray):
        prob_true_, prob_pred_ = calibration_curve(y_true, y_cali, n_bins=n_bins)
        sns.lineplot(
            x=prob_pred_,
            y=prob_true_,
            color=palette[-5],
            label=f"{calib_method} Calibration",
            ax=ax[0],
            linestyle="solid",
        )

    sns.histplot(y_preds, bins=n_bins * 5, color=palette[1], ax=ax[1])

    for i, _ in enumerate(ax):
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].xaxis.grid(False)
        ax[i].yaxis.grid(True)

    ax[0].set_title("Probability calibration plot", fontdict={"fontweight": "bold"})
    ax[1].set_title("Histogram of predictions", fontdict={"fontweight": "bold"})

    ax[0].set_xticks(list(np.arange(0, xylim + tick, tick)))
    ax[0].set_yticks(list(np.arange(0, xylim + tick, tick)))
    ax[0].set(xlabel="predicted", ylabel="actual")
    fig.suptitle(
        f"{calib_method} Predictions in range {(0, xylim)}",
        ha="center",
        fontweight="bold",
        fontsize=16,
    )
    plt.tight_layout()


def plot_pred(pred_df):
    """予測の精度などを可視化する関数

    Args:
        pred_df (_type_): _description_

    Returns:
        _type_: _description_
    """
    palette = [
        "#302c36",
        "#037d97",
        "#E4591E",
        "#C09741",
        "#EC5B6D",
        "#90A6B1",
        "#6ca957",
        "#D8E3E2",
    ]

    if "1着馬番確率" in pred_df.columns:
        pred_df["1着"] = pred_df["着順"].apply(lambda x: 1 if x == 1 else 0)
        probability_calibration_plot(
            y_true=pred_df["1着"], y_preds=pred_df["1着馬番確率"]
        )

    if "予測単勝" in pred_df.columns:
        sns.jointplot(
            x="予測単勝", y="単勝", data=pred_df, kind="hist", color="#037d97"
        )
        plt.show()

    if "予測5着着差" in pred_df.columns:
        bins = np.linspace(
            pred_df["予測5着着差"].min(), pred_df["予測5着着差"].max(), 50
        )
        pred_df["予測5着着差bin"] = pd.cut(pred_df["予測5着着差"], bins=bins)
        grouped_stats = pred_df.groupby("予測5着着差bin")["5着着差"].agg(
            ["mean", "std"]
        )
        grouped_stats["one_sigma_upper"] = grouped_stats["mean"] + grouped_stats["std"]
        grouped_stats["one_sigma_lower"] = grouped_stats["mean"] - grouped_stats["std"]
        grouped_stats["two_sigma_upper"] = (
            grouped_stats["mean"] + 2 * grouped_stats["std"]
        )
        grouped_stats["two_sigma_lower"] = (
            grouped_stats["mean"] - 2 * grouped_stats["std"]
        )
        grouped_stats["prob_below_zero"] = grouped_stats.apply(
            lambda row: norm.cdf(0, loc=row["mean"], scale=row["std"]), axis=1
        )

        display(grouped_stats.isnull().sum())
        display(pred_df.isnull().sum())
        grouped_stats = grouped_stats.fillna(0)
        pred_df = pred_df.fillna(method="ffill")

        plt.figure(figsize=(15, 8))
        sns.boxplot(
            x="予測5着着差bin",
            y="5着着差",
            data=pred_df,
            palette=palette,
            showfliers=False,
        )
        plt.errorbar(
            x=grouped_stats.index.astype(str),
            y=grouped_stats["mean"],
            yerr=grouped_stats["std"],
            fmt="o",
            label="1σ",
            color=palette[2],
        )
        plt.fill_between(
            grouped_stats.index.astype(str),
            grouped_stats["one_sigma_lower"],
            grouped_stats["one_sigma_upper"],
            color=palette[2],
            alpha=0.2,
        )
        plt.fill_between(
            grouped_stats.index.astype(str),
            grouped_stats["two_sigma_lower"],
            grouped_stats["two_sigma_upper"],
            color=palette[3],
            alpha=0.2,
        )
        plt.xticks(rotation=45)
        plt.legend()
        plt.title("Predicted vs True 5着着差 with 1σ and 2σ intervals")
        plt.show()

        def add_statistics(row):
            bin_value = row["予測5着着差bin"]
            row["負の着差確率"] = grouped_stats.loc[bin_value, "prob_below_zero"]
            row["+1σ"] = row["予測5着着差"] + grouped_stats.loc[bin_value, "std"]
            row["-1σ"] = row["予測5着着差"] - grouped_stats.loc[bin_value, "std"]
            row["+2σ"] = row["予測5着着差"] + 2 * grouped_stats.loc[bin_value, "std"]
            row["-2σ"] = row["予測5着着差"] - 2 * grouped_stats.loc[bin_value, "std"]
            return row

        pred_df = pred_df.apply(add_statistics, axis=1)

    for cols in [
        ("1C通過順位", "1C"),
        ("2C通過順位", "2C"),
        ("3C通過順位", "3C"),
        ("4C通過順位", "4C"),
    ]:
        if f"予測{cols[1]}通過順位" in pred_df.columns:
            sns.boxplot(
                x=f"予測{cols[1]}通過順位",
                y=f"{cols[0]}",
                data=pred_df,
                palette=palette,
            )
            plt.title(f"Boxplot of 予測{cols[1]}通過順位 for each value of {cols[0]}")
            plt.show()

    return pred_df


def calculate_recovery_rates(pred_df):
    """期待値の閾値ごとの回収率とデータ数を計算する関数

    Args:
        pred_df (_type_): _description_

    Returns:
        _type_: _description_
    """
    expected_value_thresholds = [2.5 * i for i in range(0, 41)]
    recovery_rates = []
    data_counts = []

    if "期待値" not in pred_df.columns:
        return expected_value_thresholds, recovery_rates, data_counts

    for threshold in expected_value_thresholds:
        filtered_df = pred_df[
            (pred_df["期待値"] >= threshold) & (pred_df["単勝"] != 999)
        ]
        total_bets = len(filtered_df)
        total_return = filtered_df[filtered_df["着順"] == 1]["単勝"].sum()

        if total_bets > 0:
            recovery_rate = total_return / total_bets
        else:
            recovery_rate = 0

        recovery_rates.append(recovery_rate)
        data_counts.append(total_bets)

    return expected_value_thresholds, recovery_rates, data_counts


def plot_recovery_rates(expected_value_thresholds, recovery_rates, data_counts):
    """期待値の閾値ごとの回収率とデータ数をプロットする関数

    Args:
        expected_value_thresholds (_type_): _description_
        recovery_rates (_type_): _description_
        data_counts (_type_): _description_
    """

    # データ数を1000で割る
    data_counts = [count / 1000 for count in data_counts]

    plt.figure(figsize=(12, 6))

    # 線グラフの色を1色に統一
    palette = [
        "#302c36",
        "#037d97",
        "#E4591E",
        "#C09741",
        "#EC5B6D",
        "#90A6B1",
        "#6ca957",
        "#D8E3E2",
    ]
    sns.lineplot(
        x=expected_value_thresholds,
        y=recovery_rates,
        color=palette[0],
        marker="o",
        linestyle="-",
        linewidth=2,
    )
    plt.axhline(y=1, color="#E4591E", linestyle="-", linewidth=1)
    plt.xlabel("Expected Value Threshold")
    plt.ylabel("Recovery Rate")

    ax2 = plt.twinx()
    ax2.bar(expected_value_thresholds, data_counts, color=palette[1])
    ax2.axhline(y=1, color="#C09741", linestyle="-", linewidth=1)
    ax2.set_ylabel("Data Count", color=palette[1])
    ax2.tick_params(axis="y", labelcolor=palette[1])

    plt.title("Recovery Rates and Data Counts by Expected Value Threshold")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_boxplot_and_recovery_rates(df):
    """1着率ごとの着順の箱ひげ図と回収率をプロットする関数

    Args:
        df (_type_): _description_
    """
    # 1着率を2.5%ごとに区切る
    df["win_rate_bin"] = pd.cut(
        df["1着馬番確率"], bins=np.arange(0, 0.5 + 0.025, 0.025)
    )

    # 各領域での着順の箱ひげ図をplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    sns.boxplot(x="win_rate_bin", y="着順", data=df, ax=ax1)
    ax1.set_ylim(18, 1)  # y軸の範囲を設定
    ax1.set_xlabel("Win Rate Range")
    ax1.set_ylabel("Rank")
    ax1.set_title("Boxplot of Rank by Win Rate Range")
    ax1.set_xticklabels(
        [f"{i.right:.3f}" for i in df.win_rate_bin.cat.categories]
    )  # x軸のラベルを区間の上限値に変更

    # 各領域での還元率とデータ数をplot
    expected_value_thresholds = [0.0125 * i for i in range(0, 41)]
    recovery_rates = []
    data_counts = []
    for threshold in expected_value_thresholds:
        filtered_df = df[(df["1着馬番確率"] >= threshold) & (df["単勝"] != 999)]
        total_bets = len(filtered_df)
        total_return = filtered_df[filtered_df["着順"] == 1]["単勝"].sum()

        if total_bets > 0:
            recovery_rate = total_return / total_bets
        else:
            recovery_rate = 0

        recovery_rates.append(recovery_rate)
        data_counts.append(total_bets)
    data_counts = [count / 1000 for count in data_counts]

    palette = [
        "#302c36",
        "#037d97",
        "#E4591E",
        "#C09741",
        "#EC5B6D",
        "#90A6B1",
        "#6ca957",
        "#D8E3E2",
    ]
    sns.lineplot(
        x=expected_value_thresholds,
        y=recovery_rates,
        color=palette[0],
        marker="o",
        linestyle="-",
        linewidth=2,
    )
    ax2.axhline(y=1, color="#E4591E", linestyle="-", linewidth=1)
    ax2.set_xlabel("Win Rate Range")
    ax2.set_ylabel("Recovery Rate")
    ax2.set_title("Recovery Rates and Data Counts by Win Rate Range")
    ax2.set_xlim(0, 0.5)  # x軸の範囲を0から0.5に設定

    ax3 = ax2.twinx()
    ax3.bar(expected_value_thresholds, data_counts, color=palette[1], width=0.01)
    ax3.set_ylabel("Data Count", color=palette[1])
    ax3.axhline(y=1, color="#C09741", linestyle="-", linewidth=1)
    ax3.tick_params(axis="y", labelcolor=palette[1])

    plt.tight_layout()
    plt.show()


def display_race_winner_info(df, race_id):
    """レース勝利馬の情報を表示する関数

    Args:
        df (_type_): _description_
        race_id (_type_): _description_
    """

    # レースidに該当するレースの一行目を取得
    race_row = df.loc[df["race_id"] == race_id].iloc[0]

    # 表示する特徴量
    feature_columns = [
        "勝利馬当該コース競走馬通算複勝率",
        "勝利馬当該グレード競走馬通算複勝率",
        "勝利馬当該距離分類競走馬通算複勝率",
        "勝利馬当該競馬場競走馬通算複勝率",
        "勝利馬当該コース重賞競走馬通算試合数",
        "勝利馬当該コース重賞競走馬通算勝率",
        "勝利馬当該コース重賞競走馬通算複勝率",
        "勝利馬1走前着順",
        "勝利馬1走前1着タイム差",
        "勝利馬1走前距離",
        "勝利馬平均連対距離",
    ]

    # feature_columnsを基に、平均、分散、最大値、最小値の列名を作成
    stats_columns = ["mean", "std", "max", "min"]
    columns = []
    for col in feature_columns:
        for stat in stats_columns:
            columns.append(f"{col}_{stat}")

    # 結果を格納するデータフレームを初期化
    result_df = pd.DataFrame(columns=feature_columns, index=stats_columns)

    # race_rowから必要な値を取得し、結果のデータフレームに格納
    for col in feature_columns:
        result_df.at["mean", col] = race_row[f"{col}_mean"]
        result_df.at["std", col] = race_row[f"{col}_std"]
        result_df.at["max", col] = race_row[f"{col}_max"]
        result_df.at["min", col] = race_row[f"{col}_min"]

    replace_dict = {
        "勝利馬当該コース競走馬通算複勝率": "コース複勝率",
        "勝利馬当該グレード競走馬通算複勝率": "グレード複勝率",
        "勝利馬当該距離分類競走馬通算複勝率": "距離複勝率",
        "勝利馬当該競馬場競走馬通算複勝率": "競馬場複勝率",
        "勝利馬当該コース重賞競走馬通算試合数": "重賞試合数",
        "勝利馬当該コース重賞競走馬通算勝率": "重賞勝率",
        "勝利馬当該コース重賞競走馬通算複勝率": "重賞複勝率",
        "勝利馬1走前着順": "1走前着順",
        "勝利馬1走前1着タイム差": "1走前1着タイム差",
        "勝利馬1走前距離": "1走前距離",
        "勝利馬平均連対距離": "平均連対距離",
    }

    result_df = result_df.rename(columns=replace_dict)

    # 結果を表示
    print(f"{race_row['コース種類']}, {race_row['性別条件']}, {race_row['年齢条件']}")
    print(f"{race_row['グレード']}, {race_row['quarter']}, {race_row['距離']}")
    display(result_df)

    ext_df = df.loc[df["race_id"] == race_id].copy()

    # 馬番を新しい列として展開し、統計量を計算
    result_df = ext_df.pivot_table(
        index="馬番", values=["馬番別平均着順", "馬番別勝率"]
    )

    # データフレームを転置
    transposed_df = result_df.T

    # 書式設定用の関数を定義
    def format_value(value):
        if pd.isna(value):
            return value
        elif isinstance(value, float):
            return f"{value:.3f}"
        else:
            return value

    # 各セルに書式設定を適用
    formatted_df = transposed_df.applymap(format_value)

    # '馬番別平均着順'の行を小数第1位まで表示
    formatted_df.loc["馬番別平均着順"] = (
        formatted_df.loc["馬番別平均着順"].astype(float).apply(lambda x: f"{x:.1f}")
    )

    # '馬番別勝率'の行をパーセント表記に変換
    formatted_df.loc["馬番別勝率"] = (
        formatted_df.loc["馬番別勝率"].astype(float).apply(lambda x: f"{x:.1%}")
    )

    display(formatted_df)


def make_output(pred_df, race_id_list):
    """レースごとの馬番ごとの予測結果を表示する関数

    Args:
        pred_df (_type_): _description_
        race_id_list (_type_): _description_
    """

    warnings.simplefilter("ignore")

    # レース情報可視化のためのデータフレームを作成
    race_df = pred_df.copy()

    columns = [
        "race_id",
        "ラウンド",
        "枠番",
        "馬番",
        "馬名",
        "騎手",
        "コース種類",
        "距離",
        "distance_category",
        "騎手レーティング",
        "逃げ",
        "先行",
        "差し",
        "追込",
        "当該コース競走馬通算勝率",
        "当該コース競走馬通算複勝率",
        "当該コース単年競走馬通算勝率",
        "当該コース単年競走馬通算複勝率",
        "当該距離分類競走馬通算勝率",
        "当該距離分類競走馬通算複勝率",
        "当該距離分類単年競走馬通算勝率",
        "当該距離分類単年競走馬通算複勝率",
        "当該距離分類競走馬レーティング",
        "当該競馬場競走馬通算勝率",
        "当該競馬場競走馬通算複勝率",
        "当該競馬場単年競走馬通算勝率",
        "当該競馬場単年競走馬通算複勝率",
        "当該競馬場競走馬レーティング",
        "当該コース騎手通算勝率",
        "当該コース騎手通算複勝率",
        "当該コース単年騎手通算勝率",
        "当該コース単年騎手通算複勝率",
        "当該距離分類騎手通算勝率",
        "当該距離分類騎手通算複勝率",
        "当該距離分類単年騎手通算勝率",
        "当該距離分類単年騎手通算複勝率",
        "当該距離分類騎手レーティング",
        "当該競馬場騎手通算勝率",
        "当該競馬場騎手通算複勝率",
        "当該競馬場単年騎手通算勝率",
        "当該競馬場単年騎手通算複勝率",
        "当該競馬場騎手レーティング",
    ]

    if "1着馬番確率" in pred_df.columns:
        columns = [
            "race_id",
            "ラウンド",
            "枠番",
            "馬番",
            "馬名",
            "騎手",
            "コース種類",
            "距離",
            "distance_category",
            "騎手レーティング",
            "逃げ",
            "先行",
            "差し",
            "追込",
            "1着馬番確率",
            "当該距離分類競走馬レーティング",
            "当該競馬場競走馬レーティング",
            "当該距離分類騎手レーティング",
            "当該競馬場騎手レーティング",
        ]

    if "予測単勝" in pred_df.columns:
        columns.extend(["予測単勝", "修正予測単勝"])
    if "予測5着着差" in pred_df.columns:
        columns.extend(["予測5着着差", "負の着差確率", "+1σ", "-1σ", "+2σ", "-2σ"])
    if "期待値" in pred_df.columns:
        columns.extend(["期待値"])

    pred_df = pred_df[columns]

    for race_id in race_id_list:
        df = pred_df.loc[pred_df["race_id"] == race_id]

        df.loc[:, "R"] = df.loc[:, "ラウンド"].astype(int).astype(str) + "R"
        df = df.drop(columns=["ラウンド"])

        df["脚質"] = df[["逃げ", "先行", "差し", "追込"]].idxmax(axis=1)
        df = df.drop(columns=["逃げ", "先行", "差し", "追込"])

        if "1着馬番確率" in df.columns:
            df["1着馬番確率"] = df["1着馬番確率"] * 100
            df["1着馬番確率"] = df["1着馬番確率"].round(1)
            df = df.rename(columns={"1着馬番確率": "1着率"})

        for cols in [
            "当該距離分類競走馬レーティング",
            "当該競馬場競走馬レーティング",
            "当該距離分類騎手レーティング",
            "当該競馬場騎手レーティング",
        ]:
            if cols in df.columns:
                df[cols] = df[cols].round(1)
        df = df.rename(
            columns={
                "当該距離分類競走馬レーティング": "馬DR",
                "当該競馬場競走馬レーティング": "馬CR",
                "当該距離分類騎手レーティング": "JDR",
                "当該競馬場騎手レーティング": "JCR",
            }
        )

        if "予測単勝" in df.columns:
            df["予測単勝"] = df["予測単勝"].round(1)
            df["修正予測単勝"] = df["修正予測単勝"].round(1)

        if "予測5着着差" in df.columns:
            df["予測5着着差"] = df["予測5着着差"].round(1)
            df["負の着差確率"] = df["負の着差確率"] * 100
            df["負の着差確率"] = df["負の着差確率"].round(1)
            df["+1σ"] = df["+1σ"].round(1)
            df["-1σ"] = df["-1σ"].round(1)
            df["+2σ"] = df["+2σ"].round(1)
            df["-2σ"] = df["-2σ"].round(1)

        if "予測_期待値" in df.columns:
            df["予測期待値"] = df["予測_期待値"].round(1)
            df = df.drop(columns=["予測_期待値"])

        print(
            f'{df["R"].iloc[0]}, {df["コース種類"].iloc[0]}, {df["距離"].iloc[0].astype(int)}m'
        )

        df = df.drop(columns=["distance_category", "race_id", "R", "距離"])
        df = df.rename(
            columns={
                "rating": "馬R",
                "騎手レーティング": "騎手R",
                "予測前半ペース": "予測前半",
                "予測後半ペース": "予測後半",
            }
        )
        df = df.drop(columns=["馬DR", "馬CR", "JDR", "JCR"])
        df = df.reset_index(drop=True)

        styled_df = df.style.background_gradient(subset=["1着率"], cmap="Reds")
        # styled_df = styled_df.background_gradient(subset=['騎手R','馬DR','馬CR','JDR','JCR'], cmap='Blues')

        styled_df = styled_df.format(subset=["1着率"], precision=1)
        # styled_df = styled_df.format(subset=['騎手R','馬DR','馬CR','JDR','JCR'], precision=0)
        styled_df = styled_df.format(subset=["騎手R"], precision=0)
        styled_df = styled_df.format(subset=["期待値"], precision=1)

        display(styled_df)

        # レース情報を書き出す
        display_race_winner_info(race_df, race_id)


def pred_analysis(pred_data_path, race_id_list):
    """全ての処理を実行する関数

    Args:
        pred_data_path (_type_): _description_
        race_id_list (_type_): _description_
    """
    pred_df = process_prediction_data(pred_data_path)
    pred_df = plot_pred(pred_df)
    expected_value_thresholds, recovery_rates, data_counts = calculate_recovery_rates(
        pred_df
    )
    plot_recovery_rates(expected_value_thresholds, recovery_rates, data_counts)
    plot_boxplot_and_recovery_rates(pred_df)
    make_output(pred_df, race_id_list)
