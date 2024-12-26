import pandas as pd
import numpy as np
import torch


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def apply_softmax_by_race(df):
    """ ソフトマックスを用いて勝率を計算する関数"""
    result_list = []
    for race_id, race_df in df.groupby("race_id"):
        scores = race_df["score"].values
        softmax_scores = softmax(scores)
        race_df["勝率"] = softmax_scores
        result_list.append(race_df)

    result_df = pd.concat(result_list, ignore_index=True)
    return result_df


# 複勝率作成
def calc_second_place_prob(df, i):
    """2着の確率を計算する関数"""
    mask = df["馬番"] != i
    scores = df["補正勝率"].values
    score_i = scores[df["馬番"] == i]
    return np.sum(scores[mask] * score_i / (1 - scores[mask]))


def calc_third_place_prob(df, i, second_place_probs):
    """3着の確率を計算する関数"""
    mask = df["馬番"] != i
    scores = df["補正勝率"].values
    score_i = scores[df["馬番"] == i]
    masked_second_place_probs = second_place_probs[
        np.in1d(df["馬番"][mask], df["馬番"][mask])
    ]
    return np.sum(
        scores[mask]
        * masked_second_place_probs
        * score_i
        / (1 - scores[mask] * masked_second_place_probs)
    )


def calc_top3_prob(ext_df, i):
    """1着、2着、3着の確率を計算する関数"""
    first_place_prob = ext_df[ext_df["馬番"] == i]["補正勝率"].values[0]
    second_place_prob = calc_second_place_prob(ext_df, i)
    second_place_probs = np.array(
        [
            calc_second_place_prob(ext_df[ext_df["馬番"] != j], k)
            for j, k in enumerate(ext_df["馬番"])
            if k != i
        ]
    )
    third_place_prob = calc_third_place_prob(ext_df, i, second_place_probs)
    return first_place_prob + second_place_prob + third_place_prob


def add_top3_prob(df):
    """複勝率を計算する関数"""
    df["複勝率"] = df.apply(lambda x: calc_top3_prob(df, x["馬番"]), axis=1)
    return df


def kelly_criterion(win_probs, odds):
    """勝率とオッズからケリー基準に基づいて購入金額割合を計算する関数"""

    # 勝率とオッズが同じ形状であることを確認
    assert win_probs.shape == odds.shape, "勝率とオッズの形状が一致しません"

    # ケリー基準の計算
    f = np.where(
        (win_probs == 0) | (odds == 0),
        0,
        np.where(odds == 1, 0, (win_probs * odds - 1) / (odds - 1)),
    )

    # 購入金額割合がマイナスの場合は0に修正
    f = np.where(f < 0, 0, f)
    # 購入金額割合が0.8より大きい場合は0.8に修正
    f = np.where(f >= 0.8, 0.8, f)

    return f


def generate_bets(race_id):
    # データの読み込み
    root_path = (
        "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2"
    )
    score_data = root_path + f"/40_automation/{race_id}_win_probs_df.csv"
    fuku_data = root_path + f"/40_automation/{race_id}_fuku_odds.csv"
    wide_data = root_path + f"/40_automation/{race_id}_wide_odds.csv"

    score_df = pd.read_csv(score_data, encoding="utf-8-sig")
    fuku_df = pd.read_csv(fuku_data)
    wide_df = pd.read_csv(wide_data)

    # それぞれ馬番を昇順でソート
    score_df = score_df.sort_values("馬番")
    fuku_df = fuku_df.sort_values("馬番")
    wide_df = wide_df.sort_values(["馬番1", "馬番2"])

    # 勝率の計算
    score_df = apply_softmax_by_race(score_df)

    # 型変換
    # "---.-"を0に変換
    fuku_df = fuku_df.replace("---.-", 0)
    fuku_df["最低オッズ"] = fuku_df["最低オッズ"].astype("float")
    fuku_df["最高オッズ"] = fuku_df["最高オッズ"].astype("float")
    fuku_df["馬番"] = fuku_df["馬番"].astype("int")
    wide_df = wide_df.replace("---.-", 0)
    wide_df["馬番1"] = wide_df["馬番1"].astype("int")
    wide_df["馬番2"] = wide_df["馬番2"].astype("int")
    wide_df["倍率下限"] = wide_df["倍率下限"].astype("float")
    wide_df["倍率上限"] = wide_df["倍率上限"].astype("float")

    # 勝率の補正を実施
    # 単勝倍率が0の場合は勝率を0にする
    # score_df.loc[score_df["単勝"] == 0, "勝率"] = 0
    score_df["補正勝率"] = score_df["勝率"]
    score_df.loc[score_df["補正勝率"] < 0.03, "補正勝率"] = 0
    # race_idごとに勝率の合計を計算
    total_win_rate_per_race = score_df["補正勝率"].sum()
    # 勝率を正規化
    score_df["補正勝率"] = score_df["補正勝率"] / total_win_rate_per_race

    # 複勝率の計算
    score_df = add_top3_prob(score_df)

    # torch作成
    # テンソル1: 勝率
    score_tensor = torch.zeros(1, 18)
    for _, row in score_df.iterrows():
        score_tensor[0, int(row["馬番"]) - 1] = row["補正勝率"]

    # テンソル2: 複勝勝率
    fuku_score_tensor = torch.zeros(1, 18)
    for _, row in score_df.iterrows():
        fuku_score_tensor[0, int(row["馬番"]) - 1] = row["複勝率"]

    # テンソル3: ワイド勝率
    wide_score_tensor = torch.zeros(1, 18 * 18)
    # 馬番と複勝率をNumPy配列に抽出
    horse_numbers = score_df["馬番"].values
    win_place_odds = score_df["複勝率"].values
    # 馬番をインデックスに変換
    horse_indices = horse_numbers - 1
    # 複勝率のベクトルを作成
    odds_vector = np.zeros(18)
    odds_vector[horse_indices] = win_place_odds
    # ベクトル同士の積で行列を作成
    odds_matrix = np.outer(odds_vector, odds_vector)
    # 行列をベクトルに変換してwide_score_tensorに格納
    wide_score_tensor[0] = torch.from_numpy(odds_matrix.ravel())

    # テンソル4: 単勝オッズ
    odds_tensor = torch.zeros(1, 18)
    for _, row in score_df.iterrows():
        odds_tensor[0, int(row["馬番"]) - 1] = 0.5
    # テンソル3: 複勝オッズ
    min_odds_tensor = torch.zeros(1, 18)
    for _, row in fuku_df.iterrows():
        min_odds_tensor[0, int(row["馬番"]) - 1] = (row['最高オッズ'] + row["最低オッズ"]) / 2

    # テンソル6: 倍率下限 (ワイド)
    wide_lower_tensor = torch.zeros(1, 18 * 18)
    for _, row in wide_df.iterrows():
        index = (int(row["馬番1"]) - 1) * 18 + (int(row["馬番2"]) - 1)
        wide_lower_tensor[0, index] = (row['倍率上限'] + row["倍率下限"]) / 2

    # numpyへの変換
    score_array = score_tensor.numpy()
    fuku_score_array = fuku_score_tensor.numpy()
    wide_score_array = wide_score_tensor.numpy()
    odds_array = odds_tensor.numpy()
    fuku_array = min_odds_tensor.numpy()
    wide_array = wide_lower_tensor.numpy()

    # 倍率に上限を与える
    odds_array = np.where(odds_array >= 30, 30, odds_array)
    fuku_array = np.where(fuku_array >= 15, 15, fuku_array)
    wide_array = np.where(wide_array >= 20, 20, wide_array)

    # ケリー基準に基づいて賭け金の割合を計算
    bet_fractions = kelly_criterion(score_array, odds_array)
    fuku_bet_fractions = kelly_criterion(fuku_score_array, fuku_array)
    wide_bet_fractions = kelly_criterion(wide_score_array, wide_array)

    # 賭け金の割合をデータフレームに格納
    if len(score_df) > 0:
        bet_fractions_row = bet_fractions[0][: len(score_df)]
        fuku_fractions_row = fuku_bet_fractions[0][: len(score_df)]
        score_df.loc[score_df["race_id"] == race_id, "単勝賭け割合"] = bet_fractions_row
        score_df.loc[score_df["race_id"] == race_id, "複勝賭け割合"] = (
            fuku_fractions_row
        )
    else:
        new_row = pd.DataFrame(
            {
                "race_id": [race_id],
                "馬番": range(1, 19),
                "単勝賭け割合": bet_fractions[0],
            }
        )
        fuku_new_row = pd.DataFrame(
            {
                "race_id": [race_id],
                "馬番": range(1, 19),
                "複勝賭け割合": fuku_bet_fractions[0],
            }
        )
        score_df = pd.concat([score_df, new_row], ignore_index=True)
        score_df = pd.concat([score_df, fuku_new_row], ignore_index=True)

    if len(score_df) > 0:
        wide_bet_matrix = wide_bet_fractions[0].reshape(18, 18)[
            : len(score_df), : len(score_df)
        ]
        wide_columns = [f"{j}_ワイド" for j in range(1, len(score_df) + 1)]
        score_df.loc[score_df["race_id"] == race_id, wide_columns] = wide_bet_matrix
    else:
        wide_columns = [f"{j}_ワイド" for j in range(1, 19)]
        score_df.loc[score_df["race_id"] == race_id, wide_columns] = wide_bet_fractions[
            0
        ]

    # 単勝賭け割合のデータフレーム
    tansho_df = score_df[["race_id", "馬番", "単勝賭け割合"]]
    tansho_df = tansho_df.rename(columns={"馬番": "馬番1"})

    # 複勝賭け割合のデータフレーム
    fukusho_df = score_df[["race_id", "馬番", "複勝賭け割合"]]
    fukusho_df = fukusho_df.rename(columns={"馬番": "馬番1"})

    wide_columns = [col for col in score_df.columns if col.endswith("_ワイド")]
    wide_data = []
    for _, row in score_df.iterrows():
        for col in wide_columns:
            if (row[col] != 0) & (pd.notnull(row[col])):
                uma2 = int(col.split("_")[0])
                wide_data.append([row["race_id"], row["馬番"], uma2, row[col]])

    wide_kelly_df = pd.DataFrame(
        wide_data, columns=["race_id", "馬番1", "馬番2", "ワイド賭け割合"]
    )

    # 各種賭け割合_dfを保存
    tansho_df.to_csv(
        root_path + f"/40_automation/{race_id}_bet_tansho.csv", index=False
    )
    fukusho_df.to_csv(
        root_path + f"/40_automation/{race_id}_bet_fukusho.csv", index=False
    )
    wide_kelly_df.to_csv(
        root_path + f"/40_automation/{race_id}_bet_wide.csv", index=False
    )

    score_df.to_csv(root_path + f"/40_automation/{race_id}_bet_all.csv", index=False)
