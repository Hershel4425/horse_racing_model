# 取得したデータを特徴量作成が可能になるように加工するプログラム

import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import math
import datetime
import os
from IPython.display import display
import glob
import traceback
import multiprocessing as mp
from functools import partial


DATA_STRING = datetime.datetime.now().strftime("%Y%m%d")

# 全体のルートパス
ROOT_PATH = "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/153_競馬_v3/data"

# idと名前の対応辞書
HORSE_DICT = ROOT_PATH + "/01_processed/99_id_name_dict/horse_dict.pkl"
JOCKEY_DICT = ROOT_PATH + "/01_processed/99_id_name_dict/jockey_dict.pkl"
TRAINER_DICT = ROOT_PATH + "/01_processed/99_id_name_dict/trainer_dict.pkl"
HORSE_DICT_BACKUP = ROOT_PATH + f"/01_processed/99_id_name_dict/backup/{DATA_STRING}_horse_dict.pkl"
JOCKEY_DICT_BACKUP = ROOT_PATH + f"/01_processed/99_id_name_dict/backup/{DATA_STRING}_horse_dict.pkl"
TRAINER_DICT_BACKUP = ROOT_PATH + f"/01_processed/99_id_name_dict/backup/{DATA_STRING}_horse_dict.pkl"

# データの保存先
RESULT_WITH_FUTURE_PATH = ROOT_PATH + "/01_processed/12_all_data_including_future/race_result_df.csv"
INFO_WITH_FUTURE_PATH = ROOT_PATH + "/01_processed/12_all_data_including_future/race_info_df.csv"
PERFORMANCE_WITH_FUTURE_PATH = ROOT_PATH + "/01_processed/12_all_data_including_future/horse_past_performance_df.csv"
# 血統データを保存するパス
PEDIGREE_ID_PATH = ROOT_PATH + "/01_processed/40_pedigree/pedigree_horse_id.csv"
PEDIGREE_NAME_PATH = ROOT_PATH + "/01_processed/40_pedigree/pedigree_horse_name.csv"

# 未来データの読み込みに使うパス
FORECAST_PATH = ROOT_PATH + "/00_raw/12_future_data/race_forecast"
FUTURE_INFO_PATH = ROOT_PATH +  "/00_raw/12_future_data/future_race_info"

# データ読み込みパス
RACE_RESULT_PATH = ROOT_PATH +  "/00_raw/11_race_result/race_result_df.csv"
ODDS_PATH = ROOT_PATH +  "/00_raw/50_odds/odds_df.csv"
PACE_PATH = ROOT_PATH +  "/00_raw/30_pace/pace_df.csv"
RACE_INFO_PATH = ROOT_PATH +  "/00_raw/10_race_info/race_info_df.csv"
HORSE_PAST_PERFORMANCE_PATH = ROOT_PATH +  "/00_raw/20_horse_past_performance/horse_past_performance_df.csv"
PEDIGREE_PATH = ROOT_PATH +  "/00_raw/40_pedigree/pedigree_df.csv"


##### 不要レース削除処理
def delete_useless_race(result_df):
    """不要なレースを削除し競争中止馬などを18着などにする処理をする関数

    Args:
        result_df (dataframe): レース結果を保存したDF

    Returns:
        result_df (dataframe): レース結果を保存したDF
    """
    print("------取り消し除外馬処理開始------")
    print("削除前データサイズ：", result_df.shape)
    result_df = result_df.loc[~result_df["着順"].isnull()]
    result_df = result_df.loc[~result_df["着順"].str.contains("取")]
    result_df = result_df.loc[~result_df["着順"].str.contains("除")]
    print("削除後データサイズ：", result_df.shape)
    print("------不要データ削除処理開始------")
    # 降着があるレース、失格があるレースは削除する
    print("総レース数：", len(list(result_df["race_id"].unique())))
    print(
        "降着あり：",
        len(
            list(
                result_df.loc[result_df["着順"].str.contains("\(")]["race_id"].unique()
            )
        ),
    )
    print(
        "失格あり：",
        len(
            list(
                result_df.loc[result_df["着順"].str.contains("失")]["race_id"].unique()
            )
        ),
    )
    print(
        "両方あり：",
        len(
            list(
                result_df.loc[
                    result_df["着順"].str.contains("失")
                    & result_df["着順"].str.contains("\(")
                ]["race_id"].unique()
            )
        ),
    )
    # 該当レース削除
    print("削除前データサイズ：", result_df.shape)
    result_df = result_df.loc[
        ~result_df["race_id"].isin(
            list(
                result_df.loc[result_df["着順"].str.contains("\(")]["race_id"].unique()
            )
        )
    ]
    result_df = result_df.loc[
        ~result_df["race_id"].isin(
            list(
                result_df.loc[result_df["着順"].str.contains("失")]["race_id"].unique()
            )
        )
    ]
    # 人気が元データに存在しないレースがいくつかあるため、それらのレースを除外する
    result_df = result_df.loc[
        ~result_df["race_id"].isin(
            list(result_df.loc[result_df["人気"].isna()]["race_id"].unique())
        )
    ]
    # 削除済みレース数
    print("削除後データサイズ：", result_df.shape)
    print("不要データ削除後総レース数：", len(list(result_df["race_id"].unique())))
    print("")
    # 競争中止は18位、取り消し除外は削除する
    print("-----競争中止馬を最下位にする処理開始-----")
    print("処理前データサイズ：", result_df.shape)
    result_df.loc[result_df["着順"].isin(["中止"]), "着順"] = "18"
    result_df["着順"] = result_df["着順"].astype("int")
    print("処理後データサイズ：", result_df.shape)
    print("着順ユニーク要素リスト：", list(result_df["着順"].unique()))
    print("着順null数：", result_df["着順"].isnull().sum())
    print("")

    return result_df


def process_bracket_number(result_df):
    """枠番をint変数に処理する関数

    Args:
        result_df (dataframe): レース結果を保存したDF

    Returns:
        result_df (dataframe): レース結果を保存したDF
    """
    print("-----枠番処理開始-----")
    # 枠番をカテゴリ変数にする
    result_df = result_df.rename(columns={"枠": "枠番"})
    result_df["枠番"] = result_df["枠番"].astype("int")
    print("枠番null数：", result_df["枠番"].isnull().sum())
    print("")

    return result_df


def process_horse_number(result_df):
    """馬番をint変数にする関数

    Args:
        result_df (dataframe): レース結果を保存したDF

    Returns:
        result_df (dataframe): レース結果を保存したDF
    """
    print("-----馬番処理開始-----")
    # 枠番をカテゴリ変数にする
    result_df["馬番"] = result_df["馬番"].astype("int")
    print("馬番null数：", result_df["馬番"].isnull().sum())
    print("")

    return result_df


def process_sex_and_age(result_df):
    """性齢を性別年齢に分ける関数

    Args:
        result_df (dataframe): レース結果を保存したDF

    Returns:
        result_df (dataframe): レース結果を保存したDF
    """
    print("-----性別年齢処理開始-----")
    result_df["性"] = result_df["性齢"].apply(lambda x: str(x[:1]))
    result_df["齢"] = result_df["性齢"].apply(lambda x: int(x[1:]))
    # 性齢列削除
    result_df = result_df.drop(columns=["性齢"])
    print("性別ユニーク要素：", result_df["性"].unique())
    print("年齢ユニーク要素：", result_df["齢"].unique())
    print("性別null数：", result_df["性"].isnull().sum())
    print("年齢null数：", result_df["齢"].isnull().sum())
    print("")

    return result_df


def process_basis_weight(result_df):
    """斤量をfloatにする関数

    Args:
        result_df (dataframe): レース結果を保存したDF

    Returns:
        result_df (dataframe): レース結果を保存したDF
    """
    print("-----斤量処理開始-----")
    # 斤量はfloatにする
    # 斤量未定は55kgにする
    result_df.loc[result_df["斤量"] == "未定", "斤量"] = 55.0
    result_df["斤量"] = result_df["斤量"].astype("float")
    print("斤量null数：", result_df["斤量"].isnull().sum())
    print("")

    return result_df


def process_time(result_df):
    """時間系の処理をする関数

    Args:
        result_df (dataframe): レース結果を保存したDF

    Returns:
        result_df (dataframe): レース結果を保存したDF
    """
    print("-----タイム処理開始-----")

    # タイム列がNaNのインデックスを取得
    nan_time_index = result_df["タイム"].isna()

    # タイム列を秒数に変換
    result_df.loc[~nan_time_index, "タイム"] = (
        result_df.loc[~nan_time_index, "タイム"]
        .str.split(":", expand=True)
        .astype(float)
        .apply(lambda x: x[0] * 60 + x[1], axis=1)
    )

    # 1着タイム差を計算
    result_df["1着タイム差"] = result_df.groupby("race_id")["タイム"].transform(
        lambda x: x - x.min()
    )

    # 先位タイム差を計算
    result_df["先位タイム差"] = result_df.groupby("race_id")["タイム"].diff().fillna(0)
    result_df.loc[result_df["着順"] == 1, "先位タイム差"] = 0

    # タイム列がNaNの行に対して、1着タイム差や先位タイム差列もNaNに設定
    result_df.loc[nan_time_index, ["タイム", "1着タイム差", "先位タイム差"]] = np.nan

    print("タイム null数：", result_df["タイム"].isnull().sum())
    print("先位タイム差 null数：", result_df["先位タイム差"].isnull().sum())
    print("")

    return result_df


def process_fifth_margin(result_df, race_id):
    """各レースの5着の着差を計算する関数"""
    df = result_df[result_df["race_id"] == race_id].copy()
    df["1着着差"] = df["着差"].cumsum()
    # 5着が存在するかどうかで分岐
    if 5 not in df["着順"].values:
        if 4 not in df["着順"].values:
            if 3 not in df["着順"].values:
                if 2 not in df["着順"].values:
                    fifth_rank = 1
                else:
                    fifth_rank = 2
            else:
                fifth_rank = 3
        else:
            fifth_rank = 4
    else:
        fifth_rank = 5
    fifth_place_gap = df[df["着順"] == fifth_rank]["1着着差"].values[0]
    df["5着着差"] = df["1着着差"] - fifth_place_gap

    # 5着着差が存在しない場合は、5着着差を10にし、５着着差の最大値を20に設定する
    df.loc[df["5着着差"].isnull(), "5着着差"] = 10
    df.loc[df["5着着差"] > 20, "5着着差"] = 20

    return df


def process_margin(result_df):
    """着差を処理する関数

    Args:
        result_df (dataframe): レース結果を保存したDF

    Returns:
        result_df (dataframe): レース結果を保存したDF
    """
    print("-----着差処理開始-----")
    print("処理前dfサイズ：", result_df.shape)
    # 着差で整数以下の秒数に補正を掛けるため着差をタイム差の平均値で変換する
    margin_list = list(result_df["着差"].unique())

    margin_dict = {}
    for margin in margin_list:
        mean = result_df.loc[result_df["着差"] == margin]["先位タイム差"].mean()
        margin_dict[margin] = mean
    # margin列を追加
    result_df["着差"] = result_df["着差"].apply(lambda x: margin_dict[x])

    print("着差処理途中null数：", result_df["着差"].isnull().sum())

    # 1着は埋めておく
    result_df.loc[result_df["着差"].isnull() & result_df["着順"] == 1, "着差"] = 0
    # 1着以外のnullは10秒差扱いとする
    result_df["着差"] = result_df["着差"].fillna(10)
    result_df["着差"] = result_df["着差"].astype("float")

    # 2着との着差を計算
    # 並列処理で2着着差を計算
    with mp.Pool(mp.cpu_count()) as pool:
        race_ids = result_df["race_id"].unique()
        # functools.partialを使用してresult_dfをprocess_race_idにバインドします
        func = partial(process_fifth_margin, result_df)
        results = pool.map(func, race_ids)

    result_df = pd.concat(results)

    print("処理後dfサイズ：", result_df.shape)

    result_df = result_df.drop(["1着着差"], axis=1)

    print("着差null数：", result_df["着差"].isnull().sum())
    print("5着着差null数：", result_df["5着着差"].isnull().sum())
    print("")

    print("着差のリスト")
    print(result_df["着差"].value_counts())

    return result_df


def process_win(result_df):
    """オッズをfloatにする関数

    Args:
        result_df (dataframe): レース結果を保存したDF

    Returns:
        result_df (dataframe): レース結果を保存したDF
    """
    print("-----単勝処理開始-----")
    result_df = result_df.rename(columns={"単勝オッズ": "単勝"})
    # ---は999にしておく
    result_df.loc[result_df["単勝"] == "--", "単勝"] = "0"
    result_df.loc[result_df["単勝"] == "---", "単勝"] = "0"
    result_df.loc[result_df["単勝"].isnull(), "単勝"] = "0"
    result_df["単勝"] = result_df["単勝"].astype("float")
    print("単勝null数：", result_df["単勝"].isnull().sum())
    print("")

    return result_df


def process_favorite(result_df):
    """人気をカテゴリ変数にする関数

    Args:
        result_df (dataframe): レース結果を保存したDF

    Returns:
        result_df (dataframe): レース結果を保存したDF
    """
    # nanは18にしておく
    result_df["人気"] = result_df["人気"].astype("int")
    result_df.loc[result_df["人気"].isnull(), "人気"] = 18
    result_df["人気"] = result_df["人気"].astype("int")

    print("-----人気処理開始-----")
    print("人気null数：", result_df["人気"].isnull().sum())
    print("")

    return result_df


def ext_weeight_increase_and_decrease(weight):
    """体重の増減を取り出す関数

    Args:
        result_df (dataframe): レース結果を保存したDF

    Returns:
        result_df (dataframe): レース結果を保存したDF
    """
    if len(weight.split("(")) == 1:
        incdec = 0
    elif len(weight.split("(")) >= 2:
        incdec = int(weight.split("(")[1].split(")")[0])

    return incdec


def process_weight(result_df):
    """体重列から馬体重と増減に分割する関数

    Args:
        result_df (dataframe): レース結果を保存したDF

    Returns:
        result_df (dataframe): レース結果を保存したDF
    """
    print("-----体重処理開始-----")
    result_df = result_df.rename(columns={"馬体重(増減)": "馬体重"})
    # 計測不能は'000(0)'にしておく
    result_df.loc[result_df["馬体重"] == "計不", "馬体重"] = "000(0)"
    result_df.loc[result_df["馬体重"].isna(), "馬体重"] = "000(0)"
    result_df["増減"] = result_df["馬体重"].apply(
        lambda x: ext_weeight_increase_and_decrease(x)
    )
    result_df["馬体重"] = result_df["馬体重"].apply(lambda x: int(x.split("(")[0]))

    print("馬体重null数：", result_df["馬体重"].isnull().sum())
    print("増減null数：", result_df["増減"].isnull().sum())
    print("")

    return result_df


def process_trainer_and_affiliation(result_df):
    """厩舎列から調教師と所属に分割する関数

    Args:
        result_df (dataframe): レース結果を保存したDF

    Returns:
        result_df (dataframe): レース結果を保存したDF
    """
    print("-----調教師処理開始-----")
    # 所属と調教師名に分ける
    result_df = result_df.rename(columns={"厩舎": "調教師"})
    result_df["所属"] = result_df["調教師"].apply(lambda x: str(x[0:2]))
    result_df["調教師"] = result_df["調教師"].apply(lambda x: str(x[2:]))

    print("所属null数：", result_df["所属"].isnull().sum())
    print("調教師null数：", result_df["調教師"].isnull().sum())
    print("")

    return result_df


def separate_turn_rank(x, c=1):
    """コーナー通過順位を文字列からリストに分割する関数

    Args:
        result_df (dataframe): レース結果を保存したDF

    Returns:
        result_df (dataframe): レース結果を保存したDF
    """
    if pd.isna(x):
        rank = np.NaN
    else:
        x = x.split("-")
        # 通過してないコーナーはnanで埋める
        if len(x) > 4 - c:
            rank = float(x[4 - c])
        else:
            rank = np.NaN
    return rank


def process_turn_passing_rank(result_df):
    """コーナー通過順位を取り出す関数

    Args:
        result_df (dataframe): レース結果を保存したDF

    Returns:
        result_df (dataframe): レース結果を保存したDF
    """
    print("-----コーナー通過順処理開始-----")
    # 各コーナーごとの通過順位を求める
    result_df["1C通過順位"] = result_df["コーナー通過順"].apply(
        lambda x: separate_turn_rank(x, 1)
    )
    result_df["2C通過順位"] = result_df["コーナー通過順"].apply(
        lambda x: separate_turn_rank(x, 2)
    )
    result_df["3C通過順位"] = result_df["コーナー通過順"].apply(
        lambda x: separate_turn_rank(x, 3)
    )
    result_df["4C通過順位"] = result_df["コーナー通過順"].apply(
        lambda x: separate_turn_rank(x, 4)
    )
    result_df = result_df.drop(columns="コーナー通過順")

    print("1C通過順位null数：", result_df["1C通過順位"].isnull().sum())
    print("2C通過順位null数：", result_df["2C通過順位"].isnull().sum())
    print("3C通過順位null数：", result_df["3C通過順位"].isnull().sum())
    print("4C通過順位null数：", result_df["4C通過順位"].isnull().sum())
    print("")

    return result_df


def process_last_three_furlong(result_df):
    """上がり３ハロンをfloatにする関数

    Args:
        result_df (dataframe): レース結果を保存したDF

    Returns:
        result_df (dataframe): レース結果を保存したDF
    """
    print("-----上がり3ハロン処理開始-----")
    result_df = result_df.rename(columns={"後3F": "上がり3F"})
    result_df["上がり3F"] = result_df["上がり3F"].astype("float")

    # 上がり3Fが30以下か50以上は異常値としてnullにする
    result_df.loc[result_df["上がり3F"] <= 30, "上がり3F"] = np.nan
    result_df.loc[result_df["上がり3F"] >= 50, "上がり3F"] = np.nan

    print("上がり3ハロンnull数:", result_df["上がり3F"].isnull().sum())
    print("")

    return result_df


def load_or_create_dict(file_path):
    """ファイルが存在する場合は読み込み、存在しない場合は空の辞書を返す関数

    Args:
        file_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}


def save_dict(dictionary, file_path):
    """辞書をファイルに保存する関数

    Args:
        dictionary (_type_): _description_
        file_path (_type_): _description_
    """
    # ディレクトリが存在しない場合にディレクトリを作成
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(file_path, "wb") as f:
        pickle.dump(dictionary, f)


def compensate_missing_ids(result_df):
    # この処理には問題がある
    # 例えば一度欠損した馬のidが0で登録されると、その後のレースで同じ馬が出走したときに0が登録されてしまう
    # また、同名の馬が複数出走した場合に、どの馬のidを登録すればいいかわからない
    """欠損しているIDを補完する関数

    Args:
        result_df (dataframe): レース結果を保存したDF

    Returns:
        result_df (dataframe): 欠損IDが補完されたレース結果を保存したDF
    """
    print("-----欠損ID処理開始-----")
    print("欠損IDレース削除前：", result_df.shape)

    # 欠損している馬ID、騎手ID、調教師IDのインデックスを取得
    missing_horse_ids = result_df["horse_id"].isna()
    missing_jockey_ids = result_df["jockey_id"].isna()
    missing_trainer_ids = result_df["trainer_id"].isna()

    # 辞書をファイルから読み込むか、新規に作成する
    horse_dict_file = HORSE_DICT
    jockey_dict_file = JOCKEY_DICT
    trainer_dict_file = TRAINER_DICT
    backup_horse_dict_file = HORSE_DICT_BACKUP
    backup_jockey_dict_file = JOCKEY_DICT_BACKUP
    backup_trainer_dict_file = TRAINER_DICT_BACKUP

    horse_dict = load_or_create_dict(horse_dict_file)
    jockey_dict = load_or_create_dict(jockey_dict_file)
    trainer_dict = load_or_create_dict(trainer_dict_file)

    # 新規のデータを辞書に追加する処理
    new_horse_entries = result_df.loc[
        ~missing_horse_ids & ~result_df["馬名"].isin(horse_dict)
    ]
    new_jockey_entries = result_df.loc[
        ~missing_jockey_ids & ~result_df["騎手"].isin(jockey_dict)
    ]
    new_trainer_entries = result_df.loc[
        ~missing_trainer_ids & ~result_df["調教師"].isin(trainer_dict)
    ]

    new_horse_dict = dict(
        zip(new_horse_entries["馬名"], new_horse_entries["horse_id"].astype("int"))
    )
    new_jockey_dict = dict(
        zip(new_jockey_entries["騎手"], new_jockey_entries["jockey_id"].astype("int"))
    )
    new_trainer_dict = dict(
        zip(
            new_trainer_entries["調教師"],
            new_trainer_entries["trainer_id"].astype("int"),
        )
    )

    horse_dict.update(new_horse_dict)
    jockey_dict.update(new_jockey_dict)
    trainer_dict.update(new_trainer_dict)

    # 欠損IDを持つ馬名、騎手名、調教師名のリスト
    missing_horse_names = result_df.loc[missing_horse_ids, "馬名"].unique()
    missing_jockey_names = result_df.loc[missing_jockey_ids, "騎手"].unique()
    missing_trainer_names = result_df.loc[missing_trainer_ids, "調教師"].unique()

    # 欠損しているIDを補完
    for name in missing_horse_names:
        if name not in horse_dict:
            horse_dict[name] = 0
    for name in missing_jockey_names:
        if name not in jockey_dict:
            jockey_dict[name] = 0
    for name in missing_trainer_names:
        if name not in trainer_dict:
            trainer_dict[name] = 0

    # 欠損しているIDを補完した辞書を適用
    result_df.loc[missing_horse_ids, "horse_id"] = result_df.loc[
        missing_horse_ids, "馬名"
    ].map(horse_dict)
    result_df.loc[missing_jockey_ids, "jockey_id"] = result_df.loc[
        missing_jockey_ids, "騎手"
    ].map(jockey_dict)
    result_df.loc[missing_trainer_ids, "trainer_id"] = result_df.loc[
        missing_trainer_ids, "調教師"
    ].map(trainer_dict)

    # IDが0の馬、騎手、調教師を含むレースを削除
    result_df = result_df[~result_df["horse_id"].isin([0])]
    result_df = result_df[~result_df["jockey_id"].isin([0])]
    result_df = result_df[~result_df["trainer_id"].isin([0])]

    # 重複するジョッキーがいるレースを削除する
    result_df = result_df[~result_df.duplicated(subset=["race_id", "jockey_id"])]

    # IDを整数型に変換
    result_df["horse_id"] = result_df["horse_id"].astype("int")
    result_df["jockey_id"] = result_df["jockey_id"].astype("int")
    result_df["trainer_id"] = result_df["trainer_id"].astype("int")

    # 辞書をファイルに保存する
    save_dict(horse_dict, horse_dict_file)
    save_dict(jockey_dict, jockey_dict_file)
    save_dict(trainer_dict, trainer_dict_file)
    save_dict(horse_dict, backup_horse_dict_file)
    save_dict(jockey_dict, backup_jockey_dict_file)
    save_dict(trainer_dict, backup_trainer_dict_file)

    print("欠損IDレース削除後：", result_df.shape)
    print("")

    return result_df


def process_date(info_df):
    """開催時刻を年月日時分に分割する関数

    Args:
        info_df (dataframe): レース情報を格納したDF

    Returns:
        info_df (dataframe): レース情報を格納したDF
    """
    print("-----時刻処理開始-----")
    # race_idはintにしておく
    info_df["race_id"] = info_df["race_id"].astype("int")
    # 不要レースは削除する
    info_df = info_df.loc[
        ~info_df["race_id"].isin(
            list(info_df.loc[info_df["日付"].isna()]["race_id"].unique())
        )
    ]
    # 年数を取り出す
    info_df["年"] = info_df["race_id"].apply(lambda x: int(str(x)[:4]))
    info_df["月"] = info_df["日付"].apply(lambda x: int(x.split("月")[0]))
    info_df["日"] = info_df["日付"].apply(
        lambda x: int(x.split("月")[1].split("日")[0])
    )
    info_df = info_df.drop(columns="日付")
    # 時刻
    info_df["発走時刻"] = info_df["発走時刻"].fillna("23:59発走")
    info_df["時"] = info_df["発走時刻"].apply(lambda x: int(x.split(":")[0]))
    info_df["分"] = info_df["発走時刻"].apply(
        lambda x: int(x.split(":")[1].split("発")[0])
    )
    info_df = info_df.drop(columns="発走時刻")

    info_df["年"] = info_df["年"].astype("int")
    info_df["月"] = info_df["月"].astype("int")
    info_df["日"] = info_df["日"].astype("int")
    info_df["date"] = pd.to_datetime(
        info_df["年"].astype(str)
        + "-"
        + info_df["月"].astype(str)
        + "-"
        + info_df["日"].astype(str)
    )
    info_df["時"] = info_df["時"].astype("int")
    info_df["分"] = info_df["分"].astype("int")

    print("年null数:", info_df["年"].isnull().sum())
    print("月null数:", info_df["月"].isnull().sum())
    print("日null数:", info_df["日"].isnull().sum())
    print("時null数:", info_df["時"].isnull().sum())
    print("分null数:", info_df["分"].isnull().sum())
    print("")

    return info_df


def ext_direction(x):
    """距離条件列から左右方向を取り出す関数

    Args:
        x (str): 距離条件列

    Returns:
        direction (str): 左右方向
    """
    direction = "直線"
    if "右" in x:
        direction = "右"
    if "左" in x:
        direction = "左"

    return direction


def ext_radius(x):
    """距離条件列から左右方向を取り出す関数

    Args:
        x (str): 距離条件列

    Returns:
        radius (str): 内外
    """
    radius = "内外なし"
    if "外" in x:
        radius = "外"
        if "内" in x:
            radius = "外内"
    if "内" in x:
        radius = "内"

    return radius


def process_direction_radius(info_df):
    """方向と内外を取り出す関数

    Args:
        info_df (dataframe): レース情報を格納したDF

    Returns:
        info_df (dataframe): レース情報を格納したDF
    """
    print("------方向内外処理開始------")
    info_df["距離条件"] = info_df["距離条件"].fillna("直線")
    info_df["方向"] = info_df["距離条件"].apply(lambda x: ext_direction(x))
    info_df["内外"] = info_df["距離条件"].apply(lambda x: ext_radius(x))

    print("方向null数:", info_df["方向"].isnull().sum())
    print("内外null数:", info_df["内外"].isnull().sum())
    print("")

    return info_df


def ext_course(x):
    """コース種を取り出す関数

    Args:
        x (str): 距離条件列

    Returns:
        corse (str): 芝かダートか障害か
    """
    course = "情報なし"
    if "障" in x:
        course = "障害"
    elif "芝" in x:
        course = "芝"
    elif "ダ" in x:
        course = "ダート"
    return course


def process_course(info_df):
    """コース上のレース種類を取り出す関数

    Args:
        info_df (dataframe): レース情報を格納したDF

    Returns:
        info_df (dataframe): レース情報を格納したDF
    """
    print("------コース種類処理開始------")
    info_df["コース種類"] = info_df["距離条件"].apply(lambda x: ext_course(x))
    # 障害は芝に分類されることがあるため、レース名に障害がある場合を障害にする
    info_df.loc[info_df["レース名"].str.contains("障", na=False), "コース種類"] = "障害"
    info_df["コース種類"] = info_df["コース種類"].astype("str")

    print("コース種類null数:", info_df["コース種類"].isnull().sum())
    print("")

    return info_df


def process_distance(info_df):
    """距離を取り出す関数

    Args:
        info_df (dataframe): レース情報を格納したDF

    Returns:
        info_df (dataframe): レース情報を格納したDF
    """
    print("------距離処理開始------")
    info_df["距離"] = info_df["距離条件"].apply(lambda x: int(x[1:5]))
    info_df = info_df.drop(columns="距離条件")

    # 距離区分列を作成
    def categorize_distance(distance):
        for category, (low, high) in dist_dict.items():
            if low <= distance <= high:
                return category
        return None  # どのカテゴリにも該当しない場合

    dist_dict = {
        "短距離": [1000, 1300],
        "マイル": [1301, 1899],
        "中距離": [1900, 2100],
        "クラシック": [2101, 2700],
        "長距離": [2700, 4000],
    }
    info_df["distance_category"] = info_df["距離"].apply(categorize_distance)

    print("距離null数:", info_df["距離"].isnull().sum())
    print("距離区分null数:", info_df["distance_category"].isnull().sum())
    print("")

    return info_df


def process_hold_racetrack(info_df):
    """競馬場の開催を処理する関数

    Args:
        info_df (dataframe): レース情報を格納したDF

    Returns:
        info_df (dataframe): レース情報を格納したDF
    """
    print("------開催レース処理開始------")
    info_df["競馬場"] = info_df["競馬場"].astype("str")
    info_df["日数"] = info_df["日数"].apply(lambda x: int(x.split("日")[0]))
    info_df["回"] = info_df["回数"].apply(lambda x: int(x[0]))
    info_df["ラウンド"] = info_df["ラウンド"].apply(lambda x: int(x.split("R")[0]))
    info_df = info_df.drop(columns="回数")

    print("競馬場null数:", info_df["競馬場"].isnull().sum())
    print("日数null数:", info_df["日数"].isnull().sum())
    print("回null数:", info_df["回"].isnull().sum())
    print("")

    return info_df


def replace_grade(grade):
    """グレードを新規名称に合わせる関数

    Args:
        grade (str): グレード名
    """
    if grade == "５００万下" or grade == "１勝クラス":
        grade = "1勝クラス"
    elif grade == "１０００万下" or grade == "２勝クラス":
        grade = "2勝クラス"
    elif grade == "１６００万下" or grade == "３勝クラス":
        grade = "3勝クラス"

    return grade


def process_grade(info_df):
    """グレードを処理する関数

    Args:
        info_df (dataframe): レース情報を格納したDF

    Returns:
        info_df (dataframe): レース情報を格納したDF
    """
    print("------グレード処理開始------")
    info_df["グレード"] = info_df["グレード"].apply(lambda x: replace_grade(x))
    # G1G2G3を特別に取りだす
    info_df.loc[info_df["重賞"].str.contains("G1", na=False), "グレード"] = "G1"
    info_df.loc[info_df["重賞"].str.contains("G2", na=False), "グレード"] = "G2"
    info_df.loc[info_df["重賞"].str.contains("G3", na=False), "グレード"] = "G3"
    info_df = info_df.drop(columns="重賞")
    print("グレードnull数:", info_df["グレード"].isnull().sum())
    # グレードが取得できていないレースは削除する
    print("グレードが削除できていないレースを削除します")
    info_df = info_df.loc[~info_df["グレード"].isnull()]
    info_df["グレード"] = info_df["グレード"].astype("str")
    print("グレードnull数:", info_df["グレード"].isnull().sum())
    print("")

    return info_df


def ext_old_condition(x):
    """競争条件から年齢条件を取り出す関数

    Args:
        x (str): 競争条件
    """
    old = "情報なし"
    x = str(x)
    if "４歳" in x:
        old = "4歳以上"
    if "３歳" in x:
        old = "3歳"
        if "以上" in x:
            old = "3歳以上"
    if "２歳" in x:
        old = "2歳"
    return old


def ext_sex_condition(x):
    sex = "情報なし"
    if pd.isna(x):
        sex = "条件なし"
    elif "牝" in x:
        sex = "牝"
        if "牡・牝" in x:
            sex = "牡・牝"
        else:
            pass
    else:
        sex = "条件なし"
    return sex


def process_old_and_sex_conditon(info_df):
    """年齢条件と性別条件を処理する関数

    Args:
        info_df (dataframe): レース情報を格納したDF

    Returns:
        info_df (dataframe): レース情報を格納したDF
    """
    print("------性別年齢条件処理開始------")
    info_df["年齢条件"] = info_df["条件"].apply(lambda x: ext_old_condition(x))
    info_df["性別条件"] = info_df["分類"].apply(lambda x: ext_sex_condition(x))

    info_df = info_df.drop(columns="条件")
    info_df = info_df.drop(columns="分類")
    info_df = info_df.drop(columns="特殊条件")

    print("年齢条件null数:", info_df["年齢条件"].isnull().sum())
    print("性別条件null数:", info_df["性別条件"].isnull().sum())
    print("")

    return info_df


def ext_added_money(x):
    """1着賞金を取り出す関数

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    x = str(x)
    x = x.split(",")[0]
    x = float(x[4:])
    return x


def process_added_money(info_df):
    """年齢条件と性別条件を処理する関数

    Args:
        info_df (dataframe): レース情報を格納したDF

    Returns:
        info_df (dataframe): レース情報を格納したDF
    """
    print("------1着賞金処理開始------")
    info_df["1着賞金"] = info_df["賞金"].apply(lambda x: ext_added_money(x))
    info_df = info_df.drop(columns="賞金")

    print("1着賞金null数:", info_df["1着賞金"].isnull().sum())
    print("")

    return info_df


def process_weather_and_going(info_df):
    """馬場と天候を処理する関数

    Args:
        info_df (dataframe): レース情報を格納したDF

    Returns:
        info_df (dataframe): レース情報を格納したDF
    """
    print("------天気馬場処理開始------")
    info_df["天気"] = info_df["天気"].apply(lambda x: str(x[3:]))
    info_df["馬場"] = info_df["馬場"].apply(lambda x: str(x[3:]))

    print("天気null数:", info_df["天気"].isnull().sum())
    print("馬場null数:", info_df["馬場"].isnull().sum())
    print("")

    return info_df


def process_number_of_starters(info_df):
    """出走頭数処理する関数

    Args:
        info_df (dataframe): レース情報を格納したDF

    Returns:
        info_df (dataframe): レース情報を格納したDF
    """
    print("------立て数処理開始------")
    info_df["立て数"] = info_df["立て数"].apply(lambda x: int(str(x).split("頭")[0]))

    print("立て数null数:", info_df["立て数"].isnull().sum())
    print("")

    return info_df


def ext_horse_name(name):
    """毛情報やページリンク込みの名前から馬名のみを取り出す関数

    Args:
        name (str): 英名込みの名前

    Returns:
        ext_name (str): 馬名
    """
    # 名前がnanの時は専用名にする
    if pd.isna(name):
        name = "名前がないよ 2022"
    for i in range(len(name)):
        # 名前の直後に出生年が記載されるためその前でfor文を終了させる
        if name[i].isdecimal():
            name = name[: i - 1]
            break
    # 海外馬などは英名が空白で切られているため空白を「_」で埋める
    name = name.split(" ")
    ext_name = "_".join(name)

    return ext_name


def create_pedigree_list():
    pedigree_list = ["馬名"]
    for i in range(62):
        pedigree_list.append(f"pedigree_{i}")
    return pedigree_list


def create_bokei_list():
    bokei_list = ["馬名"]
    indices = [
        0,
        2,
        3,
        6,
        7,
        8,
        9,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
    ]
    for index in indices:
        bokei_list.append(f"pedigree_{index}")
    return bokei_list


def create_hinkei_list():
    hinkei_list = ["馬名"]
    indices = [
        1,
        4,
        5,
        10,
        11,
        12,
        13,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
    ]
    for index in indices:
        hinkei_list.append(f"pedigree_{index}")
    return hinkei_list


def create_pedigree_direct():
    pedigree_direct = []
    for i in range(30, 62):
        direct = ["馬名"]
        fifth = i
        forth = fifth // 2 - 1
        third = forth // 2 - 1
        second = third // 2 - 1
        first = second // 2 - 1
        for generation in [first, second, third, forth, fifth]:
            direct.append(f"pedigree_{generation}")
        pedigree_direct.append(direct)
    return pedigree_direct


def make_pedigree_text(df):
    # horse_idのまま特徴量にするときのためにhorse_idをキーにしたままのDFを保存
    pedigree_df_path = PEDIGREE_ID_PATH
    df.to_csv(pedigree_df_path, encoding="utf_8_sig", index=False)
    # pedigree_dfは馬名ではなく馬IDをキーにしているため馬名をキーにしたDFを作成
    # 馬名とhorse_idの辞書を読み込む
    horse_dict_file = HORSE_DICT
    with open(horse_dict_file, "rb") as f:
        horse_dict = pickle.load(f)
    # idを馬名に変換するためにhorse_dict のキーと値を反転
    horse_dict_reversed = {v: k for k, v in horse_dict.items()}
    # 'horse_id' 列の要素を horse_dict_reversed を使って変換
    df["horse_id"] = df["horse_id"].map(horse_dict_reversed)
    df.rename(columns={"horse_id": "馬名"}, inplace=True)

    # 名前を修正したdfを保存
    pedigree_df_path = PEDIGREE_NAME_PATH
    df.to_csv(pedigree_df_path, encoding="utf_8_sig", index=False)

    # (親情報のモデル変更により一旦削除 20241205)
    # # fast_txt利用のために馬名をテキスト化する

    # pedigree_list = create_pedigree_list()
    # bokei_list = create_bokei_list()
    # hinkei_list = create_hinkei_list()
    # pedigree_direct = create_pedigree_direct()

    # txt_list = []
    # for _, row in df.iterrows():
    #     # All pedigrees
    #     txt_list.append(" ".join([str(row[pedigree]) for pedigree in pedigree_list]))

    #     # Bokei
    #     txt_list.append(" ".join([str(row[bokei]) for bokei in bokei_list]))

    #     # Hinkei
    #     txt_list.append(" ".join([str(row[hinkei]) for hinkei in hinkei_list]))

    #     # Direct pedigrees
    #     for direct in pedigree_direct:
    #         txt_list.append(" ".join([str(row[ped]) for ped in direct]))

    # print(df.shape[0] * 3 + df.shape[0] * 32)
    # print(len(txt_list))
    # date_string = datetime.date.today().strftime("%Y%m%d")
    # output_file = os.path.join(
    #     root_path, "20_data_processing/processed_data/pedigree/fasttext/pedigree.txt"
    # )
    # backup_output_file = os.path.join(
    #     root_path,
    #     f"20_data_processing/processed_data/pedigree/backup/{date_string}_pedigree.txt",
    # )
    # with open(output_file, "w", encoding="utf-8") as f:
    #     for txt in txt_list:
    #         f.write(txt + "\n")
    # with open(backup_output_file, "w", encoding="utf-8") as f:
    #     for txt in txt_list:
    #         f.write(txt + "\n")


def process_pedigree(pedigree_df):
    """血統DFの親名から毛情報やページリンクなどを削除し馬名のみを取り出したDFにする関数

    Args:
        pedigree_df (daragrame): 血統データが保存されたDF

    Returns:
        pedigree_df (daragrame): 血統データが保存されたDF
    """
    # 新しい列名
    pedigree_list = ["pedigree_" + str(i) for i in range(62)]
    for ped in pedigree_list:
        pedigree_df[ped] = pedigree_df[ped].apply(lambda x: ext_horse_name(x))
    make_pedigree_text(pedigree_df)


def ext_hold_racetrack(x, type="racetrack"):
    """競走馬成績DFの開催列から開催地(中央競馬のみ)と回数と日数を取り出す関数

    Args:
        x (str)): 開催列

    Returns:
        racetrack (開催地): 中央競馬開催地
        times (回数): 開催回数
        days (日数): 開催日数
    """
    x = str(x)
    field_list = [
        "京都",
        "中山",
        "小倉",
        "東京",
        "阪神",
        "中京",
        "福島",
        "新潟",
        "函館",
        "札幌",
    ]
    for field in field_list:
        if field in x:
            if "地" in x:
                break
            else:
                x = x.split(field)
                racetrack = field
                times = int(x[0])
                days = int(x[1])
            break
        # 海外開催や地方開催は1回1日目とする
        else:
            racetrack = x
            times = 1
            days = 1
    if type == "racetrack":
        return racetrack
    elif type == "times":
        return times
    elif type == "days":
        return days
    else:
        return np.NaN


def process_horse_past_performance(horse_past_performance_df):
    """競走馬成績を処理する関数

    Args:
        horse_past_performance_df (dataframe): 競走馬成績情報が保存されたDF

    Returns:
        horse_past_performance_df (dataframe): 競走馬成績情報が保存されたDF
    """

    horse_past_performance_df = horse_past_performance_df.rename(
        columns={"race_id": "horse_id"}
    )
    # 日付処理
    print("------競走馬情報処理開始------")
    print("------日付情報処理開始------")
    horse_past_performance_df[["年", "月", "日"]] = (
        horse_past_performance_df["日付"].str.split("/", expand=True).astype(int)
    )
    horse_past_performance_df = horse_past_performance_df.drop(columns="日付")
    horse_past_performance_df = horse_past_performance_df.sort_values(
        ["horse_id", "年", "月", "日"]
    )

    print("年null数:", horse_past_performance_df["年"].isnull().sum())
    print("月null数:", horse_past_performance_df["月"].isnull().sum())
    print("日null数:", horse_past_performance_df["日"].isnull().sum())
    print("")

    # 出走間隔列追加
    print("------出走間隔列処理開始------")
    print("処理前：", horse_past_performance_df.shape)
    horse_past_performance_df["出走間隔"] = (
        horse_past_performance_df.groupby("horse_id")
        .apply(lambda x: (x["年"] * 12 + x["月"] + x["日"] / 30).diff().fillna(0))
        .reset_index(level=0, drop=True)
    )
    horse_past_performance_df.loc[
        horse_past_performance_df["出走間隔"] < 0, "出走間隔"
    ] = 0
    print("処理後：", horse_past_performance_df.shape)

    print("出走間隔null数:", horse_past_performance_df["出走間隔"].isnull().sum())
    print("")

    # 開催処理
    print("------開催地処理開始------")
    horse_past_performance_df["競馬場"] = horse_past_performance_df["開催"].apply(
        lambda x: ext_hold_racetrack(x, "racetrack")
    )
    horse_past_performance_df["回"] = horse_past_performance_df["開催"].apply(
        lambda x: ext_hold_racetrack(x, "times")
    )
    horse_past_performance_df["日数"] = horse_past_performance_df["開催"].apply(
        lambda x: ext_hold_racetrack(x, "days")
    )
    horse_past_performance_df = horse_past_performance_df.drop(columns="開催")

    print("競馬場null数:", horse_past_performance_df["競馬場"].isnull().sum())
    print("回null数:", horse_past_performance_df["回"].isnull().sum())
    print("日数null数:", horse_past_performance_df["日数"].isnull().sum())
    print("")

    # ラウンド処理
    print("------ラウンド処理開始------")
    # 海外のラウンド情報がないものは第0R扱いにする
    horse_past_performance_df["R"] = horse_past_performance_df["R"].fillna(0)
    horse_past_performance_df["ラウンド"] = horse_past_performance_df["R"].apply(
        lambda x: int(x)
    )
    horse_past_performance_df = horse_past_performance_df.drop(columns="R")

    print("ラウンドnull数:", horse_past_performance_df["ラウンド"].isnull().sum())
    print("")

    # ペース処理
    print("------ペース処理開始------")
    horse_past_performance_df["ペース"] = horse_past_performance_df["ペース"].fillna(
        "0-0"
    )
    horse_past_performance_df["前半ペース"] = horse_past_performance_df["ペース"].apply(
        lambda x: float(x.split("-")[0])
    )
    horse_past_performance_df["後半ペース"] = horse_past_performance_df["ペース"].apply(
        lambda x: float(x.split("-")[1])
    )
    horse_past_performance_df.loc[
        horse_past_performance_df["ペース"] == "0-0", ["前半ペース", "後半ペース"]
    ] = np.NaN
    horse_past_performance_df = horse_past_performance_df.drop(columns="ペース")

    print("前半ペースnull数:", horse_past_performance_df["前半ペース"].isnull().sum())
    print("後半ペースnull数:", horse_past_performance_df["後半ペース"].isnull().sum())
    print("")

    # 賞金処理
    print("------賞金処理開始------")
    horse_past_performance_df["賞金"] = horse_past_performance_df["賞金"].fillna(0)

    print("賞金null数:", horse_past_performance_df["賞金"].isnull().sum())
    print("------賞金処理終了------")
    print("")

    display(horse_past_performance_df.head())

    return horse_past_performance_df


def process_odds(odds_df):
    """odds_dfを一行一馬券に書きかえる関数

    Args:
        odds_df (dataframe): 払戻情報が保存されたDF

    Returns:
        odds_df (daragrame): 払戻情報が保存されたDF
    """
    columns_list = ["券種", "馬番1", "馬番2", "馬番3", "払戻金額", "人気", "race_id"]
    horse_num_dict = {
        "単勝": 1,
        "複勝": 1,
        "枠連": 2,
        "馬連": 2,
        "ワイド": 2,
        "馬単": 2,
        "3連複": 3,
        "3連単": 3,
    }  # 一枚の馬券に何頭の馬が絡むのか
    df_list = list()

    # データが取れていない行を削除
    odds_df = odds_df.loc[~odds_df["人気"].isnull()]

    for idx, row in tqdm(odds_df.iterrows()):
        list_list = list()  # odds_dfを一行一馬券にするために使うリスト
        # 一行から情報を取り出す
        horse_number_list = row["馬番"].split(" ")

        # エラーチェック用
        # if '' in horse_number_list:
        #     print(row['race_id'])
        #     print(horse_number_list)
        #     print(row)

        # 馬番のsplitはたまに空白が入ってしまうため取り除く
        horse_number_list = [h for h in horse_number_list if h != ""]
        payoff_list = row["払戻金額"].split("円")[:-1]
        favorite_list = row["人気"].split("人気")[:-1]
        # dataframe作成のために券種を格納したリストを作る
        ticket_list = [row["券種"] for i in range(len(favorite_list))]
        list_list.append(ticket_list)
        # 馬券に絡んだ上位3頭をリストに入れる
        if horse_num_dict[row["券種"]] == 1:
            horse_one_list = horse_number_list[0::1]
            horse_two_list = [np.NaN for i in range(len(favorite_list))]
            horse_three_list = [np.NaN for i in range(len(favorite_list))]
        elif horse_num_dict[row["券種"]] == 2:
            horse_one_list = horse_number_list[0::2]
            horse_two_list = horse_number_list[1::2]
            horse_three_list = [np.NaN for i in range(len(favorite_list))]
        elif horse_num_dict[row["券種"]] == 3:
            horse_one_list = horse_number_list[0::3]
            horse_two_list = horse_number_list[1::3]
            horse_three_list = horse_number_list[2::3]
        else:
            print(idx, row)
        # リストに格納
        list_list.append(horse_one_list)
        list_list.append(horse_two_list)
        list_list.append(horse_three_list)
        list_list.append(payoff_list)
        list_list.append(favorite_list)
        # dataframe作成のためにrace_idを格納したリストを作る
        race_id_list = [row["race_id"] for i in range(len(favorite_list))]
        list_list.append(race_id_list)
        # dataframe作成
        if len(list_list) == 7:
            df = pd.DataFrame(list_list, index=columns_list).T
            df_list.append(df)
        else:
            print(idx)

    odds_df = pd.concat(df_list)
    odds_df = odds_df[~odds_df["券種"].isna()]

    odds_df["券種"] = odds_df["券種"].astype("str")
    odds_df["払戻金額"] = odds_df["払戻金額"].apply(lambda x: int(x.replace(",", "")))
    odds_df["人気"] = odds_df["人気"].apply(lambda x: int(x.replace(",", "")))
    odds_df["race_id"] = odds_df["race_id"].astype("int")

    return odds_df


def process_pace(pace_df):
    """ペース情報を処理する関数

    Args:
        pace_df (dataframe): ペース情報が保存されたDF

    Returns:
        pace_df (dataframe): ペース情報が保存されたDF
    """
    # 同じ race_id を持つ複数行のうち、二行目を残す
    pace_df = pace_df.groupby("race_id").nth(1).reset_index()

    # 列の順序を race_id, 100m, 200m, ..., 3600m に変更し、その他の列を取り除く
    column_order = ["race_id"] + [f"{i}m" for i in range(100, 3601, 100)]
    for m in [2700, 2900, 3100, 3300, 3500]:
        column_order.remove(f"{m}m")
    pace_df = pace_df[column_order]

    # 100m, 200m の値がともに null の行を削除する
    pace_df = pace_df.dropna(subset=["100m", "200m"], how="all")

    # 〇〇mの値は、過去の数値の累積にする(300mの値は100m + 300m)
    pace_columns = [
        f"{i}m" for i in range(100, 3601, 100) if f"{i}m" in pace_df.columns
    ]
    pace_df[pace_columns] = pace_df[pace_columns].astype("float").cumsum(axis=1)

    return pace_df


def process_result(result_df):
    """レース結果をまとめて処理する関数

    Args:
        result_df (dataframe): レース結果を保存したDF

    Returns:
        result_df (dataframe): レース結果を保存したDF
    """
    result_df = delete_useless_race(result_df)
    result_df = process_bracket_number(result_df)
    result_df = process_horse_number(result_df)
    result_df = process_sex_and_age(result_df)
    result_df = process_basis_weight(result_df)
    result_df = process_time(result_df)
    result_df = process_margin(result_df)
    result_df = process_win(result_df)
    result_df = process_favorite(result_df)
    result_df = process_weight(result_df)
    result_df = process_trainer_and_affiliation(result_df)
    result_df = process_turn_passing_rank(result_df)
    result_df = process_last_three_furlong(result_df)
    result_df = compensate_missing_ids(result_df)
    result_df = result_df.loc[~result_df.duplicated()]
    print("最終加工後：", result_df.shape)
    display(result_df.head())
    result_df.info()

    return result_df


def process_info(info_df):
    """レース情報をまとめて処理する関数

    Args:
        info_df (dataframe): レース情報を格納したDF

    Returns:
        info_df (dataframe): レース情報を格納したDF
    """
    info_df = info_df.sort_values(["race_id"])
    info_df = process_date(info_df)
    info_df = process_direction_radius(info_df)
    info_df = process_course(info_df)
    info_df = process_distance(info_df)
    info_df = process_hold_racetrack(info_df)
    info_df = process_grade(info_df)
    info_df = process_old_and_sex_conditon(info_df)
    info_df = process_added_money(info_df)
    info_df = process_weather_and_going(info_df)
    info_df = process_number_of_starters(info_df)
    info_df = info_df.loc[~info_df.duplicated()]
    print("最終加工後：", info_df.shape)
    display(info_df.head())
    info_df.info()

    return info_df


def load_latest_csv(folder_path):
    """最新のCSVファイルを読み込む関数"""
    # 指定したフォルダ内のすべてのCSVファイルを取得
    if "info" in folder_path:
        csv_files = glob.glob(os.path.join(folder_path, "*_future_race_info_df.csv"))
    elif "forecast" in folder_path:
        csv_files = glob.glob(os.path.join(folder_path, "*_race_forecast_df.csv"))

    # ファイル名から作成日を抽出し、ソートする
    sorted_files = sorted(
        csv_files, key=lambda x: x.split("/")[-1].split("_")[0], reverse=True
    )

    # 最新のファイルを読み込む
    latest_file = sorted_files[0]
    df = pd.read_csv(latest_file, encoding="utf_8_sig")

    return df


def process_future_data():
    """未来のレース情報をまとめて処理する関数"""
    # データ読み込みパスの設定
    race_forecast_df_path = FORECAST_PATH
    future_race_info_df_path = FUTURE_INFO_PATH

    # データ読み込み
    race_forecast_df = load_latest_csv(race_forecast_df_path)
    future_race_info_df = load_latest_csv(future_race_info_df_path)

    def assign_rank(group):
        # 未来情報の変換処理のためのもの、最終的には予測のフラグ付に関係する
        # race_idごとに、着順に2と3を1回ずつ代入し、それ以外に1を代入する
        group.loc[group.index[0], "着順"] = "2"
        group.loc[group.index[1], "着順"] = "3"
        group.loc[group.index[2:], "着順"] = "1"
        return group

    # 未来のデータのフォーマットを過去のものと合わせる
    future_race_info_df["天気"] = "天候:晴"
    future_race_info_df["馬場"] = "馬場:良"
    race_forecast_df["人気"] = 1
    race_forecast_df["馬番"].fillna("1", inplace=True)
    race_forecast_df["枠"].fillna("1", inplace=True)
    display(race_forecast_df.head())
    race_forecast_df = race_forecast_df.groupby("race_id").apply(assign_rank).reset_index(drop=True)
    display(race_forecast_df.head())
    race_forecast_df["タイム"] = "9:99:9"
    race_forecast_df["着差"] = "大差"
    race_forecast_df["後3F"] = 33.4
    race_forecast_df["コーナー通過順"] = "1-1-1-1"
    race_forecast_df["馬体重(増減)"] = "999(-99)"
    race_forecast_df = race_forecast_df.reindex(
        columns=[
            "着順",
            "枠",
            "馬番",
            "馬名",
            "性齢",
            "斤量",
            "騎手",
            "タイム",
            "着差",
            "人気",
            "単勝オッズ",
            "後3F",
            "コーナー通過順",
            "厩舎",
            "馬体重(増減)",
            "horse_id",
            "jockey_id",
            "trainer_id",
            "race_id",
        ]
    )
    display(race_forecast_df.head())

    # 未来情報を過去同様処理
    race_forecast_df = process_result(race_forecast_df)
    future_race_info_df = process_info(future_race_info_df)

    # 競走馬過去成績を未来データから作成する
    horse_ids = [
        int(h_id)
        for h_id in race_forecast_df["horse_id"].unique()
        if not math.isnan(h_id)
    ]
    df_list = []
    for horse_id in horse_ids:
        race_id = race_forecast_df.loc[
            race_forecast_df["horse_id"] == horse_id, "race_id"
        ].iloc[0]
        race_info = future_race_info_df.loc[future_race_info_df["race_id"] == race_id]

        # レース情報がある時
        if race_info.shape[0] > 0:
            df = pd.DataFrame(
                {
                    "horse_id": [horse_id],
                    "日付": [race_info["date"].iloc[0].strftime("%Y/%m/%d")],
                    "開催": [
                        f"{race_info['回'].iloc[0]}{race_info['競馬場'].iloc[0]}{race_info['日数'].iloc[0]}"
                    ],
                    "R": [race_info["ラウンド"].iloc[0]],
                    "レース名": [race_info["レース名"].iloc[0]],
                    "ペース": ["33.4-33.4"],
                    "賞金": [race_info["1着賞金"].iloc[0]],
                }
            )
        else:
            df = pd.DataFrame(
                {
                    "horse_id": [horse_id],
                    "日付": ["9999/99/99"],
                    "開催": ["1東京1"],
                    "R": ["1"],
                    "レース名": ["未定"],
                    "ペース": ["33.4-33.4"],
                    "賞金": [0],
                }
            )

        df_list.append(df)
    try:
        horse_future_performance_df = pd.concat(df_list, axis=0, ignore_index=True)
        # 競走馬過去成績の処理は出走間隔処理のために過去データと結合してから行う
        # horse_future_performance_df = process_horse_past_performance(horse_future_performance_df)
    except Exception:
        print(f"Exception\n{traceback.format_exc()}")
        # 早くデータを取り込みすぎた時は、まだ未来のデータがないので、空のデータフレームを返す
        horse_future_performance_df = pd.DataFrame()

    return race_forecast_df, future_race_info_df, horse_future_performance_df


def run_process():
    """一連の処理を実行する関数"""
    # データ読み込みパスの設定
    data_paths = {
        "race_result": RACE_RESULT_PATH,
        "odds": ODDS_PATH,
        "pace": PACE_PATH,
        "race_info": RACE_INFO_PATH,
        "horse_past_performance": HORSE_PAST_PERFORMANCE_PATH,
        "pedigree": PEDIGREE_PATH,
    }

    # データの読み込み
    dfs = {
        name: pd.read_csv(path, encoding="utf_8_sig")
        for name, path in data_paths.items()
    }

    # 各データの処理
    processed_dfs = {
        "race_result": process_result(dfs["race_result"]),
        "odds": process_odds(dfs["odds"]),
        "pace": process_pace(dfs["pace"]),
        "race_info": process_info(dfs["race_info"]),
        # horse_past_performanceは、出走間隔を計算するために未来のデータと結合した後に実行する
        # "horse_past_performance": process_horse_past_performance(dfs["horse_past_performance"]),
    }
    process_pedigree(dfs["pedigree"])

    # 処理後のデータを保存
    df_pass_dict = {
        "race_result": "11_race_result",
        "odds": "50_odds",
        "pace": "30_pace",
        "race_info": "10_race_info",
        "horse_past_performance": "20_horse_past_performance"}
    for name, df in processed_dfs.items():
        output_path = os.path.join(
            ROOT_PATH, f"01_processed/{df_pass_dict[name]}/{name}_df.csv"
        )
        df.to_csv(output_path, encoding="utf_8_sig", index=False)

    # 未来データの処理
    race_forecast_df, future_race_info_df, horse_future_performance_df = (
        process_future_data()
    )

    # 過去データとの結合
    race_result_df = pd.concat(
        [processed_dfs["race_result"], race_forecast_df], axis=0, ignore_index=True
    )
    race_info_df = pd.concat(
        [processed_dfs["race_info"], future_race_info_df], axis=0, ignore_index=True
    )
    horse_past_performance_df = pd.concat(
        [dfs["horse_past_performance"], horse_future_performance_df],
        axis=0,
        ignore_index=True,
    )
    horse_past_performance_df = process_horse_past_performance(
        horse_past_performance_df
    )

    # データの保存
    race_result_df_path = RESULT_WITH_FUTURE_PATH
    race_info_df_path = INFO_WITH_FUTURE_PATH
    horse_past_performance_df_path = PERFORMANCE_WITH_FUTURE_PATH

    race_result_df.to_csv(race_result_df_path, encoding="utf_8_sig", index=False)
    race_info_df.to_csv(race_info_df_path, encoding="utf_8_sig", index=False)
    horse_past_performance_df.to_csv(
        horse_past_performance_df_path, encoding="utf_8_sig", index=False
    )