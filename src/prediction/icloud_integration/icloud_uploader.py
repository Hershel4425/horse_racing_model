import pandas as pd

import os

import logging
import traceback

import pickle


def upload_to_icloud(race_id):
    try:
        # レースidに対応するファイルを読み込む
        # データの読み込み
        root_path = (
            "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2"
        )
        tansho_path = root_path + f"/40_automation/{race_id}_bet_tansho.csv"
        fukusho_path = root_path + f"/40_automation/{race_id}_bet_fukusho.csv"
        wide_path = root_path + f"/40_automation/{race_id}_bet_wide.csv"
        score_path = root_path + f"/40_automation/{race_id}_bet_all.csv"

        tansho_df = pd.read_csv(tansho_path)
        fukusho_df = pd.read_csv(fukusho_path)
        wide_df = pd.read_csv(wide_path)
        score_df = pd.read_csv(score_path)

        icloud_path = "/Users/okamuratakeshi/Library/Mobile Documents/com~apple~CloudDocs/競馬予測"

        # idを用いてファイルを作成
        # 最初の4文字が年
        # 次の2文字が競馬場
        # 01:札幌,02:函館,03:福島,04:新潟,05:東京,06:中山,07:中京,08:京都,09:阪神,10:小倉
        # 次の2文字が開催回数
        # 次の2文字が日数
        # 次の2文字がレース番号
        # 年と競馬場のファイルを作成し書き込む
        race_id = str(int(race_id))
        year = race_id[:4]
        place = race_id[4:6]
        place_dict = {
            1: "札幌",
            2: "函館",
            3: "福島",
            4: "新潟",
            5: "東京",
            6: "中山",
            7: "中京",
            8: "京都",
            9: "阪神",
            10: "小倉",
        }
        place = place_dict[int(place)]

        # 年のファイルを作成
        year_path = f"{icloud_path}/{year}"
        if not os.path.exists(year_path):
            os.makedirs(year_path)

        # 競馬場のファイルを作成
        place_path = f"{year_path}/{place}"
        if not os.path.exists(place_path):
            os.makedirs(place_path)

        # 日付のファイルを作成
        today = pd.to_datetime("today").strftime("%m%d")
        date_path = f"{place_path}/{today}"
        if not os.path.exists(date_path):
            os.makedirs(date_path)

        # ファイルの処理
        # tansho_df, fukusho_df, wide_df, score_dfの「〇〇_賭け割合」列について、10000円を掛けて十円単位を四捨五入
        # その後、列名を「〇〇_賭け数」に変更
        for df in [tansho_df, fukusho_df, wide_df, score_df]:
            for col in df.columns:
                if ("賭け割合" in col) | ("ワイド" in col):
                    df[col] = (df[col] * 5000).round(-2).astype(int)
                    df.rename(
                        columns={col: col.replace("賭け割合", "賭け数")}, inplace=True
                    )

        # race_id列、date列、score列、勝率列、着順列を削除
        for df in [score_df]:
            df.drop(columns=["race_id", "date", "score", "勝率", "着順"], inplace=True)

        # race_id列を削除
        for df in [tansho_df, fukusho_df, wide_df]:
            df.drop(columns=["race_id"], inplace=True)

        # 補正勝率、複勝率列を、%にして小数第一位まで表示
        for df in [score_df]:
            for col in df.columns:
                if "率" in col:
                    df[col] = (df[col] * 100).round(1).astype(str) + "%"

        # score_dfに、馬名やレーティングなどの情報を追加する
        # feature.pickleを読み込む
        root_path = (
        "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2"
        )
        with open(
            root_path + "/20_data_processing/feature_data/feature_df.pickle", "rb"
        ) as f:
            feature_df = pickle.load(f)
        
        # feature_dfのrace_id部分のみ取り出し、馬番列、馬名列、レーティング列を取り出す
        feature_df = feature_df[feature_df['race_id'] == int(race_id)][['馬番', '馬名', '騎手', '騎手レーティング']]
        # 馬番で結合
        score_df = pd.merge(score_df, feature_df, on='馬番', how='left')

        # ファイルを書き込む
        # ファイル名はレース番号
        score_df.to_csv(
            f"{date_path}/{race_id}_score.csv", index=False, encoding="utf_8_sig"
        )

    except Exception as e:
        logging.error(
            f"race_id: {race_id} のiCloudアップロード処理中にエラーが発生しました: {str(e)}"
        )
        logging.error(traceback.format_exc())
