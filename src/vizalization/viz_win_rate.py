import os

import pandas as pd
import numpy as np

from IPython.display import display
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

# ファイルパス設定
ROOT_PATH = "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/153_競馬_v3"
DATA_PATH = os.path.join(ROOT_PATH, "data/02_features/feature.csv")  # 特徴量CSVのパス
SAVE_PATH_PRED = os.path.join(ROOT_PATH, "result/predictions/test.csv") 

def visualize_win_rates(race_id, df1_path = DATA_PATH, df2_path = SAVE_PATH_PRED):
    # CSV読み込み
    df1 = pd.read_csv(df1_path) # race_id, 枠番, 馬番, 馬名, 騎手名, レーティング
    df2 = pd.read_csv(df2_path) # race_id, 馬番, 1着率, 3着以内率, 5着以内率

    # データ結合
    merged = pd.merge(df1, df2, on=["race_id", "馬番"], how="inner")

    # 騎手レーティングの整数化
    merged["馬レーティング"] = merged["horse_mu_before"].round(1)
    merged["騎手レーティング"] = merged["jockey_mu_before"].round(1)

    # 背景グラデーション用に確率列を数値化＆小数点1桁までに丸める
    merged["P_top1_val"] = (merged["P_top1"] * 100).round(1)
    merged["P_top3_val"] = (merged["P_top3"] * 100).round(1)
    merged["P_top5_val"] = (merged["P_top5"] * 100).round(1)
    merged["P_pop1_val"] = (merged["P_pop1"] * 100).round(1)
    merged["P_pop3_val"] = (merged["P_pop3"] * 100).round(1)
    merged["P_pop5_val"] = (merged["P_pop5"] * 100).round(1)

    # race_id から race_id+11 までループして可視化
    for rid in range(race_id, race_id + 12):
        df_show = merged[merged["race_id"] == rid].copy()
        if df_show.empty:
            continue
        
        # 表示用にパーセント文字列へ変換（小数点1桁）
        df_show["1着以内率"] = df_show["P_top1_val"].apply(lambda x: f"{x:.1f}%")
        df_show["3着以内率"] = df_show["P_top3_val"].apply(lambda x: f"{x:.1f}%")
        df_show["5着以内率"] = df_show["P_top5_val"].apply(lambda x: f"{x:.1f}%")
        df_show["1人気以内率"] = df_show["P_pop1_val"].apply(lambda x: f"{x:.1f}%")
        df_show["3人気以内率"] = df_show["P_pop3_val"].apply(lambda x: f"{x:.1f}%")
        df_show["5人気以内率"] = df_show["P_pop5_val"].apply(lambda x: f"{x:.1f}%")

        # 表示するカラムを抽出しつつ、背景用の_valカラムは後ろに残しておく
        display_cols = ["race_id", "枠番", "馬番", "馬名", '馬レーティング', "騎手", "騎手レーティング",        
                        "horse_ability_mean",
                        "1着以内率", "3着以内率", "5着以内率",
                        "1人気以内率", "3人気以内率", "5人気以内率" ]
        # 表示
        display(df_show[display_cols].sort_values('馬番'))

        # レース全馬が同じ条件と想定して先頭行を参照
        course_type = df_show["コース種類"].iloc[0]
        dist = df_show["距離"].iloc[0]
        direction = df_show["方向"].iloc[0]
        weather = df_show["天気"].iloc[0]
        
        # レーダーチャート用に表示したいフラグを選択
        selected_flags = []
        
        # コース種類
        if course_type == "芝":
            selected_flags.append("芝")
        elif course_type == "ダート":
            selected_flags.append("ダート")
        
        # 距離
        if 1000 <= dist <= 1400:
            selected_flags.append("短距離")
        elif 1401 <= dist <= 1799:
            selected_flags.append("マイル")
        elif 1800 <= dist <= 2200:
            selected_flags.append("中距離")
        elif 2201 <= dist <= 2600:
            selected_flags.append("クラシック")
        elif 2601 <= dist <= 4000:
            selected_flags.append("長距離")

        # 方向
        if direction == "右":
            selected_flags.append("方向_右")
        elif direction == "左":
            selected_flags.append("方向_左")
        elif direction == "直線":
            selected_flags.append("方向_直線")

        # 天気
        if weather == "晴":
            selected_flags.append("天気_晴")
        elif weather == "雨":
            selected_flags.append("天気_雨")
        elif weather == "曇":
            selected_flags.append("天気_曇")
        elif weather == "雪":
            selected_flags.append("天気_雪")

        # 馬場
        track = df_show["馬場"].iloc[0]
        if track == "良":
            selected_flags.append("馬場_良")
        elif track == "不":
            selected_flags.append("馬場_不")
        elif track == "重":
            selected_flags.append("馬場_重")
        elif track == "稍":
            selected_flags.append("馬場_稍")

        # カーブ
        curve = df_show["カーブ"].iloc[0]
        if curve == "大回り":
            selected_flags.append("大回り")
        if curve == "小回り":
            selected_flags.append("小回り")
        if curve == "急":
            selected_flags.append("急")

        # ゴール前坂
        slope = df_show["ゴール前坂"].iloc[0]
        if slope == "急坂":
            selected_flags.append("急坂")
        if slope == "平坦":
            selected_flags.append("平坦")
        if slope == "緩坂":
            selected_flags.append("緩坂")

        # 芝タイプ
        course_type = df_show["コース種類"].iloc[0]
        grass_type = df_show["芝タイプ"].iloc[0]
        if (grass_type == "重") & (course_type == "芝"):
            selected_flags.append("重")
        if (grass_type == "中") & (course_type == "芝"):
            selected_flags.append("中")
        if (grass_type == "軽") & (course_type == "芝"):
            selected_flags.append("軽")

        # ダートタイプ
        course_type = df_show["コース種類"].iloc[0]
        dirt_type = df_show["ダートタイプ"].iloc[0]
        if (dirt_type == "重") & (course_type == "ダート"):
            selected_flags.append("重")
        if (dirt_type == "中") & (course_type == "ダート"):
            selected_flags.append("中")
        if (dirt_type == "軽") & (course_type == "ダート"):
            selected_flags.append("軽")

        spart_type = df_show["スパートタイプ"].iloc[0]
        if spart_type == "ロンスパ":
            selected_flags.append("ロンスパ")
        if spart_type == "瞬発力":
            selected_flags.append("瞬発力")

        spart_vel = df_show["スパート速度"].iloc[0]
        if spart_vel == "低速":
            selected_flags.append("低速")
        if spart_vel == "中速":
            selected_flags.append("中速")
        if spart_vel == "高速":
            selected_flags.append("高速")

        # グレード
        grade = df_show["グレード"].iloc[0]
        if (grade == "G1") | (grade == "G2") | (grade == "G3"):
            selected_flags.append("重賞")
        elif (grade != "G1") & (grade != "G2") & (grade != "G3"):
            selected_flags.append("平場")

        # 描画用データフレーム作成
        if not selected_flags:
            continue
        race = df_show[["馬名"] + selected_flags].copy()

        values = race[selected_flags].values
        labels = selected_flags
        horse_names = race["馬名"].values

        # レーダーチャートの描画
        fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(20, 20), facecolor="w", subplot_kw=dict(polar=True))
        ax = ax.flatten()
        
        for i, name in enumerate(horse_names):
            if i >= len(ax):
                break
            radar_values = np.concatenate([values[i], [values[i][0]]])
            angles = np.linspace(0, 2 * np.pi, len(labels) + 1, endpoint=True)
            ax[i].plot(angles, radar_values)
            ax[i].fill(angles, radar_values, alpha=0.2)
            ax[i].set_thetagrids(angles[:-1] * 180 / np.pi, labels)
            ax[i].set_ylim([0.0, 1.0])
            ax[i].set_title(name, pad=20)

        plt.tight_layout()
        plt.show()