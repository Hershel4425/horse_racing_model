import os

import pandas as pd
import numpy as np

from IPython.display import display
import matplotlib.pyplot as plt
# ▼ 馬のシルエットを貼り付けるために追加
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


# ファイルパス設定
ROOT_PATH = "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/153_競馬_v3"
DATA_PATH = os.path.join(ROOT_PATH, "data/02_features/feature.csv")  # 特徴量CSVのパス
SAVE_PATH_PRED = os.path.join(ROOT_PATH, "result/predictions/transformer/20250517222912.csv") 

HORSE_IMG_PATH = os.path.join(ROOT_PATH, "result/visals/horse-2.png")
JOCKEY_IMG_PATH = os.path.join(ROOT_PATH, "result/visals/upper_body-2.png")

def visualize_win_rates(race_id, df1_path = DATA_PATH, df2_path = SAVE_PATH_PRED):
    # CSV読み込み
    df1 = pd.read_csv(df1_path) # race_id, 枠番, 馬番, 馬名, 騎手名, レーティング
    df2 = pd.read_csv(df2_path) # race_id, 馬番, 1着率, 3着以内率, 5着以内率

    # データ結合
    merged = pd.merge(df1, df2, on=["race_id", "馬番"], how="inner")

    # 騎手レーティングの整数化
    merged["馬レーティング"] = merged["馬レーティング"].round(1)
    merged["騎手レーティング"] = merged["騎手レーティング"].round(1)

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
        print('10%以上の勝率')
        display(df_show.loc[df_show['P_top1_val']>=10][['race_id','馬番','馬名']].sort_values('馬番'))
        print('回収率1以上領域')
        cons_0 = (df_show['P_top1_val']>=10)&(df_show['P_pop1_val']<=10)
        cons_1 = (df_show['P_top1_val']>=20)&(df_show['P_pop1_val']<=30)
        cons_2 = (df_show['P_top1_val']>=30)&(df_show['P_pop1_val']<=60)
        cons_3 = (df_show['P_top1_val']>=40)&(df_show['P_pop1_val']<=80)
        display(df_show.loc[cons_0|cons_1|cons_2|cons_3][display_cols].sort_values('馬番'))
        df_html = df_show[display_cols].sort_values('馬番').to_html(index=False)
        with open(ROOT_PATH + f'/result/visals/2025FS/{rid}.html', "w", encoding="utf-8") as f:
            f.write(df_html)

        # #############################
        # 可視化表示
        # #############################   
        # ───────────── カラーマッピングの準備 ─────────────
        df_map = df_show[display_cols].copy()
        # 可視化用のデータフレームのindexをリセットする
        df_map = df_map.reset_index(drop=True)

        # 馬と騎手のシルエット画像を読み込む(馬は必須で、騎手はなければNone)
        horse_img = plt.imread(HORSE_IMG_PATH)

        # ───────────── プロットの準備 ─────────────
        # ---------------------------
        # 表示用パラメータの設定
        # ---------------------------
        row_gap = 0.5          # 各行の間隔（縦方向）を縮める
        scale = 40             # 横方向のスケール
        left_margin = 5        # 基準となる左端の位置

        # 画像の拡大率設定（お好みで調整してね）
        horse_image_zoom = 0.10
        jockey_image_zoom = 0.10

        # 馬と騎手の画像は、各パスから読み込んでおくわ
        horse_img = plt.imread(HORSE_IMG_PATH)
        jockey_img = plt.imread(JOCKEY_IMG_PATH)

        # プロットの準備
        fig, ax = plt.subplots(figsize=(12, row_gap * len(df_map) + 2))

        # ───────── 5%ごとの縦の点線を描く ─────────
        # 5%から100%まで、5%刻みで描画するわ
        for perc in np.arange(0, 105, 5):
            x_pos = left_margin + (perc / 100) * scale
            ax.axvline(x=x_pos, color='gray', linestyle=':', linewidth=1)

        # ───────── 各馬（行）ごとの描画 ─────────
        for i, row in df_map.iterrows():
            # 各行の基準ラインのy座標
            y_horse = len(df_map) * row_gap - i * row_gap
            # 馬の1着以内率からx座標を算出
            top1_value = float(row["1着以内率"].strip('%')) / 100
            horse_x = left_margin + top1_value * scale
            # 騎手画像は、馬画像とほぼ同じx、馬画像のすぐ上（間隔を縮める）
            jockey_y = y_horse + 0.15

            # ─ 馬の画像配置（枠線・色分けは無し）
            horse_offset_img = OffsetImage(horse_img, zoom=horse_image_zoom)
            ab_horse = AnnotationBbox(horse_offset_img, (horse_x, y_horse), frameon=False)
            ax.add_artist(ab_horse)

            # ─ 騎手の画像配置（こちらも枠線無し）
            jockey_offset_img = OffsetImage(jockey_img, zoom=jockey_image_zoom)
            ab_jockey = AnnotationBbox(jockey_offset_img, (horse_x, jockey_y), frameon=False)
            ax.add_artist(ab_jockey)

            # ─ テキストの配置 ─
            # 馬名は馬画像より右にずらす（名前と画像が重ならないように調整）
            name_offset = 2.5
            ax.text(horse_x + name_offset, y_horse - 0.05, f'{row["馬名"]}',
                    va='center', fontsize=9, color='black')
            # 馬名の右側に馬レーティングを表示
            rating_offset = name_offset + 6.0
            ax.text(horse_x + rating_offset, y_horse - 0.05, f'{row["馬レーティング"]}',
                    va='center', fontsize=9, color='blue')

            # 枠番と馬番は、各馬の基準ラインの左側に「N枠M番」の形式で表示
            ax.text(left_margin - 4, y_horse, f'{row["枠番"]}枠{row["馬番"]}番',
                    va='center', fontsize=12, color='purple')

            # 騎手の名前は、騎馬画像より右にずらす（名前と画像が重ならないように調整）
            ax.text(horse_x + name_offset, y_horse + 0.05, f'{row["騎手"]}',
                    ha='center', va='bottom', fontsize=9, color='black')
            # 騎手の名前の右側に騎手レーティングを表示
            ax.text(horse_x + rating_offset, y_horse + 0.05, f'{row["騎手レーティング"]}',
                    ha='left', va='bottom', fontsize=9, color='orangered')

            # 右端に、1着以内率と1人気以内率の差分を表示
            top1_percent = float(row["1着以内率"].strip('%'))
            pop1_percent = float(row["1人気以内率"].strip('%'))
            diff = top1_percent - pop1_percent
            ax.text(left_margin + scale * 1.1 - 8, y_horse + 0.1, f'1着以内率: {top1_percent:.1f}%',
                    va='center', fontsize=12, color='green')
            ax.text(left_margin + scale * 1.1 - 8, y_horse - 0.1, f'差: {diff:.1f}%',
                    va='center', fontsize=12, color='green')

        # ───────── 軸・レイアウトの調整 ─────────
        ax.set_xlim(0 , left_margin + scale * 1.1)
        ax.set_ylim(0, len(df_map) * row_gap + 1)
        ax.set_xlabel('パフォーマンス指標（基準は左端）')
        ax.set_yticks([])
        ax.set_title('馬・騎手パフォーマンスの可視化', fontsize=14)

        plt.tight_layout()
        plt.show()


        # #############################
        # レーダーチャート表示
        # #############################    

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
            selected_flags.append("芝重")
        if (grass_type == "中") & (course_type == "芝"):
            selected_flags.append("芝中")
        if (grass_type == "軽") & (course_type == "芝"):
            selected_flags.append("芝軽")

        # ダートタイプ
        course_type = df_show["コース種類"].iloc[0]
        dirt_type = df_show["ダートタイプ"].iloc[0]
        if (dirt_type == "重") & (course_type == "ダート"):
            selected_flags.append("ダート重")
        if (dirt_type == "中") & (course_type == "ダート"):
            selected_flags.append("ダート中")
        if (dirt_type == "軽") & (course_type == "ダート"):
            selected_flags.append("ダート軽")

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
        new_selected_flags = ['競走馬レーティング_' + flag for flag in selected_flags]
        race = df_show[["馬名"] + new_selected_flags].copy()

        values = race[new_selected_flags].values
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
            ax[i].set_ylim([0.0, 50.0])
            ax[i].set_title(name, pad=20)

        plt.savefig(ROOT_PATH + f'/result/visals/2024HS/2024HS_{rid}.png', dpi=600, bbox_inches='tight')

        plt.tight_layout()
        plt.show()