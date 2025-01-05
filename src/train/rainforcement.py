import os
import re
import datetime
import random
from tqdm import tqdm

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# このコードは競馬のレースデータを強化学習で学習・推論し、ROI（投資収益率）を高めるための手法を実装したものです。
# 「なぜこの処理を行うのか？」を強調する形でコメントを追加しています。

ROOT_PATH = "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/153_競馬_v3"
# 日時文字列を作成する理由: モデルの保存先や予測結果の保存先を実行日時ごとに管理することで、過去の実験結果を整理しやすくするため
DATE_STRING = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

# モデル保存ディレクトリのパスを決定。
# なぜ日時付きでディレクトリを作るか？ -> 各実行時のモデルを時系列で管理しやすくするため
MODEL_SAVE_DIR = os.path.join(ROOT_PATH, f"models/transormer予測モデル/{DATE_STRING}")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

# 予測結果の保存先を決定。なぜ分けるか？ -> 予測結果とモデルを管理しやすくし、モデルのバージョンに対応する結果を明確にするため
SAVE_PATH_PRED = os.path.join(ROOT_PATH, f"result/predictions/強化学習/{DATE_STRING}.csv")
pred_dir = os.path.dirname(SAVE_PATH_PRED)
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)

# データのパスを指定
DATA_PATH = os.path.join(ROOT_PATH, "data/02_features/feature.csv")


def split_data(df, id_col="race_id", test_ratio=0.05, valid_ratio=0.05):
    """
    データをtrain/valid/testに分割する関数
    Why:
      - データが時系列で並んでいるため、日時の古い順に並べてから分割を行う
      - 時系列を考慮することで情報漏洩(データリーク)のリスクを下げる
      - テストデータやバリデーションデータとの重複を避ける
    """
    # 日付順に並べることで、未来のデータが学習データに混在するのを防ぐ
    df = df.sort_values('date').reset_index(drop=True)
    
    # race_idをユニークに取得し、数を把握する
    race_ids = df[id_col].unique()
    dataset_len = len(race_ids)
    
    # テストやバリデーションの分割位置を決定
    test_cut = int(dataset_len * (1 - test_ratio))
    valid_cut = int(test_cut * (1 - valid_ratio))
    
    # 各データのrace_idを抽出
    train_ids = race_ids[:valid_cut]
    valid_ids = race_ids[valid_cut:test_cut]
    test_ids = race_ids[test_cut:]
    
    # race_idにもとづいてデータを分割
    train_df = df[df[id_col].isin(train_ids)].copy()
    valid_df = df[df[id_col].isin(valid_ids)].copy()
    test_df = df[df[id_col].isin(test_ids)].copy()
    
    return train_df, valid_df, test_df


def prepare_data(
    data_path,
    id_col='race_id',
    test_ratio=0.1,
    valid_ratio=0.1,
    pca_dim_horse=50,
    pca_dim_jockey=50,
    cat_cols=None,
    finishing_col='着順',
    single_odds_col='単勝'
):
    """
    データの読み込み、リークを起こす可能性のある列の削除、数値列の欠損補完、
    PCAによる次元圧縮、標準化などを行い、学習に適した形へ整形する。
    Why:
      - リークの可能性のある列を削除しないと、学習の過程で実際には使えない未来情報を使ってしまうリスクがある
      - PCAによる次元圧縮で特徴量をコンパクトにまとめ、高次元のデータでも学習を安定させやすくする
      - 標準化を行うことで、モデルが扱いやすいスケールにデータを合わせる（大きい値の特徴量が学習を支配しないようにする）
    """
    if cat_cols is None:
        cat_cols = []
    
    # CSVデータを読み込む
    df = pd.read_csv(data_path, encoding="utf_8_sig")
    
    # デフォルトでリークにつながる可能性がある列をまとめて削除
    # Why: これらの列は結果やレースの本質的情報（順位など）に直結しているため、
    # 学習に使うと実際には得られないはずの情報を活用してしまうリスクがある
    default_leakage_cols = [
        '斤量','タイム','着差','上がり3F','馬体重','人気','horse_id','jockey_id','trainer_id','順位点',
        '入線','1着タイム差','先位タイム差','5着着差','増減','1C通過順位','2C通過順位','3C通過順位',
        '4C通過順位','賞金','前半ペース','後半ペース','ペース','上がり3F順位','100m','200m','300m',
        '400m','500m','600m','700m','800m','900m','1000m','1100m','1200m','1300m','1400m','1500m',
        '1600m','1700m','1800m','1900m','2000m','2100m','2200m','2300m','2400m','2500m','2600m',
        '2700m','2800m','2900m','3000m','3100m','3200m','3300m','3400m','3500m','3600m','horse_ability'
    ]
    df.drop(columns=default_leakage_cols, errors='ignore', inplace=True)
    
    # 数値カラムをリストアップ
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 目的変数（着順）やオッズの列はnum_colsから外しておき、別途取り扱いたい
    if finishing_col in num_cols:
        num_cols.remove(finishing_col)
    if single_odds_col in num_cols:
        num_cols.remove(single_odds_col)
    
    # カテゴリ列が指定されていても、実際に存在しない列は排除する
    cat_cols = [c for c in cat_cols if c in df.columns]
    
    # 数値カラムの欠損を0で補完
    # Why: 強化学習やPCAでNaNがあるとエラーになることが多い。シンプルに0埋めで対応
    for c in num_cols:
        df[c] = df[c].fillna(0)
    
    # データをsplitしてtrain/valid/testに分割
    train_df, valid_df, test_df = split_data(df, id_col=id_col, test_ratio=test_ratio, valid_ratio=valid_ratio)
    
    # PCA対象を馬関連と騎手関連に振り分けるための正規表現パターン
    # Why: 馬と騎手で特徴のまとまりを分けることで、異なる視点(馬性能・騎手性能)を圧縮しやすい
    pca_pattern_horse = r'^(競走馬芝|競走馬ダート|単年競走馬芝|単年競走馬ダート)'
    pca_pattern_jockey = r'^(騎手芝|騎手ダート|単年騎手芝|単年騎手ダート)'
    
    # 馬関連と騎手関連の数値列を抽出
    pca_horse_target_cols = [c for c in num_cols if re.match(pca_pattern_horse, c)]
    pca_jockey_target_cols = [c for c in num_cols if re.match(pca_pattern_jockey, c)]
    
    # それ以外の数値列
    other_num_cols = [
        c for c in num_cols
        if c not in pca_horse_target_cols
        and c not in pca_jockey_target_cols
    ]
    
    # 以下、馬関連・騎手関連・その他数値列に分けて標準化→PCA処理
    scaler_horse = StandardScaler()
    if len(pca_horse_target_cols) > 0:
        # 馬関連の特徴量を学習データでfitし、バリデーションとテストはtransformのみ
        horse_train_scaled = scaler_horse.fit_transform(train_df[pca_horse_target_cols])
        horse_valid_scaled = scaler_horse.transform(valid_df[pca_horse_target_cols])
        horse_test_scaled = scaler_horse.transform(test_df[pca_horse_target_cols])
    else:
        # 馬関連の特徴量がない場合は、形だけ0行列を用意
        horse_train_scaled = np.zeros((len(train_df), 0))
        horse_valid_scaled = np.zeros((len(valid_df), 0))
        horse_test_scaled = np.zeros((len(test_df), 0))
    
    # 次元数指定が、実際の特徴量数を超えていないかチェック
    # Why: PCAは実際の特徴量数以上にコンポーネントを取れない
    pca_dim_horse = min(pca_dim_horse, horse_train_scaled.shape[1]) if horse_train_scaled.shape[1] > 0 else 0
    
    # PCAによる馬関連次元圧縮
    # Why: 馬の傾向を圧縮して特徴をまとめることで、学習の高速化＆高次元での過学習リスクを低減
    if pca_dim_horse > 0:
        pca_model_horse = PCA(n_components=pca_dim_horse)
        horse_train_pca = pca_model_horse.fit_transform(horse_train_scaled)
        horse_valid_pca = pca_model_horse.transform(horse_valid_scaled)
        horse_test_pca = pca_model_horse.transform(horse_test_scaled)
    else:
        pca_model_horse = None
        horse_train_pca = horse_train_scaled
        horse_valid_pca = horse_valid_scaled
        horse_test_pca = horse_test_scaled
    
    # 騎手関連も同様
    scaler_jockey = StandardScaler()
    if len(pca_jockey_target_cols) > 0:
        jockey_train_scaled = scaler_jockey.fit_transform(train_df[pca_jockey_target_cols])
        jockey_valid_scaled = scaler_jockey.transform(valid_df[pca_jockey_target_cols])
        jockey_test_scaled = scaler_jockey.transform(test_df[pca_jockey_target_cols])
    else:
        jockey_train_scaled = np.zeros((len(train_df), 0))
        jockey_valid_scaled = np.zeros((len(valid_df), 0))
        jockey_test_scaled = np.zeros((len(test_df), 0))
    
    pca_dim_jockey = min(pca_dim_jockey, jockey_train_scaled.shape[1]) if jockey_train_scaled.shape[1] > 0 else 0
    
    if pca_dim_jockey > 0:
        pca_model_jockey = PCA(n_components=pca_dim_jockey)
        jockey_train_pca = pca_model_jockey.fit_transform(jockey_train_scaled)
        jockey_valid_pca = pca_model_jockey.transform(jockey_valid_scaled)
        jockey_test_pca = pca_model_jockey.transform(jockey_test_scaled)
    else:
        pca_model_jockey = None
        jockey_train_pca = jockey_train_scaled
        jockey_valid_pca = jockey_valid_scaled
        jockey_test_pca = jockey_test_scaled
    
    # その他の数値列も標準化のみを行う
    scaler_other = StandardScaler()
    if len(other_num_cols) > 0:
        other_train = scaler_other.fit_transform(train_df[other_num_cols])
        other_valid = scaler_other.transform(valid_df[other_num_cols])
        other_test = scaler_other.transform(test_df[other_num_cols])
    else:
        other_train = np.zeros((len(train_df), 0))
        other_valid = np.zeros((len(valid_df), 0))
        other_test = np.zeros((len(test_df), 0))
    
    # カテゴリ特徴量はそのまま扱う（one-hotエンコードなどをしていない点に注意）
    cat_features_train = train_df[cat_cols].values if cat_cols else np.zeros((len(train_df), 0))
    cat_features_valid = valid_df[cat_cols].values if cat_cols else np.zeros((len(valid_df), 0))
    cat_features_test = test_df[cat_cols].values if cat_cols else np.zeros((len(test_df), 0))
    
    # 馬関連、騎手関連、その他数値、カテゴリのベクトルを結合して最終的な特徴量とする
    # Why: 異なるブロックに分割し、それらを最終的に一つの入力ベクトルにまとめることで、
    #      各要素が欠損していてもコードが煩雑にならずにすむ
    X_train = np.concatenate([cat_features_train, other_train, horse_train_pca, jockey_train_pca], axis=1)
    X_valid = np.concatenate([cat_features_valid, other_valid, horse_valid_pca, jockey_valid_pca], axis=1)
    X_test = np.concatenate([cat_features_test, other_test, horse_test_pca, jockey_test_pca], axis=1)
    
    # それぞれのDataFrameに特徴量ベクトルを格納
    train_df["X"] = list(X_train)
    valid_df["X"] = list(X_valid)
    test_df["X"] = list(X_test)
    
    # 特徴量の次元数を記録し、後の環境構築に活用する
    actual_num_dim = X_train.shape[1]
    
    return (
        train_df, valid_df, test_df,
        (scaler_horse, pca_model_horse), (scaler_jockey, pca_model_jockey), scaler_other,
        cat_cols, actual_num_dim
    )


class MultiRaceEnv(gym.Env):
    """
    強化学習用の環境クラス。
    なぜ独自環境を作るのか？ -> 競馬の各レースで馬を選択し、その報酬を得るという形にしたいから。
    ここではレースごとに馬を観察→アクションとして「どの馬に賭けるか」を選択し、着順による報酬を得る。
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_dim: int,
        id_col="race_id",
        horse_col="馬番",
        horse_name_col="馬名",
        single_odds_col="単勝",
        finishing_col="着順",
        cost=100,
        races_per_episode=128
    ):
        """
        Why:
          - df: 強化学習に使うレースデータ
          - feature_dim: 1頭あたりの特徴量ベクトルの次元数
          - id_col, horse_col, horse_name_col, single_odds_col, finishing_col: データ内の列名指定
          - cost: 1レースに賭けるコスト。報酬計算に使用
          - races_per_episode: 1エピソードあたりに回すレースの数
        """
        super().__init__()
        self.df = df
        self.id_col = id_col
        self.horse_col = horse_col
        self.horse_name_col = horse_name_col
        self.single_odds_col = single_odds_col
        self.finishing_col = finishing_col
        self.cost = cost
        self.feature_dim = feature_dim
        self.races_per_episode = races_per_episode

        # race_idをキーに、それぞれのレースのデータを保持する
        self.race_ids = df[id_col].unique().tolist()
        self.race_map = {}

        max_horses = 0
        # 全レースのうち最大の出走頭数を探し、観測空間を一律の長さ(最大頭数)に揃える
        # Why: Gymの観測空間は固定長が望ましいため、足りない分はパディングする設計とする
        for rid in self.race_ids:
            subdf = df[df[id_col] == rid].copy().sort_values(self.horse_col)
            max_horses = max(max_horses, len(subdf))
            self.race_map[rid] = subdf
        self.max_horses = max_horses

        # 観測空間を定義。馬最大頭数×特徴量次元
        # Why: 「何頭出走でも強制的に配列を同じ次元にして扱う」設計
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_horses * self.feature_dim,),
            dtype=np.float32
        )
        
        # 行動空間: 
        # 修正後（複数馬・変動掛け金）
        # なぜ MultiDiscrete にするのか？
        #  -> 各馬番に対して「掛けない(0)〜複数単位賭ける(n)」を同時に表現するため。
        #     例えば [4]*self.max_horses とすれば 0〜3 の4パターンの賭け方が取れる。
        max_bet = 3  # 例として、「1馬に対して最大3単位賭けられる」とする
        self.action_space = spaces.MultiDiscrete([max_bet + 1] * self.max_horses)


        # エピソード内でのレース順序をシャッフルするためのリストや状態を初期化
        self.sampled_races = []
        self.current_race_idx = 0
        self.current_obs = None
        self.terminated = False

    def _get_obs_for_race(self, race_df: pd.DataFrame):
        """
        指定されたレースの馬ごとの特徴量を取り出し、最大頭数に合わせてパディングし、
        1次元にflattenして返す
        Why:
          - Gymの標準的な扱いやすいフォーマットに合わせるため
          - 出走頭数が少ないレースでも同じ次元の観測空間を確保する
        """
        n_horses = len(race_df)
        feats = []
        for i in range(n_horses):
            feats.append(race_df.iloc[i]["X"])
        feats = np.array(feats, dtype=np.float32)

        # max_horsesに満たない場合は0埋め
        if n_horses < self.max_horses:
            pad_len = self.max_horses - n_horses
            pad = np.zeros((pad_len, self.feature_dim), dtype=np.float32)
            feats = np.vstack([feats, pad])
        
        # flattenして返す
        return feats.flatten()

    def _select_races_for_episode(self):
        """
        1エピソード中に使用するレースをランダムに選択し、シャッフルして保持
        Why:
          - 同じレース順で回していると学習が偏る可能性があり、ランダムサンプリングで汎化性能を上げる
        """
        self.sampled_races = random.sample(self.race_ids, k=self.races_per_episode)
        random.shuffle(self.sampled_races)

    def reset(self, seed=None, options=None):
        """
        エピソードの最初に呼ばれるメソッド。
        レースのサンプリングと初期状態のリセットを行い、最初の観測を返す。
        Why:
          - 毎エピソードで新しくレースの順序を変えることで、多様な学習を行う
        """
        super().reset(seed=seed)
        self._select_races_for_episode()
        self.current_race_idx = 0
        self.terminated = False

        # 最初のレースの観測を返す
        race_df = self.race_map[self.sampled_races[self.current_race_idx]]
        self.current_obs = self._get_obs_for_race(race_df)
        return self.current_obs, {}

    def step(self, action):
        """
        エージェントからの行動(action: 賭ける馬のインデックス)を受け取り、報酬を計算して次の状態を返す。
        Why:
          - どの馬に賭けたかに応じて払い戻し（当選時） or 賭け金損失を計算する
        """
        if self.terminated:
            return self.current_obs, 0.0, True, False, {}

        # 現在のレースID
        rid = self.sampled_races[self.current_race_idx]
        race_df = self.race_map[rid]
        n_horses = len(race_df)

        # action は長さ self.max_horses のベクトル（各要素が0〜max_bet）。
        # 例: action = [2, 0, 3, 1, 0, ...] -> 馬1に2単位、馬3に3単位、馬4に1単位のように賭ける。
        reward = 0.0

        # 各馬に対して賭け金を処理
        for i in range(self.max_horses):
            bet_units = action[i]  # i番目の馬への「何単位賭けるか」
            cost_i = bet_units * self.cost  # その馬への実コスト (例: 1単位=100円相当、など)

            # まず賭けた分だけ支出(マイナス)とする
            reward -= cost_i

            # 出走馬数(n_horses)より i が小さければ実在する馬なので、勝利判定
            if i < n_horses:
                row = race_df.iloc[i]
                if row[self.finishing_col] == 1:
                    # 勝った馬(1着)ならオッズ分の払い戻しを受ける
                    odds = row[self.single_odds_col]
                    reward += cost_i * (odds / 100)


        # 次のレースへ
        self.current_race_idx += 1
        terminated = (self.current_race_idx >= self.races_per_episode)
        truncated = False
        self.terminated = terminated

        if not terminated:
            # 次のレースの観測を生成
            next_rid = self.sampled_races[self.current_race_idx]
            next_race_df = self.race_map[next_rid]
            obs = self._get_obs_for_race(next_race_df)
            self.current_obs = obs
        else:
            # エピソード終了の場合は現在の状態をそのまま返す
            obs = self.current_obs

        return obs, float(reward), terminated, truncated, {}


def evaluate_model(env: MultiRaceEnv, model):
    """
    学習後のモデルを使って全レースで行動を取り、ROIを算出する関数。
    Why:
      - 学習の成果が投資収益率(ROI)でどれだけ高まったかを確認し、モデルの良し悪しを評価するため
    """
    original_ids = env.race_ids
    cost_sum = 0.0
    profit_sum = 0.0
    results = []

    for rid in tqdm(original_ids):
        # レースごとのデータを取り出し、観測を作成してモデルに入力
        subdf = env.race_map[rid].sort_values(env.horse_col).reset_index(drop=True)
        obs = env._get_obs_for_race(subdf)
        # なぜ複数ベットを展開するのか？
        #  -> "action" は [馬1への掛け単位, 馬2への掛け単位, ...] のベクトルだから。
        action, _ = model.predict(obs, deterministic=True)
        n_horses = len(subdf)

        # このレース全体のコスト・利益を集計
        race_cost = 0.0
        race_profit = 0.0

        # 報酬計算を模擬（step関数を呼ばずに、ここでそのまま計算する）
        for i in range(env.max_horses):
            bet_units = action[i]
            bet_amount = bet_units * env.cost  # 実際に掛けた金額
            race_cost += bet_amount

            if i < n_horses:
                row_i = subdf.iloc[i]
                finishing = row_i[env.finishing_col]
                odds = row_i[env.single_odds_col]

                # 着順が1位であればオッズ×bet_amount の払い戻し
                if finishing == 1:
                    race_profit += bet_amount * (odds / 100)

        # 総利益 = 払い戻し - 掛け金
        net_race_profit = race_profit - race_cost
        profit_sum += net_race_profit
        cost_sum += race_cost

        # どの馬を選択したかを記録
        # 結果保存 (各馬ごとに "bet_amount" として入れる)
        for i in range(n_horses):
            row_i = subdf.iloc[i]
            bet_units = action[i]
            bet_amount = bet_units * env.cost

            results.append({
                "race_id": rid,
                "馬番": row_i[env.horse_col],
                "馬名": row_i[env.horse_name_col],
                "着順": row_i[env.finishing_col],
                "単勝": row_i[env.single_odds_col],
                # 修正前は selected_flag だったが、今は bet_amount を保存
                "bet_amount": bet_amount
            })
    
    # ROI = (利益合計 / コスト合計)
    roi = (profit_sum / cost_sum) if cost_sum > 0 else 0.0
    return roi, pd.DataFrame(results)


class StatsCallback(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)
        self.iteration_logs = {
            "iteration": [],
            "timesteps": [],
            "time_elapsed": [],
            "fps": [],
            "approx_kl": [],
            "clip_fraction": [],
            "entropy_loss": [],
            "explained_variance": [],
            "learning_rate": [],
            "loss": [],
            "policy_gradient_loss": [],
            "value_loss": []
        }
        self.iter_count = 0

    # on_step (or _on_step) メソッドを実装しないと TypeError になる
    # 新しめのバージョンでは先頭にアンダースコア (_on_step) がつく場合があります
    def _on_step(self) -> bool:
        # ここでは必要最低限の True を返すだけでOK
        return True

    def _on_rollout_end(self):
        self.iter_count += 1
        self.iteration_logs["iteration"].append(self.iter_count)
        self.iteration_logs["timesteps"].append(self.model.num_timesteps)
        self.iteration_logs["time_elapsed"].append(self.model.logger.name_to_value.get("time/total_timesteps", 0))
        self.iteration_logs["fps"].append(self.model.logger.name_to_value.get("time/fps", 0))
        self.iteration_logs["approx_kl"].append(self.model.logger.name_to_value.get("train/approx_kl", 0))
        self.iteration_logs["clip_fraction"].append(self.model.logger.name_to_value.get("train/clip_fraction", 0))
        self.iteration_logs["entropy_loss"].append(self.model.logger.name_to_value.get("train/entropy_loss", 0))
        self.iteration_logs["explained_variance"].append(self.model.logger.name_to_value.get("train/explained_variance", 0))
        self.iteration_logs["learning_rate"].append(self.model.logger.name_to_value.get("train/learning_rate", 0))
        self.iteration_logs["loss"].append(self.model.logger.name_to_value.get("train/loss", 0))
        self.iteration_logs["policy_gradient_loss"].append(self.model.logger.name_to_value.get("train/policy_gradient_loss", 0))
        self.iteration_logs["value_loss"].append(self.model.logger.name_to_value.get("train/value_loss", 0))

    def plot_logs(self):
        """
        学習ログを可視化するためのメソッド。
        Why:
          - 訓練の安定性や過学習をグラフで把握しやすくする
        """
        fig, axs = plt.subplots(4, 3, figsize=(18, 18))
        logs = self.iteration_logs

        idx_map = [
            ("timesteps", 0, 0), 
            ("fps", 0, 1), 
            ("approx_kl", 0, 2), 
            ("clip_fraction", 1, 0),
            ("entropy_loss", 1, 1), 
            ("explained_variance", 1, 2), 
            ("learning_rate", 2, 0), 
            ("loss", 2, 1),
            ("policy_gradient_loss", 2, 2), 
            ("value_loss", 3, 0)
        ]

        # グラフを10種類描画
        for key, row, col in idx_map:
            axs[row, col].plot(logs["iteration"], logs[key], marker='o', label=key)
            axs[row, col].set_xlabel("iteration")
            axs[row, col].set_ylabel(key)
            axs[row, col].legend()
        
        # 残りのスペースはoffにしてレイアウトを整える
        axs[3, 1].axis("off")
        axs[3, 2].axis("off")

        plt.tight_layout()
        plt.show()


def run_training_and_inference(
    data_path=DATA_PATH,
    id_col='race_id',
    horse_col='馬番',
    horse_name_col='馬名',
    single_odds_col='単勝',
    finishing_col='着順',
    cost=0.01,
    total_timesteps=500000,
    races_per_episode=128
):
    """
    学習データの準備からモデルの学習、評価、予測結果の保存、ログの可視化までを一括で実行する関数。
    Why:
      - スクリプトを1回実行するだけで全ステップを行えるように設計し、再現性・再利用性を高める
    """
    # データを読み込み＆前処理
    train_df, valid_df, test_df, _, _, _, _, dim = prepare_data(
        data_path=data_path,
        id_col=id_col,
        test_ratio=0.1,
        valid_ratio=0.1,
        pca_dim_horse=50,
        pca_dim_jockey=50,
        cat_cols=[]
    )

    # 強化学習環境をtrain, valid用に作成
    train_env = MultiRaceEnv(
        df=train_df,
        feature_dim=dim,
        id_col=id_col,
        horse_col=horse_col,
        horse_name_col=horse_name_col,
        single_odds_col=single_odds_col,
        finishing_col=finishing_col,
        cost=cost,
        races_per_episode=races_per_episode
    )
    valid_env = MultiRaceEnv(
        df=valid_df,
        feature_dim=dim,
        id_col=id_col,
        horse_col=horse_col,
        horse_name_col=horse_name_col,
        single_odds_col=single_odds_col,
        finishing_col=finishing_col,
        cost=cost,
        races_per_episode=races_per_episode
    )

    # VecEnvに変換（PPOなど多くのRLアルゴリズムは並列環境を前提にしているため）
    vec_train_env = DummyVecEnv([lambda: train_env])

    # 学習ログの可視化などに使うコールバックを準備
    stats_callback = StatsCallback()

    # PPOモデルを初期化。なぜPPOか？
    # -> シンプルで安定した強化学習アルゴリズムであり、ハイパーパラメータをある程度簡単に調整できる
    ppo_hyperparams = {
        "learning_rate": 1e-4,
        "n_steps": 1024,
        "batch_size": 256,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "n_epochs": 10,
        # 必要に応じて vf_coef, max_grad_norm なども追加
    }
    model = PPO(
        "MlpPolicy",
        env=vec_train_env,
        # device="mps",  # ← ここで MPS を指定
        verbose=1,
        **ppo_hyperparams
    )

    # 学習を実行
    model.learn(total_timesteps=total_timesteps, callback=stats_callback)

    # 学習データでのROI確認
    train_roi, _ = evaluate_model(train_env, model)
    print(f"Train ROI: {train_roi:.2f}%")

    # バリデーションデータでのROI確認
    valid_roi, _ = evaluate_model(valid_env, model)
    print(f"Valid ROI: {valid_roi:.2f}%")

    # テストデータで最終評価
    test_env = MultiRaceEnv(
        df=test_df,
        feature_dim=dim,
        id_col=id_col,
        horse_col=horse_col,
        horse_name_col=horse_name_col,
        single_odds_col=single_odds_col,
        finishing_col=finishing_col,
        cost=cost,
        races_per_episode=races_per_episode
    )
    test_roi, test_df_out = evaluate_model(test_env, model)
    print(f"Test ROI: {test_roi:.2f}%")

    # 推論結果の保存
    test_df_out.to_csv(SAVE_PATH_PRED, index=False, encoding='utf_8_sig')

    # 学習ログを可視化
    stats_callback.plot_logs()


if __name__ == "__main__":
    # スクリプトを直接実行したときにトレーニングから推論までを実行
    run_training_and_inference(
        data_path=DATA_PATH,
        id_col='race_id',
        horse_col='馬番',
        horse_name_col='馬名',
        single_odds_col='単勝',
        finishing_col='着順',
        cost=0.01, # 学習を安定させるため、costのスケールを減らす
        total_timesteps=200000,
        races_per_episode=128
    )
