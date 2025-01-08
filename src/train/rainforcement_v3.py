import os
import re
import datetime
import random
from tqdm import tqdm

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

# Off-Policy実装として、Stable-Baselines3-ContribのQR-DQNを使用
# なぜQR-DQNか？
#   - Off-Policy手法の中でもDQN系列は実績があり、
#   - さらに分布強化学習の要素を取り入れたQuantile Regression DQN (QR-DQN)は
#     通常のDQNよりも報酬分布を学習できる点でパフォーマンスの安定が期待できるため。
#   - SB3-contribに含まれており、Stable-Baselines3と同様に扱いやすい。
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# 日付文字列を作成して実行単位を管理（なぜ必要か？ -> 実験ごとに成果物を時系列で区分管理）
ROOT_PATH = "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/153_競馬_v3"
DATE_STRING = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

# モデルや結果を保存するディレクトリを用意（なぜ分離するか？ -> 過去実験分と混ざるのを防ぎ、再現性を高める）
MODEL_SAVE_DIR = os.path.join(ROOT_PATH, f"models/offpolicy_QRDQN/{DATE_STRING}")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

SAVE_PATH_PRED = os.path.join(ROOT_PATH, f"result/predictions/offpolicy_QRDQN/{DATE_STRING}.csv")
pred_dir = os.path.dirname(SAVE_PATH_PRED)
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)

DATA_PATH = os.path.join(ROOT_PATH, "result/predictions/transformer/20250106194932.csv")
FEATURE_PATH = os.path.join(ROOT_PATH, "data/02_features/feature.csv")

def split_data(df, id_col="race_id", test_ratio=0.05, valid_ratio=0.05):
    """
    データをtrain/valid/testに分割するための関数
    ------------------------------------------------------------------------------------------
    Why:
      - 競馬データは時系列的要素（開催日）があるため、古い→新しい順に並べてから
        データを分割することで情報漏洩を防ぐ
      - race_id単位でまとめて分割することで、同じレースがtrainとtestに分散しないようにする
    ------------------------------------------------------------------------------------------
    """
    df = df.sort_values('date').reset_index(drop=True)
    race_ids = df[id_col].unique()
    dataset_len = len(race_ids)

    test_cut = int(dataset_len * (1 - test_ratio))
    valid_cut = int(test_cut * (1 - valid_ratio))

    train_ids = race_ids[:valid_cut]
    valid_ids = race_ids[valid_cut:test_cut]
    test_ids = race_ids[test_cut:]

    train_df = df[df[id_col].isin(train_ids)].copy()
    valid_df = df[df[id_col].isin(valid_ids)].copy()
    test_df = df[df[id_col].isin(test_ids)].copy()

    return train_df, valid_df, test_df

def prepare_data(
    data_path,
    feature_path,
    test_ratio=0.1,
    valid_ratio=0.1,
    id_col="race_id",
    single_odds_col="単勝",
    finishing_col="着順",
    pca_dim_horse=50,
    pca_dim_jockey=50
):
    """
    データの読み込みから、リーク対策・欠損処理・PCAでの次元圧縮・標準化などをまとめる
    ------------------------------------------------------------------------------------------
    Why:
      - 過去データを扱う上で、未来情報を含む列は削除（リーク対策）
      - PCAで高次元の馬・騎手統計情報を圧縮し、学習の負荷と過学習リスクを軽減
      - 標準化により、特定の大きな値を持つ特徴量が学習を支配しないようにする
    ------------------------------------------------------------------------------------------
    """
    df1 = pd.read_csv(feature_path, encoding='utf-8-sig')
    df2 = pd.read_csv(data_path, encoding='utf-8-sig')

    # df2には追加予測列(P_top1など)が入っている想定
    df = pd.merge(
        df1,
        df2[["race_id", "馬番", "P_top1", "P_top3", "P_top5", "P_pop1", "P_pop3", "P_pop5"]],
        on=["race_id", "馬番"],
        how="inner"
    )

    # 今回リークの可能性が高い列を削除
    default_leakage_cols = [
        '斤量','タイム','着差','上がり3F','馬体重','人気','horse_id','jockey_id','trainer_id','順位点','入線',
        '1着タイム差','先位タイム差','5着着差','増減','1C通過順位','2C通過順位','3C通過順位','4C通過順位',
        '賞金','前半ペース','後半ペース','ペース','上がり3F順位','100m','200m','300m','400m','500m','600m','700m',
        '800m','900m','1000m','1100m','1200m','1300m','1400m','1500m','1600m','1700m','1800m','1900m','2000m','2100m',
        '2200m','2300m','2400m','2500m','2600m','2700m','2800m','2900m','3000m','3100m','3200m','3300m','3400m','3500m',
        '3600m','horse_ability'
    ]
    df.drop(columns=default_leakage_cols, errors='ignore', inplace=True)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # 目的列や単勝オッズ列は分析用に残すが、スケーリング対象から除外
    if finishing_col in num_cols:
        num_cols.remove(finishing_col)
    if single_odds_col in num_cols:
        num_cols.remove(single_odds_col)

    # 数値列: 欠損は0埋め（簡易実装だが、より洗練された方法が必要な場合も）
    for c in num_cols:
        df[c] = df[c].fillna(0)
    # カテゴリ列: "missing"というカテゴリを追加
    for c in cat_cols:
        df[c] = df[c].fillna("missing").astype(str)

    # 学習・バリデーション・テストに分割
    train_df, valid_df, test_df = split_data(df, id_col=id_col, test_ratio=test_ratio, valid_ratio=valid_ratio)

    # 馬・騎手に関連する列を正規表現でまとめ、PCAを適用する
    pca_pattern_horse = r'^(競走馬芝|競走馬ダート|単年競走馬芝|単年競走馬ダート)'
    pca_pattern_jockey = r'^(騎手芝|騎手ダート|単年騎手芝|単年騎手ダート)'

    pca_horse_target_cols = [c for c in num_cols if re.match(pca_pattern_horse, c)]
    pca_jockey_target_cols = [c for c in num_cols if re.match(pca_pattern_jockey, c)]
    other_num_cols = [
        c for c in num_cols
        if c not in pca_horse_target_cols
        and c not in pca_jockey_target_cols
    ]

    # 馬関係列にPCA
    scaler_horse = StandardScaler()
    if len(pca_horse_target_cols) > 0:
        horse_train_scaled = scaler_horse.fit_transform(train_df[pca_horse_target_cols])
        horse_valid_scaled = scaler_horse.transform(valid_df[pca_horse_target_cols])
        horse_test_scaled = scaler_horse.transform(test_df[pca_horse_target_cols])
    else:
        horse_train_scaled = np.zeros((len(train_df), 0))
        horse_valid_scaled = np.zeros((len(valid_df), 0))
        horse_test_scaled = np.zeros((len(test_df), 0))

    pca_dim_horse = min(pca_dim_horse, horse_train_scaled.shape[1]) if horse_train_scaled.shape[1] > 0 else 0
    if pca_dim_horse > 0:
        pca_model_horse = PCA(n_components=pca_dim_horse)
        horse_train_pca = pca_model_horse.fit_transform(horse_train_scaled)
        horse_valid_pca = pca_model_horse.transform(horse_valid_scaled)
        horse_test_pca = pca_model_horse.transform(horse_test_scaled)
    else:
        horse_train_pca = horse_train_scaled
        horse_valid_pca = horse_valid_scaled
        horse_test_pca = horse_test_scaled

    # 騎手関係列にPCA
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
        jockey_train_pca = jockey_train_scaled
        jockey_valid_pca = jockey_valid_scaled
        jockey_test_pca = jockey_test_scaled

    # その他数値列を標準化
    scaler_other = StandardScaler()
    if len(other_num_cols) > 0:
        other_train = scaler_other.fit_transform(train_df[other_num_cols])
        other_valid = scaler_other.transform(valid_df[other_num_cols])
        other_test = scaler_other.transform(test_df[other_num_cols])
    else:
        other_train = np.zeros((len(train_df), 0))
        other_valid = np.zeros((len(valid_df), 0))
        other_test = np.zeros((len(test_df), 0))

    # カテゴリ列はcategory変換 -> category codesに変換 (trainのカテゴリを基準にvalid/testを合わせる)
    for c in cat_cols:
        train_df[c] = train_df[c].astype('category')
        valid_df[c] = valid_df[c].astype('category')
        test_df[c] = test_df[c].astype('category')
        train_cat = train_df[c].cat.categories
        valid_df[c] = pd.Categorical(valid_df[c], categories=train_cat)
        test_df[c] = pd.Categorical(test_df[c], categories=train_cat)
        train_df[c] = train_df[c].cat.codes
        valid_df[c] = valid_df[c].cat.codes
        test_df[c] = test_df[c].cat.codes

    cat_features_train = train_df[cat_cols].values
    cat_features_valid = valid_df[cat_cols].values
    cat_features_test = test_df[cat_cols].values

    # 上記3つの数値配列を結合
    X_train_num = np.concatenate([other_train, horse_train_pca, jockey_train_pca], axis=1)
    X_valid_num = np.concatenate([other_valid, horse_valid_pca, jockey_valid_pca], axis=1)
    X_test_num = np.concatenate([other_test, horse_test_pca, jockey_test_pca], axis=1)

    X_train = np.concatenate([cat_features_train, X_train_num], axis=1)
    X_valid = np.concatenate([cat_features_valid, X_valid_num], axis=1)
    X_test = np.concatenate([cat_features_test, X_test_num], axis=1)

    # P_top1, P_top3, P_top5, P_pop1, P_pop3, P_pop5 も追加
    p_cols = ["P_top1", "P_top3", "P_top5", "P_pop1", "P_pop3", "P_pop5"]
    p_train = train_df[p_cols].fillna(0.0).values
    p_valid = valid_df[p_cols].fillna(0.0).values
    p_test = test_df[p_cols].fillna(0.0).values

    X_train = np.concatenate([X_train, p_train], axis=1)
    X_valid = np.concatenate([X_valid, p_valid], axis=1)
    X_test = np.concatenate([X_test, p_test], axis=1)

    train_df["X"] = list(X_train)
    valid_df["X"] = list(X_valid)
    test_df["X"] = list(X_test)

    dim = X_train.shape[1]
    return train_df, valid_df, test_df, dim

class MultiRaceEnv(gym.Env):
    """
    競馬レース用の強化学習環境 (Off-Policy対応版)
    ------------------------------------------------------------------------------------------
    なぜ独自環境が必要か？
      - 標準のAtariやMuJoCoなどの環境と異なり「馬券購入 → レース結果での払い戻し」
        という独特の流れをシミュレートする必要があるため。
    ------------------------------------------------------------------------------------------
    ※ 今回はOff-Policyアルゴリズム(QRDQN)を用いるため、経験を再利用しやすいように環境設計をそのまま継承。
      ただしMultiDiscreteアクションは標準DQN系が対応しにくいため、
      bet_mode="single"（単勝1点買い）モードのみ対応可能とし、複数馬への同時ベットはNotImplementedErrorとする。
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
        races_per_episode=16,
        initial_capital=5000,
        bet_mode="single"
    ):
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
        self.initial_capital = initial_capital

        # オフポリシー用に、複数ベット（MultiDiscrete）は標準DQN系列が扱いにくいため未対応とする
        if bet_mode != "single":
            raise NotImplementedError("このサンプルコードではbet_mode='single'のみ対応です。")

        self.bet_mode = bet_mode

        # race_id毎にDataFrameを保管
        self.race_ids = df[id_col].unique().tolist()
        self.race_map = {}
        self.max_horses = 0

        # 全レース中の最大出走頭数を特定→観測を固定次元に（パディング方式）
        for rid in self.race_ids:
            subdf = df[df[id_col] == rid].copy().sort_values(self.horse_col)
            self.max_horses = max(self.max_horses, len(subdf))
            self.race_map[rid] = subdf

        # 観測空間: 馬最大頭数×特徴量 + 所持金1次元
        obs_dim = self.max_horses * self.feature_dim + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # 行動空間: 単勝1点買いのみ
        #   -> 0〜(max_horses-1)の離散行動
        self.action_space = spaces.Discrete(self.max_horses)

        # 1エピソード内のレースをシャッフルする
        self.sampled_races = []
        self.current_race_idx = 0
        self.current_obs = None
        self.terminated = False

        # 持ち金
        self.capital = self.initial_capital

    def _get_obs_for_race(self, race_df: pd.DataFrame):
        """
        race_df（特定のレース）内の馬ごとの特徴量を取り出し、最大頭数まで0パディングしてflatten
        最後に所持金を連結して観測ベクトルを作る
        ------------------------------------------------------------------------------------------
        Why:
          - Off-Policyでも基本的な「状態(s)」は同じく馬情報＋所持金という構成
        ------------------------------------------------------------------------------------------
        """
        n_horses = len(race_df)
        feats = []
        for i in range(n_horses):
            feats.append(race_df.iloc[i]["X"])
        feats = np.array(feats, dtype=np.float32)

        # 頭数が最大に満たない場合はパディング
        if n_horses < self.max_horses:
            pad_len = self.max_horses - n_horses
            pad = np.zeros((pad_len, self.feature_dim), dtype=np.float32)
            feats = np.vstack([feats, pad])

        # flattenして所持金を最後に追加
        feats = feats.flatten()
        feats_with_capital = np.concatenate([feats, [self.capital]], axis=0)
        return feats_with_capital

    def _select_races_for_episode(self):
        """
        1エピソードで使用するレースをランダムサンプリング（オフポリシーでもサンプリングは同様に実施）
        ------------------------------------------------------------------------------------------
        Why:
          - 常に同じ順序や同じレースを学習すると偏りが生じやすいためシャッフル
        ------------------------------------------------------------------------------------------
        """
        self.sampled_races = random.sample(self.race_ids, k=self.races_per_episode)
        random.shuffle(self.sampled_races)

    def reset(self, seed=None, options=None):
        """
        エピソード開始前に呼ばれる
        ------------------------------------------------------------------------------------------
        Why:
          - 毎エピソード、レースをシャッフルして初期化
        ------------------------------------------------------------------------------------------
        """
        super().reset(seed=seed)
        self._select_races_for_episode()
        self.current_race_idx = 0
        self.terminated = False
        self.capital = self.initial_capital

        race_df = self.race_map[self.sampled_races[self.current_race_idx]]
        self.current_obs = self._get_obs_for_race(race_df)
        return self.current_obs, {}

    def step(self, action):
        """
        行動: 購入する馬のindex(単勝1点)が与えられる
        報酬: 当たった場合の払い戻し - 購入コスト
        ------------------------------------------------------------------------------------------
        Why:
          - オフポリシーの場合もステップ実行は同様に「状態→行動→報酬→次状態」を生成して経験を蓄積しやすい
        ------------------------------------------------------------------------------------------
        """
        if self.terminated:
            return self.current_obs, 0.0, True, False, {}

        rid = self.sampled_races[self.current_race_idx]
        race_df = self.race_map[rid]
        n_horses = len(race_df)

        # 賭け金
        race_cost = self.cost
        race_profit = 0.0

        # 持ち金を消費
        self.capital -= race_cost

        # 当たったら払い戻し
        if action < n_horses:
            row = race_df.iloc[action]
            if row[self.finishing_col] == 1:
                odds = row[self.single_odds_col]
                race_profit = race_cost * odds

        self.capital += race_profit

        reward = race_profit - race_cost  # 純増をそのまま報酬

        self.current_race_idx += 1
        terminated = (self.current_race_idx >= self.races_per_episode)

        # 所持金が尽きたら終了
        if self.capital <= 0:
            terminated = True

        self.terminated = terminated
        truncated = False

        if not terminated:
            next_rid = self.sampled_races[self.current_race_idx]
            next_race_df = self.race_map[next_rid]
            obs = self._get_obs_for_race(next_race_df)
            self.current_obs = obs
        else:
            obs = self.current_obs

        return obs, float(reward), terminated, truncated, {}

def evaluate_model(env: MultiRaceEnv, model):
    """
    学習したモデルを用いて全レースで推論を行い、ROIを算出する
    ------------------------------------------------------------------------------------------
    Why:
      - Off-Policy手法ではオンライン学習の途中でも評価可能だが、最終的な性能評価として
        全レースを通した収益率(ROI)を計算し、汎化能力を測る
    ------------------------------------------------------------------------------------------
    """
    original_ids = env.race_ids
    cost_sum = 0.0
    profit_sum = 0.0
    results = []

    for rid in tqdm(original_ids):
        # レースデータ抽出
        subdf = env.race_map[rid].sort_values(env.horse_col).reset_index(drop=True)
        obs = env._get_obs_for_race(subdf)

        # QRDQNで行動を決定（deterministic=True）
        action, _ = model.predict(obs, deterministic=True)
        n_horses = len(subdf)

        race_cost = env.cost
        race_profit = 0.0

        if action < n_horses:
            row = subdf.iloc[action]
            finishing = row[env.finishing_col]
            odds = row[env.single_odds_col]
            if finishing == 1:
                race_profit = race_cost * odds

        cost_sum += race_cost
        profit_sum += race_profit

        # 賭けた馬・賭け金等を記録（分析用）
        for i in range(n_horses):
            bet_amount = env.cost if i == action else 0
            row_i = subdf.iloc[i]
            results.append({
                "race_id": rid,
                "馬番": row_i[env.horse_col],
                "馬名": row_i[env.horse_name_col],
                "着順": row_i[env.finishing_col],
                "単勝": row_i[env.single_odds_col],
                "bet_amount": bet_amount
            })

    roi = (profit_sum / cost_sum) if cost_sum > 0 else 0.0
    return roi, pd.DataFrame(results)

class StatsCallback(BaseCallback):
    """
    学習過程のメトリクスを収集・可視化するコールバッククラス
    ------------------------------------------------------------------------------------------
    Why:
      - Off-Policyの場合でも学習過程を可視化し、Q値の安定性・収束状況を把握する
    ------------------------------------------------------------------------------------------
    """
    def __init__(self):
        super().__init__(verbose=0)
        self.iteration_logs = {
            "iteration": [],
            "timesteps": [],
            "time_elapsed": [],
            "fps": [],
            "train_loss": []
        }
        self.iter_count = 0

    def _on_step(self) -> bool:
        # 定期的に呼ばれ、ログを追加するなどの処理が可能
        return True

    def _on_rollout_end(self):
        # DQN系列の場合、rollout_endが呼ばれるタイミングがPPOと異なる場合があるので要注意
        self.iter_count += 1
        self.iteration_logs["iteration"].append(self.iter_count)
        self.iteration_logs["timesteps"].append(self.model.num_timesteps)
        # time/total_timesteps や train/loss などはloggerから取得
        self.iteration_logs["time_elapsed"].append(self.model.logger.name_to_value.get("time/total_timesteps", 0))
        self.iteration_logs["fps"].append(self.model.logger.name_to_value.get("time/fps", 0))
        # QRDQNの場合、train/lossがキーとして存在していれば取得
        self.iteration_logs["train_loss"].append(self.model.logger.name_to_value.get("train/loss", 0))

    def plot_logs(self):
        """
        ログをグラフで可視化
        ------------------------------------------------------------------------------------------
        Why:
          - 学習が進むにつれてLossが減少しているか、学習時間とともにどれだけステップが進んだか確認
        ------------------------------------------------------------------------------------------
        """
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        logs = self.iteration_logs

        # iteration vs timesteps
        axs[0, 0].plot(logs["iteration"], logs["timesteps"], marker='o', label="timesteps")
        axs[0, 0].set_xlabel("iteration")
        axs[0, 0].set_ylabel("timesteps")
        axs[0, 0].legend()

        # iteration vs fps
        axs[0, 1].plot(logs["iteration"], logs["fps"], marker='o', color='green', label="fps")
        axs[0, 1].set_xlabel("iteration")
        axs[0, 1].set_ylabel("fps")
        axs[0, 1].legend()

        # iteration vs train_loss
        axs[1, 0].plot(logs["iteration"], logs["train_loss"], marker='o', color='red', label="train_loss")
        axs[1, 0].set_xlabel("iteration")
        axs[1, 0].set_ylabel("train_loss")
        axs[1, 0].legend()

        # iteration vs time_elapsed
        axs[1, 1].plot(logs["iteration"], logs["time_elapsed"], marker='o', color='orange', label="time_elapsed")
        axs[1, 1].set_xlabel("iteration")
        axs[1, 1].set_ylabel("time_elapsed")
        axs[1, 1].legend()

        plt.tight_layout()
        plt.show()

def run_training_and_inference_offpolicy(
    data_path=DATA_PATH,
    feature_path=FEATURE_PATH,
    id_col='race_id',
    horse_col='馬番',
    horse_name_col='馬名',
    single_odds_col='単勝',
    finishing_col='着順',
    cost=100,
    total_timesteps=100000,  # Off-PolicyはReplayBufferで経験を再利用できるため少なめでも十分学習可
    races_per_episode=32,
    seed_value=42,
    bet_mode="single"
):
    """
    オフポリシー手法(QRDQN)を使った強化学習の実行フロー
    ------------------------------------------------------------------------------------------
    Why:
      - prepare_dataでデータ準備
      - Env構築→QRDQN学習→評価ROI算出
      - 結果およびモデルを保存し、後から再評価できるようにする
    ------------------------------------------------------------------------------------------
    """
    set_random_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)

    train_df, valid_df, test_df, dim = prepare_data(
        data_path=data_path,
        feature_path=feature_path,
        id_col=id_col,
        test_ratio=0.1,
        valid_ratio=0.1,
        single_odds_col=single_odds_col,
        finishing_col=finishing_col,
        pca_dim_horse=50,
        pca_dim_jockey=50
    )

    # 学習環境（train用）とバリデーション環境（valid用）を構築
    train_env = MultiRaceEnv(
        df=train_df,
        feature_dim=dim,
        id_col=id_col,
        horse_col=horse_col,
        horse_name_col=horse_name_col,
        single_odds_col=single_odds_col,
        finishing_col=finishing_col,
        cost=cost,
        races_per_episode=races_per_episode,
        bet_mode=bet_mode
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
        races_per_episode=races_per_episode,
        bet_mode=bet_mode
    )

    # DummyVecEnvでラップ
    vec_train_env = DummyVecEnv([lambda: train_env])

    # コールバック初期化 (ログ可視化用)
    stats_callback = StatsCallback()

    # QRDQNモデルのハイパラ設定
    # なぜQRDQNか？
    #   - オフポリシー+DQNの強化版として、分布強化学習を取り入れたQRDQNは収束が安定しやすい
    #   - SACなどは連続アクション向けのため、今回は離散アクションに対応するQRDQNを採用
    qrdqn_hyperparams = {
        "learning_rate": 1e-4,
        "buffer_size": 100000,   # Replay Bufferサイズ（十分大きく取る）
        "batch_size": 256,
        "tau": 0.8,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "exploration_fraction": 0.2,  # ε-greedyのεを徐々に下げる
        "exploration_final_eps": 0.05,
        "policy_kwargs": dict(
            net_arch=[256, 256]
        )
    }

    model = QRDQN(
        "MlpPolicy",
        env=vec_train_env,
        verbose=1,
        seed=seed_value,
        tensorboard_log=None,  # ログディレクトリを指定したい場合はここにパスを入れる
        **qrdqn_hyperparams
    )

    # 学習開始
    model.learn(total_timesteps=total_timesteps, callback=stats_callback)
    model.save(os.path.join(MODEL_SAVE_DIR, "qrdqn_model"))

    # 学習データでのROI
    train_roi, _ = evaluate_model(train_env, model)
    print(f"Train ROI: {train_roi*100:.2f}%")

    # バリデーションデータでのROI
    valid_roi, _ = evaluate_model(valid_env, model)
    print(f"Valid ROI: {valid_roi*100:.2f}%")

    # テストで最終評価
    test_env = MultiRaceEnv(
        df=test_df,
        feature_dim=dim,
        id_col=id_col,
        horse_col=horse_col,
        horse_name_col=horse_name_col,
        single_odds_col=single_odds_col,
        finishing_col=finishing_col,
        cost=cost,
        races_per_episode=races_per_episode,
        bet_mode=bet_mode
    )
    test_roi, test_df_out = evaluate_model(test_env, model)
    print(f"Test ROI: {test_roi*100:.2f}%")

    # 結果をCSVで保存
    test_df_out.to_csv(SAVE_PATH_PRED, index=False, encoding='utf_8_sig')

    # ログ可視化
    stats_callback.plot_logs()

if __name__ == "__main__":
    run_training_and_inference_offpolicy()
