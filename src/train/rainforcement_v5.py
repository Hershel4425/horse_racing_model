import os
import re
import datetime
import random
from tqdm import tqdm

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

# ここではSACを使うため、off-policyアルゴリズムを読み込む
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

ROOT_PATH = "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/153_競馬_v3"
DATE_STRING = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

MODEL_SAVE_DIR = os.path.join(ROOT_PATH, f"models/SAC_offpolicy/{DATE_STRING}")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

SAVE_PATH_PRED = os.path.join(ROOT_PATH, f"result/predictions/SAC_offpolicy/{DATE_STRING}.csv")
pred_dir = os.path.dirname(SAVE_PATH_PRED)
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)

DATA_PATH = os.path.join(ROOT_PATH, "result/predictions/transformer/20250109221743.csv")
FEATURE_PATH = os.path.join(ROOT_PATH, "data/02_features/feature.csv")
FUKUSHO_PATH = os.path.join(ROOT_PATH, "data/01_processed/50_odds/odds_df.csv")


##### ------------######
#    設定パラメータ一覧   #
##### ------------######
TEST_RATIO = 0.05 # test分割時のtestの割合
PCA_DIM_HORSE = 50 # 馬情報のpca圧縮次元
PCA_DIM_JOCKEY = 50 # 騎手情報のpca圧縮次元

COST = 100 # 掛け金の一単位
TOTAL_TIMESTEPS = 100000 # 学習の総timestep
RACES_PER_EPISODE = 16 # 1エピソードのレース数

INITIAL_CAPITAL = 10000 # エピソード最初の所持金
MAX_TOTAL_BET_COST = 1000 # 1レースの最大掛け金

WIN_RATIO = 0.5 # 単勝にかける割合
PLACE_RATIO = 0.5 # 複勝にかける割合

SAC_HYPERPARAMS = {
    "learning_rate": 3e-4,
    "buffer_size": 100000,
    "batch_size": 512,
    "ent_coef": "auto",
    "gamma": 0.99,
    "tau": 0.005,
    "train_freq": 1,
    "gradient_steps": 1,
    "policy_kwargs": dict(net_arch=[256, 256, 128])
}


def split_data(df, id_col="race_id", test_ratio=0.05):
    """
    既存のsplit_data関数を流用
    """
    df = df.sort_values('date').reset_index(drop=True)
    race_ids = df[id_col].unique()
    dataset_len = len(race_ids)
    print(f'total race_id : {dataset_len}')

    test_cut = int(dataset_len * (1 - test_ratio))
    train_ids = race_ids[:test_cut]
    test_ids = race_ids[test_cut:]

    train_df = df[df[id_col].isin(train_ids)].copy()
    test_df = df[df[id_col].isin(test_ids)].copy()

    return train_df, test_df


def prepare_data(
    data_path,
    feature_path,
    test_ratio=0.1,
    id_col="race_id",
    single_odds_col="単勝",
    place_odds_col="複勝",   # ★ 複勝カラム
    finishing_col="着順",
    pca_dim_horse=50,
    pca_dim_jockey=50
):
    """
    前処理＆ train/test split を行う関数。
    """
    df1 = pd.read_csv(feature_path, encoding='utf-8-sig')
    df2 = pd.read_csv(data_path, encoding='utf-8-sig')
    fukusho_df = pd.read_csv(FUKUSHO_PATH, encoding='utf-8-sig')
    fukusho_df = fukusho_df.rename(columns={'馬番1': '馬番'})
    fukusho_df = fukusho_df.loc[fukusho_df['券種'] == '複勝']
    # 複勝を倍率で扱うために 100 で割る
    fukusho_df["複勝"] = fukusho_df["払戻金額"] / 100.0

    # df2 に複勝列をマージ
    df_temp = pd.merge(
        df2,
        fukusho_df[["race_id", "馬番", "複勝"]],
        on=["race_id", "馬番"],
        how="left"
    )
    # feature.csv(df1) とマージ
    df = pd.merge(
        df1,
        df_temp[["race_id", "馬番", "P_top1", "P_top3", "P_top5", "P_pop1", "P_pop3", "P_pop5", "複勝"]],
        on=["race_id", "馬番"],
        how="inner"
    )

    # NaN埋め
    df["複勝"] = df["複勝"].fillna(0.0)

    default_leakage_cols = [
        '斤量','タイム','着差','上がり3F','馬体重','人気',
        'horse_id','jockey_id','trainer_id',
        '順位点',
        '入線','1着タイム差','先位タイム差','5着着差','増減',
        '1C通過順位','2C通過順位','3C通過順位','4C通過順位','賞金',
        '前半ペース','後半ペース','ペース','上がり3F順位',
        '100m','200m','300m',
        '400m','500m','600m','700m','800m','900m','1000m','1100m','1200m','1300m','1400m','1500m',
        '1600m','1700m','1800m','1900m','2000m','2100m','2200m','2300m','2400m','2500m','2600m',
        '2700m','2800m','2900m','3000m','3100m','3200m','3300m','3400m','3500m','3600m',
        'horse_ability'
    ]
    df.drop(columns=default_leakage_cols, errors='ignore', inplace=True)

    # 馬名を退避
    df["馬名_raw"] = df["馬名"].astype(str)

    # 数値列 / カテゴリ列
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if finishing_col in num_cols:
        num_cols.remove(finishing_col)
    if single_odds_col in num_cols:
        num_cols.remove(single_odds_col)
    if place_odds_col in num_cols:
        num_cols.remove(place_odds_col)

    for c in num_cols:
        df[c] = df[c].fillna(0)
    for c in cat_cols:
        df[c] = df[c].fillna("missing").astype(str)

    # train/test split
    train_df, test_df = split_data(df, id_col=id_col, test_ratio=test_ratio)

    # PCA対象の列抽出
    pca_pattern_horse = r'^(競走馬芝|競走馬ダート|単年競走馬芝|単年競走馬ダート)'
    pca_pattern_jockey = r'^(騎手芝|騎手ダート|単年騎手芝|単年騎手ダート)'

    pca_horse_target_cols = [c for c in num_cols if re.match(pca_pattern_horse, c)]
    pca_jockey_target_cols = [c for c in num_cols if re.match(pca_pattern_jockey, c)]
    other_num_cols = [
        c for c in num_cols
        if c not in pca_horse_target_cols
        and c not in pca_jockey_target_cols
    ]

    # --- horse系PCA ---
    scaler_horse = StandardScaler()
    if len(pca_horse_target_cols) > 0:
        horse_train_scaled = scaler_horse.fit_transform(train_df[pca_horse_target_cols])
        horse_test_scaled = scaler_horse.transform(test_df[pca_horse_target_cols])
    else:
        horse_train_scaled = np.zeros((len(train_df), 0))
        horse_test_scaled = np.zeros((len(test_df), 0))

    pca_dim_horse = min(pca_dim_horse, horse_train_scaled.shape[1]) if horse_train_scaled.shape[1] > 0 else 0
    if pca_dim_horse > 0:
        pca_model_horse = PCA(n_components=pca_dim_horse)
        horse_train_pca = pca_model_horse.fit_transform(horse_train_scaled)
        horse_test_pca = pca_model_horse.transform(horse_test_scaled)
    else:
        horse_train_pca = horse_train_scaled
        horse_test_pca = horse_test_scaled

    # --- jockey系PCA ---
    scaler_jockey = StandardScaler()
    if len(pca_jockey_target_cols) > 0:
        jockey_train_scaled = scaler_jockey.fit_transform(train_df[pca_jockey_target_cols])
        jockey_test_scaled = scaler_jockey.transform(test_df[pca_jockey_target_cols])
    else:
        jockey_train_scaled = np.zeros((len(train_df), 0))
        jockey_test_scaled = np.zeros((len(test_df), 0))

    pca_dim_jockey = min(pca_dim_jockey, jockey_train_scaled.shape[1]) if jockey_train_scaled.shape[1] > 0 else 0
    if pca_dim_jockey > 0:
        pca_model_jockey = PCA(n_components=pca_dim_jockey)
        jockey_train_pca = pca_model_jockey.fit_transform(jockey_train_scaled)
        jockey_test_pca = pca_model_jockey.transform(jockey_test_scaled)
    else:
        jockey_train_pca = jockey_train_scaled
        jockey_test_pca = jockey_test_scaled

    # --- other数値 ---
    scaler_other = StandardScaler()
    if len(other_num_cols) > 0:
        other_train = scaler_other.fit_transform(train_df[other_num_cols])
        other_test = scaler_other.transform(test_df[other_num_cols])
    else:
        other_train = np.zeros((len(train_df), 0))
        other_test = np.zeros((len(test_df), 0))

    # カテゴリ列をエンコード
    for c in cat_cols:
        train_df[c] = train_df[c].astype('category')
        test_df[c] = test_df[c].astype('category')
        train_cat = train_df[c].cat.categories
        test_df[c] = pd.Categorical(test_df[c], categories=train_cat)
        train_df[c] = train_df[c].cat.codes
        test_df[c] = test_df[c].cat.codes

    cat_features_train = train_df[cat_cols].values
    cat_features_test = test_df[cat_cols].values

    X_train_num = np.concatenate([other_train, horse_train_pca, jockey_train_pca], axis=1)
    X_test_num = np.concatenate([other_test, horse_test_pca, jockey_test_pca], axis=1)

    X_train = np.concatenate([cat_features_train, X_train_num], axis=1)
    X_test = np.concatenate([cat_features_test, X_test_num], axis=1)

    p_cols = ["P_top1", "P_top3", "P_top5", "P_pop1", "P_pop3", "P_pop5"]
    p_train = train_df[p_cols].fillna(0.0).values
    p_test = test_df[p_cols].fillna(0.0).values

    X_train = np.concatenate([X_train, p_train], axis=1)
    X_test = np.concatenate([X_test, p_test], axis=1)

    train_df["X"] = list(X_train)
    test_df["X"] = list(X_test)

    dim = X_train.shape[1]

    return train_df, test_df, dim


def get_max_horses_for_env(train_df, test_df, id_col="race_id"):
    """
    train_df と test_df の両方を見て、最大頭数を求める。
    """
    # レースIDごとに何頭いるかカウント
    train_size = train_df.groupby(id_col).size().max()
    test_size = test_df.groupby(id_col).size().max()
    return max(train_size, test_size)


class MultiRaceEnvContinuous(gym.Env):
    """
    連続アクション空間対応の強化学習環境。
    単勝 & 複勝を想定しており、アクションベクトルの前半を単勝ベット、後半を複勝ベットに対応。
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
        place_odds_col="複勝",
        finishing_col="着順",
        cost=100,
        races_per_episode=16,
        initial_capital=INITIAL_CAPITAL,
        max_total_bet_cost=MAX_TOTAL_BET_COST,
        max_horses=None
    ):
        super().__init__()
        self.df = df
        self.id_col = id_col
        self.horse_col = horse_col
        self.horse_name_col = horse_name_col
        self.single_odds_col = single_odds_col
        self.place_odds_col = place_odds_col
        self.finishing_col = finishing_col
        self.cost = cost
        self.feature_dim = feature_dim
        self.races_per_episode = races_per_episode
        self.initial_capital = initial_capital
        self.max_total_bet_cost = max_total_bet_cost

        self.race_ids = df[id_col].unique().tolist()
        self.race_map = {}
        # ---- ここで「最大頭数」固定 ----
        if max_horses is not None:
            self.max_horses = max_horses
        else:
            # df 内での最大頭数を計算
            self.max_horses = 0
            for rid in self.race_ids:
                subdf = df[df[id_col] == rid]
                self.max_horses = max(self.max_horses, len(subdf))

        for rid in self.race_ids:
            subdf = df[df[id_col] == rid].copy().sort_values(self.horse_col)
            self.race_map[rid] = subdf

        # 観測次元 = max_horses * feature_dim + 1(所持金)
        obs_dim = self.max_horses * self.feature_dim + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # アクション次元 = 2 * max_horses (単勝 + 複勝)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2 * self.max_horses,),
            dtype=np.float32
        )

        self.sampled_races = []
        self.current_race_idx = 0
        self.terminated = False
        self.capital = self.initial_capital
        self.current_obs = None

    def _get_obs_for_race(self, race_df: pd.DataFrame):
        """
        レース内の馬の特徴量を flatten() して所持金を追加。
        """
        n_horses = len(race_df)
        feats = []
        for i in range(n_horses):
            feats.append(race_df.iloc[i]["X"])
        feats = np.array(feats, dtype=np.float32)

        # 足りない頭数分をゼロパディング
        if n_horses < self.max_horses:
            pad_len = self.max_horses - n_horses
            pad = np.zeros((pad_len, self.feature_dim), dtype=np.float32)
            feats = np.vstack([feats, pad])

        # flatten & 所持金を末尾に追加
        feats = feats.flatten()
        feats_with_capital = np.concatenate([feats, [self.capital]], axis=0)
        return feats_with_capital

    def _select_races_for_episode(self):
        self.sampled_races = random.sample(self.race_ids, k=self.races_per_episode)
        random.shuffle(self.sampled_races)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._select_races_for_episode()
        self.current_race_idx = 0
        self.terminated = False
        self.capital = self.initial_capital

        rid = self.sampled_races[self.current_race_idx]
        race_df = self.race_map[rid]
        self.current_obs = self._get_obs_for_race(race_df)
        return self.current_obs, {}

    def step(self, action):
        if self.terminated:
            return self.current_obs, 0.0, True, False, {}

        rid = self.sampled_races[self.current_race_idx]
        race_df = self.race_map[rid]
        n_horses = len(race_df)

        # アクションを前半(単勝)・後半(複勝)に分割
        half = self.max_horses
        win_action_raw = action[:half]
        place_action_raw = action[half:]

        # 1) まず clip [0,1]
        win_action = np.clip(win_action_raw, 0.0, 1.0)
        place_action = np.clip(place_action_raw, 0.0, 1.0)

        # 2) ソートして上位を取る
        top_k_win = min(1, n_horses)   # 例: 単勝はトップ1頭
        top_k_place = min(3, n_horses) # 例: 複勝はトップ3頭
        idx_top_win = np.argsort(win_action)[-top_k_win:]
        idx_top_place = np.argsort(place_action)[-top_k_place:]

        # 3) 上位馬の合計値で正規化
        sum_win = np.sum(win_action)
        sum_place = np.sum(place_action)
        bet_ratio_win = np.zeros_like(win_action)
        bet_ratio_place = np.zeros_like(place_action)
        if sum_win > 0:
            bet_ratio_win[idx_top_win] = win_action[idx_top_win] / sum_win
        if sum_place > 0:
            bet_ratio_place[idx_top_place] = place_action[idx_top_place] / sum_place

        # 4) 賭け可能総額
        total_bet = min(self.max_total_bet_cost, self.capital)
        bet_win = total_bet * WIN_RATIO
        bet_place = total_bet * PLACE_RATIO

        bet_amounts_win = bet_win * bet_ratio_win
        bet_amounts_place = bet_place * bet_ratio_place

        bet_amounts_win = np.floor(bet_amounts_win / self.cost) * self.cost
        bet_amounts_place = np.floor(bet_amounts_place / self.cost) * self.cost

        race_cost = np.sum(bet_amounts_win) + np.sum(bet_amounts_place)
        self.capital -= race_cost

        # 払戻計算
        race_profit_win = 0.0
        race_profit_place = 0.0
        for i in range(self.max_horses):
            if i < n_horses:
                row = race_df.iloc[i]
                finishing = row[self.finishing_col]

                # 単勝的中
                if finishing == 1:
                    race_profit_win += bet_amounts_win[i] * row[self.single_odds_col]
                # 複勝的中(着順<=3 例)
                if finishing <= 3:
                    race_profit_place += bet_amounts_place[i] * row[self.place_odds_col]

        race_profit = race_profit_win + race_profit_place
        self.capital += race_profit

        # 報酬計算 (log(1 + 利益率)) など
        if race_cost > 0:
            ratio = (race_profit - race_cost) / race_cost
            ratio_clamped = max(ratio, -0.99)
            reward = np.log1p(ratio_clamped)
        else:
            reward = 0.0

        # 追加ペナルティ例: ベットが少なすぎる場合
        if race_cost < 100:
            reward -= 0.1

        self.current_race_idx += 1
        terminated = (self.current_race_idx >= self.races_per_episode)
        if self.capital <= 500:
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


def evaluate_model(
        env: MultiRaceEnvContinuous,
        model,
        capital_reset_threshold=MAX_TOTAL_BET_COST,
        capital_reset_value=INITIAL_CAPITAL
):
    """
    学習済みモデルで全レースを通し推論し、ROIを計算。馬券ごとの結果を DataFrame で返す。
    """
    # 評価用に別途キャピタルを用意して試算する想定
    capital = env.initial_capital

    original_ids = env.race_ids
    cost_sum = 0.0
    profit_sum = 0.0
    results = []

    for rid in tqdm(original_ids):
        subdf = env.race_map[rid].sort_values(env.horse_col).reset_index(drop=True)
        # 環境に合わせたサイズにパディングした観測を得る
        obs = env._get_obs_for_race(subdf)

        # SB3モデルのpredictは (n_env, obs_dim) の shape を期待する場合が多いので
        # 1次元 -> (1, obs_dim) に reshape してから予測
        action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
        # 返ってくる action も (1, 2*self.max_horses) なので squeeze
        action = action.squeeze(0)

        half = env.max_horses
        win_action_raw = action[:half]
        place_action_raw = action[half:]

        # まず clip
        win_action = np.clip(win_action_raw, 0.0, 1.0)
        place_action = np.clip(place_action_raw, 0.0, 1.0)

        n_horses = len(subdf)
        top_k = min(3, n_horses)
        idx_top_win = np.argsort(win_action)[-1:]   # 単勝 1頭
        idx_top_place = np.argsort(place_action)[-top_k:]

        sum_win = np.sum(win_action)
        sum_place = np.sum(place_action)

        bet_ratio_win = np.zeros_like(win_action)
        bet_ratio_place = np.zeros_like(place_action)
        if sum_win > 0:
            bet_ratio_win[idx_top_win] = win_action[idx_top_win] / sum_win
        if sum_place > 0:
            bet_ratio_place[idx_top_place] = place_action[idx_top_place] / sum_place

        total_bet = min(env.max_total_bet_cost, capital)
        bet_win = total_bet * WIN_RATIO
        bet_place = total_bet * PLACE_RATIO

        bet_amounts_win = bet_win * bet_ratio_win
        bet_amounts_place = bet_place * bet_ratio_place

        bet_amounts_win = np.floor(bet_amounts_win / env.cost) * env.cost
        bet_amounts_place = np.floor(bet_amounts_place / env.cost) * env.cost

        race_cost = np.sum(bet_amounts_win) + np.sum(bet_amounts_place)
        race_profit = 0.0

        # 各馬の bet_amount_win, bet_amount_place を記録するために回す
        for i in range(env.max_horses):
            if i < n_horses:
                row = subdf.iloc[i]
                finishing = row[env.finishing_col]

                # 払戻
                single_hit = 0.0
                place_hit = 0.0
                if finishing == 1:
                    single_hit = bet_amounts_win[i] * row[env.single_odds_col]
                if finishing <= 3:
                    place_hit = bet_amounts_place[i] * row[env.place_odds_col]
                race_profit += single_hit + place_hit

                # 保存
                results.append({
                    "race_id": rid,
                    "馬番": row[env.horse_col],
                    "馬名": row[env.horse_name_col + "_raw"],
                    "着順": row[env.finishing_col],
                    "単勝": row[env.single_odds_col],
                    "複勝": row[env.place_odds_col],
                    "bet_amount_win": bet_amounts_win[i],
                    "bet_amount_place": bet_amounts_place[i],
                })

        # 所持金更新
        capital -= race_cost
        capital += race_profit

        # 閾値より下がったらリセット
        if capital < capital_reset_threshold:
            capital = capital_reset_value

        cost_sum += race_cost
        profit_sum += race_profit

    roi = (profit_sum / cost_sum) if cost_sum > 0 else 0.0
    df_out = pd.DataFrame(results)
    return roi, df_out


class OffPolicyStatsCallback(BaseCallback):
    """
    SACやTD3の学習中にロスなどを拾うためのコールバック。
    """
    def __init__(self):
        super().__init__(verbose=0)
        self.iteration_logs = {
            "iteration": [],
            "timesteps": [],
            "time_elapsed": [],
            "actor_loss": [],
            "critic_loss": []
        }
        self.iter_count = 0

    def _on_step(self) -> bool:
        return True

    def _on_training_start(self):
        pass

    def _on_rollout_end(self):
        self.iter_count += 1
        self.iteration_logs["iteration"].append(self.iter_count)
        self.iteration_logs["timesteps"].append(self.model.num_timesteps)

        # SB3 の logger から値を取得
        self.iteration_logs["time_elapsed"].append(
            self.model.logger.name_to_value.get("time/total_timesteps", 0)
        )
        actor_loss = self.model.logger.name_to_value.get("train/actor_loss")
        critic_loss = self.model.logger.name_to_value.get("train/critic_loss")
        import math
        self.iteration_logs["actor_loss"].append(actor_loss if actor_loss is not None else math.nan)
        self.iteration_logs["critic_loss"].append(critic_loss if critic_loss is not None else math.nan)

    def plot_logs(self):
        logs = self.iteration_logs
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # timesteps
        axs[0, 0].plot(logs["iteration"], logs["timesteps"], marker='o', label="timesteps")
        axs[0, 0].set_xlabel("iteration")
        axs[0, 0].set_ylabel("timesteps")
        axs[0, 0].legend()

        # time/total_timesteps
        axs[0, 1].plot(logs["iteration"], logs["time_elapsed"], marker='o', label="time/total_timesteps")
        axs[0, 1].set_xlabel("iteration")
        axs[0, 1].set_ylabel("time/total_timesteps")
        axs[0, 1].legend()

        # actor_loss
        axs[1, 0].plot(logs["iteration"], logs["actor_loss"], marker='o', color='red', label="actor_loss")
        axs[1, 0].set_xlabel("iteration")
        axs[1, 0].set_ylabel("actor_loss")
        axs[1, 0].legend()

        # critic_loss
        axs[1, 1].plot(logs["iteration"], logs["critic_loss"], marker='o', color='blue', label="critic_loss")
        axs[1, 1].set_xlabel("iteration")
        axs[1, 1].set_ylabel("critic_loss")
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
    place_odds_col="複勝",
    finishing_col='着順',
    cost=COST,
    total_timesteps=TOTAL_TIMESTEPS,
    races_per_episode=RACES_PER_EPISODE,
    seed_value=42
):
    set_random_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)

    # train/test へ前処理
    train_df, test_df, dim = prepare_data(
        data_path=data_path,
        feature_path=feature_path,
        id_col=id_col,
        test_ratio=TEST_RATIO,
        single_odds_col=single_odds_col,
        finishing_col=finishing_col,
        place_odds_col=place_odds_col,
        pca_dim_horse=PCA_DIM_HORSE,
        pca_dim_jockey=PCA_DIM_JOCKEY
    )

    # train/test 両方を見て最大頭数を算出し、同じ次元の環境を作る
    global_max_horses = get_max_horses_for_env(train_df, test_df, id_col=id_col)

    # ---- train_env ----
    train_env = MultiRaceEnvContinuous(
        df=train_df,
        feature_dim=dim,
        id_col=id_col,
        horse_col=horse_col,
        horse_name_col=horse_name_col,
        single_odds_col=single_odds_col,
        place_odds_col=place_odds_col,
        finishing_col=finishing_col,
        cost=cost,
        races_per_episode=races_per_episode,
        max_horses=global_max_horses  # ここで最大頭数を固定
    )
    vec_train_env = DummyVecEnv([lambda: train_env])

    stats_callback = OffPolicyStatsCallback()

    model = SAC(
        "MlpPolicy",
        env=vec_train_env,
        verbose=1,
        **SAC_HYPERPARAMS
    )
    model.learn(total_timesteps=total_timesteps, callback=stats_callback)

    # ---- evaluate (train) ----
    train_roi, train_df_out = evaluate_model(train_env, model)
    print(f"Train ROI: {train_roi*100:.2f}%")

    # ---- evaluate (test) ----
    test_env = MultiRaceEnvContinuous(
        df=test_df,
        feature_dim=dim,
        id_col=id_col,
        horse_col=horse_col,
        horse_name_col=horse_name_col,
        single_odds_col=single_odds_col,
        place_odds_col=place_odds_col,
        finishing_col=finishing_col,
        cost=cost,
        races_per_episode=races_per_episode,
        max_horses=global_max_horses  # テスト環境も同じ最大頭数
    )
    test_roi, test_df_out = evaluate_model(test_env, model)
    print(f"Test ROI: {test_roi*100:.2f}%")

    # 推論結果のCSV書き出し
    # bet_amount_win + bet_amount_place を合計カラムにしておく
    test_df_out["bet_amount_total"] = test_df_out["bet_amount_win"] + test_df_out["bet_amount_place"]
    # 単勝・複勝の的中払い戻しをまとめた payout
    test_df_out["payout_total"] = 0.0
    # 1着なら単勝的中
    cond_single = (test_df_out["着順"] == 1)
    test_df_out.loc[cond_single, "payout_total"] += (
        test_df_out.loc[cond_single, "bet_amount_win"] * test_df_out.loc[cond_single, "単勝"]
    )
    # 3着以内なら複勝的中(例)
    cond_place = (test_df_out["着順"] <= 3)
    test_df_out.loc[cond_place, "payout_total"] += (
        test_df_out.loc[cond_place, "bet_amount_place"] * test_df_out.loc[cond_place, "複勝"]
    )
    # ROI (馬券1件ごとの水準)
    test_df_out["roi"] = test_df_out.apply(
        lambda row: (row["payout_total"] / row["bet_amount_total"]) if row["bet_amount_total"] > 0 else 0,
        axis=1
    )
    test_df_out.to_csv(SAVE_PATH_PRED, index=False, encoding='utf_8_sig')

    # 可視化用に train_df_out も同様に合計カラムを付与
    train_df_out["bet_amount_total"] = train_df_out["bet_amount_win"] + train_df_out["bet_amount_place"]
    train_df_out["payout_total"] = 0.0
    cond_single_tr = (train_df_out["着順"] == 1)
    train_df_out.loc[cond_single_tr, "payout_total"] += (
        train_df_out.loc[cond_single_tr, "bet_amount_win"] * train_df_out.loc[cond_single_tr, "単勝"]
    )
    cond_place_tr = (train_df_out["着順"] <= 3)
    train_df_out.loc[cond_place_tr, "payout_total"] += (
        train_df_out.loc[cond_place_tr, "bet_amount_place"] * train_df_out.loc[cond_place_tr, "複勝"]
    )
    train_df_out["roi"] = train_df_out.apply(
        lambda row: (row["payout_total"] / row["bet_amount_total"]) if row["bet_amount_total"] > 0 else 0,
        axis=1
    )

    # レース単位の集計
    train_race_agg = train_df_out.groupby("race_id").agg(
        race_bet_amount=("bet_amount_total", "sum"),
        race_payout=("payout_total", "sum")
    ).reset_index()
    train_race_agg["race_roi"] = train_race_agg["race_payout"] / (train_race_agg["race_bet_amount"] + 1e-15)

    test_race_agg = test_df_out.groupby("race_id").agg(
        race_bet_amount=("bet_amount_total", "sum"),
        race_payout=("payout_total", "sum")
    ).reset_index()
    test_race_agg["race_roi"] = test_race_agg["race_payout"] / (test_race_agg["race_bet_amount"] + 1e-15)

    # ----------------
    # (1) train用ヒストグラム
    # ----------------
    plt.figure(figsize=(10, 6))
    bins_bet = range(0, int(train_race_agg["race_bet_amount"].max()) + 100, 100)
    plt.hist(train_race_agg["race_bet_amount"], bins=bins_bet, edgecolor="black")
    plt.title("【Train】レース単位の賭け金合計ヒストグラム (100円区切り)")
    plt.xlabel("race_bet_amount (円)")
    plt.ylabel("レース件数")
    plt.xticks(bins_bet, rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(train_race_agg["race_roi"], bins=50, edgecolor="black")
    plt.title("【Train】レース単位の回収率ヒストグラム")
    plt.xlabel("race_roi (払い戻し / 賭け金)")
    plt.ylabel("レース件数")
    plt.tight_layout()
    plt.show()

    # ----------------
    # (2) test用ヒストグラム
    # ----------------
    plt.figure(figsize=(10, 6))
    bins_bet_test = range(0, int(test_race_agg["race_bet_amount"].max()) + 100, 100)
    plt.hist(test_race_agg["race_bet_amount"], bins=bins_bet_test, edgecolor="black")
    plt.title("【Test】レース単位の賭け金合計ヒストグラム (100円区切り)")
    plt.xlabel("race_bet_amount (円)")
    plt.ylabel("レース件数")
    plt.xticks(bins_bet_test, rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(test_race_agg["race_roi"], bins=50, edgecolor="black")
    plt.title("【Test】レース単位の回収率ヒストグラム")
    plt.xlabel("race_roi (払い戻し / 賭け金)")
    plt.ylabel("レース件数")
    plt.tight_layout()
    plt.show()

    stats_callback.plot_logs()

    # モデルの保存
    # stable-baselines3 の標準的な保存形式(model.zip)を使う方が安全です。
    model_save_path = os.path.join(MODEL_SAVE_DIR, "model.zip")
    model.save(model_save_path)


if __name__ == "__main__":
    run_training_and_inference_offpolicy(total_timesteps=10000)
