import os
import re
import datetime
import random

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

ROOT_PATH = "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/153_競馬_v3"
DATE_STRING = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
MODEL_SAVE_DIR = os.path.join(ROOT_PATH, f"models/transormer予測モデル/{DATE_STRING}")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)
SAVE_PATH_PRED = os.path.join(ROOT_PATH, f"result/predictions/強化学習/{DATE_STRING}.csv")
pred_dir = os.path.dirname(SAVE_PATH_PRED)
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)
DATA_PATH = os.path.join(ROOT_PATH, "data/02_features/feature.csv")


def split_data(df, id_col="race_id", test_ratio=0.05, valid_ratio=0.05):
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
    id_col='race_id',
    test_ratio=0.1,
    valid_ratio=0.1,
    pca_dim_horse=50,
    pca_dim_jockey=50,
    cat_cols=None,
    finishing_col='着順',
    single_odds_col='単勝'
):
    if cat_cols is None:
        cat_cols = []
    df = pd.read_csv(data_path, encoding="utf_8_sig")
    default_leakage_cols = [
        '斤量','タイム','着差','上がり3F','馬体重','人気','horse_id','jockey_id','trainer_id','順位点',
        '入線','1着タイム差','先位タイム差','5着着差','増減','1C通過順位','2C通過順位','3C通過順位',
        '4C通過順位','賞金','前半ペース','後半ペース','ペース','上がり3F順位','100m','200m','300m',
        '400m','500m','600m','700m','800m','900m','1000m','1100m','1200m','1300m','1400m','1500m',
        '1600m','1700m','1800m','1900m','2000m','2100m','2200m','2300m','2400m','2500m','2600m',
        '2700m','2800m','2900m','3000m','3100m','3200m','3300m','3400m','3500m','3600m','horse_ability'
    ]
    df.drop(columns=default_leakage_cols, errors='ignore', inplace=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if finishing_col in num_cols:
        num_cols.remove(finishing_col)
    if single_odds_col in num_cols:
        num_cols.remove(single_odds_col)
    cat_cols = [c for c in cat_cols if c in df.columns]
    for c in num_cols:
        df[c] = df[c].fillna(0)
    train_df, valid_df, test_df = split_data(df, id_col=id_col, test_ratio=test_ratio, valid_ratio=valid_ratio)
    pca_pattern_horse = r'^(競走馬芝|競走馬ダート|単年競走馬芝|単年競走馬ダート)'
    pca_pattern_jockey = r'^(騎手芝|騎手ダート|単年騎手芝|単年騎手ダート)'
    pca_horse_target_cols = [c for c in num_cols if re.match(pca_pattern_horse, c)]
    pca_jockey_target_cols = [c for c in num_cols if re.match(pca_pattern_jockey, c)]
    other_num_cols = [
        c for c in num_cols
        if c not in pca_horse_target_cols
        and c not in pca_jockey_target_cols
    ]
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
        pca_model_horse = None
        horse_train_pca = horse_train_scaled
        horse_valid_pca = horse_valid_scaled
        horse_test_pca = horse_test_scaled
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
    scaler_other = StandardScaler()
    if len(other_num_cols) > 0:
        other_train = scaler_other.fit_transform(train_df[other_num_cols])
        other_valid = scaler_other.transform(valid_df[other_num_cols])
        other_test = scaler_other.transform(test_df[other_num_cols])
    else:
        other_train = np.zeros((len(train_df), 0))
        other_valid = np.zeros((len(valid_df), 0))
        other_test = np.zeros((len(test_df), 0))
    cat_features_train = train_df[cat_cols].values if cat_cols else np.zeros((len(train_df), 0))
    cat_features_valid = valid_df[cat_cols].values if cat_cols else np.zeros((len(valid_df), 0))
    cat_features_test = test_df[cat_cols].values if cat_cols else np.zeros((len(test_df), 0))
    X_train = np.concatenate([cat_features_train, other_train, horse_train_pca, jockey_train_pca], axis=1)
    X_valid = np.concatenate([cat_features_valid, other_valid, horse_valid_pca, jockey_valid_pca], axis=1)
    X_test = np.concatenate([cat_features_test, other_test, horse_test_pca, jockey_test_pca], axis=1)
    train_df["X"] = list(X_train)
    valid_df["X"] = list(X_valid)
    test_df["X"] = list(X_test)
    actual_num_dim = X_train.shape[1]
    return (
        train_df, valid_df, test_df,
        (scaler_horse, pca_model_horse), (scaler_jockey, pca_model_jockey), scaler_other,
        cat_cols, actual_num_dim
    )


class MultiRaceEnv(gym.Env):
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

        self.race_ids = df[id_col].unique().tolist()
        self.race_map = {}
        max_horses = 0
        for rid in self.race_ids:
            subdf = df[df[id_col] == rid].copy().sort_values(self.horse_col)
            max_horses = max(max_horses, len(subdf))
            self.race_map[rid] = subdf
        self.max_horses = max_horses

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_horses * self.feature_dim,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.max_horses)

        self.sampled_races = []
        self.current_race_idx = 0
        self.current_obs = None
        self.terminated = False

    def _get_obs_for_race(self, race_df: pd.DataFrame):
        n_horses = len(race_df)
        feats = []
        for i in range(n_horses):
            feats.append(race_df.iloc[i]["X"])
        feats = np.array(feats, dtype=np.float32)
        if n_horses < self.max_horses:
            pad_len = self.max_horses - n_horses
            pad = np.zeros((pad_len, self.feature_dim), dtype=np.float32)
            feats = np.vstack([feats, pad])
        return feats.flatten()

    def _select_races_for_episode(self):
        self.sampled_races = random.sample(self.race_ids, k=self.races_per_episode)
        random.shuffle(self.sampled_races)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._select_races_for_episode()
        self.current_race_idx = 0
        self.terminated = False
        race_df = self.race_map[self.sampled_races[self.current_race_idx]]
        self.current_obs = self._get_obs_for_race(race_df)
        return self.current_obs, {}

    def step(self, action):
        if self.terminated:
            return self.current_obs, 0.0, True, False, {}

        rid = self.sampled_races[self.current_race_idx]
        race_df = self.race_map[rid]
        n_horses = len(race_df)

        reward = 0.0
        if action < n_horses:
            row = race_df.iloc[action]
            if row[self.finishing_col] == 1:
                odds = row[self.single_odds_col]
                reward = odds * 100 - self.cost
            else:
                reward = -self.cost
        else:
            reward = -self.cost

        self.current_race_idx += 1
        terminated = (self.current_race_idx >= self.races_per_episode)
        truncated = False
        self.terminated = terminated
        if not terminated:
            next_rid = self.sampled_races[self.current_race_idx]
            next_race_df = self.race_map[next_rid]
            obs = self._get_obs_for_race(next_race_df)
            self.current_obs = obs
        else:
            obs = self.current_obs
        return obs, float(reward), terminated, truncated, {}


def evaluate_model(env: MultiRaceEnv, model):
    original_ids = env.race_ids
    cost_sum = 0.0
    profit_sum = 0.0
    results = []
    for rid in original_ids:
        subdf = env.race_map[rid].sort_values(env.horse_col).reset_index(drop=True)
        obs = env._get_obs_for_race(subdf)
        action, _ = model.predict(obs, deterministic=True)
        n_horses = len(subdf)
        if action < n_horses:
            row = subdf.iloc[action]
            odds = row[env.single_odds_col]
            finishing = row[env.finishing_col]
            if finishing == 1:
                profit_sum += odds * 100 - env.cost
            else:
                profit_sum -= env.cost
        else:
            profit_sum -= env.cost
        cost_sum += env.cost
        for i in range(len(subdf)):
            row_i = subdf.iloc[i]
            selected_flag = (i == action and action < n_horses)
            results.append({
                "race_id": rid,
                "馬番": row_i[env.horse_col],
                "馬名": row_i[env.horse_name_col],
                "着順": row_i[env.finishing_col],
                "単勝": row_i[env.single_odds_col],
                "selected_flag": selected_flag
            })
    roi = (profit_sum / cost_sum * 100) if cost_sum > 0 else 0.0
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
        fig, axs = plt.subplots(3, 4, figsize=(18, 10))
        logs = self.iteration_logs
        idx_map = [
            ("timesteps", 0, 0), ("fps", 0, 1), ("approx_kl", 0, 2), ("clip_fraction", 0, 3),
            ("entropy_loss", 1, 0), ("explained_variance", 1, 1), ("learning_rate", 1, 2), ("loss", 1, 3),
            ("policy_gradient_loss", 2, 0), ("value_loss", 2, 1)
        ]
        for key, row, col in idx_map:
            axs[row, col].plot(logs["iteration"], logs[key], marker='o', label=key)
            axs[row, col].set_xlabel("iteration")
            axs[row, col].set_ylabel(key)
            axs[row, col].legend()
        axs[2, 2].axis("off")
        axs[2, 3].axis("off")
        plt.tight_layout()
        plt.show()


def run_training_and_inference(
    data_path=DATA_PATH,
    id_col='race_id',
    horse_col='馬番',
    horse_name_col='馬名',
    single_odds_col='単勝',
    finishing_col='着順',
    cost=100,
    total_timesteps=200000,
    races_per_episode=128
):
    train_df, valid_df, test_df, _, _, _, _, dim = prepare_data(
        data_path=data_path,
        id_col=id_col,
        test_ratio=0.1,
        valid_ratio=0.1,
        pca_dim_horse=50,
        pca_dim_jockey=50,
        cat_cols=[]
    )
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
    vec_train_env = DummyVecEnv([lambda: train_env])
    stats_callback = StatsCallback()
    model = PPO(
        "MlpPolicy",
        env=vec_train_env,
        verbose=1,
        batch_size=256,
        n_steps=2048
    )
    model.learn(total_timesteps=total_timesteps, callback=stats_callback)

    train_roi, _ = evaluate_model(train_env, model)
    print(f"Train ROI: {train_roi:.2f}%")
    valid_roi, _ = evaluate_model(valid_env, model)
    print(f"Valid ROI: {valid_roi:.2f}%")

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
    test_df_out.to_csv(SAVE_PATH_PRED, index=False, encoding='utf_8_sig')

    stats_callback.plot_logs()


if __name__ == "__main__":
    run_training_and_inference(
        data_path=DATA_PATH,
        id_col='race_id',
        horse_col='馬番',
        horse_name_col='馬名',
        single_odds_col='単勝',
        finishing_col='着順',
        cost=100,
        total_timesteps=200000,
        races_per_episode=128
    )
