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
RESET_THRESHOLD = 500 # test時に所持金をリセットする値

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

    # 重複行削除
    df = df.drop_duplicates(subset=["race_id", "馬番"])
    # ソート
    df = df.sort_values(['date', 'race_id', '馬番']).reset_index(drop=True)

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

    # 馬の頭数が18を超えるレースを削除
    horse_count_df = df.groupby(id_col)["馬番"].nunique().reset_index()
    horse_count_df = horse_count_df.rename(columns={"馬番": "horse_count"})
    valid_race_ids = horse_count_df.loc[horse_count_df["horse_count"] <= 18, id_col]
    df = df[df[id_col].isin(valid_race_ids)]


    # 数値列 / カテゴリ列
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    # 馬名を退避
    df["馬名_raw"] = df["馬名"].astype(str)

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

class MultiRaceEnvContinuous(gym.Env):
    """
    オフポリシーを前提とした連続アクション空間対応の強化学習環境サンプル。
    今後は単勝・複勝など複数の賭け方を扱う拡張が想定されるため、
    行動空間を連続値として設計（bet率や馬ごとの配分を連続値で指定）にしている。
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
        initial_capital=INITIAL_CAPITAL,
        max_total_bet_cost=MAX_TOTAL_BET_COST
    ):
        """
        - df: レースデータ
        - feature_dim: 特徴量次元
        - cost: 一単位ベットのコスト
        - max_total_bet_cost: 1レースに賭けられる最大総額
        今後、複勝用の払い戻し計算を拡張しやすいように馬ごとの情報を整理しやすい形に構成する。
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
        self.initial_capital = initial_capital
        self.max_total_bet_cost = max_total_bet_cost

        self.race_ids = df[id_col].unique().tolist()
        self.race_map = {}
        self.max_horses = 0
        for rid in self.race_ids:
            subdf = df[df[id_col] == rid].copy().sort_values(self.horse_col)
            self.max_horses = max(self.max_horses, len(subdf))
            self.race_map[rid] = subdf
        print(f'最大頭数：{self.max_horses}')
        
        # 観測空間: max_horses頭 × feature_dim + 所持金1次元
        obs_dim = self.max_horses * self.feature_dim + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        print(f'観測次元：{obs_dim}')

        # 行動空間:
        # 今後複勝にも対応予定 -> "各馬に何％賭けるか" を連続で出力する設計にすると拡張しやすい
        # 例: 各馬のactionは[0.0 ~ 1.0]で、総和<=1.0に正規化する運用など
        # ここでは簡単に [-1,1] を馬ごとに出す形にしておき、後ほど正規化して賭け金を計算
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.max_horses,),
            dtype=np.float32
        )

        self.sampled_races = []
        self.current_race_idx = 0
        self.terminated = False
        self.capital = self.initial_capital
        self.current_obs = None

    def _get_obs_for_race(self, race_df: pd.DataFrame):
        """
        レース内の馬の特徴量をflattenし、所持金を追加。
        """
        n_horses = len(race_df)
        feats = []
        for i in range(n_horses):
            feats.append(race_df.iloc[i]["X"])
        feats = np.array(feats, dtype=np.float32)

        if n_horses < self.max_horses:
            pad_len = self.max_horses - n_horses
            pad = np.zeros((pad_len, self.feature_dim), dtype=np.float32)
            feats = np.vstack([feats, pad])

        feats = feats.flatten()
        feats_with_capital = np.concatenate([feats, [self.capital]], axis=0)
        return feats_with_capital

    def _select_races_for_episode(self):
        """
        1エピソードで使用するレースをランダムにシャッフル。
        今後、複勝ベースのタスクと併用する場合にも同様のロジックを流用可能。
        """
        self.sampled_races = random.sample(self.race_ids, k=self.races_per_episode)
        random.shuffle(self.sampled_races)

    def reset(self, seed=None, options=None):
        """
        エピソード開始時に呼ばれる。
        各種状態を初期化し、最初のレースを返す。
        """
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
        """
        連続アクションで受け取ったベット配分を基に、
        単勝の払い戻しを計算して次状態に進む。
        今後の複勝計算拡張時に、ここで複勝払い戻しロジックを追加してもよい。
        """
        if self.terminated:
            return self.current_obs, 0.0, True, False, {}

        rid = self.sampled_races[self.current_race_idx]
        race_df = self.race_map[rid]
        n_horses = len(race_df)

        # 行動を [0,1] に正規化 -> ベット配分率に変換
        ## 1) アクションを [0, 1] にクリップ
        clipped_action = np.clip(action, 0.0, 1.0)

        # 2) bet_amountが大きい上位3頭を選択
        #   -> ソートして最後の3つを上位とみなす (3頭未満なら全頭を賭け対象)
        top_k = min(3, n_horses)  
        idx_top = np.argsort(clipped_action)[-top_k:]

        # 3) 選択された上位馬の合計アクション値でベット配分を正規化
        selected_action = clipped_action[idx_top]
        sum_action = np.sum(clipped_action)
        bet_ratio = np.zeros_like(clipped_action)
        if sum_action > 0:
            bet_ratio[idx_top] = selected_action / sum_action

        # 4) 賭け可能総額を計算 (所持金と上限の小さい方)
        total_bet = min(self.max_total_bet_cost, self.capital)
        bet_amounts = total_bet * bet_ratio

        #   cost の整数倍に丸める
        bet_amounts = np.floor(bet_amounts / self.cost) * self.cost
        race_cost = np.sum(bet_amounts)
        self.capital -= race_cost

        # 5) 単勝での払戻額を計算
        race_profit = 0.0
        for i in range(self.max_horses):
            if i < n_horses:
                row = race_df.iloc[i]
                if row[self.finishing_col] == 1:
                    race_profit += bet_amounts[i] * row[self.single_odds_col]

        # 複勝にも対応するにはここで「もし複勝対応なら...」とブロックを追加予定

        # 6) 所持金を更新
        self.capital += race_profit
        
        # 7) 報酬: 「(race_profit - race_cost / race_cost)」を対数スケーリングして過剰フィットを緩和
        #   (ベット額0のときは0報酬に)
        if race_cost > 0:
            ratio = (race_profit - race_cost) / race_cost
            ratio_clamped = max(ratio, -0.99999)  # -1よりは少し上(-0.99など)を下限に設定
            ratio_clamped = min(ratio_clamped, 10)  # 最大値を20に設定
            # 例: log(1 + ratio) で極端な大勝ちへの感度を下げる
            reward = np.log1p(ratio_clamped)
        else:
            reward = 0.0
    
        # 追加: race_costが100未満のときにペナルティ
        if race_cost < 100:
            reward -= 1

        self.current_race_idx += 1
        terminated = (self.current_race_idx >= self.races_per_episode)
        if self.capital <= 500:
            reward -= 100
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
        capital_reset_threshold=RESET_THRESHOLD,
        capital_reset_value=INITIAL_CAPITAL):
    """
    学習後のモデルを全レースに適用してROIを計算。
    もし所持金が一定額を下回った場合、所持金を指定額にリセットする。
    
    - capital_reset_threshold: リセットを発動する閾値
    - capital_reset_value: リセット後の所持金額
    """
    # 評価用の一時的な所持金
    capital = env.initial_capital

    cost_sum = 0.0
    profit_sum = 0.0
    results = []

    # 元の環境が保持しているレースIDリストを利用
    for rid in tqdm(env.race_ids):
        # レースのデータ取得
        subdf = env.race_map[rid].sort_values(env.horse_col).reset_index(drop=True)
        # 環境の観測作成ロジックを流用
        obs = env._get_obs_for_race(subdf)

        # SACは連続アクションを出力
        action, _ = model.predict(obs, deterministic=True)
        clipped_action = np.clip(action, 0.0, 1.0)

        n_horses = len(subdf)
        # 学習時と同様: アクション上位3頭だけ選択して正規化
        top_k = min(3, n_horses)
        idx_top = np.argsort(clipped_action)[-top_k:]
        selected_action = clipped_action[idx_top]
        sum_action = np.sum(clipped_action)

        bet_ratio = np.zeros_like(clipped_action)
        if sum_action > 0:
            bet_ratio[idx_top] = selected_action / sum_action

        # 賭けられる総額
        total_bet = min(env.max_total_bet_cost, capital)
        bet_amounts = total_bet * bet_ratio
        # コスト単位で丸める
        bet_amounts = np.floor(bet_amounts / env.cost) * env.cost

        race_cost = np.sum(bet_amounts)
        race_profit = 0.0

        # 着順1位に賭けていたら払戻しを取得
        for i in range(env.max_horses):
            if i < n_horses:
                row = subdf.iloc[i]
                finishing = row[env.finishing_col]
                odds = row[env.single_odds_col]
                horse_name = row[env.horse_name_col + "_raw"]
                # 単勝が当たった場合
                if finishing == 1:
                    race_profit += bet_amounts[i] * odds

                # 結果を保存
                results.append({
                    "race_id": rid,
                    "馬番": row[env.horse_col],
                    "馬名": horse_name,
                    "着順": finishing,
                    "単勝": odds,
                    "bet_amount": bet_amounts[i]
                })

        # 賭け金を差し引き、払戻しを上乗せ
        capital -= race_cost
        capital += race_profit

        # 所持金が閾値より下がったらリセット
        if capital < capital_reset_threshold:
            capital = capital_reset_value

        cost_sum += race_cost
        profit_sum += race_profit

    # 最終的なROI
    roi = (profit_sum / cost_sum) if cost_sum > 0 else 0.0
    return roi, pd.DataFrame(results)

class OffPolicyStatsCallback(BaseCallback):
    """
    SACやTD3の学習中の統計を収集するコールバック。
    PPO時のStatsCallbackを流用。
    """
    def __init__(self):
        super().__init__(verbose=0)
        self.iteration_logs = {
            "iteration": [],
            "timesteps": [],
            "time_elapsed": [],
            "train_reward": [],
            # 追加: actor_loss / critic_loss を格納するリスト
            "actor_loss": [],
            "critic_loss": []
        }
        self.iter_count = 0

    def _on_step(self) -> bool:
        return True

    def _on_training_start(self):
        pass

    def _on_rollout_end(self):
        """
        SAC/TD3ではrolloutが終わるたびに呼ばれるので、このタイミングでロスを拾う。
        """
        self.iter_count += 1
        self.iteration_logs["iteration"].append(self.iter_count)
        self.iteration_logs["timesteps"].append(self.model.num_timesteps)

        # SB3 の logger に格納された値を取得:
        self.iteration_logs["time_elapsed"].append(
            self.model.logger.name_to_value.get("time/total_timesteps", 0)
        )
        # actor_loss / critic_loss を取得 (ない場合は None が返るので np.nan に変換)
        actor_loss = self.model.logger.name_to_value.get("train/actor_loss")
        critic_loss = self.model.logger.name_to_value.get("train/critic_loss")
        self.iteration_logs["actor_loss"].append(actor_loss if actor_loss is not None else np.nan)
        self.iteration_logs["critic_loss"].append(critic_loss if critic_loss is not None else np.nan)

        # もし他に報酬情報などを追う場合は合わせて記録する

    def plot_logs(self):
        """
        ログを可視化。
        actor_loss / critic_loss を追加で可視化するため、2行×2列に変更。
        """
        logs = self.iteration_logs

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # 1段目左: timesteps
        axs[0, 0].plot(logs["iteration"], logs["timesteps"], marker='o', label="timesteps")
        axs[0, 0].set_xlabel("iteration")
        axs[0, 0].set_ylabel("timesteps")
        axs[0, 0].legend()

        # 1段目右: time/total_timesteps
        axs[0, 1].plot(logs["iteration"], logs["time_elapsed"], marker='o', label="time/total_timesteps")
        axs[0, 1].set_xlabel("iteration")
        axs[0, 1].set_ylabel("time/total_timesteps")
        axs[0, 1].legend()

        # 2段目左: actor_loss
        axs[1, 0].plot(logs["iteration"], logs["actor_loss"], marker='o', color='red', label="actor_loss")
        axs[1, 0].set_xlabel("iteration")
        axs[1, 0].set_ylabel("actor_loss")
        axs[1, 0].legend()

        # 2段目右: critic_loss
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
    finishing_col='着順',
    cost=COST,
    total_timesteps=TOTAL_TIMESTEPS,
    races_per_episode=RACES_PER_EPISODE,
    seed_value=42
):
    """
    SACを使ったオフポリシー学習の実行関数。
    今後、単勝・複勝の拡張を念頭において、同様の構成を複勝に適用できるようコメントを充実させる。
    """
    set_random_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)

    train_df, test_df, dim = prepare_data(
        data_path=data_path,
        feature_path=feature_path,
        id_col="race_id",
        test_ratio=TEST_RATIO,
        single_odds_col="単勝",
        place_odds_col="複勝",   # ★ 複勝カラム
        finishing_col="着順",
        pca_dim_horse=PCA_DIM_HORSE,
        pca_dim_jockey=PCA_DIM_JOCKEY
    )

    train_env = MultiRaceEnvContinuous(
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

    vec_train_env = DummyVecEnv([lambda: train_env])

    stats_callback = OffPolicyStatsCallback()

    # SACアルゴリズム（オフポリシー）を使用
    # 今後複勝を組み込む場合でも、このアルゴリズム設定はほぼ共通で流用可能
    model = SAC(
        "MlpPolicy",
        env=vec_train_env,
        verbose=1,
        **SAC_HYPERPARAMS
    )

    model.learn(total_timesteps=total_timesteps, callback=stats_callback)

    # 学習データでのROI
    train_roi, train_df_out = evaluate_model(train_env, model)
    print(f"Train ROI: {train_roi*100:.2f}%")

    # テストデータでのROI
    test_env = MultiRaceEnvContinuous(
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
    print(f"Test ROI: {test_roi*100:.2f}%")

    test_df_out.to_csv(SAVE_PATH_PRED, index=False, encoding='utf_8_sig')

    stats_callback.plot_logs()

    # 回収率と掛け金の可視化
    # まず、train用・test用の結果DataFrameそれぞれに払戻と個別ROI列を追加します。
    # 例として、すでに evaluate_model(train_env, model) で得られた train_df_out、
    # evaluate_model(test_env, model) で得られた test_df_out があるものとします。
    # ここでは train_df_out / test_df_out に "bet_amount", "着順", "単勝" カラムが含まれている想定です。

    # 払戻とROI(1レース1馬券単位)を列追加
    for df_out in [train_df_out, test_df_out]:
        df_out["payout"] = df_out.apply(
            lambda row: row["bet_amount"] * row["単勝"] if row["着順"] == 1 and row["bet_amount"] > 0 else 0,
            axis=1
        )
        df_out["roi"] = df_out.apply(
            lambda row: (row["payout"] / row["bet_amount"]) if row["bet_amount"] > 0 else 0,
            axis=1
        )

    # race_id単位で集計: 賭け金合計、払戻合計、ROI(レース単位)
    train_race_agg = train_df_out.groupby("race_id").agg(
        race_bet_amount=("bet_amount", "sum"),
        race_payout=("payout", "sum")
    ).reset_index()
    train_race_agg["race_roi"] = train_race_agg["race_payout"] / (train_race_agg["race_bet_amount"] + 1e-15)

    test_race_agg = test_df_out.groupby("race_id").agg(
        race_bet_amount=("bet_amount", "sum"),
        race_payout=("payout", "sum")
    ).reset_index()
    test_race_agg["race_roi"] = test_race_agg["race_payout"] / (test_race_agg["race_bet_amount"]+ 1e-15)

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
    plt.hist(train_race_agg["race_roi"], bins=500, edgecolor="black")
    plt.title("【Train】レース単位の回収率ヒストグラム")
    plt.xlabel("race_roi (払い戻し合計 / 賭け金合計)")
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
    plt.hist(test_race_agg["race_roi"], bins=500, edgecolor="black")
    plt.title("【Test】レース単位の回収率ヒストグラム")
    plt.xlabel("race_roi (払い戻し合計 / 賭け金合計)")
    plt.ylabel("レース件数")
    plt.tight_layout()
    plt.show()

    # 掛け金が100円以上の行を抽出し、train_df_out / test_df_out それぞれで「単勝」列の分布を可視化します。
    # (1) まず条件に合う行のみ抽出
    train_df_out_100 = train_df_out[train_df_out["bet_amount"] >= 100].copy()
    test_df_out_100 = test_df_out[test_df_out["bet_amount"] >= 100].copy()

    # (2) 単勝オッズの分布をヒストグラムで可視化 (Train)
    plt.figure(figsize=(10, 5))
    plt.hist(train_df_out_100["単勝"], bins=50, edgecolor="black")
    plt.title("【Train】掛け金>=100円の単勝オッズ分布")
    plt.xlabel("単勝オッズ")
    plt.ylabel("件数")
    plt.tight_layout()
    plt.show()

    # (3) 単勝オッズの分布をヒストグラムで可視化 (Test)
    plt.figure(figsize=(10, 5))
    plt.hist(test_df_out_100["単勝"], bins=50, edgecolor="black")
    plt.title("【Test】掛け金>=100円の単勝オッズ分布")
    plt.xlabel("単勝オッズ")
    plt.ylabel("件数")
    plt.tight_layout()
    plt.show()

    # モデル保存
    # stable-baselines3 の標準的な保存形式(model.zip)を使う方が安全です。
    model_save_path = os.path.join(MODEL_SAVE_DIR, "model.zip")
    model.save(model_save_path)

if __name__ == "__main__":
    run_training_and_inference_offpolicy()
