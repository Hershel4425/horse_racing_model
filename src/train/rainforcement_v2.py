import os
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
from stable_baselines3.common.utils import set_random_seed

import matplotlib.pyplot as plt

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
DATA_PATH = os.path.join(ROOT_PATH, "result/predictions/transformer/20250106194932.csv")
FEATURE_PATH = os.path.join(ROOT_PATH, "data/02_features/feature.csv")  # 特徴量CSVのパス


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
    feature_path,
    test_ratio=0.1,
    valid_ratio=0.1,
):
    """
    データの読み込み、リークを起こす可能性のある列の削除、数値列の欠損補完、
    PCAによる次元圧縮、標準化などを行い、学習に適した形へ整形する。
    Why:
      - リークの可能性のある列を削除しないと、学習の過程で実際には使えない未来情報を使ってしまうリスクがある
      - PCAによる次元圧縮で特徴量をコンパクトにまとめ、高次元のデータでも学習を安定させやすくする
      - 標準化を行うことで、モデルが扱いやすいスケールにデータを合わせる（大きい値の特徴量が学習を支配しないようにする）
    """
    # CSVデータを読み込む
    df1 = pd.read_csv(feature_path, encoding='utf-8-sig') # race_id, 馬番, 馬名, 単勝, 着順, 各種特徴量
    df2 = pd.read_csv(data_path, encoding='utf-8-sig') # race_id, 馬番, transformerを用いた予測値

    # マージ
    df = pd.merge(
        df1,
        df2[["race_id", "馬番", "P_top1", "P_top3", "P_top5", "P_pop1", "P_pop3", "P_pop5"]],
        on=["race_id", "馬番"],
        how="inner"
    )

    # 無駄な列や直接リークになる列を削除するならここで対応
    # 今回は最低限の列だけ残す（馬名, 単勝, 着順, date, 各P_系）
    # 必要に応じて他の列をdrop
    keep_cols = [
        "race_id", "馬番", "馬名", "単勝", "着順", "date",
        "P_top1", "P_top3", "P_top5", "P_pop1", "P_pop3", "P_pop5"
    ]
    df = df[keep_cols].copy()

    # 欠損埋め
    prob_cols = ["P_top1", "P_top3", "P_top5", "P_pop1", "P_pop3", "P_pop5"]
    for c in prob_cols:
        df[c] = df[c].fillna(0.0)

    # train/valid/testに分割
    train_df, valid_df, test_df = split_data(df, id_col="race_id", test_ratio=test_ratio, valid_ratio=valid_ratio)

    # 特徴量は P_top1, P_top3, P_top5, P_pop1, P_pop3, P_pop5 のみ
    # X は 6次元
    def assign_features(dataframe):
        x_list = []
        for idx, row in dataframe.iterrows():
            x_list.append([
                row["P_top1"], row["P_top3"], row["P_top5"],
                row["P_pop1"], row["P_pop3"], row["P_pop5"]
            ])
        return x_list

    
    # それぞれのDataFrameに特徴量ベクトルを格納
    train_df["X"] = assign_features(train_df)
    valid_df["X"] = assign_features(valid_df)
    test_df["X"] = assign_features(test_df)
    
    # 特徴量次元6
    dim = 6
    return train_df, valid_df, test_df, dim

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
        races_per_episode=16,
        initial_capital=5000,  # <-- 追加: 初期所持金
        max_total_bet_cost: float = 1000.0,  # <-- 追加: 1レースあたりのベット上限額
        bet_mode="multi",            # ← 追加: "single" or "multi"
        max_bet_units=5               # ← 追加: 複数馬の場合の最大ベット単位
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
        self.initial_capital = initial_capital
        # 追加パラメータ
        self.bet_mode = bet_mode
        self.max_bet_units = max_bet_units
        self.max_total_bet_cost = max_total_bet_cost

        # race_idをキーに、それぞれのレースのデータを保持する
        self.race_ids = df[id_col].unique().tolist()
        self.race_map = {}

        self.max_horses = 0
        # 全レースのうち最大の出走頭数を探し、観測空間を一律の長さ(最大頭数)に揃える
        # Why: Gymの観測空間は固定長が望ましいため、足りない分はパディングする設計とする
        for rid in self.race_ids:
            subdf = df[df[id_col] == rid].copy().sort_values(self.horse_col)
            self.max_horses = max(self.max_horses, len(subdf))
            self.race_map[rid] = subdf

        # 観測空間を定義。馬最大頭数×特徴量次元
        # Why: 「何頭出走でも強制的に配列を同じ次元にして扱う」設計
        # 馬特徴ベクトル (max_horses * feature_dim) + 所持金 (1次元) = 総合計
        obs_dim = self.max_horses * self.feature_dim + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # 行動空間: 
        # bet_modeに応じて行動空間を切り替える
        if self.bet_mode == "single":
            # 単勝1点 = 「どの馬に賭けるか」だけを離散で選択
            # action は 0〜(max_horses-1)
            self.action_space = spaces.Discrete(self.max_horses)
        # （複数馬・変動掛け金）
        else:
        # なぜ MultiDiscrete にするのか？
        #  -> 各馬番に対して「掛けない(0)〜複数単位賭ける(n)」を同時に表現するため。
        #     例えば [4]*self.max_horses とすれば 0〜3 の4パターンの賭け方が取れる。
            # action は長さ self.max_horses のベクトル
            self.action_space = spaces.MultiDiscrete(
                [self.max_bet_units + 1] * self.max_horses
            )


        # エピソード内でのレース順序をシャッフルするためのリストや状態を初期化
        self.sampled_races = []
        self.current_race_idx = 0
        self.current_obs = None
        self.terminated = False

        # 追加：所持金
        self.capital = self.initial_capital

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
        
        # flatten
        feats = feats.flatten()

        # 所持金を最後に付与 (float32でキャストして連結)
        feats_with_capital = np.concatenate([feats, [self.capital]], axis=0)

        return feats_with_capital

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

        # 所持金を初期化
        self.capital = self.initial_capital

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
        # まず賭け金を算出
        race_cost = 0.0
        race_profit = 0.0

        if self.bet_mode == "single":
            # 単勝1点モードの場合、action は「賭ける馬のindex (0〜max_horses-1)」
            chosen_horse_idx = action
            # コスト (1単位だけ賭ける想定であれば)
            race_cost = self.cost

            # 実際に賭けた分を所持金から差し引く
            self.capital -= race_cost

            if chosen_horse_idx < n_horses:
                row = race_df.iloc[chosen_horse_idx]
                if row[self.finishing_col] == 1:
                    odds = row[self.single_odds_col]
                    race_profit = race_cost * odds
        else:
            # 各馬に対して賭け金を処理
            for i in range(self.max_horses):
                bet_units = action[i]  # i番目の馬への「何単位賭けるか」
                cost_i = bet_units * self.cost  # その馬への実コスト (例: 1単位=100円相当、など)
                # 所持金が足りなければ賭け金を0にするなど
                if self.capital < (race_cost + cost_i):
                    cost_i = 0
                race_cost += cost_i


            # 合計賭け金が上限を超えたらペナルティ etc.
            if race_cost > self.max_total_bet_cost:
                penalty = (race_cost - self.max_total_bet_cost) * 5
                # この時点でペナルティを reward に加えておく (別管理でもOK)
                race_cost += penalty
                # もしくは total_cost を max_total_bet_cost で打ち切るなどの処理

            # 実際に賭けた分を所持金から差し引く
            self.capital -= race_cost

            # その後に払い戻し計算
            # 勝ち馬への払い戻し
            for i in range(self.max_horses):
                bet_units = action[i]
                cost_i = bet_units * self.cost
                # 上記と同様にcapitalチェックや無効化があった場合は cost_iが変わる(管理要注意)
                if i < len(race_df) and cost_i > 0:
                    row = race_df.iloc[i]
                    if row[self.finishing_col] == 1:
                        race_profit += cost_i * row[self.single_odds_col]

        # 払い戻しを所持金に加える
        self.capital += race_profit

        # 報酬: 「今回の純増分」をベースにするなら
        reward = (race_profit - race_cost)

        # 次のレースへ
        self.current_race_idx += 1
        terminated = (self.current_race_idx >= self.races_per_episode)

        # 所持金が尽きたら強制終了
        if self.capital <= 0:
            terminated = True

        self.terminated = terminated
        truncated = False

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

        if env.bet_mode == "single":
            # 単勝1点賭けモード
            chosen_horse_idx = action  # actionは整数一つ
            # コスト
            race_cost = env.cost

            # 出走頭数内なら払い戻し計算
            if chosen_horse_idx < n_horses:
                row = subdf.iloc[chosen_horse_idx]
                finishing = row[env.finishing_col]
                odds = row[env.single_odds_col]
                if finishing == 1:
                    race_profit = race_cost * odds

            # 結果保存
            for i in range(n_horses):
                bet_amount = env.cost if i == chosen_horse_idx else 0
                row_i = subdf.iloc[i]
                results.append({
                    "race_id": rid,
                    "馬番": row_i[env.horse_col],
                    "馬名": row_i[env.horse_name_col],
                    "着順": row_i[env.finishing_col],
                    "単勝": row_i[env.single_odds_col],
                    "bet_amount": bet_amount
                })

        else:
            # 複数馬・複数単位賭けモード
            # actionは長さ self.max_horses のベクトル
            for i in range(env.max_horses):
                bet_units = action[i]
                bet_amount = bet_units * env.cost
                race_cost += bet_amount

                # 出走頭数内だけ払い戻し計算
                if i < n_horses:
                    row_i = subdf.iloc[i]
                    finishing = row_i[env.finishing_col]
                    odds = row_i[env.single_odds_col]
                    if finishing == 1:
                        race_profit += bet_amount * odds

                # 結果保存
                if i < n_horses:
                    row_i = subdf.iloc[i]
                    results.append({
                        "race_id": rid,
                        "馬番": row_i[env.horse_col],
                        "馬名": row_i[env.horse_name_col],
                        "着順": row_i[env.finishing_col],
                        "単勝": row_i[env.single_odds_col],
                        "bet_amount": bet_amount
                    })

        cost_sum += race_cost
        profit_sum += race_profit
    
    # ROI = (払い戻し合計 / コスト合計)
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
    feature_path = FEATURE_PATH,
    id_col='race_id',
    horse_col='馬番',
    horse_name_col='馬名',
    single_odds_col='単勝',
    finishing_col='着順',
    cost=100,
    total_timesteps=200000,
    races_per_episode=32,
    seed_value=42,
    bet_mode = "multi",
    max_bet_units=5    
):
    """
    学習データの準備からモデルの学習、評価、予測結果の保存、ログの可視化までを一括で実行する関数。
    Why:
      - スクリプトを1回実行するだけで全ステップを行えるように設計し、再現性・再利用性を高める
    """
    # seed固定
    set_random_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)


    # データを読み込み＆前処理
    train_df, valid_df, test_df, dim = prepare_data(
        data_path=data_path,
        feature_path=feature_path,
        test_ratio=0.1,
        valid_ratio=0.1,
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
        races_per_episode=races_per_episode,
        bet_mode=bet_mode,
        max_bet_units = max_bet_units
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
        bet_mode=bet_mode,
        max_bet_units=max_bet_units
    )

    # VecEnvに変換（PPOなど多くのRLアルゴリズムは並列環境を前提にしているため）
    vec_train_env = DummyVecEnv([lambda: train_env])

    # 学習ログの可視化などに使うコールバックを準備
    stats_callback = StatsCallback()

    # PPOモデルを初期化。なぜPPOか？
    # -> シンプルで安定した強化学習アルゴリズムであり、ハイパーパラメータをある程度簡単に調整できる
    ppo_hyperparams = {
        "learning_rate": 1e-4,
        "n_steps": 2048,
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
    print(f"Train ROI: {train_roi*100:.2f}%")

    # バリデーションデータでのROI確認
    valid_roi, _ = evaluate_model(valid_env, model)
    print(f"Valid ROI: {valid_roi*100:.2f}%")

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
        races_per_episode=races_per_episode,
        bet_mode=bet_mode,
        max_bet_units=max_bet_units
    )
    test_roi, test_df_out = evaluate_model(test_env, model)
    print(f"Test ROI: {test_roi*100:.2f}%")

    # 推論結果の保存
    test_df_out.to_csv(SAVE_PATH_PRED, index=False, encoding='utf_8_sig')

    # 学習ログを可視化
    stats_callback.plot_logs()


if __name__ == "__main__":
    # スクリプトを直接実行したときにトレーニングから推論までを実行
    run_training_and_inference()
