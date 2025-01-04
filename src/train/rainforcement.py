import os
import re
import datetime

import numpy as np
import pandas as pd

import gym
from gym import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------
# ファイルパスやディレクトリ、設定の定義
#------------------------------------------------------------------------------------
ROOT_PATH = "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/153_競馬_v3"
DATE_STRING = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

MODEL_SAVE_DIR = os.path.join(ROOT_PATH, f"models/transormer予測モデル/{DATE_STRING}")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

# predictionを保存するパス
SAVE_PATH_PRED = os.path.join(ROOT_PATH, f"result/predictions/{DATE_STRING}.csv")
# 保存先のディレクトリがなければ作成（os.path.existsだとファイルorディレクトリのどちらかだけのチェックになるので修正）
pred_dir = os.path.dirname(SAVE_PATH_PRED)
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)

# データ読み込み先
DATA_PATH = os.path.join(ROOT_PATH, "data/02_features/feature.csv")

#------------------------------------------------------------------------------------
# データ分割用関数
#------------------------------------------------------------------------------------
def split_data(df, id_col="race_id", test_ratio=0.1, valid_ratio=0.1):
    """
    日付順にソートしてから、レースIDのリストをシャッフルなしで分割。
    デフォルトはtrain:0.8, valid:0.1, test:0.1のイメージ。
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

#------------------------------------------------------------------------------------
# PCAやスケーリングを行い、特徴量を作成するための関数
#------------------------------------------------------------------------------------
def prepare_data(
    data_path,
    id_col='race_id',
    test_ratio=0.1,
    valid_ratio=0.1,
    pca_dim_horse=50,
    pca_dim_jockey=50,
    cat_cols=None
):
    """
    データを読み込み、PCAやスケーリングなどの前処理を実施。
    この関数が出力するtrain_df, valid_df, test_dfには「X」列に特徴量が格納される。
    """
    if cat_cols is None:
        cat_cols = []

    # CSV読み込み
    df = pd.read_csv(data_path, encoding="utf_8_sig")

    # 不要列をリストアップ（リークを防ぐための削除候補）
    default_leakage_cols = [
            '着順', '斤量', 'タイム', '着差',
            '単勝',
            '上がり3F','馬体重','人気',
            'horse_id','jockey_id',
            'trainer_id',
            '順位点','入線','1着タイム差','先位タイム差','5着着差','増減',
            '1C通過順位','2C通過順位','3C通過順位','4C通過順位','賞金','前半ペース','後半ペース','ペース',
            '上がり3F順位','100m','200m','300m','400m','500m','600m','700m','800m','900m','1000m',
            '1100m','1200m','1300m','1400m','1500m','1600m','1700m','1800m','1900m','2000m',
            '2100m','2200m','2300m','2400m','2500m','2600m','2700m','2800m','2900m','3000m',
            '3100m','3200m','3300m','3400m','3500m','3600m','horse_ability'
        ]
    drop_candidates = set(default_leakage_cols)

    # 数値列・カテゴリ列を分ける
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in cat_cols if c in df.columns]

    # 数値列の欠損埋めを0で実施
    for c in num_cols:
        df[c] = df[c].fillna(0)

    # 学習/検証/テストに分割
    train_df, valid_df, test_df = split_data(
        df, id_col=id_col,
        test_ratio=test_ratio, valid_ratio=valid_ratio
    )

    # Horse / Jockey特徴量に対してPCAを実施したいのでパターン定義
    pca_pattern_horse = r'^(競走馬芝|競走馬ダート|単年競走馬芝|単年競走馬ダート)'
    pca_pattern_jockey = r'^(騎手芝|騎手ダート|単年騎手芝|単年騎手ダート)'

    pca_horse_target_cols = [c for c in num_cols if re.match(pca_pattern_horse, c)]
    pca_jockey_target_cols = [c for c in num_cols if re.match(pca_pattern_jockey, c)]
    other_num_cols = [
        c for c in num_cols 
        if (c not in pca_horse_target_cols) 
        and (c not in pca_jockey_target_cols) 
        and (c not in drop_candidates)
    ]

    # Horseに対するPCA
    scaler_horse = StandardScaler()
    horse_features_train_scaled = scaler_horse.fit_transform(train_df[pca_horse_target_cols].values)
    horse_features_valid_scaled = scaler_horse.transform(valid_df[pca_horse_target_cols].values)
    horse_features_test_scaled = scaler_horse.transform(test_df[pca_horse_target_cols].values)

    pca_dim_horse = min(pca_dim_horse, horse_features_train_scaled.shape[1])
    pca_model_horse = PCA(n_components=pca_dim_horse)
    horse_features_train_pca = pca_model_horse.fit_transform(horse_features_train_scaled)
    horse_features_valid_pca = pca_model_horse.transform(horse_features_valid_scaled)
    horse_features_test_pca = pca_model_horse.transform(horse_features_test_scaled)

    # Jockeyに対するPCA
    scaler_jockey = StandardScaler()
    jockey_features_train_scaled = scaler_jockey.fit_transform(train_df[pca_jockey_target_cols].values)
    jockey_features_valid_scaled = scaler_jockey.transform(valid_df[pca_jockey_target_cols].values)
    jockey_features_test_scaled = scaler_jockey.transform(test_df[pca_jockey_target_cols].values)

    pca_dim_jockey = min(pca_dim_jockey, jockey_features_train_scaled.shape[1])
    pca_model_jockey = PCA(n_components=pca_dim_jockey)
    jockey_features_train_pca = pca_model_jockey.fit_transform(jockey_features_train_scaled)
    jockey_features_valid_pca = pca_model_jockey.transform(jockey_features_valid_scaled)
    jockey_features_test_pca = pca_model_jockey.transform(jockey_features_test_scaled)

    # その他数値列
    scaler_other = StandardScaler()
    other_features_train = scaler_other.fit_transform(train_df[other_num_cols].values)
    other_features_valid = scaler_other.transform(valid_df[other_num_cols].values)
    other_features_test = scaler_other.transform(test_df[other_num_cols].values)

    # カテゴリ列(今回は単純にone-hotなどせずに、cat_featuresのままstack)
    cat_features_train = train_df[cat_cols].values if cat_cols else np.array([]).reshape(len(train_df), 0)
    cat_features_valid = valid_df[cat_cols].values if cat_cols else np.array([]).reshape(len(valid_df), 0)
    cat_features_test = test_df[cat_cols].values if cat_cols else np.array([]).reshape(len(test_df), 0)

    # 上記特徴量を結合してXに格納
    X_train = np.concatenate([
        cat_features_train,
        other_features_train,
        horse_features_train_pca,
        jockey_features_train_pca
    ], axis=1)
    X_valid = np.concatenate([
        cat_features_valid,
        other_features_valid,
        horse_features_valid_pca,
        jockey_features_valid_pca
    ], axis=1)
    X_test = np.concatenate([
        cat_features_test,
        other_features_test,
        horse_features_test_pca,
        jockey_features_test_pca
    ], axis=1)

    actual_num_dim = other_features_train.shape[1] + pca_dim_horse + pca_dim_jockey

    # 各データフレームにX列を追加（各サンプルが1次元np.arrayになるように保持）
    train_df["X"] = list(X_train)
    valid_df["X"] = list(X_valid)
    test_df["X"] = list(X_test)

    return train_df, valid_df, test_df, (scaler_horse, pca_model_horse), (scaler_jockey, pca_model_jockey), scaler_other, cat_cols, actual_num_dim

#------------------------------------------------------------------------------------
# PPOで使用する環境クラス
#------------------------------------------------------------------------------------
class SingleWinBetEnvVariableHorses(gym.Env):
    """
    単勝馬券を1レースにつき1頭だけ買うというシミュレーション環境。
    各レースの馬ごとの特徴量をまとめてflattenしたものをobsとし、
    アクションは「何番目の馬を買うか」をDiscretで指定する。
    max_horsesまでパディングしているので、action_spaceはmax_horses。
    """
    def __init__(
        self, 
        df: pd.DataFrame,
        feature_cols: list,
        id_col="race_id",
        horse_col="馬番",
        horse_name_col="馬名",
        single_odds_col="単勝",
        finishing_col="着順", 
        cost=100
    ):
        super().__init__()
        
        # 受け取ったパラメータを保存
        self.df = df
        self.id_col = id_col
        self.horse_col = horse_col
        self.horse_name_col = horse_name_col
        self.single_odds_col = single_odds_col
        self.finishing_col = finishing_col
        self.feature_cols = feature_cols
        self.cost = cost

        # 全レースIDを取得
        self.race_ids = df[id_col].unique().tolist()
        self.race_dfs = []
        
        # 最大頭数を調べる(環境のobservation_spaceを固定するため)
        self.max_horses = 0
        for rid in self.race_ids:
            sub = df[df[id_col] == rid].copy()
            n_horses = len(sub)
            self.max_horses = max(self.max_horses, n_horses)
            self.race_dfs.append(sub)
        
        # 特徴量次元
        self.feat_dim = len(self.feature_cols)
        
        # observation_space( [最大頭数×特徴量] のflatten )
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(self.max_horses * self.feat_dim,),
            dtype=np.float32
        )
        
        # action_space(馬を一頭選択:最大頭数の離散)
        self.action_space = spaces.Discrete(self.max_horses)
        
        self.current_race_idx = 0
        self.done = False
        
        # 最初のレースをセットアップ
        self._setup_next_race()

    def _setup_next_race(self):
        """
        次のレースの状態を環境にセットする。最大頭数に満たない場合は0埋めして次元を合わせる。
        """
        if self.current_race_idx >= len(self.race_dfs):
            self.done = True
            self.current_obs = np.zeros((self.max_horses * self.feat_dim,), dtype=np.float32)
            self.current_subdf = None
            return
        
        sub = self.race_dfs[self.current_race_idx].copy()
        n_horses = len(sub)
        
        # 馬番でソート(アクションと馬番を紐づけるため)
        sub = sub.sort_values(self.horse_col)
        
        # 各馬の特徴量をスタック
        sub_feats_list = []
        for i in range(len(sub)):
            # ここでDataFrameのX列(1つ1つがnp.array)を取り出してflatten
            sub_feats_list.append(sub.iloc[i]["X"])
        sub_feats = np.array(sub_feats_list, dtype=np.float32)
        
        # max_horsesに満たない場合はパディング
        if n_horses < self.max_horses:
            pad_len = self.max_horses - n_horses
            pad_array = np.zeros((pad_len, self.feat_dim), dtype=np.float32)
            sub_feats = np.vstack([sub_feats, pad_array])
        
        # Flattenしてobservationにする
        self.current_obs = sub_feats.flatten()
        self.current_subdf = sub.reset_index(drop=True)
        self.done = False

    def reset(self):
        """
        レースの最初の状態に戻す。
        """
        if self.current_race_idx >= len(self.race_dfs):
            self.current_race_idx = 0
        self._setup_next_race()
        return self.current_obs

    def step(self, action):
        """
        馬を選択(action)し、その的中払い戻しをrewardとして返す。
        レースが1回終了したらdone=Trueにして次レースへ。
        """
        if self.done or self.current_subdf is None:
            return self.current_obs, 0.0, True, {}
        
        sub = self.current_subdf
        n_horses = len(sub)
        
        if action < n_horses:
            # 選択した馬が勝ったかどうか(着順が1なら勝利馬)
            row = sub.iloc[action]
            if row[self.finishing_col] == 1:
                reward = row[self.single_odds_col]  # 払戻金
            else:
                reward = 0.0
        else:
            # アクションが馬数より大きい場合は無効とみなし報酬0
            reward = 0.0
        
        self.done = True
        self.current_race_idx += 1
        
        return self.current_obs, reward, True, {}

#------------------------------------------------------------------------------------
# モデルの評価用関数
#------------------------------------------------------------------------------------
def evaluate_model(env: SingleWinBetEnvVariableHorses, model, cost=100):
    """
    学習済みmodelを使ってenvを順番に進めながら、ROI(回収率)を計算する。
    """
    results = []
    total_reward = 0.0
    total_cost = 0.0
    
    env.current_race_idx = 0
    obs = env.reset()
    
    while True:
        # 現在のobsからモデルがactionを予測
        action, _states = model.predict(obs, deterministic=True)
        next_obs, reward, done, info = env.step(action)
        
        sub = env.current_subdf
        if sub is not None and len(sub) > 0:
            race_id_value = sub.iloc[0][env.id_col]
            if action < len(sub):
                row_action = sub.iloc[action]
                bet_horse_num = row_action[env.horse_col]
                bet_horse_name = row_action[env.horse_name_col]
                bet_finishing = row_action[env.finishing_col]
                bet_single_odds = row_action[env.single_odds_col]
                this_cost = cost
            else:
                race_id_value = sub.iloc[0][env.id_col]
                bet_horse_num = -1
                bet_horse_name = "INVALID"
                bet_finishing = -1
                bet_single_odds = 0.0
                this_cost = 0
            
            results.append({
                "race_id": race_id_value,
                "馬番": bet_horse_num,
                "馬名": bet_horse_name,
                "着順": bet_finishing,
                "単勝": bet_single_odds,
                "モデルによる掛け金": this_cost
            })
            
            total_reward += reward
            total_cost += this_cost
        
        if env.current_race_idx >= len(env.race_dfs):
            break
        
        if done:
            obs = env.reset()
        else:
            obs = next_obs
    
    # ROI(回収率)を計算
    roi = (total_reward / total_cost * 100.0) if total_cost > 0 else 0.0
    result_df = pd.DataFrame(results)
    return roi, result_df

#------------------------------------------------------------------------------------
# 学習および推論を実行する関数
#------------------------------------------------------------------------------------
def run_training_and_inference(
    data_path=DATA_PATH,
    id_col='race_id',
    horse_col='馬番',
    horse_name_col='馬名',
    single_odds_col='単勝',
    finishing_col='着順',
    cost=100,
    n_epochs=10
):
    """
    学習→検証→学習曲線表示→テスト評価までを一通り実施する関数。
    最終的にテスト予測結果をCSVに保存する。
    """

    #--------------------------------------------------------------------------------
    # (1) データの読み込みと前処理
    #--------------------------------------------------------------------------------
    # コメント: train_df, valid_df, test_df にレース単位で分割されたデータが作られ、
    # 各行のX列に特徴量が格納される形となる。
    train_df, valid_df, test_df, _, _, _, _, _ = prepare_data(
        data_path=data_path,
        id_col=id_col,
        test_ratio=0.1,
        valid_ratio=0.1,
        pca_dim_horse=50,
        pca_dim_jockey=50,
        cat_cols=[]
    )

    # 環境を構築するためのfeature_cols(=Xに格納されている次元)を定義
    if len(train_df) > 0:
        feat_dim = len(train_df.iloc[0]["X"])  # 最初のサンプルから次元数を取得
        feature_cols = [f"feat_{i}" for i in range(feat_dim)]
    else:
        feature_cols = []

    # コメント: 環境を作成するときにはDataFrameにXが入っているが、
    # どのようにobsを構築するかはSingleWinBetEnvVariableHorsesの実装に依存している。
    
    #--------------------------------------------------------------------------------
    # (2) 学習環境, 検証環境を作成
    #--------------------------------------------------------------------------------
    train_env = SingleWinBetEnvVariableHorses(
        df=train_df, 
        feature_cols=feature_cols,
        id_col=id_col,
        horse_col=horse_col,
        horse_name_col=horse_name_col,
        single_odds_col=single_odds_col,
        finishing_col=finishing_col,
        cost=cost
    )
    valid_env = SingleWinBetEnvVariableHorses(
        df=valid_df, 
        feature_cols=feature_cols,
        id_col=id_col,
        horse_col=horse_col,
        horse_name_col=horse_name_col,
        single_odds_col=single_odds_col,
        finishing_col=finishing_col,
        cost=cost
    )

    # こちらでベクトル化(DummyVecEnvでラップ) - PPOは通常VecEnvを使うので
    vec_train_env = DummyVecEnv([lambda: train_env])

    #--------------------------------------------------------------------------------
    # (3) モデル構築(PPO)
    #--------------------------------------------------------------------------------
    # コメント: ニューラルネットの設定をpolicy_kwargsなどで細かく設定可能だが、ここではデフォルト。
    model = PPO("MlpPolicy", vec_train_env, verbose=0)

    #--------------------------------------------------------------------------------
    # (4) 学習ループを作成（epochごとにtrain_envとvalid_envのROIを計算）
    #--------------------------------------------------------------------------------
    train_rois = []
    valid_rois = []
    for epoch in range(n_epochs):
        # 学習を1epoch(=ある程度ステップ)実施
        model.learn(total_timesteps=1000)
        
        # 学習後、train回収率を評価
        train_roi, _ = evaluate_model(train_env, model, cost=cost)
        # valid回収率を評価
        valid_roi, _ = evaluate_model(valid_env, model, cost=cost)

        train_rois.append(train_roi)
        valid_rois.append(valid_roi)

        print(f"Epoch {epoch+1}/{n_epochs} - Train ROI: {train_roi:.2f}%, Valid ROI: {valid_roi:.2f}%")

    #--------------------------------------------------------------------------------
    # (5) 学習曲線を表示（trainとvalidのROI推移を可視化）
    #--------------------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_epochs+1), train_rois, label='Train ROI')
    plt.plot(range(1, n_epochs+1), valid_rois, label='Valid ROI')
    plt.xlabel('Epoch')
    plt.ylabel('ROI(%)')
    plt.title('Learning Curve')
    plt.legend()
    plt.tight_layout()
    plt.show()

    #--------------------------------------------------------------------------------
    # (6) テスト評価
    #--------------------------------------------------------------------------------
    test_env = SingleWinBetEnvVariableHorses(
        df=test_df, 
        feature_cols=feature_cols,
        id_col=id_col,
        horse_col=horse_col,
        horse_name_col=horse_name_col,
        single_odds_col=single_odds_col,
        finishing_col=finishing_col,
        cost=cost
    )

    test_roi, test_result_df = evaluate_model(test_env, model, cost=cost)
    print(f"Test ROI: {test_roi:.2f}%")

    #--------------------------------------------------------------------------------
    # (7) テスト結果をCSVに保存
    #--------------------------------------------------------------------------------
    # コメント: race_id, 馬番、馬名、着順、単勝、モデルによる掛け金(単勝購入額)を保存
    test_result_df.to_csv(SAVE_PATH_PRED, index=False, encoding='utf_8_sig')

#------------------------------------------------------------------------------------
# メイン呼び出し例（スクリプトとして実行する場合は下記を呼ぶ）
#------------------------------------------------------------------------------------
if __name__ == "__main__":
    run_training_and_inference(
        data_path=DATA_PATH,
        id_col='race_id',
        horse_col='馬番',
        horse_name_col='馬名',
        single_odds_col='単勝',
        finishing_col='着順',
        cost=100,
        n_epochs=10
    )
