import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import re
import datetime
import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import copy
import pickle
import random
from sklearn.calibration import calibration_curve

# 乱数固定(再現性確保)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ファイルパス設定（必要に応じて変更してください）
ROOT_PATH = "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/153_競馬_v3"
DATE_STRING = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
MODEL_SAVE_DIR = os.path.join(ROOT_PATH, f"models/transormer予測モデル/{DATE_STRING}")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

DATA_PATH = os.path.join(ROOT_PATH, "data/02_features/feature.csv")
SAVE_PATH_PRED = os.path.join(ROOT_PATH, f"result/predictions/transformer/{DATE_STRING}.csv")
SAVE_PATH_FULL_PRED = os.path.join(ROOT_PATH, f"result/predictions/transformer/{DATE_STRING}_full.csv")
SAVE_PATH_MODEL = os.path.join(MODEL_SAVE_DIR, "model.pickle")
SAVE_PATH_PCA_MODEL_HORSE = os.path.join(MODEL_SAVE_DIR, "pcamodel_horse.pickle")
SAVE_PATH_PCA_MODEL_JOCKEY = os.path.join(MODEL_SAVE_DIR, "pcamodel_jockey.pickle")
SAVE_PATH_SCALER_HORSE = os.path.join(MODEL_SAVE_DIR, "scaler_horse.pickle")
SAVE_PATH_SCALER_JOCKEY = os.path.join(MODEL_SAVE_DIR, "scaler_jockey.pickle")
SAVE_PATH_SCALER_OTHER = os.path.join(MODEL_SAVE_DIR, "scaler_other.pickle")

# =====================================================
# Datasetクラス
# =====================================================
class HorseRaceDataset(Dataset):
    """
    sequences: (num_races, max_seq_len, feature_dim)
    labels:    (num_races, max_seq_len, 6)
    masks:     (num_races, max_seq_len)
    race_ids:  (num_races, max_seq_len)  # レースIDを各馬行に割り当て（同一レースなら同じ値）
    horse_nums:(num_races, max_seq_len)  # 馬番

    ### diff追加
    is_wins    : (num_races, max_seq_len) -> (着順=1なら1)
    base_sups  : (num_races, max_seq_len) -> 0.8 / 単勝
    """
    def __init__(self, sequences, labels, masks, race_ids, horse_nums,
                 is_wins, base_sups):  ### diff追加
        self.sequences = sequences
        self.labels = labels
        self.masks = masks
        self.race_ids = race_ids
        self.horse_nums = horse_nums

        self.is_wins = is_wins            # (B,L)
        self.base_sups = base_sups        # (B,L)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        lab = torch.tensor(self.labels[idx], dtype=torch.float32)
        m = torch.tensor(self.masks[idx], dtype=torch.bool)
        rid = torch.tensor(self.race_ids[idx], dtype=torch.long)
        hn = torch.tensor(self.horse_nums[idx], dtype=torch.long)

        isw = torch.tensor(self.is_wins[idx],    dtype=torch.float32)  # (L,)
        bas = torch.tensor(self.base_sups[idx],  dtype=torch.float32)  # (L,)

        return seq, lab, m, rid, hn, isw, bas


# =====================================================
# Embedding + Transformerモデルクラス
#    => 出力次元を 6→7 に拡張 (diff + 6タスク)
# =====================================================
class FeatureEmbedder(nn.Module):
    """
    カテゴリ特徴量と数値特徴量を取り込み、
    Embedding + Linear で最終的に d_model 次元へ変換する層
    """
    def __init__(self, cat_unique, cat_cols, cat_emb_dim=16, num_dim=50, feature_dim=None):
        super().__init__()
        self.cat_cols = cat_cols
        self.emb_layers = nn.ModuleDict()
        for c in cat_cols:
            unique_count = cat_unique[c]
            # why: Embedding次元は適度に小さくすることで過学習を防ぎつつ、
            #      カテゴリ特徴を圧縮・表現可能にする。過度に大きいEmbeddingは学習が不安定になりやすい。
            emb_dim_real = min(cat_emb_dim, unique_count // 2 + 1)
            emb_dim_real = max(emb_dim_real, 4)
            self.emb_layers[c] = nn.Embedding(unique_count, emb_dim_real)
        # why: 数値特徴量にも簡単な変換をかけて表現力を持たせる（学習における線形変換のメリット）
        self.num_linear = nn.Linear(num_dim, num_dim)
        # Embedding出力と数値特徴量を結合するための最終線形層
        cat_out_dim = sum([self.emb_layers[c].embedding_dim for c in self.cat_cols])
        self.out_linear = nn.Linear(cat_out_dim + num_dim, feature_dim)

    def forward(self, x):
        """
        x: (batch, seq_len, feature_dim)
        ただしcat_cols部分が先頭に、後ろに数値特徴量という構造を前提
        """
        cat_len = len(self.cat_cols)
        cat_x = x[..., :cat_len].long()  # カテゴリ部分
        num_x = x[..., cat_len:]         # 数値部分
        embs = []
        for i, c in enumerate(self.cat_cols):
            embs.append(self.emb_layers[c](cat_x[..., i]))
        cat_emb = torch.cat(embs, dim=-1)
        num_emb = self.num_linear(num_x)
        out = torch.cat([cat_emb, num_emb], dim=-1)
        out = self.out_linear(out)
        return out


class PositionalEncoding(nn.Module):
    """
    トランスフォーマのSelf-Attentionで「系列順序」を学習させるために位置情報を付与
    """
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        # why: d_modelが奇数の場合はcos用に一つ短いdiv_termを用意してマッチさせるための条件分岐
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return x


class HorseTransformer(nn.Module):
    """
    Embedding + PositionalEncoding + TransformerEncoder -> 最後にLinearで6ターゲット出力
    """
    def __init__(self, cat_unique, cat_cols, max_seq_len, num_dim=50,
                 d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.feature_embedder = FeatureEmbedder(
            cat_unique, cat_cols, cat_emb_dim=16, num_dim=num_dim, feature_dim=d_model
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 7)  # [0]=diff, [1..6]=top1..pop5

    def forward(self, src, src_key_padding_mask=None):
        """
        src_key_padding_mask=Trueの部分をアテンションから除外することで
        パディング部分に対する誤学習を防ぐ
        """
        emb = self.feature_embedder(src)
        emb = self.pos_encoder(emb)
        out = self.transformer_encoder(emb, src_key_padding_mask=src_key_padding_mask)
        logits = self.fc_out(out) # (B,L,7)
        return logits


# =====================================================
# データ分割と前処理周りの関数
# =====================================================
def split_data(df, id_col="race_id", target_col="着順", test_ratio=0.1, valid_ratio=0.1):
    """
    ※ データを日付順でソートし、前のレースをtrain→valid→testと分割
    why: 未来の情報が混入しないよう、時系列的に分割する目的
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
    target_col="着順",
    pop_col="人気",
    id_col="race_id",
    leakage_cols=None,
    pca_dim_horse=50,
    pca_dim_jockey=50,
    test_ratio=0.1,
    valid_ratio=0.1
):
    """
    データ読み込み・前処理・PCA適用・Dataset作成を行う統合関数
    """
    if leakage_cols is None:
        leakage_cols = [
            '斤量','タイム','着差','単勝','上がり3F','馬体重','人気',
            'horse_id','jockey_id',
            'trainer_id',
            '順位点','入線','1着タイム差','先位タイム差','5着着差','増減',
            '1C通過順位','2C通過順位','3C通過順位','4C通過順位','賞金','前半ペース','後半ペース','ペース',
            '上がり3F順位','100m','200m','300m','400m','500m','600m','700m','800m','900m','1000m',
            '1100m','1200m','1300m','1400m','1500m','1600m','1700m','1800m','1900m','2000m',
            '2100m','2200m','2300m','2400m','2500m','2600m','2700m','2800m','2900m','3000m',
            '3100m','3200m','3300m','3400m','3500m','3600m','horse_ability',
            'is_win',
            'base_sups'
        ]

    # CSV読み込み
    df = pd.read_csv(data_path, encoding="utf_8_sig")

    # ### diff追加: is_win / base_sups
    df["is_win"] = (df[target_col] == 1).astype(int)
    df["base_sups"] = 0.8 / (df["単勝"] + 1e-9)

    # 時系列Split
    train_df, valid_df, test_df = split_data(df, id_col=id_col, target_col=target_col,
                                             test_ratio=test_ratio, valid_ratio=valid_ratio)

    # 数値列・カテゴリ列を取得
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in leakage_cols and c not in [target_col, id_col, pop_col]]
    num_cols = [c for c in num_cols if c not in leakage_cols and c not in [target_col, id_col, pop_col]]

    print('カテゴリ特徴量数：', len(cat_cols))
    print('数値特徴量数：', len(num_cols))

    # 欠損埋め・型変換
    for c in cat_cols:
        for d in [train_df, valid_df, test_df]:
            d[c] = d[c].fillna("missing").astype(str)
    for n in num_cols:
        for d in [train_df, valid_df, test_df]:
            d[n] = d[n].fillna(0)

    # カテゴリをcodes化
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

    # PCA対象列を抽出（例：馬の芝成績系、騎手の芝成績系など）
    pca_pattern_horse = r'^(競走馬芝|競走馬ダート|単年競走馬芝|単年競走馬ダート)'
    pca_pattern_jockey = r'^(騎手芝|騎手ダート|単年騎手芝|単年騎手ダート)'
    pca_horse_target_cols = [c for c in num_cols if re.match(pca_pattern_horse, c)]
    pca_jockey_target_cols = [c for c in num_cols if re.match(pca_pattern_jockey, c)]
    other_num_cols = [c for c in num_cols if (c not in pca_horse_target_cols)
                      and (c not in pca_jockey_target_cols)]

    # 1) Horse用のスケーリング + PCA
    scaler_horse = StandardScaler()
    horse_features_train_scaled = scaler_horse.fit_transform(train_df[pca_horse_target_cols].values)
    horse_features_valid_scaled = scaler_horse.transform(valid_df[pca_horse_target_cols].values)
    horse_features_test_scaled = scaler_horse.transform(test_df[pca_horse_target_cols].values)

    pca_dim_horse = min(pca_dim_horse, horse_features_train_scaled.shape[1])
    pca_model_horse = PCA(n_components=pca_dim_horse)
    horse_features_train_pca = pca_model_horse.fit_transform(horse_features_train_scaled)
    horse_features_valid_pca = pca_model_horse.transform(horse_features_valid_scaled)
    horse_features_test_pca = pca_model_horse.transform(horse_features_test_scaled)

    # 2) Jockey用のスケーリング + PCA
    scaler_jockey = StandardScaler()
    jockey_features_train_scaled = scaler_jockey.fit_transform(train_df[pca_jockey_target_cols].values)
    jockey_features_valid_scaled = scaler_jockey.transform(valid_df[pca_jockey_target_cols].values)
    jockey_features_test_scaled = scaler_jockey.transform(test_df[pca_jockey_target_cols].values)

    pca_dim_jockey = min(pca_dim_jockey, jockey_features_train_scaled.shape[1])
    pca_model_jockey = PCA(n_components=pca_dim_jockey)
    jockey_features_train_pca = pca_model_jockey.fit_transform(jockey_features_train_scaled)
    jockey_features_valid_pca = pca_model_jockey.transform(jockey_features_valid_scaled)
    jockey_features_test_pca = pca_model_jockey.transform(jockey_features_test_scaled)

    # 3) その他の数値特徴量のスケーリング
    scaler_other = StandardScaler()
    other_features_train = scaler_other.fit_transform(train_df[other_num_cols].values)
    other_features_valid = scaler_other.transform(valid_df[other_num_cols].values)
    other_features_test = scaler_other.transform(test_df[other_num_cols].values)

    # 4) カテゴリ特徴量
    cat_features_train = train_df[cat_cols].values
    cat_features_valid = valid_df[cat_cols].values
    cat_features_test = test_df[cat_cols].values

    # 特徴量結合
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

    # 数値次元を計算
    actual_num_dim = other_features_train.shape[1] + pca_dim_horse + pca_dim_jockey

    print(f'X_train shape: {X_train.shape}')
    print(f'X_valid shape: {X_valid.shape}')
    print(f'X_test  shape: {X_test.shape}')
    print(f'Actual numerical dimensions: {actual_num_dim}')

    # ラベルなどを元にSequence形式に変換
    def create_sequences(_df, X):
        """
        labels: [top1, top3, top5, pop1, pop3, pop5]
        """
        ranks = _df[target_col].values  # 着順
        pops = _df[pop_col].values      # 人気
        rids = _df[id_col].values       # race_id
        horse_nums = _df["馬番"].values if "馬番" in _df.columns else np.arange(len(_df))

        # diff追加
        is_wins   = _df["is_win"].values
        base_sups = _df["base_sups"].values

        groups = _df.groupby(id_col)
        max_seq_len = groups.size().max()
        feature_dim = X.shape[1]

        # 6タスク (top1..pop5)
        top1 = (ranks == 1).astype(int)
        top3 = (ranks <= 3).astype(int)
        top5 = (ranks <= 5).astype(int)
        pop1 = (pops == 1).astype(int)
        pop3 = (pops <= 3).astype(int)
        pop5 = (pops <= 5).astype(int)

        sequences, labels, masks = [], [], []
        race_ids_seq, horse_nums_seq = [], []
        is_win_seq, base_sups_seq = [], []


        for unique_rid in _df[id_col].unique():
            idx = np.where(rids == unique_rid)[0]
            feat = X[idx]
            seq_len = len(idx)

            tar = np.stack([top1[idx], top3[idx], top5[idx],
                            pop1[idx], pop3[idx], pop5[idx]], axis=-1)  # (seq_len, 6)

            rid_array = rids[idx]
            horse_array = horse_nums[idx]

            # diff追加
            isw_arr = is_wins[idx]
            bas_arr = base_sups[idx]

            pad_len = max_seq_len - seq_len
            if pad_len > 0:
                feat = np.vstack([feat, np.zeros((pad_len, feature_dim))])
                pad_label = np.zeros((pad_len, 6), dtype=int)
                tar = np.concatenate([tar, pad_label], axis=0)
                mask = [1]*seq_len + [0]*pad_len

                rid_pad = np.full((pad_len,), fill_value=-1, dtype=rid_array.dtype)
                horse_pad = np.full((pad_len,), fill_value=-1, dtype=horse_array.dtype)
                rid_array = np.concatenate([rid_array, rid_pad])
                horse_array = np.concatenate([horse_array, horse_pad])

                # diff追加
                isw_pad = np.zeros((pad_len,), dtype=isw_arr.dtype)
                bas_pad = np.zeros((pad_len,), dtype=bas_arr.dtype)
                isw_arr = np.concatenate([isw_arr, isw_pad])
                bas_arr = np.concatenate([bas_arr, bas_pad])
            else:
                mask = [1]*seq_len

            sequences.append(feat)
            labels.append(tar)
            masks.append(mask)
            race_ids_seq.append(rid_array)
            horse_nums_seq.append(horse_array)
            is_win_seq.append(isw_arr)
            base_sups_seq.append(bas_arr)

        return sequences, labels, masks, max_seq_len, race_ids_seq, horse_nums_seq, is_win_seq, base_sups_seq

    train_seq, train_lab, train_mask, msl_train, train_rids_seq, train_horses_seq, \
        train_iswin_seq, train_basesup_seq = create_sequences(train_df, X_train)
    valid_seq, valid_lab, valid_mask, msl_valid, valid_rids_seq, valid_horses_seq, \
        valid_iswin_seq, valid_basesup_seq = create_sequences(valid_df, X_valid)
    test_seq, test_lab, test_mask,  msl_test,  test_rids_seq,  test_horses_seq,  \
        test_iswin_seq,  test_basesup_seq  = create_sequences(test_df,  X_test)

    max_seq_len = max(msl_train, msl_valid, msl_test)

    cat_unique = {}
    for c in cat_cols:
        cat_unique[c] = len(train_df[c].unique())

    # Dataset化 (diff対応)
    train_dataset = HorseRaceDataset(
        train_seq, train_lab, train_mask, train_rids_seq, train_horses_seq,
        train_iswin_seq, train_basesup_seq
    )
    valid_dataset = HorseRaceDataset(
        valid_seq, valid_lab, valid_mask, valid_rids_seq, valid_horses_seq,
        valid_iswin_seq, valid_basesup_seq
    )
    test_dataset  = HorseRaceDataset(
        test_seq,  test_lab,  test_mask,  test_rids_seq,  test_horses_seq,
        test_iswin_seq,  test_basesup_seq
    )

    return (train_dataset, valid_dataset, test_dataset,
            cat_cols, cat_unique, max_seq_len,
            pca_dim_horse, pca_dim_jockey,
            6,  # (multiタスク6)
            actual_num_dim, df, cat_cols, num_cols,
            pca_horse_target_cols, pca_jockey_target_cols, other_num_cols,
            scaler_horse, scaler_jockey, scaler_other,
            pca_model_horse, pca_model_jockey,
            X_train.shape[1],
            id_col, target_col,
            train_df)


# =====================================================
# Train Time Split を用いた学習 + 重み付きアンサンブル
# =====================================================
def run_train_time_split(
    data_path=DATA_PATH,
    target_col="着順",
    pop_col="人気",
    id_col="race_id",
    batch_size=256,
    lr=0.001,
    num_epochs=50,
    pca_dim_horse=50,
    pca_dim_jockey=50,
    test_ratio=0.2,
    valid_ratio=0.1,
    d_model=128,
    nhead=8,
    num_layers=6,
    dropout=0.15,
    weight_decay=1e-5,
    patience=10,
    # 今回のポイント: 複数の時系列Splitをさらに作ってモデルを増やす
    # 例として train-valid を time split し、その区間をずらしながら複数のモデルを作る想定
    # ここでは5回に分割する例
    n_splits=5
):
    """
    時系列を使った再分割 → 学習 → 各モデルの精度を計測 → 精度に応じて重み付け → 推論平均
    """
    # -------------------------------------------------
    # 1) メインとなるデータを prepare_data で1回読み込み
    #    （全体をさらに段階的に time split して使う）
    # -------------------------------------------------
    """
    - Diffタスク (出力[0]番) + 6分類 (出力[1..6]) のマルチタスク学習
    - Time-splitをずらして複数モデルを学習→加重アンサンブル
    """
    (base_train_dataset, base_valid_dataset, base_test_dataset,
     cat_cols, cat_unique, max_seq_len, pca_dim_horse, pca_dim_jockey, _,
     actual_num_dim, df_all, _, _,
     _, _, _,
     scaler_horse, scaler_jockey, scaler_other,
     pca_model_horse, pca_model_jockey, _,
     id_col, target_col,
     df_train_part) = prepare_data(
        data_path=data_path,
        target_col=target_col,
        pop_col=pop_col,
        id_col=id_col,
        pca_dim_horse=pca_dim_horse,
        pca_dim_jockey=pca_dim_jockey,
        test_ratio=test_ratio,
        valid_ratio=valid_ratio
    )

    # デバイス選択
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # -------------------------------------------------
    # 2) train_part(= train_dataset + valid_datasetの元データ)を
    #    n_splits 個にさらに時系列分割するロジック
    #    ※ 既に split_data で train/valid/test は分かれているが、
    #      ここでさらに "train内部" を細かく split → それぞれで学習・評価
    #    ※ 実際には train_df やdf_train_part からレースID順に区切る
    # -------------------------------------------------
    # base_train_dataset は df_train_part がベースになっているが、
    #  ここでさらに race_id 日付順にいくつかに分割して学習する例を簡易実装

    race_ids_sorted = df_train_part.sort_values('date')[id_col].unique()
    chunk_size = len(race_ids_sorted) // n_splits

    # コンテナ
    models = []
    valid_losses_per_model = []

    # -- カスタム損失関数 (diff + multi(6)) --
    def custom_loss_diff_multi(outputs, labels_multi, is_win, base_sups, masks):
        """
        outputs: (B,L,7) -> [0]: diff_out, [1..6]: multi_out
        labels_multi: (B,L,6)
        is_win: (B,L)
        base_sups: (B,L)
        masks: (B,L)  (True=有効)
        """
        eps = 1e-7
        diff_out  = outputs[..., 0]      # (B,L)
        multi_out = outputs[..., 1:7]    # (B,L,6)

        # (1) diff 用の BCE
        #     pred_prob_win = sigmoid(logit(base_sups) + diff_out)
        logit_bs = torch.logit(base_sups, eps=eps)  # (B,L)
        pred_win = torch.sigmoid(logit_bs + diff_out)  # (B,L)
        # BCE( pred_win, is_win )
        bce_diff = - (is_win*torch.log(pred_win+eps) + (1 - is_win)*torch.log(1-pred_win+eps))  # (B,L)

        # (2) multi(6) 用 BCEWithLogits
        bce_multi = nn.functional.binary_cross_entropy_with_logits(multi_out, labels_multi, reduction='none')
        # (B,L,6)

        # パディング除外
        masks_3d = masks.unsqueeze(-1)  # (B,L,1)
        bce_diff  = bce_diff * masks
        bce_multi = bce_multi * masks_3d

        denom = masks.sum() + eps
        loss_diff  = bce_diff.sum() / denom
        loss_multi = bce_multi.sum() / (denom*6)

        loss = loss_diff + loss_multi
        return loss, loss_diff.item(), loss_multi.item()

    for i in range(n_splits):
        # 例: i-th split の開始～終了
        start_idx = i * chunk_size
        end_idx = (i+1)*chunk_size if i < (n_splits-1) else len(race_ids_sorted)
        # train用, valid用にレースIDを割り当て（例: 最後のチャンクをvalid相当にする等、いろいろ方法あり）
        # ここでは単純に "最初のi個を除いた部分" を train, "該当のチャンク" を valid としてみる
        # why: time-basedに前方をtrain、後方をvalidationにするなど色々な方法が考えられるが、
        #      ここでは各チャンクを順番にvalidationとして扱う。より本格的な方法では
        #      train: 0～i-1チャンク, valid: iチャンク, test: i+1以降 などの形式も可能。
        valid_race_ids = race_ids_sorted[start_idx:end_idx]
        train_race_ids = race_ids_sorted[:start_idx]

        # ただしi=0の場合は train_race_idsが空にならないようにするなど実運用では工夫
        if len(train_race_ids) == 0:
            # i=0 ならvalidも小さくしてスキップ or 次のsplitに飛ばすなど
            continue

        # DatasetのSubsetを作る
        train_indices = []
        valid_indices = []
        for idx in range(len(base_train_dataset)):
            # race_idsにはパディングで-1が含まれている可能性があるので除外
            race_id_tensor = base_train_dataset[idx][3]  # 3番目がrace_ids
            # 同一レース内の要素(パディング以外)はどれでも同じ値なので先頭を使う
            rid_val = race_id_tensor[0].item()
            if rid_val in train_race_ids:
                train_indices.append(idx)
            elif rid_val in valid_race_ids:
                valid_indices.append(idx)

        # Subset
        sub_train_dataset = torch.utils.data.Subset(base_train_dataset, train_indices)
        sub_valid_dataset = torch.utils.data.Subset(base_train_dataset, valid_indices)

        # Dataloader
        train_loader = DataLoader(sub_train_dataset, batch_size=batch_size, shuffle=False)
        valid_loader = DataLoader(sub_valid_dataset, batch_size=batch_size, shuffle=False)

        # モデル初期化
        model = HorseTransformer(
            cat_unique, cat_cols, max_seq_len,
            num_dim=actual_num_dim, d_model=d_model,
            nhead=nhead, num_layers=num_layers, dropout=dropout
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        # 早期終了用
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = float('inf')
        epochs_no_improve = 0

        print(f"\n====== Time-split {i+1}/{n_splits} ======")
        for epoch in range(num_epochs):
            # --- train ---
            model.train()
            total_loss = 0.0
            total_count = 0.0

            for (seq, lab6, masks, _, _, is_win, base_sups) in train_loader:
                seq = seq.to(device)
                lab6= lab6.to(device)
                masks= masks.to(device)
                is_win = is_win.to(device)
                base_sups = base_sups.to(device)

                optimizer.zero_grad()
                outputs = model(seq, src_key_padding_mask=~masks)  # (B,L,7)
                loss_batch, ld, lm = custom_loss_diff_multi(outputs, lab6, is_win, base_sups, masks)
                loss_batch.backward()
                optimizer.step()

                total_loss += loss_batch.item()
                total_count += 1

            scheduler.step()
            avg_train_loss = total_loss / max(total_count,1)

            # --- valid ---
            model.eval()
            total_loss_val = 0.0
            total_count_val = 0.0
            with torch.no_grad():
                for (seq, lab6, masks, _, _, is_win, base_sups) in valid_loader:
                    seq = seq.to(device)
                    lab6= lab6.to(device)
                    masks= masks.to(device)
                    is_win= is_win.to(device)
                    base_sups= base_sups.to(device)

                    outputs = model(seq, src_key_padding_mask=~masks)
                    loss_val, _, _ = custom_loss_diff_multi(outputs, lab6, is_win, base_sups, masks)
                    total_loss_val += loss_val.item()
                    total_count_val += 1

            avg_valid_loss = total_loss_val / max(total_count_val,1)
            print(f"Epoch [{epoch+1}/{num_epochs}]  TrainLoss={avg_train_loss:.4f}  ValidLoss={avg_valid_loss:.4f}")

            # 早期終了判定
            if avg_valid_loss < best_loss:
                best_loss = avg_valid_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

        # モデルの学習結果を保存
        model.load_state_dict(best_model_wts)
        models.append(model)
        valid_losses_per_model.append(best_loss)

    # アンサンブル重み
    valid_losses_np = np.array(valid_losses_per_model)
    eps = 1e-6
    inv_losses = 1.0 / (valid_losses_np + eps)
    weights = inv_losses / inv_losses.sum()

    for i, (loss_val, w) in enumerate(zip(valid_losses_per_model, weights)):
        print(f" Model{i+1}: best_valid_loss={loss_val:.5f}, weight={w:.3f}")

    # -------------------------------------------------
    # 4) テストデータでアンサンブル
    # -------------------------------------------------
    test_loader  = DataLoader(base_test_dataset, batch_size=batch_size, shuffle=False)

    def test_evaluate_ensemble_diff(loader, models, weights):
        """
        diff + 6タスク の予測を加重平均
          - diffは logit空間で加重平均 → pred_prob_win を計算
          - 6タスクはシグモイド確率を加重平均
        """
        for m in models:
            m.eval()

        # 各タスクのロス集計用
        loss_diff_sum = 0.0
        loss_multi_sum= 0.0
        count = 0

        all_pred_multi_list = []
        all_pred_diff_list  = []  # diff
        all_label_multi_list= []
        all_label_win_list  = []
        all_rids_list = []
        all_hnums_list = []

        with torch.no_grad():
            for (seq, lab6, masks, rids, hnums, is_win, base_sups) in loader:
                seq     = seq.to(device)
                lab6    = lab6.to(device)
                masks   = masks.to(device)
                is_win  = is_win.to(device)
                base_sups= base_sups.to(device)

                # (A) 全モデルで出力を得て加重平均
                ensemble_diff = None
                ensemble_multi= None
                for w, model in zip(weights, models):
                    out = model(seq, src_key_padding_mask=~masks)  # (B,L,7)
                    diff_out  = out[..., 0]
                    multi_out = out[..., 1:]  # (B,L,6)

                    # diff_out は "ログit空間に加算する" 形なので、そのまま w加重
                    if ensemble_diff is None:
                        ensemble_diff = w * diff_out
                        ensemble_multi= w * torch.sigmoid(multi_out)
                    else:
                        ensemble_diff += w * diff_out
                        ensemble_multi+= w * torch.sigmoid(multi_out)

                # ensemble_diff => pred_win
                # pred_win = sigmoid( logit_bs + ensemble_diff )
                logit_bs = torch.logit(base_sups, eps=1e-7)
                pred_win = torch.sigmoid(logit_bs + ensemble_diff)  # (B,L)

                # (B) 損失を計算 (diff + multi)
                # diff BCE
                eps_ = 1e-7
                bce_diff = - ( is_win*torch.log(pred_win+eps_) + (1-is_win)*torch.log(1-pred_win+eps_) )
                bce_diff = bce_diff * masks
                denom = masks.sum() + eps_
                ldiff = bce_diff.sum() / denom

                # multi BCE
                # ensemble_multiは(確率)
                # logitsに戻す場合: ensemble_logits = logit( ensemble_multi )
                # ただし 0or1 は危険なので clamp
                multi_out_logits = torch.logit(ensemble_multi.clamp(min=1e-7, max=1-1e-7))
                bce_multi = nn.functional.binary_cross_entropy_with_logits(
                    multi_out_logits, lab6, reduction='none'
                )
                bce_multi = bce_multi * masks.unsqueeze(-1)
                lm = bce_multi.sum() / (denom*6)

                # 合計
                loss_diff_sum  += ldiff.item()
                loss_multi_sum += lm.item()
                count += 1

                # (C) マスク除外して最終確率を保存
                valid_mask_2d = masks.cpu().numpy().astype(bool)
                pd_np = base_sups[masks].view(-1).cpu().numpy()
                pm_np = ensemble_multi[masks].view(-1, 6).cpu().numpy()
                lw_np = is_win[masks].view(-1).cpu().numpy()
                lm_np = lab6[masks].view(-1, 6).cpu().numpy()

                rids_np  = rids.cpu().numpy()
                hnums_np = hnums.cpu().numpy()
                rid_valid   = rids_np[valid_mask_2d]
                horse_valid = hnums_np[valid_mask_2d]

                all_pred_diff_list.append(pd_np)
                all_pred_multi_list.append(pm_np)
                all_label_win_list.append(lw_np)
                all_label_multi_list.append(lm_np)
                all_rids_list.append(rid_valid)
                all_hnums_list.append(horse_valid)

        avg_loss_diff  = loss_diff_sum / max(count,1)
        avg_loss_multi = loss_multi_sum / max(count,1)

        pd_all = np.concatenate(all_pred_diff_list, axis=0)
        pm_all = np.concatenate(all_pred_multi_list, axis=0)
        lw_all = np.concatenate(all_label_win_list, axis=0)
        lm_all = np.concatenate(all_label_multi_list, axis=0)
        rr_all = np.concatenate(all_rids_list, axis=0)
        hh_all = np.concatenate(all_hnums_list, axis=0)

        return avg_loss_diff, avg_loss_multi, pd_all, pm_all, lw_all, lm_all, rr_all, hh_all

    # テスト評価
    ld_test, lm_test, pd_test, pm_test, lw_test, lm_test_, rr_test, hh_test = \
        test_evaluate_ensemble_diff(test_loader, models, weights)


    print(f"[Test] BCE Loss - diff={ld_test:.4f}, multi={lm_test:.4f}")

    # 結果をDataFrame化 (pred_win, pred_top1..pop5 etc.)
    test_df_out = pd.DataFrame({
        "race_id": rr_test.astype(int),
        "馬番":    hh_test.astype(int),
        "pred_diff":  pd_test,    # diff
    })
    # 6タスクを列追加
    task_names = ["top1","top3","top5","pop1","pop3","pop5"]
    for i, nm in enumerate(task_names):
        test_df_out[f"pred_{nm}"]  = pm_test[:, i]
        test_df_out[f"label_{nm}"] = lm_test_[:, i]

    test_df_out.to_csv(SAVE_PATH_PRED, index=False)
    print("Saved test predictions:", SAVE_PATH_PRED)

    # Full dataset に対しても推論
    full_dataset = ConcatDataset([base_train_dataset, base_valid_dataset, base_test_dataset])
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    ld_full, lm_full, pd_full, pm_full, lw_full, lm_full_, rr_full, hh_full = \
        test_evaluate_ensemble_diff(full_loader, models, weights)
    print(f"[Full] BCE Loss - diff={ld_full:.4f}, multi={lm_full:.4f}")

    full_df_out = pd.DataFrame({
        "race_id": rr_full.astype(int),
        "馬番":    hh_full.astype(int),
        "pred_diff":  pd_full,
    })
    for i, nm in enumerate(task_names):
        full_df_out[f"pred_{nm}"]  = pm_full[:, i]
        full_df_out[f"label_{nm}"] = lm_full_[:, i]

    full_df_out.to_csv(SAVE_PATH_FULL_PRED, index=False)
    print("Saved full predictions:", SAVE_PATH_FULL_PRED)

    # キャリブレーション曲線表示
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    target_names = ["Top1", "Top3", "Top5", "Pop1", "Pop3", "Pop5"]
    for i in range(6):
        prob_true, prob_pred = calibration_curve(lm_test_[:, i], pm_test[:, i], n_bins=10)
        ax = axes[i]
        ax.plot(prob_pred, prob_true, marker='o', label='Calibration')
        ax.plot([0,1],[0,1], '--', color='gray', label='Perfect')
        ax2 = ax.twinx()
        ax2.hist(pm_test[:, i], bins=20, range=(0,1), alpha=0.3, color='gray')
        ax.set_title(f'Calibration Curve ({target_names[i]})')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('True Probability')
        ax.legend()

    plt.suptitle("Weighted Ensemble Calibration Curves", fontsize=16)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,6))
    plt.hist(pd_test, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title("Distribution of Ensemble Diff")
    plt.xlabel("Ensemble Diff")
    plt.ylabel("Frequency")
    plt.show()

    # -------------------------------------------------
    # 6) 代表的なモデル or 全モデルを保存 (ここでは重み付きアンサンブル想定)
    #    最終的に"平均"の概念が強いので、単一モデルとしての保存は必要に応じて
    # -------------------------------------------------
    # 例: 最良のモデル(一番valid_lossが小さいモデル)を保存
    # モデル保存 (best単体モデルなど)
    best_model_idx = np.argmin(valid_losses_np)
    best_model = models[best_model_idx]
    with open(SAVE_PATH_MODEL, "wb") as f:
        pickle.dump(best_model.state_dict(), f)
    with open(SAVE_PATH_PCA_MODEL_HORSE, "wb") as f:
        pickle.dump(pca_model_horse, f)
    with open(SAVE_PATH_PCA_MODEL_JOCKEY, "wb") as f:
        pickle.dump(pca_model_jockey, f)
    with open(SAVE_PATH_SCALER_HORSE, "wb") as f:
        pickle.dump(scaler_horse, f)
    with open(SAVE_PATH_SCALER_JOCKEY, "wb") as f:
        pickle.dump(scaler_jockey, f)
    with open(SAVE_PATH_SCALER_OTHER, "wb") as f:
        pickle.dump(scaler_other, f)

    print(f"\n=== Final Ensemble done. Best single model index: {best_model_idx+1}")
    return 0
