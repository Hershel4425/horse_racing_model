import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import re
import datetime
import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import copy
import pickle
import random
from sklearn.calibration import calibration_curve

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

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
    labels:    (num_races, max_seq_len, 1)   ← 1着率のみ
    masks:     (num_races, max_seq_len)
    race_ids:  (num_races, max_seq_len)  # レースIDを各馬行に割り当て（同一レースなら同じ値）
    horse_nums:(num_races, max_seq_len)  # 馬番
    """
    def __init__(self, sequences, labels, masks, race_ids, horse_nums):
        self.sequences = sequences
        self.labels = labels
        self.masks = masks
        self.race_ids = race_ids
        self.horse_nums = horse_nums

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        lab = torch.tensor(self.labels[idx], dtype=torch.float32)
        m = torch.tensor(self.masks[idx], dtype=torch.bool)
        rid = torch.tensor(self.race_ids[idx], dtype=torch.long)
        hn = torch.tensor(self.horse_nums[idx], dtype=torch.long)
        return seq, lab, m, rid, hn


# =====================================================
# Embedding + Transformerモデルクラス
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
            emb_dim_real = min(cat_emb_dim, unique_count // 2 + 1)
            emb_dim_real = max(emb_dim_real, 4)
            self.emb_layers[c] = nn.Embedding(unique_count, emb_dim_real)
        self.num_linear = nn.Linear(num_dim, num_dim)
        cat_out_dim = sum([self.emb_layers[c].embedding_dim for c in self.cat_cols])
        self.out_linear = nn.Linear(cat_out_dim + num_dim, feature_dim)

    def forward(self, x):
        cat_len = len(self.cat_cols)
        cat_x = x[..., :cat_len].long()
        num_x = x[..., cat_len:]
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
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return x
    

class CustomTransformerEncoderLayer(nn.Module):
    """
    TransformerEncoderLayerを参考に、
    FeedForwardネットワークを深く拡張した例
    """
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = "relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim_feedforward)
        self.linear3 = nn.Linear(dim_feedforward, d_model)
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn  = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        if activation == "relu":
            self.act_fn = F.relu
        elif activation == "gelu":
            self.act_fn = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        src2, _ = self.self_attn(src, src, src,
                                 attn_mask=src_mask,
                                 key_padding_mask=src_key_padding_mask)
        src = src + self.dropout_attn(src2)
        src = self.norm1(src)
        ffn_out = self.linear1(src)
        ffn_out = self.act_fn(ffn_out)
        ffn_out = self.linear2(ffn_out)
        ffn_out = self.act_fn(ffn_out)
        ffn_out = self.linear3(ffn_out)
        src = src + self.dropout_ffn(ffn_out)
        src = self.norm2(src)
        return src

# ★ 新しいモデルクラス：1着率のみ予測する（シングルヘッド）
class HorseTransformerSingleHead(nn.Module):
    def __init__(self, cat_unique, cat_cols, max_seq_len,
                 num_dim=50, d_model=128, nhead=8, num_layers=4,
                 dropout=0.1, dim_feedforward=512):
        super().__init__()
        self.feature_embedder = FeatureEmbedder(
            cat_unique, cat_cols, cat_emb_dim=16, num_dim=num_dim, feature_dim=d_model
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        encoder_layer = CustomTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.fc_top1 = nn.Linear(d_model, 1)

    def forward(self, src, src_key_padding_mask=None):
        emb = self.feature_embedder(src)
        emb = self.pos_encoder(emb)
        out = self.transformer_encoder(emb, src_key_padding_mask=src_key_padding_mask)
        top1 = self.fc_top1(out)
        return top1
    

########################################
# Auto Encoder
########################################
class HorseFeatureAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=50):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
    
class JockeyFeatureAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=50):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
    

# =====================================================
# データ分割と前処理周りの関数
# =====================================================
def split_data(df, id_col="race_id", target_col="着順", test_ratio=0.1, valid_ratio=0.1):
    df = df.sort_values('date').reset_index(drop=True)
    race_ids = df[id_col].unique()
    dataset_len = len(race_ids)
    test_cut = int(dataset_len*(1 - test_ratio))
    train_ids = race_ids[:test_cut]
    test_ids  = race_ids[test_cut:]
    train_df  = df[df[id_col].isin(train_ids)].copy()
    test_df   = df[df[id_col].isin(test_ids)].copy()
    valid_df = pd.DataFrame([])
    
    return train_df, valid_df, test_df



def prepare_data(
    data_path,
    target_col="着順",
    pop_col="人気",  # 今回は使用しないけど引数は残す
    id_col="race_id",
    leakage_cols=None,
    ae_latent_horse=50,
    ae_latent_jockey=50,
    test_ratio=0.1,
    valid_ratio=0.1
):
    if leakage_cols is None:
        leakage_cols = [
            '斤量','タイム','着差','単勝','上がり3F','人気',
            'horse_id','jockey_id',
            'trainer_id',
            '馬体重', '増減','単勝',
            '順位点','入線','1着タイム差','先位タイム差','5着着差',
            '1C通過順位','2C通過順位','3C通過順位','4C通過順位','賞金','前半ペース','後半ペース','ペース','ペース_脚質',
            '上がり3F順位', '上がり3F順位_missing',
            '100m','200m','300m','400m','500m','600m','700m','800m','900m','1000m',
            '1100m','1200m','1300m','1400m','1500m','1600m','1700m','1800m','1900m','2000m',
            '2100m','2200m','2300m','2400m','2500m','2600m','2700m','2800m','2900m','3000m',
            '3100m','3200m','3300m','3400m','3500m','3600m','馬_ability'
        ]

    df = pd.read_csv(data_path, encoding="utf_8_sig")
    df = df[df['date'] >= '2017-01-01'].copy()
    print('使用データサイズ：', df.shape)

    train_df, _, test_df = split_data(df, id_col=id_col, target_col=target_col,
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
        for d in [train_df, test_df]:
            d[c] = d[c].fillna("missing").astype(str)
    for n in num_cols:
        for d in [train_df, test_df]:
            d[n] = d[n].fillna(0)

    # ★ ダウンサンプリング ★
    # train_df内の各レースで、勝ち馬（着順==1）の単勝値に基づいて
    # 単勝<=3, 3<単勝<=5, 5<単勝<=10, 10<単勝<=30, 30<単勝 のグループに分け、
    # 最も件数が少ない「30<単勝」の件数に揃える（testは対象外）
    if not train_df.empty:
        winners = train_df[train_df["着順"] == 1].copy()
        conditions = [
            (winners["単勝"] <= 3),
            (winners["単勝"] > 3) & (winners["単勝"] <= 5),
            (winners["単勝"] > 5) & (winners["単勝"] <= 10),
            (winners["単勝"] > 10) & (winners["単勝"] <= 30),
            (winners["単勝"] > 30)
        ]
        bins_labels = ["<=3", "3-5", "5-10", "10-30", ">30"]
        winners["odds_bin"] = np.select(conditions, bins_labels, default="unknown")
        bin_counts = winners["odds_bin"].value_counts()
        target_count = bin_counts.get(">30", 0)
        sampled_race_ids = []
        for label in bins_labels:
            race_ids_in_bin = winners[winners["odds_bin"] == label]["race_id"].unique()
            if len(race_ids_in_bin) > target_count:
                sampled_ids = np.random.choice(race_ids_in_bin, target_count, replace=False)
            else:
                sampled_ids = race_ids_in_bin
            sampled_race_ids.extend(sampled_ids)
        train_df = train_df[train_df["race_id"].isin(sampled_race_ids)].copy()
        print("Downsampled training data shape:", train_df.shape)

    # カテゴリをcodes化
    for c in cat_cols:
        train_df[c] = train_df[c].astype('category')
        test_df[c] = test_df[c].astype('category')
        train_cat = train_df[c].cat.categories
        test_df[c] = pd.Categorical(test_df[c], categories=train_cat)
        train_df[c] = train_df[c].cat.codes
        test_df[c] = test_df[c].cat.codes

    # PCA対象列はそのまま（AutoEncoderで圧縮）
    pattern_horse = r'^(競走馬芝|競走馬ダート|単年競走馬芝|単年競走馬ダート)'
    pattern_jockey = r'^(騎手芝|騎手ダート|単年騎手芝|単年騎手ダート)'

    horse_cols  = [c for c in num_cols if re.match(pattern_horse, c)]
    jockey_cols = [c for c in num_cols if re.match(pattern_jockey, c)]
    other_num_cols = [c for c in num_cols if c not in horse_cols + jockey_cols]

    cat_train = train_df[cat_cols].values
    cat_test  = test_df[cat_cols].values

    # ----------------------------------------------------
    # 1) 標準化
    # ----------------------------------------------------
    scaler_horse  = StandardScaler()
    scaler_jockey = StandardScaler()
    scaler_other  = StandardScaler()

    if len(horse_cols)>0:
        scaler_horse.fit(train_df[horse_cols])
        horse_train_arr = scaler_horse.transform(train_df[horse_cols])
        horse_test_arr  = scaler_horse.transform(test_df[horse_cols])
    else:
        horse_train_arr = np.zeros((len(train_df),0))
        horse_test_arr  = np.zeros((len(test_df),0))

    if len(jockey_cols)>0:
        scaler_jockey.fit(train_df[jockey_cols])
        jockey_train_arr= scaler_jockey.transform(train_df[jockey_cols])
        jockey_test_arr = scaler_jockey.transform(test_df[jockey_cols])
    else:
        jockey_train_arr= np.zeros((len(train_df),0))
        jockey_test_arr = np.zeros((len(test_df),0))

    if len(other_num_cols)>0:
        scaler_other.fit(train_df[other_num_cols])
        other_train_arr = scaler_other.transform(train_df[other_num_cols])
        other_test_arr  = scaler_other.transform(test_df[other_num_cols])
    else:
        other_train_arr = np.zeros((len(train_df),0))
        other_test_arr  = np.zeros((len(test_df),0))

    # ----------------------------------------------------
    # 2) AutoEncoder で馬特徴を圧縮
    # ----------------------------------------------------
    from torch.utils.data import DataLoader, TensorDataset

    if horse_train_arr.shape[1] > 0:
        horse_input_dim = horse_train_arr.shape[1]
        ae_horse = HorseFeatureAutoEncoder(input_dim=horse_input_dim, latent_dim=ae_latent_horse)
        horse_train_dataset = TensorDataset(torch.tensor(horse_train_arr, dtype=torch.float32))
        horse_train_loader  = DataLoader(horse_train_dataset, batch_size=512, shuffle=False)
        optimizer = torch.optim.Adam(ae_horse.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.MSELoss()
        ae_horse.train()
        num_epochs_ae = 10
        for epoch in range(num_epochs_ae):
            total_loss = 0
            for (batch_x,) in horse_train_loader:
                optimizer.zero_grad()
                x_recon, z = ae_horse(batch_x)
                loss = criterion(x_recon, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[AE-horse] Epoch {epoch+1}/{num_epochs_ae}  Loss: {total_loss/len(horse_train_loader):.4f}")
        ae_horse.eval()
        with torch.no_grad():
            x_train_tensor = torch.tensor(horse_train_arr, dtype=torch.float32)
            _, z_train_horse_t = ae_horse(x_train_tensor)
            z_train_horse = z_train_horse_t.numpy()
            x_test_tensor = torch.tensor(horse_test_arr, dtype=torch.float32)
            _, z_test_horse_t = ae_horse(x_test_tensor)
            z_test_horse = z_test_horse_t.numpy()
    else:
        z_train_horse = np.zeros((len(train_df), 0))
        z_test_horse  = np.zeros((len(test_df),  0))

    # ----------------------------------------------------
    # 3) AutoEncoder で騎手特徴を圧縮
    # ----------------------------------------------------
    if jockey_train_arr.shape[1] > 0:
        ae_jockey = JockeyFeatureAutoEncoder(input_dim=jockey_train_arr.shape[1], latent_dim=ae_latent_jockey)
        jockey_train_dataset = TensorDataset(torch.tensor(jockey_train_arr, dtype=torch.float32))
        jockey_train_loader  = DataLoader(jockey_train_dataset, batch_size=512, shuffle=False)
        optimizer = torch.optim.Adam(ae_jockey.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.MSELoss()
        ae_jockey.train()
        for epoch in range(num_epochs_ae):
            total_loss = 0
            for (batch_x,) in jockey_train_loader:
                optimizer.zero_grad()
                x_recon, z = ae_jockey(batch_x)
                loss = criterion(x_recon, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[AE-jockey] Epoch {epoch+1}/{num_epochs_ae}  Loss: {total_loss/len(jockey_train_loader):.4f}")
        ae_jockey.eval()
        with torch.no_grad():
            x_train_tensor = torch.tensor(jockey_train_arr, dtype=torch.float32)
            _, z_train_jockey_t = ae_jockey(x_train_tensor)
            z_train_jockey = z_train_jockey_t.numpy()
            x_test_tensor = torch.tensor(jockey_test_arr, dtype=torch.float32)
            _, z_test_jockey_t = ae_jockey(x_test_tensor)
            z_test_jockey = z_test_jockey_t.numpy()
    else:
        z_train_jockey = np.zeros((len(train_df), 0))
        z_test_jockey  = np.zeros((len(test_df),  0))

    # ----------------------------------------------------
    # 4) 上記の圧縮ベクトル + その他数値 + カテゴリ特徴 を結合
    # ----------------------------------------------------
    X_train = np.concatenate([cat_train, other_train_arr, z_train_horse, z_train_jockey], axis=1)
    X_test  = np.concatenate([cat_test,  other_test_arr,  z_test_horse,  z_test_jockey],  axis=1)
    actual_num_dim = other_train_arr.shape[1] + z_train_horse.shape[1] + z_train_jockey.shape[1]
    print(f'X_train shape: {X_train.shape}')
    print(f'X_test  shape: {X_test.shape}')
    print(f'Actual numerical dimensions: {actual_num_dim}')

    # ラベル作成：1着率のみ
    def create_sequences(_df, X):
        ranks = _df[target_col].values  # 着順
        rids = _df[id_col].values       # race_id
        horse_nums = _df["馬番"].values if "馬番" in _df.columns else np.arange(len(_df))
        groups = _df.groupby(id_col)
        max_seq_len = groups.size().max()
        feature_dim = X.shape[1]
        top1 = (ranks == 1).astype(int)
        tar = np.stack([top1], axis=-1)  # (N, 1)
        sequences = []
        labels = []
        masks = []
        race_ids_seq = []
        horse_nums_seq = []
        for unique_rid in _df[id_col].unique():
            idx = np.where(rids == unique_rid)[0]
            feat = X[idx]
            seq_len = len(idx)
            tar_seq = tar[idx]
            rid_array = rids[idx]
            horse_array = horse_nums[idx]
            pad_len = max_seq_len - seq_len
            if pad_len > 0:
                feat = np.vstack([feat, np.zeros((pad_len, feature_dim))])
                pad_label = np.zeros((pad_len, 1), dtype=int)
                tar_seq = np.concatenate([tar_seq, pad_label], axis=0)
                mask = [1]*seq_len + [0]*pad_len
                rid_pad = np.full((pad_len,), fill_value=-1, dtype=rid_array.dtype)
                horse_pad = np.full((pad_len,), fill_value=-1, dtype=horse_array.dtype)
                rid_array = np.concatenate([rid_array, rid_pad])
                horse_array = np.concatenate([horse_array, horse_pad])
            else:
                mask = [1]*seq_len
            sequences.append(feat)
            labels.append(tar_seq)
            masks.append(mask)
            race_ids_seq.append(rid_array)
            horse_nums_seq.append(horse_array)
        return sequences, labels, masks, max_seq_len, race_ids_seq, horse_nums_seq

    train_seq, train_lab, train_mask, max_seq_len_train, train_rids_seq, train_horses_seq = create_sequences(train_df, X_train)
    test_seq, test_lab, test_mask, max_seq_len_test, test_rids_seq, test_horses_seq = create_sequences(test_df, X_test)
    max_seq_len = max(max_seq_len_train, max_seq_len_test)

    def pad_sequences(sequences, labels, masks, rids_seq, horses_seq, seq_len_target):
        feature_dim = sequences[0].shape[1]
        new_seqs = []
        new_labs = []
        new_masks = []
        new_rids = []
        new_horses = []
        for feat, tar, m, r_arr, h_arr in zip(sequences, labels, masks, rids_seq, horses_seq):
            cur_len = len(feat)
            if cur_len < seq_len_target:
                pad_len = seq_len_target - cur_len
                feat = np.vstack([feat, np.zeros((pad_len, feature_dim))])
                pad_label = np.zeros((pad_len, 1), dtype=int)
                tar = np.concatenate([tar, pad_label], axis=0)
                m = m + [0]*pad_len
                rid_pad = np.full((pad_len,), fill_value=-1, dtype=r_arr.dtype)
                h_pad = np.full((pad_len,), fill_value=-1, dtype=h_arr.dtype)
                r_arr = np.concatenate([r_arr, rid_pad])
                h_arr = np.concatenate([h_arr, h_pad])
            new_seqs.append(feat)
            new_labs.append(tar)
            new_masks.append(m)
            new_rids.append(r_arr)
            new_horses.append(h_arr)
        return new_seqs, new_labs, new_masks, new_rids, new_horses

    train_seq, train_lab, train_mask, train_rids_seq, train_horses_seq = pad_sequences(
        train_seq, train_lab, train_mask, train_rids_seq, train_horses_seq, max_seq_len
    )
    test_seq, test_lab, test_mask, test_rids_seq, test_horses_seq = pad_sequences(
        test_seq, test_lab, test_mask, test_rids_seq, test_horses_seq, max_seq_len
    )

    cat_unique = {}
    for c in cat_cols:
        cat_unique[c] = len(train_df[c].unique())

    train_dataset = HorseRaceDataset(train_seq, train_lab, train_mask, train_rids_seq, train_horses_seq)
    test_dataset  = HorseRaceDataset(test_seq,  test_lab,  test_mask,  test_rids_seq,  test_horses_seq)

    return (train_dataset, _, test_dataset,
            cat_cols, cat_unique, max_seq_len, 
            ae_latent_horse,         
            ae_latent_jockey,        
            1,                       # ターゲット数は1に変更
            actual_num_dim, df, cat_cols, num_cols,
            scaler_horse,
            scaler_jockey,
            scaler_other,
            ae_horse if horse_train_arr.shape[1]>0 else None,
            ae_jockey if jockey_train_arr.shape[1]>0 else None,
            X_train.shape[1],
            id_col,
            target_col,
            train_df,
            test_df
        )


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
    ae_latent_horse=50,
    ae_latent_jockey=50,
    test_ratio=0.2,
    d_model=128,
    nhead=8,
    num_layers=6,
    dropout=0.15,
    weight_decay=1e-5,
    patience=10,
    n_splits=5,
):
    (base_train_dataset, _, base_test_dataset,
     cat_cols, cat_unique, 
     max_seq_len, 
     ae_dim_horse, ae_dim_jockey,
     _,
     actual_num_dim, _, _, _,
     scaler_horse, scaler_jockey, scaler_other,
     _, _, _,
     id_col, target_col, df_train_part, test_df) = prepare_data(
        data_path=data_path,
        target_col=target_col,
        pop_col=pop_col,
        id_col=id_col,
        ae_latent_horse=ae_latent_horse,
        ae_latent_jockey=ae_latent_jockey,
        test_ratio=test_ratio,
        valid_ratio=0.0
    )

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    race_ids_sorted = df_train_part.sort_values('date')[id_col].unique()
    chunk_size = len(race_ids_sorted) // (n_splits + 1)
    models = []
    valid_losses_per_model = []
    
    # 今回はLightGBMは使わず、Transformerのみで学習する例
    for i in range(n_splits):
        end_idx = (i+1)*chunk_size
        valid_race_ids = race_ids_sorted[end_idx:]
        train_race_ids = race_ids_sorted[:end_idx]
        if len(train_race_ids) == 0:
            continue
        train_indices = []
        valid_indices = []
        for idx in range(len(base_train_dataset)):
            race_id_tensor = base_train_dataset[idx][3]
            rid_val = race_id_tensor[0].item()
            if rid_val in train_race_ids:
                train_indices.append(idx)
            elif rid_val in valid_race_ids:
                valid_indices.append(idx)
        sub_train_dataset = torch.utils.data.Subset(base_train_dataset, train_indices)
        sub_valid_dataset = torch.utils.data.Subset(base_train_dataset, valid_indices)
        train_loader = DataLoader(sub_train_dataset, batch_size=batch_size, shuffle=False)
        valid_loader = DataLoader(sub_valid_dataset, batch_size=batch_size, shuffle=False)
        model = HorseTransformerSingleHead(
            cat_unique, cat_cols, max_seq_len,
            num_dim=actual_num_dim, d_model=d_model,
            nhead=nhead, num_layers=num_layers, dropout=dropout
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        total_steps = len(train_loader)*num_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                                            optimizer,
                                            max_lr=1e-2,
                                            total_steps=total_steps,
                                            pct_start=0.3,
                                            anneal_strategy='cos'
                                        )
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = float('inf')
        epochs_no_improve = 0
        print(f"\n====== Time-split {i+1}/{n_splits} ======")
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            total_count = 0.0
            for sequences, labels, masks, _, _ in train_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
                optimizer.zero_grad()
                scheduler.step()
                outputs = model(sequences, src_key_padding_mask=~masks)
                loss_raw = criterion(outputs, labels)
                valid_mask = masks.unsqueeze(-1).expand_as(loss_raw)
                loss = (loss_raw * valid_mask).sum() / valid_mask.sum()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_count += 1
            avg_train_loss = total_loss / total_count if total_count > 0 else 0
            model.eval()
            loss_sum = 0.0
            valid_count = 0.0
            with torch.no_grad():
                for sequences, labels, masks, _, _ in valid_loader:
                    sequences = sequences.to(device)
                    labels = labels.to(device)
                    masks = masks.to(device)
                    outputs = model(sequences, src_key_padding_mask=~masks)
                    loss_raw = criterion(outputs, labels)
                    valid_mask = masks.unsqueeze(-1).expand_as(loss_raw)
                    loss_val = (loss_raw * valid_mask).sum() / valid_mask.sum()
                    loss_sum += loss_val.item()
                    valid_count += 1
            avg_valid_loss = loss_sum / valid_count if valid_count > 0 else 0
            print(f"Epoch [{epoch+1}/{num_epochs}]  TrainLoss: {avg_train_loss:.4f}  ValidLoss: {avg_valid_loss:.4f}")
            if avg_valid_loss < best_loss:
                best_loss = avg_valid_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break
        model.load_state_dict(best_model_wts)
        models.append(model)
        valid_losses_per_model.append(best_loss)

    valid_losses_np = np.array(valid_losses_per_model)
    eps = 1e-6
    inv_losses = 1.0 / (valid_losses_np + eps)
    weights_transformer = inv_losses / inv_losses.sum()
    all_models = []
    all_weights  = []
    for i, m in enumerate(models):
        all_models.append(m)
        all_weights.append(weights_transformer[i])
    print("\n=== Models & Weights (based on valid loss) ===")
    for i, (loss_val, w) in enumerate(zip(valid_losses_per_model, weights_transformer)):
        print(f" Model{i+1}: best_valid_loss={loss_val:.5f}, weight={w:.3f}")

    test_loader  = DataLoader(base_test_dataset, batch_size=batch_size, shuffle=False)
    criterion_eval = nn.BCEWithLogitsLoss(reduction='none')

    def test_evaluate_ensemble(loader, models, weights):
        for m in models:
            m.eval()
        loss_sum = 0.0
        valid_count = 0.0
        all_probs_list = []
        all_labels_list = []
        all_rids_list = []
        all_hnums_list = []
        def _predict_proba_transformer(model, sequences, masks):
            logits = model(sequences, src_key_padding_mask=~masks)
            return torch.sigmoid(logits)
        with torch.no_grad():
            for sequences, labels, masks, rids, hnums in loader:
                sequences = sequences.to(device)
                labels    = labels.to(device)
                masks     = masks.to(device)
                ensemble_probs = None
                for w, model_any in zip(weights, models):
                    probs = _predict_proba_transformer(model_any, sequences, masks)
                    if ensemble_probs is None:
                        ensemble_probs = w * probs
                    else:
                        ensemble_probs += w * probs
                ensemble_logits = torch.logit(ensemble_probs.clamp(min=1e-7, max=1-1e-7))
                loss_raw = criterion_eval(ensemble_logits, labels)
                valid_mask = masks.unsqueeze(-1).expand_as(loss_raw)
                loss_sum += (loss_raw * valid_mask).sum().item()
                valid_count += valid_mask.sum().item()
                valid_mask_2d = valid_mask[..., 0].cpu().numpy().astype(bool)
                ensemble_probs_valid = ensemble_probs[valid_mask].view(-1, 1).cpu().numpy()
                labels_valid = labels[valid_mask].view(-1, 1).cpu().numpy()
                rids_np  = rids.numpy()
                hnums_np = hnums.numpy()
                rids_valid   = rids_np[valid_mask_2d].reshape(-1)
                hnums_valid  = hnums_np[valid_mask_2d].reshape(-1)
                all_probs_list.append(ensemble_probs_valid)
                all_labels_list.append(labels_valid)
                all_rids_list.append(rids_valid)
                all_hnums_list.append(hnums_valid)
        avg_loss = loss_sum / valid_count if valid_count > 0 else 0
        all_probs = np.concatenate(all_probs_list, axis=0)
        all_labels = np.concatenate(all_labels_list, axis=0)
        all_rids = np.concatenate(all_rids_list, axis=0)
        all_hnums = np.concatenate(all_hnums_list, axis=0)
        return avg_loss, all_probs, all_labels, all_rids, all_hnums

    test_loss, all_probs, all_trues, all_rids, all_horses = test_evaluate_ensemble(
        test_loader, models, all_weights
    )
    print("Test BCE Logloss (Weighted Ensemble):", f"{test_loss:.4f}")

    pred_df = pd.DataFrame({
        'race_id': all_rids,
        '馬番': all_horses,
        'P_top1': all_probs[:, 0],
        'T_top1': all_trues[:, 0],
    })
    pred_df.to_csv(SAVE_PATH_PRED, index=False)

    from torch.utils.data import ConcatDataset
    full_dataset = ConcatDataset([base_train_dataset, base_test_dataset])
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    full_loss, all_probs_full, all_trues_full, all_rids_full, all_horses_full = \
        test_evaluate_ensemble(full_loader, models, all_weights)
    print("Full BCE Logloss (Weighted Ensemble):", f"{full_loss:.4f}")
    full_pred_df = pd.DataFrame({
        'race_id': all_rids_full,
        '馬番': all_horses_full,
        'P_top1': all_probs_full[:, 0],
        'T_top1': all_trues_full[:, 0],
    })
    full_pred_df.to_csv(SAVE_PATH_FULL_PRED, index=False)

    visualize_predictions_and_return(test_df, pred_df)
    best_model_idx = np.argmin(valid_losses_np)
    best_model = models[best_model_idx]
    with open(SAVE_PATH_MODEL, "wb") as f:
        pickle.dump(best_model.state_dict(), f)
    with open(SAVE_PATH_SCALER_HORSE, "wb") as f:
        pickle.dump(scaler_horse, f)
    with open(SAVE_PATH_SCALER_JOCKEY, "wb") as f:
        pickle.dump(scaler_jockey, f)
    with open(SAVE_PATH_SCALER_OTHER, "wb") as f:
        pickle.dump(scaler_other, f)
    visualize_predictions_and_return(test_df, pred_df)
    print("\n=== Final Ensemble done. Best single model index:", best_model_idx+1)
    return 0


def visualize_predictions_and_return(test_df, pred_df):
    # 1) test_dfとpred_dfをマージ
    merged_df = pd.merge(
        test_df,
        pred_df,
        on=["race_id", "馬番"],
        how="inner"
    )
    merged_df = merged_df.dropna(subset=["単勝"])
    merged_df = merged_df[merged_df["単勝"] > 0].copy()
    n_races = merged_df["race_id"].nunique()

    # 2) 予測値しきい別の購入数と回収率プロット（1着率のみ）
    pred_target_pairs = [("P_top1", "T_top1")]
    thresholds = np.arange(0.0, 1.01, 0.05)
    fig, ax = plt.subplots(figsize=(8,6))
    for pcol, tcol in pred_target_pairs:
        purchase_counts = []
        rois = []
        for th in thresholds:
            sub_df = merged_df[merged_df[pcol] >= th]
            count_ = len(sub_df)
            purchase_counts.append(count_)
            if count_ > 0:
                payoff = (sub_df[tcol] * sub_df["単勝"] * 100).sum()
                cost = count_ * 100
                roi_ = payoff / cost
            else:
                roi_ = 0.0
            rois.append(roi_)
        purchase_counts_per_race = [c / n_races for c in purchase_counts]
        ax.bar(thresholds, purchase_counts_per_race, width=0.03, alpha=0.6, label="購入数", color="steelblue")
        ax.set_xlabel("予測値しきい")
        ax.set_ylabel("購入数/レース数")
        ax2 = ax.twinx()
        ax2.plot(thresholds, rois, marker="o", color="darkorange", label="回収率")
        ax2.set_ylabel("回収率")
        ax.legend(loc="upper left")
    plt.title("予測値しきい別の購入数と回収率")
    plt.tight_layout()
    plt.show()

    # 3) 単勝倍率別のCalibration Curve & ROI Curve（ビンは2.5刻み）
    bins_odds = list(np.arange(0, 30, 2.5)) + [np.inf]
    
    # Calibration Curve
    plt.figure(figsize=(8,6))
    colors = plt.cm.tab10(np.linspace(0,1,len(bins_odds)-1))
    for i in range(len(bins_odds)-1):
        lower = bins_odds[i]
        upper = bins_odds[i+1]
        bin_mask = (merged_df['単勝'] >= lower) & (merged_df['単勝'] < upper)
        if bin_mask.sum() == 0:
            continue
        true_labels = merged_df.loc[bin_mask, 'T_top1'].values
        pred_probs = merged_df.loc[bin_mask, 'P_top1'].values
        prob_true, prob_pred = calibration_curve(true_labels, pred_probs, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', color=colors[i],
                label=f'{lower}〜{upper}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('単勝倍率別 P_top1 の Calibration Curve')
    plt.legend(title='単勝倍率')
    plt.grid(True)
    plt.show()

    # ROI Curve
    plt.figure(figsize=(8,6))
    for i in range(len(bins_odds)-1):
        lower = bins_odds[i]
        upper = bins_odds[i+1]
        bin_mask = (merged_df['単勝'] >= lower) & (merged_df['単勝'] < upper)
        if bin_mask.sum() == 0:
            continue
        sub_df_bin = merged_df[bin_mask]
        roi_list = []
        for th in thresholds:
            df_th = sub_df_bin[sub_df_bin['P_top1'] >= th]
            count_ = len(df_th)
            if count_ > 0:
                payoff = (df_th['T_top1'] * df_th['単勝'] * 100).sum()
                cost = count_ * 100
                roi = payoff / cost
            else:
                roi = np.nan
            roi_list.append(roi)
        plt.plot(thresholds, roi_list, marker='o', label=f'{lower}〜{upper}')
    plt.xlabel('P_top1 のしきい値')
    plt.ylabel('回収率 (ROI)')
    plt.title('単勝倍率別の ROI Curve (P_top1)')
    plt.legend(title='単勝倍率')
    plt.grid(True)
    plt.show()

# 実行例
if __name__ == "__main__":
    run_train_time_split()
