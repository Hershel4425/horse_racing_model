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
import lightgbm as lgb
import seaborn as sns

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
    labels:    (num_races, max_seq_len, 6)
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
    

class CustomTransformerEncoderLayer(nn.Module):
    """
    TransformerEncoderLayerを参考に、
    FeedForwardネットワークを「2層」→「3層 or 4層」などに拡張した例
    """
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = "relu"):
        super().__init__()
        # --- Self-Attention部 ---
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # --- FeedForward部(以下「deep FFN」に拡張) ---
        # 例: 3層構成にしているイメージ
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim_feedforward)
        self.linear3 = nn.Linear(dim_feedforward, d_model)

        # DropoutやLayerNorm
        self.dropout_attn = nn.Dropout(dropout)  # Attentionの残差用
        self.dropout_ffn  = nn.Dropout(dropout)  # FFNの残差用
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 活性化関数
        if activation == "relu":
            self.act_fn = F.relu
        elif activation == "gelu":
            self.act_fn = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, src,
                src_mask=None,
                src_key_padding_mask=None,
                is_causal=False):
        # ----------------------------
        # 1) Self-Attention
        # ----------------------------
        # src: [batch_size, seq_len, d_model]
        src2, _ = self.self_attn(src, src, src,
                                 attn_mask=src_mask,
                                 key_padding_mask=src_key_padding_mask)
        # 残差 + LayerNorm
        src = src + self.dropout_attn(src2)
        src = self.norm1(src)

        # ----------------------------
        # 2) Deep FeedForward Network
        # ----------------------------
        # linear1 -> relu/gelu -> linear2 -> relu/gelu -> linear3
        # 残差 + LayerNorm
        ffn_out = self.linear1(src)
        ffn_out = self.act_fn(ffn_out)
        ffn_out = self.linear2(ffn_out)
        ffn_out = self.act_fn(ffn_out)
        ffn_out = self.linear3(ffn_out)

        src2 = ffn_out
        src = src + self.dropout_ffn(src2)
        src = self.norm2(src)

        return src


class HorseTransformer(nn.Module):
    def __init__(self, cat_unique, cat_cols, max_seq_len,
                 num_dim=50, d_model=128, nhead=8, num_layers=4,
                 dropout=0.1, dim_feedforward=512):
        super().__init__()
        self.feature_embedder = FeatureEmbedder(
            cat_unique, cat_cols, cat_emb_dim=16, num_dim=num_dim, feature_dim=d_model
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)

        # ここでは「同じCustomTransformerEncoderLayerを num_layers 回繰り返す」設定
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

        self.fc_out = nn.Linear(d_model, 6)

    def forward(self, src, src_key_padding_mask=None):
        emb = self.feature_embedder(src)
        emb = self.pos_encoder(emb)
        out = self.transformer_encoder(emb, src_key_padding_mask=src_key_padding_mask)
        logits = self.fc_out(out)
        return logits
    

########################################
# Auto Encorder
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
        # x: (batch_size, input_dim)
        z = self.encoder(x)       # 圧縮ベクトル
        x_recon = self.decoder(z) # 再構築
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
        # x: (batch_size, input_dim)
        z = self.encoder(x)       # 圧縮ベクトル
        x_recon = self.decoder(z) # 再構築
        return x_recon, z
    

# =====================================================
# データ分割と前処理周りの関数
# =====================================================
def split_data(df, id_col="race_id", target_col="着順", test_ratio=0.1, valid_ratio=0.1):
    """
    ※ データを日付順でソートし、前のレースをtrain→testと分割
    why: 未来の情報が混入しないよう、時系列的に分割する目的
    """
    df = df.sort_values('date').reset_index(drop=True)
    race_ids = df[id_col].unique()
    dataset_len = len(race_ids)
    test_cut = int(dataset_len*(1 - test_ratio))
    train_ids = race_ids[:test_cut]
    test_ids  = race_ids[test_cut:]
    train_df  = df[df[id_col].isin(train_ids)].copy()
    test_df   = df[df[id_col].isin(test_ids)].copy()
    # valid_df は空の DataFrame を返す
    valid_df = pd.DataFrame([])
    
    return train_df, valid_df, test_df



def prepare_data(
    data_path,
    target_col="着順",
    pop_col="人気",
    id_col="race_id",
    leakage_cols=None,
    ae_latent_horse=50,   # PCAの次元の代りに使う: 馬のAEの潜在次元
    ae_latent_jockey=50,  # 騎手のAEの潜在次元
    test_ratio=0.1,
    valid_ratio=0.1
):
    """
    データ読み込み・前処理・PCA適用・Dataset作成を行う統合関数
    """
    if leakage_cols is None:
        leakage_cols = [
            '斤量','タイム','着差','単勝','上がり3F','人気',
            'horse_id','jockey_id',
            'trainer_id',
            '馬体重', '増減', '単勝', # 未来データで取得が難しいもの
            '順位点','入線','1着タイム差','先位タイム差','5着着差',
            '1C通過順位','2C通過順位','3C通過順位','4C通過順位','賞金','前半ペース','後半ペース','ペース','ペース_脚質'
            '上がり3F順位', '上がり3F順位_missing',
            '100m','200m','300m','400m','500m','600m','700m','800m','900m','1000m',
            '1100m','1200m','1300m','1400m','1500m','1600m','1700m','1800m','1900m','2000m',
            '2100m','2200m','2300m','2400m','2500m','2600m','2700m','2800m','2900m','3000m',
            '3100m','3200m','3300m','3400m','3500m','3600m','horse_ability'
        ]

    # CSV読み込み
    df = pd.read_csv(data_path, encoding="utf_8_sig")

    # 古いデータを参考にしない
    df = df[df['date'] >= '2017-01-01'].copy()
    print('使用データサイズ：', df.shape)

    # 時系列Split
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

    # カテゴリをcodes化
    for c in cat_cols:
        train_df[c] = train_df[c].astype('category')
        test_df[c] = test_df[c].astype('category')
        train_cat = train_df[c].cat.categories
        test_df[c] = pd.Categorical(test_df[c], categories=train_cat)
        train_df[c] = train_df[c].cat.codes
        test_df[c] = test_df[c].cat.codes

    # ★ ダウンサンプリング ★
    # train_df内の各レースで、勝ち馬（着順==1）の単勝値に基づいて
    # 単勝<=3, 3<単勝<=5, 5<単勝<=10, 10<単勝<=10, 20<単勝 のグループに分け、
    # 最も件数が少ない「20<単勝」の件数に揃える（testは対象外）
    if not train_df.empty:
        winners = train_df[train_df["着順"] == 1].copy()
        conditions = [
            (winners["単勝"] <= 3),
            (winners["単勝"] > 3) & (winners["単勝"] <= 5),
            (winners["単勝"] > 5) & (winners["単勝"] <= 10),
            (winners["単勝"] > 10) & (winners["単勝"] <= 20),
            (winners["単勝"] > 20)
        ]
        bins_labels = ["<=3", "3-5", "5-10", "10-20", ">20"]
        winners["odds_bin"] = np.select(conditions, bins_labels, default="unknown")
        bin_counts = winners["odds_bin"].value_counts()
        target_count = bin_counts.get(">20", 0)
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

    # PCA対象列を抽出（例：馬の芝成績系、騎手の芝成績系など）
    pattern_horse = r'^(競走馬芝|競走馬ダート|単年競走馬芝|単年競走馬ダート)'
    pattern_jockey = r'^(騎手芝|騎手ダート|単年騎手芝|単年騎手ダート)'

    horse_cols  = [c for c in num_cols if re.match(pattern_horse, c)]
    jockey_cols = [c for c in num_cols if re.match(pattern_jockey, c)]
    other_num_cols = [c for c in num_cols if c not in horse_cols + jockey_cols]

    # カテゴリ特徴量(後でconcat)
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
    # HorseFeatureAutoEncoder は別途定義しておく (HorseTransformer等と同じファイルに置いてOK)
    # 今回は trainデータのみで学習する例。必要に応じてtrain+validでも可。
    from torch.utils.data import DataLoader, TensorDataset

    if horse_train_arr.shape[1] > 0:
        # AEインスタンス
        horse_input_dim = horse_train_arr.shape[1]
        ae_horse = HorseFeatureAutoEncoder(input_dim=horse_input_dim, latent_dim=ae_latent_horse)

        # 学習データをTorchDataset化
        horse_train_dataset = TensorDataset(torch.tensor(horse_train_arr, dtype=torch.float32))
        horse_train_loader  = DataLoader(horse_train_dataset, batch_size=512, shuffle=False)

        # 学習
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

        # 推論(Encoder部出力)
        ae_horse.eval()
        with torch.no_grad():
            # train
            x_train_tensor = torch.tensor(horse_train_arr, dtype=torch.float32)
            _, z_train_horse_t = ae_horse(x_train_tensor)
            z_train_horse = z_train_horse_t.numpy()

            # test
            x_test_tensor = torch.tensor(horse_test_arr, dtype=torch.float32)
            _, z_test_horse_t = ae_horse(x_test_tensor)
            z_test_horse = z_test_horse_t.numpy()
    else:
        # 馬の特徴がそもそもない場合
        z_train_horse = np.zeros((len(train_df), 0))
        z_test_horse  = np.zeros((len(test_df),  0))

    # ----------------------------------------------------
    # 3) AutoEncoder で騎手特徴を圧縮 (同様に)
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

        # 推論
        ae_jockey.eval()
        with torch.no_grad():
            # train
            x_train_tensor = torch.tensor(jockey_train_arr, dtype=torch.float32)
            _, z_train_jockey_t = ae_jockey(x_train_tensor)
            z_train_jockey = z_train_jockey_t.numpy()

            # test
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

    # actual_num_dim = (otherの次元) + (馬の潜在次元) + (騎手の潜在次元)
    actual_num_dim = other_train_arr.shape[1] + z_train_horse.shape[1] + z_train_jockey.shape[1]

    print(f'X_train shape: {X_train.shape}')
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

        groups = _df.groupby(id_col)
        max_seq_len = groups.size().max()
        feature_dim = X.shape[1]

        # 各ターゲットを2値化
        top1 = (ranks == 1).astype(int)
        top3 = (ranks <= 3).astype(int)
        top5 = (ranks <= 5).astype(int)
        pop1 = (pops == 1).astype(int)
        pop3 = (pops <= 3).astype(int)
        pop5 = (pops <= 5).astype(int)

        sequences = []
        labels = []
        masks = []
        race_ids_seq = []
        horse_nums_seq = []

        for unique_rid in _df[id_col].unique():
            idx = np.where(rids == unique_rid)[0]
            feat = X[idx]
            seq_len = len(idx)

            tar = np.stack([top1[idx], top3[idx], top5[idx],
                            pop1[idx], pop3[idx], pop5[idx]], axis=-1)  # (seq_len, 6)

            rid_array = rids[idx]
            horse_array = horse_nums[idx]

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
            else:
                mask = [1]*seq_len

            sequences.append(feat)
            labels.append(tar)
            masks.append(mask)
            race_ids_seq.append(rid_array)
            horse_nums_seq.append(horse_array)

        return sequences, labels, masks, max_seq_len, race_ids_seq, horse_nums_seq

    train_seq, train_lab, train_mask, max_seq_len_train, train_rids_seq, train_horses_seq = create_sequences(train_df, X_train)
    test_seq, test_lab, test_mask, max_seq_len_test, test_rids_seq, test_horses_seq = create_sequences(test_df, X_test)

    max_seq_len = max(max_seq_len_train, max_seq_len_test)

    # パディングをそろえるための処理
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
                pad_label = np.zeros((pad_len, 6), dtype=int)
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

    # Dataset化
    train_dataset = HorseRaceDataset(train_seq, train_lab, train_mask, train_rids_seq, train_horses_seq)
    test_dataset  = HorseRaceDataset(test_seq,  test_lab,  test_mask,  test_rids_seq,  test_horses_seq)

    return (train_dataset, _, test_dataset,
            cat_cols, cat_unique, max_seq_len, 
            ae_latent_horse,         # ここは pca_dim_horse の代り
            ae_latent_jockey,        # 同様
            6,                       # ターゲット数
            actual_num_dim, df, cat_cols, num_cols,
            # 今回はスケーラだけ返す例にする
            scaler_horse,
            scaler_jockey,
            scaler_other,
            # AEモデルも返せる
            ae_horse if horse_train_arr.shape[1]>0 else None,
            ae_jockey if jockey_train_arr.shape[1]>0 else None,
            X_train.shape[1],  # 全特徴次元
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
    # 今回のポイント: 複数の時系列Splitをさらに作ってモデルを増やす
    # 例として train-valid を time split し、その区間をずらしながら複数のモデルを作る想定
    # ここでは5回に分割する例
    n_splits=5,
    use_lightgbm=False,
):
    """
    時系列を使った再分割 → 学習 → 各モデルの精度を計測 → 精度に応じて重み付け → 推論平均
    """
    # -------------------------------------------------
    # 1) メインとなるデータを prepare_data で1回読み込み
    #    （全体をさらに段階的に time split して使う）
    # -------------------------------------------------
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
        valid_ratio=0.0 # split_data で valid は作らないようにする
    )

    # デバイス選択
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # -------------------------------------------------
    # 2) train_part(= train_df) をさらに n_splits 回に分割し、
    #    i番目スプリットを valid として残りを train にする時系列分割
    #    (test_df はすでに tail から分割済みなので除外)
    # -------------------------------------------------
    # base_train_dataset は df_train_part がベースになっているが、
    #  ここでさらに race_id 日付順にいくつかに分割して学習する例を簡易実装

    race_ids_sorted = df_train_part.sort_values('date')[id_col].unique()
    chunk_size = len(race_ids_sorted) // (n_splits + 1)

    # コンテナ
    models = []
    valid_losses_per_model = []

    # LightGBM 用に別途モデルと valid_loss を管理する場合
    lgb_models = []
    lgb_valid_losses = []
    
    # 簡易的に multi-label を 6ターゲット同時に学習するため、
    # 下記のように target を展開して各馬行に対するラベル列を作るなど
    # (詳細ロジックは省略し、必要最小限の例のみ示す
    
    def _train_lgb_binary(X_train, y_train, X_val, y_val):
        """
        二値分類用のLightGBMを1つ学習し、valid損失(BCE)を返す簡易例。
        """
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'verbosity': -1,
        }
        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        gbm = lgb.train(params,
                        dtrain,
                        num_boost_round=500,
                        valid_sets=[dtrain, dvalid])
        # validセットでの最終logloss
        best_score = gbm.best_score['valid_1']['binary_logloss']
        return gbm, best_score
    
    def _train_lgb_for_all6(X_train, Y_train, X_val, Y_val):
        """
        6種類のターゲット(Top1,Top3,Top5,Pop1,Pop3,Pop5)を個別に学習してまとめる例。
        返り値: (list_of_gbm, mean_valid_loss)
        """
        g_list = []
        losses = []
        for i in range(6):
            gbm_i, loss_i = _train_lgb_binary(X_train, Y_train[:, i], X_val, Y_val[:, i])
            g_list.append(gbm_i)
            losses.append(loss_i)
        return g_list, float(np.mean(losses))


    for i in range(n_splits):
        # 例: i-th split の開始～終了
        end_idx = (i+1)*chunk_size
        # train用, valid用にレースIDを割り当て（例: 最後のチャンクをvalid相当にする等、いろいろ方法あり）
        # ここでは単純に "最初のi個を除いた部分" を train, "該当のチャンク" を valid としてみる
        # why: time-basedに前方をtrain、後方をvalidationにするなど色々な方法が考えられるが、
        #      ここでは各チャンクを順番にvalidationとして扱う。より本格的な方法では
        #      train: 0～i-1チャンク, valid: iチャンク, test: i+1以降 などの形式も可能。
        valid_race_ids = race_ids_sorted[end_idx:]
        train_race_ids = race_ids_sorted[:end_idx]

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
        total_steps = len(train_loader)*num_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                                            optimizer,
                                            max_lr=1e-2,          # ピーク時の最大LR
                                            total_steps=total_steps,
                                            pct_start=0.3,        # どのタイミングでピークになるか(全体の0.3)
                                            anneal_strategy='cos' # 後半の下がり方(コサイン)
                                        )
        criterion = nn.BCEWithLogitsLoss(reduction='none')

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

            # --- valid ---
            model.eval()
            total_loss_val = 0.0
            total_count_val = 0.0
            with torch.no_grad():
                for sequences, labels, masks, _, _ in valid_loader:
                    sequences = sequences.to(device)
                    labels = labels.to(device)
                    masks = masks.to(device)
                    outputs = model(sequences, src_key_padding_mask=~masks)
                    loss_raw = criterion(outputs, labels)
                    valid_mask = masks.unsqueeze(-1).expand_as(loss_raw)
                    loss_val = (loss_raw * valid_mask).sum() / valid_mask.sum()
                    total_loss_val += loss_val.item()
                    total_count_val += 1
            avg_valid_loss = total_loss_val / total_count_val if total_count_val > 0 else 0

            print(f"Epoch [{epoch+1}/{num_epochs}]  "
                  f"TrainLoss: {avg_train_loss:.4f}  ValidLoss: {avg_valid_loss:.4f}")

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

        # ------------------------------
        # (Optional) LightGBM の学習・検証
        # ------------------------------
        if use_lightgbm:
            # サンプルとして、train/valid データセットから 2次元 (batch*seq, features) に変換し、
            # ラベル(top1~pop5)も同様に展開して学習するイメージ
            def dataset_to_2d(subset):
                """
                与えられた subset(HorseRaceDataset のサブセット) から
                (N, feature_dim), (N,6) を取り出す簡易例
                (マスクされたパディング部分を除外し、一列に詰める)
                """
                arr_X = []
                arr_Y = []
                for seq, lab, m, _, _ in subset:
                    # m=Trueの部分のみ取り出し
                    valid_idx = m.nonzero().squeeze(-1)
                    arr_X.append(seq[valid_idx].numpy())
                    arr_Y.append(lab[valid_idx].numpy())
                X_2d = np.concatenate(arr_X, axis=0)
                Y_2d = np.concatenate(arr_Y, axis=0)
                return X_2d, Y_2d
            X_train_2d, Y_train_2d = dataset_to_2d(sub_train_dataset)
            X_val_2d,   Y_val_2d   = dataset_to_2d(sub_valid_dataset)

            gbms, avg_loss_lgb = _train_lgb_for_all6(X_train_2d, Y_train_2d, X_val_2d, Y_val_2d)
            lgb_models.append(gbms)            # gbms は [gbm_top1, gbm_top3, ..., gbm_pop5] のリスト
            lgb_valid_losses.append(avg_loss_lgb)

    # -------------------------------------------------
    # 3) ここまでで n_splits 個のモデルを取得、各モデルのvalid loss(または精度など)
    #    に応じて重み付きアンサンブルを行う
    #    ※ BCE Loss の場合は「低いほど良い」ので weight = 1/(loss+ε) などにする
    # -------------------------------------------------
    valid_losses_np = np.array(valid_losses_per_model)
    # 万が一0割を防ぐ
    eps = 1e-6
    inv_losses = 1.0 / (valid_losses_np + eps)
    weights_transformer = inv_losses / inv_losses.sum()

    # LightGBM 用の重み
    if use_lightgbm and len(lgb_valid_losses) > 0:
        lgb_losses_np = np.array(lgb_valid_losses)
        inv_losses_lgb = 1.0 / (lgb_losses_np + eps)
        weights_lgb = inv_losses_lgb / inv_losses_lgb.sum()
    else:
        weights_lgb = []

    # 全モデルまとめて重みづけするために配列を合体
    all_models   = []
    all_weights  = []
    for i, m in enumerate(models):
        all_models.append(m)
        all_weights.append(weights_transformer[i])
    if use_lightgbm:
        for i, gbm_list in enumerate(lgb_models):
            all_models.append(gbm_list)  # gbm_list は6モデル同梱
            all_weights.append(weights_lgb[i])

    print("\n=== Models & Weights (based on valid loss) ===")
    for i, (loss_val, w) in enumerate(zip(valid_losses_per_model, weights_transformer)):
        print(f" Model{i+1}: best_valid_loss={loss_val:.5f}, weight={w:.3f}")

    if use_lightgbm:
        for i, (loss_val, w) in enumerate(zip(lgb_valid_losses, weights_lgb)):
            print(f" LGB Model{i+1}: best_valid_loss={loss_val:.5f}, weight={w:.3f}")


    # -------------------------------------------------
    # 4) テストデータでアンサンブル
    # -------------------------------------------------
    test_loader  = DataLoader(base_test_dataset, batch_size=batch_size, shuffle=False)
    criterion_eval = nn.BCEWithLogitsLoss(reduction='none')

    def test_evaluate_ensemble(loader, models, weights):
        """
        各モデルの出力(=logits)をシグモイドしてweighted average -> 損失を計算
        """
        for m in models:
            m.eval()

        import numpy as np
        loss_sum6 = np.zeros(6, dtype=np.float64)
        valid_count6 = np.zeros(6, dtype=np.float64)

        all_probs_list = []
        all_labels_list = []
        all_rids_list = []
        all_hnums_list = []

        def _predict_proba_transformer(model, sequences, masks):
            logits = model(sequences, src_key_padding_mask=~masks)
            return torch.sigmoid(logits)
        
        def _predict_proba_lgb(gbm_list, X_np):
            """
            gbm_list は [gbm_top1, gbm_top3, gbm_top5, gbm_pop1, gbm_pop3, gbm_pop5]
            6モデル分の predict_proba[:,1] を結合して返す
            """
            # shape: (N,6)
            preds = []
            for gbm_i in gbm_list:
                proba_i = gbm_i.predict(X_np)  # 2クラス想定 => shape: (N,) or (N,2)
                # 二値分類の場合 predict_proba が (N,2) になることが多いので注意
                if proba_i.ndim == 2:
                    proba_i = proba_i[:,1]
                preds.append(proba_i.reshape(-1,1))
            return np.concatenate(preds, axis=1)

        with torch.no_grad():
            for sequences, labels, masks, rids, hnums in loader:
                sequences = sequences.to(device)
                labels    = labels.to(device)
                masks     = masks.to(device)
                # 各モデルの予測確率をweighted sum
                ensemble_probs = None
                for w, model_any in zip(weights, models):
                    if isinstance(model_any, HorseTransformer):
                         # Transformerの予測
                         probs = _predict_proba_transformer(model_any, sequences, masks)
                         if ensemble_probs is None:
                             ensemble_probs = w * probs
                         else:
                             ensemble_probs += w * probs

                    elif isinstance(model_any, list):
                        # これは LGB の 6モデルリスト
                        # まずマスクを除去した 2次元で sequences をまとめる
                        # (batch, seq_len, feat) => (valid部分, feat)
                        # パディング部分は無視しないといけないので、後で一括処理。
                        # ここでは簡便化のために、いったん全要素を展開して予測 → 順番を再度詰めるイメージ
                        # ただし、テスト時にはラベルチェックだけでOKなので
                        # 予測確率だけマスク位置に合せて再配置すればよい。
                        # 省略のため、とりあえず全sequenceを一度に縦に並べる例を示す:
                        seq_np = sequences.cpu().numpy()
                        mask_np= masks.cpu().numpy()
                        n_batch, seq_len, feat_dim = seq_np.shape
                        valid_flat = []
                        idx_map = []  # (batch_i, seq_i) -> flat_idx
                        for b in range(n_batch):
                            for s in range(seq_len):
                                if mask_np[b,s]:
                                    idx_map.append((b,s))
                                    valid_flat.append(seq_np[b,s])
                        X_flat = np.stack(valid_flat, axis=0)  # shape (total_valid, feat_dim)

                        prob_lgb = _predict_proba_lgb(model_any, X_flat)  # shape (total_valid, 6)
                        # seq_lenごとに戻す(パディング0は確率0で埋める等)
                        prob_tensor = torch.zeros((n_batch, seq_len, 6), dtype=torch.float32, device=device)
                        for i_pt, (b_i, s_i) in enumerate(idx_map):
                            prob_tensor[b_i, s_i] = torch.from_numpy(prob_lgb[i_pt])

                        if ensemble_probs is None:
                            ensemble_probs = w * prob_tensor
                        else:
                            ensemble_probs += w * prob_tensor

                # 損失計算
                # why: アンサンブル後の確率に対するBCE lossを計算することで、
                #      全モデルの総合的な性能を評価している
                ensemble_logits = torch.logit(ensemble_probs.clamp(min=1e-7, max=1-1e-7))
                loss_raw = criterion_eval(ensemble_logits, labels)
                valid_mask = masks.unsqueeze(-1).expand_as(loss_raw)

                for i in range(6):
                    loss_sum6[i] += (loss_raw[..., i] * valid_mask[..., i]).sum().item()
                    valid_count6[i] += valid_mask[..., i].sum().item()

                # 有効部だけ抽出して保存
                valid_mask_2d = valid_mask[..., 0].cpu().numpy().astype(bool)
                ensemble_probs_valid = ensemble_probs[valid_mask].view(-1, 6).cpu().numpy()
                labels_valid = labels[valid_mask].view(-1, 6).cpu().numpy()
                rids_np  = rids.numpy()
                hnums_np = hnums.numpy()
                rids_valid   = rids_np[valid_mask_2d].reshape(-1)
                hnums_valid  = hnums_np[valid_mask_2d].reshape(-1)

                all_probs_list.append(ensemble_probs_valid)
                all_labels_list.append(labels_valid)
                all_rids_list.append(rids_valid)
                all_hnums_list.append(hnums_valid)

        # 平均損失
        avg_loss_each = loss_sum6 / np.maximum(valid_count6, 1e-15)

        # 全サンプルの予測確率・ラベル・race_id等を結合
        all_probs = np.concatenate(all_probs_list, axis=0)
        all_labels = np.concatenate(all_labels_list, axis=0)
        all_rids = np.concatenate(all_rids_list, axis=0)
        all_hnums = np.concatenate(all_hnums_list, axis=0)

        return avg_loss_each, all_probs, all_labels, all_rids, all_hnums

    test_loss_each6, all_probs, all_trues, all_rids, all_horses = test_evaluate_ensemble(
        test_loader, models, all_weights
    )

    print("Test BCE Logloss each of 6 targets (Weighted Ensemble):",
          [f"{v:.4f}" for v in test_loss_each6])

    # -------------------------------------------------
    # 5) 学習曲線については分割学習の都合上、分割ごとの表示は割愛。
    #    必要なら収集し、折れ線を重ね書き等する。
    #    ここでは最終的にアンサンブルしたテストの Calibration curve を表示する
    # -------------------------------------------------
    # DataFrame化
    pred_df = pd.DataFrame({
        'race_id': all_rids,
        '馬番': all_horses,
        'P_top1': all_probs[:, 0],
        'P_top3': all_probs[:, 1],
        'P_top5': all_probs[:, 2],
        'P_pop1': all_probs[:, 3],
        'P_pop3': all_probs[:, 4],
        'P_pop5': all_probs[:, 5],
        'T_top1': all_trues[:, 0],
        'T_top3': all_trues[:, 1],
        'T_top5': all_trues[:, 2],
        'T_pop1': all_trues[:, 3],
        'T_pop3': all_trues[:, 4],
        'T_pop5': all_trues[:, 5],
    })
    pred_df.to_csv(SAVE_PATH_PRED, index=False)

    #####################################
    # 1) 3つのDatasetをConcatする
    #####################################
    from torch.utils.data import ConcatDataset

    # train, test のデータセットを結合
    full_dataset = ConcatDataset([base_train_dataset, base_test_dataset])

    # すべてまとめた DataLoader
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    #####################################
    # 2) まとめたDatasetで推論する
    #####################################
    # すでに学習済みの models, weights を流用し、
    # test_evaluate_ensemble と同じ要領で予測を実施
    full_loss_each6, all_probs_full, all_trues_full, all_rids_full, all_horses_full = \
        test_evaluate_ensemble(full_loader, models, all_weights)

    print("Full BCE Logloss each of 6 targets (Weighted Ensemble):",
        [f"{v:.4f}" for v in full_loss_each6])

    #####################################
    # 3) 推論結果をDataFrameにまとめて保存
    #####################################
    full_pred_df = pd.DataFrame({
        'race_id': all_rids_full,
        '馬番': all_horses_full,
        'P_top1': all_probs_full[:, 0],
        'P_top3': all_probs_full[:, 1],
        'P_top5': all_probs_full[:, 2],
        'P_pop1': all_probs_full[:, 3],
        'P_pop3': all_probs_full[:, 4],
        'P_pop5': all_probs_full[:, 5],
        'T_top1': all_trues_full[:, 0],
        'T_top3': all_trues_full[:, 1],
        'T_top5': all_trues_full[:, 2],
        'T_pop1': all_trues_full[:, 3],
        'T_pop3': all_trues_full[:, 4],
        'T_pop5': all_trues_full[:, 5],
    })

    full_pred_df.to_csv(SAVE_PATH_FULL_PRED, index=False)


    # キャリブレーション曲線表示
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    target_names = ["Top1", "Top3", "Top5", "Pop1", "Pop3", "Pop5"]
    for i in range(6):
        prob_true, prob_pred = calibration_curve(all_trues[:, i], all_probs[:, i], n_bins=20)
        ax = axes[i]
        ax.plot(prob_pred, prob_true, marker='o', label='Calibration')
        ax.plot([0,1],[0,1], '--', color='gray', label='Perfect')
        ax2 = ax.twinx()
        ax2.hist(all_probs[:, i], bins=40, range=(0,1), alpha=0.3, color='gray')
        ax.set_title(f'Calibration Curve ({target_names[i]})')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('True Probability')
        ax.legend()

    plt.suptitle("Weighted Ensemble Calibration Curves", fontsize=16)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------
    # 6) 代表的なモデル or 全モデルを保存 (ここでは重み付きアンサンブル想定)
    #    最終的に"平均"の概念が強いので、単一モデルとしての保存は必要に応じて
    # -------------------------------------------------
    # 例: 最良のモデル(一番valid_lossが小さいモデル)を保存
    best_model_idx = np.argmin(valid_losses_np)
    best_model = models[best_model_idx]
    with open(SAVE_PATH_MODEL, "wb") as f:
        pickle.dump(best_model.state_dict(), f)
    # with open(SAVE_PATH_PCA_MODEL_HORSE, "wb") as f:
    #     pickle.dump(pca_model_horse, f)
    # with open(SAVE_PATH_PCA_MODEL_JOCKEY, "wb") as f:
    #     pickle.dump(pca_model_jockey, f)
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
    """
    test_df と pred_df を引数にとって、以下の可視化を行う。
    
    1) 6つの予測値（P_top1, P_top3, P_top5, P_pop1, P_pop3, P_pop5）について、
       0から1まで0.05刻みで「予測値がX以上の馬をすべて買った場合の回収率」と
       「購入数」を二重軸グラフでプロット

    2) 着順予測(P_topX)と人気予測(P_popX) について、それぞれ (x=着順予測, y=人気予測) の
       2次元ヒートマップを作成。
       ・ヒートマップ1: そのビンに該当するデータ数
       ・ヒートマップ2: そのビンに該当する馬をすべて買った時の回収率

    【前提】
    - 単勝オッズを '単勝' 列として持っていることを想定
    - 回収率は単勝馬券(1着的中時)の払い戻し想定で計算
      （例えば P_top3 で的中しても同じく「単勝」で計算しちゃうので、実際の馬券とは違うから注意ね）
    """
    #-------------------------------------------------
    # 1) test_df と pred_df を結合して作業用の DataFrame を作る
    #    （"race_id" と "馬番" で紐づけ）
    #-------------------------------------------------
    merged_df = pd.merge(
        test_df,
        pred_df,
        on=["race_id", "馬番"],
        how="inner"
    )
    # 念のため、単勝オッズが欠損なら除外 or 0埋めなど
    merged_df = merged_df.dropna(subset=["単勝"])
    # 単勝オッズが0だと回収率の計算ができないので、0を含む行があれば除く
    # (もしオッズ=0 があり得ないなら気にしなくていい)
    merged_df = merged_df[merged_df["単勝"] > 0].copy()

    n_races = merged_df["race_id"].nunique()

    #-------------------------------------------------
    # 2) しきい値を変えたときの回収率と購入数をグラフ化
    #-------------------------------------------------
    pred_target_pairs = [
        ("P_top1", "T_top1"),
        ("P_top3", "T_top1"),
        ("P_top5", "T_top1"),
        ("P_pop1", "T_pop1"),
        ("P_pop3", "T_pop1"),
        ("P_pop5", "T_pop1")
    ]
    thresholds = np.arange(0.0, 1.01, 0.05)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("予測値しきい別の購入数と回収率", fontsize=16)

    for i, (pcol, tcol) in enumerate(pred_target_pairs):
        row_idx = i // 3
        col_idx = i % 3
        ax = axes[row_idx, col_idx]

        # 結果を記録するリスト
        purchase_counts = []
        rois = []

        for th in thresholds:
            # しきい値を超える行を購入
            sub_df = merged_df[merged_df[pcol] >= th]
            count_ = len(sub_df)
            purchase_counts.append(count_)
            purchase_counts_per_race = [c / n_races for c in purchase_counts]


            if count_ > 0:
                # （単勝馬券前提で）的中したら sub_df["単勝"]倍の払い戻し
                # ここでは tcol=1 の行のみ的中とみなす
                payoff = (sub_df[tcol] * sub_df["単勝"] * 100).sum()  # 購入額は100円想定
                cost = count_ * 100
                roi_ = payoff / cost
            else:
                roi_ = 0.0
            rois.append(roi_)

        # 購入数の棒グラフ
        ax.bar(thresholds, purchase_counts_per_race, width=0.03, alpha=0.6, label="購入数", color="steelblue")
        ax.set_ylabel("購入数/レース数")
        ax.set_xlabel("予測値しきい")
        ax.set_ylim([0, max(purchase_counts_per_race)*1.1 if len(purchase_counts_per_race) > 0 else 1])

        # 回収率の折れ線グラフ（twinx）
        ax2 = ax.twinx()
        ax2.plot(thresholds, rois, marker="o", color="darkorange", label="回収率")
        ax2.set_ylim([0, max(rois)*1.2 if len(rois) > 0 else 1])
        ax2.set_ylabel("回収率")

        # 的中率 ( = 的中馬数 / 購入馬数 )
        # ここで tcol=1 の馬だけ数えて「的中数」を出す
        accs = []
        for th in thresholds:
            sub_df = merged_df[merged_df[pcol] >= th]
            count_ = len(sub_df)
            if count_ > 0:
                # tcol=1 になってる馬の数
                hit_count = sub_df[tcol].sum()
                acc_ = hit_count / count_
            else:
                acc_ = 0
            accs.append(acc_)

        ax2.plot(thresholds, accs, marker="x", color="green", label="的中率")

        ax2.legend(loc="upper left")  # 凡例の位置は適当に

        ax.set_title(pcol)
        # 凡例をまとめて表示したいなら工夫が必要だけど、とりあえず省略する

    plt.tight_layout()
    plt.show()


    # --- 単勝倍率別のCalibration Curve をプロットする ---
    # 単勝倍率のビン設定：0〜5, 5〜10, 10〜15, 15〜20, 20以上
    bins_odds = [0, 5, 10, 15, 20, np.inf]
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    plt.figure(figsize=(8, 6))

    for i in range(len(bins_odds)-1):
        lower = bins_odds[i]
        upper = bins_odds[i+1]
        # 該当ビンのデータを抽出
        bin_mask = (merged_df['単勝'] >= lower) & (merged_df['単勝'] < upper)
        if bin_mask.sum() == 0:
            continue  # 該当データがなければスキップするわ
        true_labels = merged_df.loc[bin_mask, 'T_top1'].values
        pred_probs = merged_df.loc[bin_mask, 'P_top1'].values

        # キャリブレーションカーブの算出（n_binsは適宜調整してね）
        prob_true, prob_pred = calibration_curve(true_labels, pred_probs, n_bins=20)
        
        plt.plot(prob_pred, prob_true, marker='o', color=colors[i % len(colors)],
                label=f'{lower}〜{upper}')

    # 完璧なキャリブレーションの基準線
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')

    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Odds別 P_top1 の Calibration Curve')
    plt.legend(title='単勝倍率')
    plt.grid(True)
    plt.show()


    # --- 単勝倍率別の ROI Curve をプロットする ---
    # ROIの計算: 各しきい値で、P_top1がその値以上の馬を購入した場合、
    #               購入馬数×100円をコストとし、T_top1が1の場合に単勝倍率×100円の払い戻しとする例よ。

    bins_odds = [0, 5, 10, 15, 20, np.inf]  # 単勝倍率のビン
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    thresholds = np.arange(0.0, 1.01, 0.05)  # P_top1のしきい値

    plt.figure(figsize=(8, 6))

    for i in range(len(bins_odds) - 1):
        lower = bins_odds[i]
        upper = bins_odds[i + 1]
        
        # 該当ビンのデータを抽出
        bin_mask = (merged_df['単勝'] >= lower) & (merged_df['単勝'] < upper)
        if bin_mask.sum() == 0:
            continue  # 該当データがなければスキップするわ
        sub_df_bin = merged_df[bin_mask]
        
        roi_list = []
        for th in thresholds:
            # しきい値 th 以上の馬を抽出
            df_th = sub_df_bin[sub_df_bin['P_top1'] >= th]
            count_ = len(df_th)
            if count_ > 0:
                # 仮に1馬あたり100円の購入とする
                payoff = (df_th['T_top1'] * df_th['単勝'] * 100).sum()
                cost = count_ * 100
                roi = payoff / cost
            else:
                roi = np.nan  # 購入馬がなければNaNにするか、0にしてもいい感じね
            roi_list.append(roi)
        
        plt.plot(thresholds, roi_list, marker='o', color=colors[i % len(colors)],
                label=f'{lower}〜{upper}')

    plt.xlabel('P_top1 のしきい値')
    plt.ylabel('回収率 (ROI)')
    plt.title('単勝倍率別の ROI Curve (P_top1)')
    plt.legend(title='単勝倍率')
    plt.grid(True)
    plt.show()


    # --- 人気と着順の予測の差分を用いる ---

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("予測勝率と予測人気の差に対する購入数と回収率", fontsize=16)

    n_races = merged_df["race_id"].nunique()

    for i, N in enumerate([1, 3, 5]):
        diff_col = merged_df[f"P_top{N}"] - merged_df[f"P_pop{N}"]
        bins = np.arange(-1.0, 1.05, 0.05)
        labels = (bins[:-1] + bins[1:]) / 2  # ビンの中央値をX軸用に

        merged_df["diff_bin"] = pd.cut(diff_col, bins=bins, right=False)

        group = merged_df.groupby("diff_bin")
        count_ = group.size()  # 各ビンの馬数
        payoff_ = group.apply(lambda g: (g["T_top1"] * g["単勝"] * 100).sum())  # T_top1で計算するならこんなイメージ
        cost_ = count_ * 100
        roi_ = payoff_ / cost_

        # 各ビンの購入数を「レース数」で割る
        purchase_counts_per_race = count_ / n_races

        ax = axes[i]

        # 購入数のバーグラフ
        ax.bar(labels, purchase_counts_per_race, width=0.08, alpha=0.6, color="steelblue")
        ax.set_ylabel("購入数/レース数")
        ax.set_xlabel("P_top{N} - P_pop{N} (ビン中央値)")
        ax.set_ylim([0, purchase_counts_per_race.max() * 1.2 if len(purchase_counts_per_race) > 0 else 1])

        # 回収率の折れ線グラフ（twinx）
        ax2 = ax.twinx()
        ax2.plot(labels, roi_, marker="o", color="darkorange", label="回収率")
        ax2.set_ylim([0, roi_.max() * 1.2 if len(roi_) > 0 else 1])
        ax2.set_ylabel("回収率")

        ax.set_title(f"(P_top{N} - P_pop{N}) vs 回収率")

    plt.tight_layout()
    plt.show()

    #-------------------------------------------------
    # 3) 着順予測(P_topX)と人気予測(P_popX)の
    #    2次元ヒートマップ（データ数＆回収率）
    #-------------------------------------------------
    # ビン定義（0～1を0.1刻み）
    bins = np.arange(0.0, 1.1, 0.1)

    # 3ペア分: (P_top1, P_pop1, T_top1), (P_top3, P_pop3, T_top1), (P_top5, P_pop5, T_top1)
    top_pop_pairs = [
        ("P_top1", "P_pop1", "T_top1"),
        ("P_top3", "P_pop3", "T_top1"),
        ("P_top5", "P_pop5", "T_top1"),
    ]

    fig2, axes2 = plt.subplots(3, 2, figsize=(12, 18))
    fig2.suptitle("着順予測×人気予測のヒートマップ", fontsize=16)

    for i, (ptop, ppop, tcol) in enumerate(top_pop_pairs):
        # i行目のaxes2
        ax_count = axes2[i, 0]
        ax_roi = axes2[i, 1]

        # x-bin, y-bin
        x_bin = pd.cut(merged_df[ptop], bins=bins, right=False, include_lowest=True)
        y_bin = pd.cut(merged_df[ppop], bins=bins, right=False, include_lowest=True)

        group = merged_df.groupby([x_bin, y_bin])

        # データ数ヒートマップ用
        count_table = group.size().unstack(fill_value=0)

        # 回収率ヒートマップ用
        # payoff は tcol=1 なら "単勝"×100、cost は group.size()*100
        sum_payoff = group.apply(lambda g: (g[tcol] * g["単勝"] * 100).sum())
        sum_payoff_table = sum_payoff.unstack(fill_value=0)

        count_table_ = count_table.reindex(index=count_table.index[::-1])  # y軸を上が大きい順に描くかどうか
        sum_payoff_table_ = sum_payoff_table.reindex(index=sum_payoff_table.index[::-1])

        # cost = count_table * 100
        # → 2次元の同じ shape で回収率 = sum_payoff / cost
        cost_table = count_table_ * 100
        roi_table = sum_payoff_table_ / cost_table
        roi_table = roi_table.fillna(0)

        # データ数ヒートマップ
        sns.heatmap(count_table_, ax=ax_count, annot=True, fmt="d", cmap="Blues")
        ax_count.set_title(f"データ数 ({ptop} vs {ppop})")
        ax_count.set_xlabel(ppop)
        ax_count.set_ylabel(ptop)

        # 回収率ヒートマップ
        sns.heatmap(roi_table, ax=ax_roi, annot=True, fmt=".2f", cmap="RdYlGn")
        ax_roi.set_title(f"回収率 ({ptop} vs {ppop})")
        ax_roi.set_xlabel(ppop)
        ax_roi.set_ylabel(ptop)

        # 軸ラベルがビン名(object型)になってて長いと邪魔だから、軸カテゴリを短くする
        # お好みで書式を整えてね
        # xとyに対して同様にやる
        x_labels = [str(lb) for lb in count_table_.columns]
        ax_count.set_xticklabels(x_labels, rotation=45, ha='right')
        ax_roi.set_xticklabels(x_labels, rotation=45, ha='right')

        y_labels = [str(lb) for lb in count_table_.index]
        ax_count.set_yticklabels(y_labels, rotation=0)
        ax_roi.set_yticklabels(y_labels, rotation=0)

    plt.tight_layout()
    plt.show()