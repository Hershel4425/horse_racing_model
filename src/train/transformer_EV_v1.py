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
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
import copy
import pickle
import random

# 乱数固定(再現性確保)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ファイルパス設定（必要に応じて変更してください）
ROOT_PATH = "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/153_競馬_v3"
DATE_STRING = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
MODEL_SAVE_DIR = os.path.join(ROOT_PATH, f"models/transformer_diffBCE/{DATE_STRING}")

DATA_PATH = os.path.join(ROOT_PATH, "data/02_features/feature.csv")
PRED_PATH  = os.path.join(ROOT_PATH, "result/predictions/transformer/20250125194112_full.csv")
SAVE_PATH_PRED      = os.path.join(ROOT_PATH, f"result/predictions/transformer_diffBCE/{DATE_STRING}.csv")
SAVE_PATH_FULL_PRED = os.path.join(ROOT_PATH, f"result/predictions/transformer_diffBCE/{DATE_STRING}_full.csv")
SAVE_PATH_MODEL     = os.path.join(MODEL_SAVE_DIR, "model_diffBCE.pickle")

SAVE_PATH_PCA_MODEL_HORSE = os.path.join(MODEL_SAVE_DIR, "pcamodel_horse_diffBCE.pickle")
SAVE_PATH_PCA_MODEL_JOCKEY= os.path.join(MODEL_SAVE_DIR, "pcamodel_jockey_diffBCE.pickle")
SAVE_PATH_SCALER_HORSE    = os.path.join(MODEL_SAVE_DIR, "scaler_horse_diffBCE.pickle")
SAVE_PATH_SCALER_JOCKEY   = os.path.join(MODEL_SAVE_DIR, "scaler_jockey_diffBCE.pickle")
SAVE_PATH_SCALER_OTHER    = os.path.join(MODEL_SAVE_DIR, "scaler_other_diffBCE.pickle")

########################################
# 1) データセットクラス
########################################
class HorseRaceDatasetDiffBCE(Dataset):
    """
    本サンプルでは
     - sequences: (num_races, max_seq_len, feature_dim)
       → 特徴量には「支持率」を含む
     - labels:    (num_races, max_seq_len) → 1着なら1、そうでなければ0
     - base_supports: (num_races, max_seq_len) → 各馬の支持率(0～1)
     - masks:     (num_races, max_seq_len)     → True=実データ, False=パディング
     - race_ids:  (num_races, max_seq_len)
     - horse_nums:(num_races, max_seq_len)
    """
    def __init__(self, sequences, labels, base_supports, masks, race_ids, horse_nums):
        self.sequences    = sequences      # (N, max_seq_len, feature_dim)
        self.labels       = labels         # (N, max_seq_len)
        self.base_supports= base_supports  # (N, max_seq_len)
        self.masks        = masks          # (N, max_seq_len)
        self.race_ids     = race_ids
        self.horse_nums   = horse_nums

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx],      dtype=torch.float32)
        lab = torch.tensor(self.labels[idx],         dtype=torch.float32)
        sup = torch.tensor(self.base_supports[idx],  dtype=torch.float32)
        m   = torch.tensor(self.masks[idx],          dtype=torch.bool)
        rid = torch.tensor(self.race_ids[idx],       dtype=torch.long)
        hn  = torch.tensor(self.horse_nums[idx],     dtype=torch.long)
        return seq, lab, sup, m, rid, hn


########################################
# 2) Embedding + Transformer モデル
########################################
class FeatureEmbedder(nn.Module):
    """
    ここは元のEmbedding処理と同様。
    カテゴリ列をEmbedding、数値列をLinearで変換して concat し、
    d_model次元にマッピングする。
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
        """
        x: (batch, seq_len, feature_dim)
        先頭 cat_len 個はカテゴリ列、それ以降は数値列
        """
        cat_len = len(self.cat_cols)
        cat_x = x[..., :cat_len].long()
        num_x = x[..., cat_len:]

        embs = []
        for i, c in enumerate(self.cat_cols):
            embs.append(self.emb_layers[c](cat_x[..., i]))

        cat_emb = torch.cat(embs, dim=-1)  # (batch, seq_len, sum_of_cat_emb_dim)
        num_emb = self.num_linear(num_x)   # (batch, seq_len, num_dim)

        out = torch.cat([cat_emb, num_emb], dim=-1)
        out = self.out_linear(out)         # (batch, seq_len, d_model)
        return out

class PositionalEncoding(nn.Module):
    """
    Transformerで位置情報を付与するための Positional Encoding
    """
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))

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

class HorseTransformerDiffBCE(nn.Module):
    """
    最終出力 = 1次元 (＝ベース支持率に加算する「差分」)
    """
    def __init__(self,
                 cat_unique, cat_cols,
                 max_seq_len,
                 num_dim=50,
                 d_model=128,
                 nhead=4,
                 num_layers=4,
                 dropout=0.1):
        super().__init__()

        self.feature_embedder = FeatureEmbedder(
            cat_unique=cat_unique,
            cat_cols=cat_cols,
            cat_emb_dim=16,
            num_dim=num_dim,
            feature_dim=d_model
        )

        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # → 差分を1次元出力（例: shape=(batch, seq_len, 1)）
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, src, src_key_padding_mask=None):
        """
        src: (batch, seq_len, feature_dim)
        src_key_padding_mask: (batch, seq_len)  True=実データ or False=pad(除外)
        """
        emb = self.feature_embedder(src)
        emb = self.pos_encoder(emb)
        out = self.transformer_encoder(emb, src_key_padding_mask=~src_key_padding_mask)
        diff = self.fc_out(out)  # shape: (batch, seq_len, 1)
        return diff
    

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


########################################
# 3) データ準備用の例関数
########################################
def split_data(df, id_col="race_id", test_ratio=0.1, valid_ratio=0.1):
    """
    日付順でsplit
    """
    df = df.sort_values('date').reset_index(drop=True)
    race_ids = df[id_col].unique()
    dataset_len = len(race_ids)

    test_cut  = int(dataset_len * (1 - test_ratio))
    valid_cut = int(test_cut * (1 - valid_ratio))

    train_ids = race_ids[:valid_cut]
    valid_ids = race_ids[valid_cut:test_cut]
    test_ids  = race_ids[test_cut:]

    train_df = df[df[id_col].isin(train_ids)].copy()
    valid_df = df[df[id_col].isin(valid_ids)].copy()
    test_df  = df[df[id_col].isin(test_ids)].copy()
    return train_df, valid_df, test_df

def prepare_data_diff_bce(
    data_path=DATA_PATH,
    pred_path=PRED_PATH,
    id_col="race_id",
    rank_col="着順",   # 勝ち(1着)を表す列
    pop_col="単勝", # ベース支持率(人気=1番人気での支持率相当)
    test_ratio=0.1,
    valid_ratio=0.1,
    ae_latent_horse=50,
    ae_latent_jockey=50
):
    """
    例: 
     1) feature.csv を読み込む
     2) pred.csv には "P_pop1" 等が入っていると仮定し、マージして「支持率」列を持つ
     3) 1着なら label=1, それ以外=0
     4) Sequence化
     5) HorseRaceDatasetDiffBCE で Dataset作成
    """
    # 1) CSV読み込み
    df_feature = pd.read_csv(data_path, encoding='utf-8-sig')
    df_pred    = pd.read_csv(pred_path, encoding='utf-8-sig')  # 例: ここにP_pop1列などがある想定

    # マージ: race_id, 馬番 がキー
    df = pd.merge(
        df_feature,
        df_pred[["race_id","馬番","P_top1","P_pop1","P_top3","P_pop3","P_top5","P_pop5"]],  # 必要な列だけ
        on=["race_id","馬番"],
        how="inner"
    )

    # 勝ったかどうか(1/0)
    df["is_win"] = (df[rank_col] == 1).astype(int)
    # 支持率を0～1に正規化した列(すでに0～1になっているならそのまま)
    # ここでは P_pop1 は「1番人気になっているかどうか(0/1)」ではなく、実際の支持率と仮定
    df["base_support"] = 0.8 / (df[pop_col] + 1e-10)
    

    # 時系列split
    train_df, valid_df, test_df = split_data(df, id_col=id_col,
                                             test_ratio=test_ratio,
                                             valid_ratio=valid_ratio)

    # ----- 以下、カテゴリ変換やPCAなどはお好みで実施 ------
    # (サンプルでは最低限の処理のみ記載)

    cat_cols_all = train_df.select_dtypes(exclude=[np.number]).columns.tolist()
    num_cols_all = train_df.select_dtypes(include=[np.number]).columns.tolist()

    # ID系・リーク系除外
    # 必要に応じてleakage_colsを定義
    leakage_cols = [
            '斤量','タイム','着差',
            '単勝',
            '上がり3F','馬体重','人気','着順',
            'horse_id','jockey_id',
            'trainer_id',
            '順位点','入線','1着タイム差','先位タイム差','5着着差','増減',
            '1C通過順位','2C通過順位','3C通過順位','4C通過順位','賞金','前半ペース','後半ペース','ペース',
            '上がり3F順位','100m','200m','300m','400m','500m','600m','700m','800m','900m','1000m',
            '1100m','1200m','1300m','1400m','1500m','1600m','1700m','1800m','1900m','2000m',
            '2100m','2200m','2300m','2400m','2500m','2600m','2700m','2800m','2900m','3000m',
            '3100m','3200m','3300m','3400m','3500m','3600m','horse_ability',
            'is_win',
            'base_support'
        ]
    cat_cols = [c for c in cat_cols_all if c not in leakage_cols and c not in [id_col]]
    num_cols = [c for c in num_cols_all if c not in leakage_cols and c not in [id_col]]

    # カテゴリ欠損埋め
    for c in cat_cols:
        for d in [train_df, valid_df, test_df]:
            d[c] = d[c].fillna("missing").astype(str)

    # 数値欠損埋め
    for n in num_cols:
        for d in [train_df, valid_df, test_df]:
            d[n] = d[n].fillna(0)

    # カテゴリ→codes
    for c in cat_cols:
        train_df[c] = train_df[c].astype('category')
        valid_df[c] = valid_df[c].astype('category')
        test_df[c]  = test_df[c].astype('category')
        base_cat = train_df[c].cat.categories
        valid_df[c] = pd.Categorical(valid_df[c], categories=base_cat)
        test_df[c]  = pd.Categorical(test_df[c], categories=base_cat)
        train_df[c] = train_df[c].cat.codes.replace(-1, 0)
        valid_df[c] = valid_df[c].cat.codes.replace(-1, 0)
        test_df[c]  = test_df[c].cat.codes.replace(-1, 0)

    # 対象探し(例)
    pattern_horse = r'^(競走馬芝|競走馬ダート)'
    pattern_jockey= r'^(騎手芝|騎手ダート)'

    # この名前はpca_pattern_*になってるけど、使い回しOK
    horse_cols   = [c for c in num_cols if re.match(pattern_horse, c)]
    jockey_cols  = [c for c in num_cols if re.match(pattern_jockey, c)]
    other_num_cols = [c for c in num_cols if c not in (horse_cols + jockey_cols)]

    # 5) 標準化 (馬・騎手・その他)
    # ---------------------------------------------------------------------
    scaler_horse  = StandardScaler()
    scaler_jockey = StandardScaler()
    scaler_other  = StandardScaler()

    # Fit
    if len(horse_cols) > 0:
        scaler_horse.fit(train_df[horse_cols])
    if len(jockey_cols) > 0:
        scaler_jockey.fit(train_df[jockey_cols])
    if len(other_num_cols) > 0:
        scaler_other.fit(train_df[other_num_cols])

    # Transform
    horse_train_arr = scaler_horse.transform(train_df[horse_cols])   if len(horse_cols)>0 else np.zeros((len(train_df),0))
    horse_valid_arr = scaler_horse.transform(valid_df[horse_cols])   if len(horse_cols)>0 else np.zeros((len(valid_df),0))
    horse_test_arr  = scaler_horse.transform(test_df[horse_cols])    if len(horse_cols)>0 else np.zeros((len(test_df),0))

    jockey_train_arr= scaler_jockey.transform(train_df[jockey_cols]) if len(jockey_cols)>0 else np.zeros((len(train_df),0))
    jockey_valid_arr= scaler_jockey.transform(valid_df[jockey_cols]) if len(jockey_cols)>0 else np.zeros((len(valid_df),0))
    jockey_test_arr = scaler_jockey.transform(test_df[jockey_cols])  if len(jockey_cols)>0 else np.zeros((len(test_df),0))

    other_train_arr = scaler_other.transform(train_df[other_num_cols]) if len(other_num_cols)>0 else np.zeros((len(train_df),0))
    other_valid_arr = scaler_other.transform(valid_df[other_num_cols]) if len(other_num_cols)>0 else np.zeros((len(valid_df),0))
    other_test_arr  = scaler_other.transform(test_df[other_num_cols])  if len(other_num_cols)>0 else np.zeros((len(test_df),0))


    # 6) ★ここで馬の数値特徴に対する AutoEncoder を学習 (Encoderだけ使う)
    # ---------------------------------------------------------------------
    # もし馬特徴がないならスキップ
    if len(horse_cols) > 0:
        # AutoEncoderインスタンス
        horse_input_dim = horse_train_arr.shape[1]  # 特徴次元
        ae_horse = HorseFeatureAutoEncoder(input_dim=horse_input_dim, latent_dim=ae_latent_horse)

        # 学習用データ (trainのみ or train+valid でもOK)
        horse_dataset = torch.tensor(horse_train_arr, dtype=torch.float32)
        horse_loader = torch.utils.data.DataLoader(horse_dataset, batch_size=512, shuffle=True)

        optimizer = torch.optim.Adam(ae_horse.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.MSELoss()

        # 学習
        ae_horse.train()
        num_epochs = 10
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_x in horse_loader:
                optimizer.zero_grad()
                x_recon, z = ae_horse(batch_x)
                loss = criterion(x_recon, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[AE-horse] Epoch {epoch+1}/{num_epochs}  Loss: {total_loss/len(horse_loader):.4f}")

        # 推論 (Train, Valid, Testを Encoder に通して潜在ベクトル z を得る)
        ae_horse.eval()
        with torch.no_grad():
            # train
            train_torch = torch.tensor(horse_train_arr, dtype=torch.float32)
            _, z_train_horse = ae_horse(train_torch)
            z_train_horse = z_train_horse.numpy()

            # valid
            valid_torch = torch.tensor(horse_valid_arr, dtype=torch.float32)
            _, z_valid_horse = ae_horse(valid_torch)
            z_valid_horse = z_valid_horse.numpy()

            # test
            test_torch = torch.tensor(horse_test_arr, dtype=torch.float32)
            _, z_test_horse = ae_horse(test_torch)
            z_test_horse = z_test_horse.numpy()
    else:
        # 馬特徴がなければ空配列
        z_train_horse = np.zeros((len(train_df), 0))
        z_valid_horse = np.zeros((len(valid_df), 0))
        z_test_horse  = np.zeros((len(test_df),  0))

    # 7) 騎手特徴も同様にAE化していいし、あるいはPCAのままでもOK
    # ---------------------------------------------------------------------
    # 以下、ざっくり同じノリで "HorseFeatureAutoEncoder" じゃなくて
    # "JockeyFeatureAutoEncoder" を用意してもいいし、
    # 同じクラスでもOK (入力dim違うだけ)。
    # ここでは省略例:
    if len(jockey_cols) > 0:
        # 同じ仕組みで学習 & 変換
        jockey_input_dim = jockey_train_arr.shape[1]
        ae_jockey = JockeyFeatureAutoEncoder(input_dim=jockey_input_dim, latent_dim=ae_latent_jockey)

        # 学習用データ (trainのみ or train+valid でもOK)
        jockey_dataset = torch.tensor(jockey_train_arr, dtype=torch.float32)
        jockey_loader = torch.utils.data.DataLoader(jockey_dataset, batch_size=512, shuffle=True)

        optimizer = torch.optim.Adam(ae_jockey.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.MSELoss()

        # 学習
        ae_jockey.train()
        num_epochs = 10
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_x in jockey_loader:
                optimizer.zero_grad()
                x_recon, z = ae_jockey(batch_x)
                loss = criterion(x_recon, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[AE-jockey] Epoch {epoch+1}/{num_epochs}  Loss: {total_loss/len(jockey_loader):.4f}")

        # 推論 (Train, Valid, Testを Encoder に通して潜在ベクトル z を得る)
        ae_jockey.eval()
        with torch.no_grad():
            # train
            train_torch = torch.tensor(jockey_train_arr, dtype=torch.float32)
            _, z_train_jockey = ae_jockey(train_torch)
            z_train_jockey = z_train_jockey.numpy()

            # valid
            valid_torch = torch.tensor(jockey_valid_arr, dtype=torch.float32)
            _, z_valid_jockey = ae_jockey(valid_torch)
            z_valid_jockey = z_valid_jockey.numpy()

            # test
            test_torch = torch.tensor(jockey_test_arr, dtype=torch.float32)
            _, z_test_jockey = ae_jockey(test_torch)
            z_test_jockey = z_test_jockey.numpy()
    else:
        z_train_jockey = np.zeros((len(train_df), 0))
        z_valid_jockey = np.zeros((len(valid_df), 0))
        z_test_jockey  = np.zeros((len(test_df),  0))

    # 8) AE結果を合わせて最終Xを作る (cat_features + other + AE-jockey + AE-jockey)
    # ---------------------------------------------------------------------
    def concat_features(df_part, cat_cols, other_arr, z_horse_arr, z_jockey_arr):
        cat_val = df_part[cat_cols].values if len(cat_cols)>0 else np.zeros((len(df_part),0))
        return np.concatenate([cat_val, other_arr, z_horse_arr, z_jockey_arr], axis=1)

    X_train = concat_features(train_df, cat_cols, other_train_arr, z_train_jockey, z_train_jockey)
    X_valid = concat_features(valid_df, cat_cols, other_valid_arr, z_valid_jockey, z_valid_jockey)
    X_test  = concat_features(test_df, cat_cols, other_test_arr,  z_test_jockey,  z_test_jockey)


    # ラベル: is_win (1/0)
    y_train = train_df["is_win"].values
    y_valid = valid_df["is_win"].values
    y_test  = test_df["is_win"].values

    # ベース支持率
    sup_train = train_df["base_support"].values
    sup_valid = valid_df["base_support"].values
    sup_test  = test_df["base_support"].values

    # Sequence 化 (同レース内でパディング揃え)
    def create_sequences(_df, X, y, base_support):
        rids = _df[id_col].values
        horses= _df["馬番"].values if "馬番" in _df.columns else np.arange(len(_df))

        groups = _df.groupby(id_col)
        max_seq_len = groups.size().max()
        feat_dim = X.shape[1]

        seq_list, label_list, sup_list, mask_list = [], [], [], []
        rid_list, horse_list = [], []

        for rid_unique in _df[id_col].unique():
            idx = np.where(rids == rid_unique)[0]
            feat = X[idx]
            lab  = y[idx]
            sup  = base_support[idx]
            length = len(idx)

            pad_len = max_seq_len - length
            if pad_len>0:
                # パディング
                feat = np.vstack([feat, np.zeros((pad_len, feat_dim))])
                lab_pad = np.zeros((pad_len,), dtype=float)
                lab  = np.concatenate([lab, lab_pad])
                sup_pad = np.zeros((pad_len,), dtype=float)
                sup  = np.concatenate([sup, sup_pad])

                mask = [True]*length + [False]*pad_len

                rid_pad   = np.full((pad_len,), -1, dtype=rids.dtype)
                horse_pad = np.full((pad_len,), -1, dtype=horses.dtype)
                rid_arr   = np.concatenate([rids[idx],   rid_pad])
                horse_arr = np.concatenate([horses[idx], horse_pad])
            else:
                mask = [True]*length
                rid_arr   = rids[idx]
                horse_arr = horses[idx]

            seq_list.append(feat)
            label_list.append(lab)
            sup_list.append(sup)
            mask_list.append(mask)
            rid_list.append(rid_arr)
            horse_list.append(horse_arr)

        return seq_list, label_list, sup_list, mask_list, max_seq_len, rid_list, horse_list

    train_seq, train_lab, train_sup, train_msk, len_tr, train_rids, train_horses = create_sequences(train_df, X_train, y_train, sup_train)
    valid_seq, valid_lab, valid_sup, valid_msk, len_vl, valid_rids, valid_horses = create_sequences(valid_df, X_valid, y_valid, sup_valid)
    test_seq,  test_lab,  test_sup,  test_msk,  len_ts, test_rids,  test_horses  = create_sequences(test_df,  X_test,  y_test,  sup_test)

    max_seq_len = max(len_tr, len_vl, len_ts)

    # さらにパディング長を合わせる（既に同じはずだが、念のため関数化可）
    # ここでは省略

    # cat_unique dict
    cat_unique = {}
    for c in cat_cols:
        cat_unique[c] = len(train_df[c].unique()) + 1

    # Dataset
    train_dataset = HorseRaceDatasetDiffBCE(train_seq, train_lab, train_sup, train_msk, train_rids, train_horses)
    valid_dataset = HorseRaceDatasetDiffBCE(valid_seq, valid_lab, valid_sup, valid_msk, valid_rids, valid_horses)
    test_dataset  = HorseRaceDatasetDiffBCE(test_seq,  test_lab,  test_sup,  test_msk,  test_rids,  test_horses)

    # 11) 戻り値 (その他、AEモデルやscalerも必要なら返す)
    # ---------------------------------------------------------------------
    actual_num_dim = other_train_arr.shape[1] + z_train_horse.shape[1] + z_train_jockey.shape[1]
    total_feature_dim = X_train.shape[1]

    return (
        train_dataset, valid_dataset, test_dataset,
        cat_cols, cat_unique, max_seq_len,
        actual_num_dim,
        # PCA モデルがいらなければ削除、代わりに AEモデルを返すとかもアリ
        None, None,
        scaler_horse, scaler_jockey, scaler_other,
        df,  # 全体DF
        total_feature_dim,
    )

########################################
# 4) 学習ルーチン (BCE + 差分)
########################################
def run_training_diff_bce(
    data_path=DATA_PATH,
    pred_path=PRED_PATH,
    rank_col="着順",
    pop_col="単勝",
    id_col="race_id",
    test_ratio=0.1,
    valid_ratio=0.1,
    pca_dim_horse=50,
    pca_dim_jockey=50,
    batch_size=64,
    lr=0.005,
    num_epochs=50,
    d_model=64,
    nhead=8,
    num_layers=6,
    dropout=0.10,
    weight_decay=1e-5,
    patience=10
):
    # ===== 1) データセット作成 =====
    (train_dataset, valid_dataset, test_dataset,
     cat_cols, cat_unique, max_seq_len,
     actual_num_dim,
     pca_model_horse, pca_model_jockey,
     scaler_horse, scaler_jockey, scaler_other,
     df_all,
     total_feature_dim) = prepare_data_diff_bce(
        data_path=data_path,
        pred_path=pred_path,
        rank_col=rank_col,
        pop_col=pop_col,
        id_col=id_col,
        test_ratio=test_ratio,
        valid_ratio=valid_ratio,
        pca_dim_horse=pca_dim_horse,
        pca_dim_jockey=pca_dim_jockey
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # ===== 2) モデル準備 =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HorseTransformerDiffBCE(
        cat_unique=cat_unique,
        cat_cols=cat_cols,
        max_seq_len=max_seq_len,
        num_dim=actual_num_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader)*num_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                                            optimizer,
                                            max_lr=1e-1,          # ピーク時の最大LR
                                            total_steps=total_steps,
                                            pct_start=0.3,        # どのタイミングでピークになるか(全体の0.3)
                                            anneal_strategy='cos' # 後半の下がり方(コサイン)
                                        )

    # 注意: PyTorch標準の BCEWithLogitsLoss は "logits (未シグモイド) → シグモイド" を内蔵するが、
    #       今回は "prob = clamp( base_support + diff, 0,1 )" を使うため、自作で実装する。
    #       つまり "loss = -[y log(prob) + (1-y) log(1-prob)]" をマスクつきで計算する。
    def custom_bce_loss(diff_outputs, base_supports, labels, masks):
        """
        diff_outputs: (batch, seq_len, 1) => 差分(モデル出力)
        base_supports:(batch, seq_len)    => ベース支持率(0～1)
        labels:       (batch, seq_len)    => 0/1
        masks:        (batch, seq_len)    => True=有効
        """
        # === ここから修正箇所 ===
        diff_outputs = diff_outputs.squeeze(-1)  # (batch, seq_len)
        eps = 1e-7
        logit_bs = torch.logit(base_supports, eps=eps)  # base_support が 0 や 1 に近いときのため epsilon考慮
        pred_prob = torch.sigmoid(logit_bs + diff_outputs)
        # === ここまで修正箇所 ===

        eps = 1e-7
        bce = - (labels * torch.log(pred_prob + eps) + (1 - labels)*torch.log(1 - pred_prob + eps))
        bce = bce * masks  # pad部分は0に
        loss = bce.sum() / (masks.sum() + eps)
        return loss

    # 早期終了
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0

    print("=== Start Training: Diff + BCE ===")
    train_loss_history = []
    valid_loss_history = []

    # ===== 3) 学習ループ =====
    for epoch in range(num_epochs):
        # --- train ---
        model.train()
        sum_loss_train = 0.0
        count_train = 0

        for sequences, labels, base_sups, masks, _, _ in train_loader:
            sequences = sequences.to(device)
            labels    = labels.to(device)
            base_sups = base_sups.to(device)
            masks     = masks.to(device)

            optimizer.zero_grad()
            diff_out = model(sequences, src_key_padding_mask=masks)  # (batch, seq_len, 1)

            loss = custom_bce_loss(diff_out, base_sups, labels, masks)
            loss.backward()
            optimizer.step()
            scheduler.step()

            sum_loss_train += loss.item()
            count_train += 1

        avg_train_loss = sum_loss_train / max(count_train,1)

        # --- valid ---
        model.eval()
        sum_loss_valid = 0.0
        count_valid = 0
        with torch.no_grad():
            for sequences, labels, base_sups, masks, _, _ in valid_loader:
                sequences = sequences.to(device)
                labels    = labels.to(device)
                base_sups = base_sups.to(device)
                masks     = masks.to(device)

                diff_out = model(sequences, src_key_padding_mask=masks)
                loss_val = custom_bce_loss(diff_out, base_sups, labels, masks)
                sum_loss_valid += loss_val.item()
                count_valid += 1

        avg_valid_loss = sum_loss_valid / max(count_valid,1)

        train_loss_history.append(avg_train_loss)
        valid_loss_history.append(avg_valid_loss)

        print(f"[Epoch {epoch+1}/{num_epochs}]  "
              f"TrainLoss = {avg_train_loss:.4f}, ValidLoss = {avg_valid_loss:.4f}")

        # 早期終了
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    # 学習終了
    print("=== Training Finished ===")
    model.load_state_dict(best_model_wts)

    # ロス推移可視化
    plt.figure(figsize=(8,5))
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(valid_loss_history, label="Valid Loss")
    plt.title("Loss Curve (Diff+BCE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # ===== 4) 推論関数 =====
    def predict_prob(loader, model):
        """
        モデル出力(差分)と、base_supportを足した確率を返す
        """
        model.eval()
        all_probs = []
        all_labels= []
        all_diffs = []
        all_rids  = []
        all_hnums = []
        with torch.no_grad():
            for sequences, labels, base_sups, masks, rids, hnums in loader:
                sequences = sequences.to(device)
                labels    = labels.to(device)
                base_sups = base_sups.to(device)
                masks     = masks.to(device)

                diff_out = model(sequences, src_key_padding_mask=masks).squeeze(-1) # (batch, seq_len)
                eps = 1e-7
                logit_bs = torch.logit(base_sups, eps=eps)  # base_support が 0 や 1 に近いときのため epsilon考慮
                pred_prob = torch.sigmoid(logit_bs + diff_out)


                # 有効部分だけ取り出し
                masks_np = masks.cpu().numpy()
                prob_np  = pred_prob.cpu().numpy()
                diff_np  = diff_out.cpu().numpy()
                labels_np= labels.cpu().numpy()
                rids_np  = rids.cpu().numpy()
                hnums_np = hnums.cpu().numpy()

                B, L = masks_np.shape
                for b in range(B):
                    valid_len = masks_np[b].sum()
                    all_probs.append(prob_np[b,:valid_len])
                    all_labels.append(labels_np[b,:valid_len])
                    all_diffs.append(diff_np[b,:valid_len])
                    all_rids.append(rids_np[b,:valid_len])
                    all_hnums.append(hnums_np[b,:valid_len])

        # 結合
        probs_concat  = np.concatenate(all_probs, axis=0)
        labels_concat = np.concatenate(all_labels, axis=0)
        diffs_concat  = np.concatenate(all_diffs, axis=0)
        rids_concat   = np.concatenate(all_rids, axis=0)
        hnums_concat  = np.concatenate(all_hnums, axis=0)

        return probs_concat, labels_concat, diffs_concat, rids_concat, hnums_concat

    # ===== テスト評価 =====
    test_probs, test_labels, test_diffs, test_rids, test_hnums = predict_prob(test_loader, model)

    # BCE (最終評価)
    eps=1e-9
    test_bce = - ( test_labels*np.log(test_probs+eps) + (1-test_labels)*np.log(1-test_probs+eps) )
    test_loss = np.mean(test_bce)
    print(f"Test BCE Loss = {test_loss:.4f}")

    # 結果DataFrame
    test_result_df = pd.DataFrame({
        "race_id": test_rids.astype(int),
        "馬番":    test_hnums.astype(int),
        "label_win": test_labels.astype(int),       # 1=勝ち,0=負け
        "diff_out":   test_diffs,                  # 差分
        "pred_prob":  test_probs,                  # 予測確率(支持率 + 差分)
    })
    # base_support を再取得したい場合、Datasetやdf_all と再マージでもOK:
    # 実オッズ列(ここでは "odds" とする)がある df_odds などがある想定
    # race_id, 馬番 でくっつける
    test_result_df = pd.merge(
        test_result_df,
        df_all[["race_id", "馬番", "単勝"]],  # 実際の単勝オッズが入ってる列
        on=["race_id","馬番"],
        how="left"
    )
    test_result_df.rename(columns={"単勝": "win_odds"}, inplace=True)

    # 保存
    test_result_df.to_csv(SAVE_PATH_PRED, index=False)

    # ===== Full(=train+valid+test) の推論 =====
    full_dataset = ConcatDataset([train_dataset, valid_dataset, test_dataset])
    full_loader  = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    full_probs, full_labels, full_diffs, full_rids, full_hnums = predict_prob(full_loader, model)
    full_bce = - ( full_labels*np.log(full_probs+eps) + (1-full_labels)*np.log(1-full_probs+eps) )
    full_loss = np.mean(full_bce)
    print(f"Full BCE Loss = {full_loss:.4f}")

    full_result_df = pd.DataFrame({
        "race_id": full_rids.astype(int),
        "馬番":    full_hnums.astype(int),
        "label_win": full_labels.astype(int),
        "diff_out":  full_diffs,
        "pred_prob": full_probs,
    })
    full_result_df.to_csv(SAVE_PATH_FULL_PRED, index=False)

    # ===== 5) モデル保存 =====
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    with open(SAVE_PATH_MODEL, "wb") as f:
        pickle.dump(model.state_dict(), f)

    # if pca_model_horse is not None:
    #     with open(SAVE_PATH_PCA_MODEL_HORSE, "wb") as f:
    #         pickle.dump(pca_model_horse, f)
    # if pca_model_jockey is not None:
    #     with open(SAVE_PATH_PCA_MODEL_JOCKEY, "wb") as f:
    #         pickle.dump(pca_model_jockey, f)
    with open(SAVE_PATH_SCALER_HORSE, "wb") as f:
        pickle.dump(scaler_horse, f)
    with open(SAVE_PATH_SCALER_JOCKEY, "wb") as f:
        pickle.dump(scaler_jockey, f)
    with open(SAVE_PATH_SCALER_OTHER, "wb") as f:
        pickle.dump(scaler_other, f)

    print("Model & preprocessors saved.")

    # ===== 6) 可視化例 =====
    # 差分 > 0 の馬（モデルが「市場より高く評価」）についての的中率や収益などを簡易集計
    analyze_diff_results(test_result_df)

    # 収益確認
    evaluate_model_performance(test_result_df)

    return 0


def analyze_diff_results(df_result):
    """
    差分>0 の馬だけを集めて、モデルが"市場より勝つ確率高い"と見た馬の実際の勝率を見たり、
    回収率を可視化したりする例。
    ここでは簡易的に "ラベル=1 の割合" を計算。
    """
    df_positive = df_result[df_result["diff_out"]>0]
    win_rate = df_positive["label_win"].mean()
    print(f"[差分>0] の馬数: {len(df_positive)}, 実際の勝率: {win_rate:.4f}")

    # プロット例: diff_out vs pred_prob
    plt.figure(figsize=(6,5))
    plt.scatter(df_result["diff_out"], df_result["pred_prob"], alpha=0.3)
    plt.title("Difference vs Predicted Probability")
    plt.xlabel("Difference (model output)")
    plt.ylabel("Predicted Probability (base+diff)")
    plt.grid(True)
    plt.show()

def evaluate_model_performance(df):
    """
    df には最低限以下の列を想定:
      - label_win: 1(勝ち) or 0(負け)
      - pred_prob: モデルの予測確率(0~1)
      - diff_out:  モデル出力の差分
      - win_odds:  実際の単勝オッズ(馬券の払戻倍率)
    """
    
    # --- 1) AUC, BCE, Brierスコア ---
    auc_value = roc_auc_score(df["label_win"], df["pred_prob"])
    bce_value = log_loss(df["label_win"], np.clip(df["pred_prob"], 1e-15, 1 - 1e-15))
    brier_value = brier_score_loss(df["label_win"], df["pred_prob"])
    
    print(f"AUC = {auc_value:.4f}")
    print(f"BCE = {bce_value:.4f}")
    print(f"Brier Score = {brier_value:.4f}")
    
    # --- 2) キャリブレーション(信頼度と実際の的中率の比較) ---
    # 予測確率を0~1に分割して、平均予測と実際のラベル(=勝率)をプロット
    prob_true, prob_pred = calibration_curve(df["label_win"], df["pred_prob"], n_bins=20)
    plt.figure(figsize=(5,5))
    plt.plot(prob_pred, prob_true, marker="o", label="Calibration")
    plt.plot([0,1], [0,1], "--", color="gray", label="Perfect")
    plt.title("Calibration Plot")
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Probability")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # --- 3) 回収率シミュレーション ---
    # 回収金額 = オッズ × 賭け金 (的中時)
    # 賭け金を100円固定と仮定する例

    # 3-1) diff>0 の馬だけ買う
    df_diff_pos = df[df["diff_out"] > 0].copy()
    stake_diff = len(df_diff_pos)*100
    # label_win=1の馬だけ、オッズ×100円の払い戻し
    returned_diff = df_diff_pos.apply(lambda row: row["win_odds"]*100 if row["label_win"]==1 else 0, axis=1).sum()
    roi_diff_pos = returned_diff / stake_diff if stake_diff>0 else 0
    print(f"[diff>0] 購入馬数: {len(df_diff_pos)}, ROI = {roi_diff_pos:.3f}")

    # 3-2) レースごとに予測確率上位N頭だけ買う
    topN = 5  # ここ変えて好きな頭数に
    def pick_topN(group, n=topN):
        return group.sort_values("pred_prob", ascending=False).head(n)

    df_topN = df.groupby("race_id", group_keys=False).apply(pick_topN)
    stake_topN = len(df_topN)*100
    returned_topN = df_topN.apply(lambda row: row["win_odds"]*100 if row["label_win"]==1 else 0, axis=1).sum()
    roi_topN = returned_topN / stake_topN if stake_topN>0 else 0
    print(f"[top{topN} strategy] 購入馬数: {len(df_topN)}, ROI = {roi_topN:.3f}")

    # 3-3) フェアオッズ戦略: fair_odds = 1 / pred_prob で計算し、
    #      実オッズの方が割高(> fair_odds)なら買う
    df["fair_odds"] = 0.8 / df["pred_prob"].clip(lower=1e-8)
    df_fair = df[df["win_odds"] > df["fair_odds"]].copy()
    stake_fair = len(df_fair)*100
    returned_fair = df_fair.apply(lambda row: row["win_odds"]*100 if row["label_win"]==1 else 0, axis=1).sum()
    roi_fair = returned_fair / stake_fair if stake_fair>0 else 0
    print(f"[fairオッズより実オッズが高い馬] 購入馬数: {len(df_fair)}, ROI = {roi_fair:.3f}")


# ===== メイン =====
if __name__ == "__main__":
    run_training_diff_bce(
        data_path=DATA_PATH,
        pred_path=PRED_PATH,
        rank_col="着順",
        pop_col="単勝",
        test_ratio=0.1,
        valid_ratio=0.1,
        batch_size=64,
        lr=0.0005,
        num_epochs=20,
        d_model=64,
        nhead=4,
        num_layers=4,
        dropout=0.1,
        patience=5
    )