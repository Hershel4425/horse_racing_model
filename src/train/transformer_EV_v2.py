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
PRED_PATH  = os.path.join(ROOT_PATH, "result/predictions/transformer/20250109221743_full.csv")
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
    sequences: (num_races, max_seq_len, feature_dim)
    masks:     (num_races, max_seq_len)
    base_supports: (num_races, max_seq_len)
    race_ids, horse_nums: (num_races, max_seq_len)

    labels_win     : (num_races, max_seq_len)          → 1着 or 0
    labels_multi   : (num_races, max_seq_len, 6)       → [top1, top3, top5, pop1, pop3, pop5]
    labels_tansho  : (num_races, max_seq_len)          → "0.8 / 単勝"
    """
    def __init__(self, sequences, labels_win, labels_multi, labels_tansho,
                 base_supports, masks, race_ids, horse_nums):
        self.sequences     = sequences
        self.labels_win    = labels_win
        self.labels_multi  = labels_multi
        self.labels_tansho = labels_tansho
        self.base_supports = base_supports
        self.masks         = masks
        self.race_ids      = race_ids
        self.horse_nums    = horse_nums

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq  = torch.tensor(self.sequences[idx],     dtype=torch.float32)
        win  = torch.tensor(self.labels_win[idx],    dtype=torch.float32)
        mlt  = torch.tensor(self.labels_multi[idx],  dtype=torch.float32)
        tns  = torch.tensor(self.labels_tansho[idx], dtype=torch.float32)
        sup  = torch.tensor(self.base_supports[idx], dtype=torch.float32)
        msk  = torch.tensor(self.masks[idx],         dtype=torch.bool)
        rid  = torch.tensor(self.race_ids[idx],      dtype=torch.long)
        hnum = torch.tensor(self.horse_nums[idx],    dtype=torch.long)
        return seq, win, mlt, tns, sup, msk, rid, hnum


########################################
# 2) Embedding + Transformer モデル
########################################
class FeatureEmbedder(nn.Module):
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
    出力: (batch, seq_len, 8)
      [0]: diff (差分)
      [1..6]: 6つの2値分類用 (top1,top3,top5,pop1,pop3,pop5)
      [7]:  0.8 / 単勝
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

        self.fc_out = nn.Linear(d_model, 8)

    def forward(self, src, src_key_padding_mask=None):
        emb = self.feature_embedder(src)
        emb = self.pos_encoder(emb)
        out = self.transformer_encoder(emb, src_key_padding_mask=~src_key_padding_mask)
        out = self.fc_out(out)  # (B,L,8)
        return out


########################################
# 3) データ準備用の例関数
########################################
def split_data(df, id_col="race_id", test_ratio=0.1, valid_ratio=0.1):
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
    rank_col="着順",   
    pop_col="単勝",    
    test_ratio=0.1,
    valid_ratio=0.1,
    pca_dim_horse=50,
    pca_dim_jockey=50
):
    df_feature = pd.read_csv(data_path, encoding='utf-8-sig')
    df_pred    = pd.read_csv(pred_path, encoding='utf-8-sig')

    df = pd.merge(
        df_feature,
        df_pred[["race_id","馬番","P_top1","P_pop1","P_top3","P_pop3","P_top5","P_pop5"]],
        on=["race_id","馬番"],
        how="inner"
    )

    # 1) is_win
    df["is_win"] = (df[rank_col] == 1).astype(int)

    # 2) 6タスク (top1, top3, top5, pop1, pop3, pop5)
    df["label_top1"] = (df[rank_col] <= 1).astype(int)
    df["label_top3"] = (df[rank_col] <= 3).astype(int)
    df["label_top5"] = (df[rank_col] <= 5).astype(int)

    # 人気をpop_col=単勝で近似している場合
    df["label_pop1"] = (df["人気"] <= 1).astype(int)
    df["label_pop3"] = (df["人気"] <= 3).astype(int)
    df["label_pop5"] = (df["人気"] <= 5).astype(int)

    # 3) base_support
    df["base_support"] = 0.8 / (df[pop_col] + 1e-10)

    # ---- split ----
    train_df, valid_df, test_df = split_data(df, id_col=id_col,
                                             test_ratio=test_ratio,
                                             valid_ratio=valid_ratio)

    cat_cols_all = train_df.select_dtypes(exclude=[np.number]).columns.tolist()
    num_cols_all = train_df.select_dtypes(include=[np.number]).columns.tolist()

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
            'base_support',
            "label_top1", "label_top3", "label_top5", "label_pop1", "label_pop3", "label_pop5"
        ]
    cat_cols = [c for c in cat_cols_all if c not in leakage_cols and c not in [id_col]]
    num_cols = [c for c in num_cols_all if c not in leakage_cols and c not in [id_col]]

    for c in cat_cols:
        for d in [train_df, valid_df, test_df]:
            d[c] = d[c].fillna("missing").astype(str)
    for n in num_cols:
        for d in [train_df, valid_df, test_df]:
            d[n] = d[n].fillna(0)

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

    pca_pattern_horse = r'^(競走馬芝|競走馬ダート)'
    pca_pattern_jockey= r'^(騎手芝|騎手ダート)'

    pca_horse_target_cols = [c for c in num_cols if re.match(pca_pattern_horse, c)]
    pca_jockey_target_cols= [c for c in num_cols if re.match(pca_pattern_jockey, c)]
    other_num_cols = [c for c in num_cols if c not in pca_horse_target_cols + pca_jockey_target_cols]

    scaler_horse = StandardScaler()
    scaler_jockey= StandardScaler()
    scaler_other = StandardScaler()

    if len(pca_horse_target_cols)>0:
        scaler_horse.fit(train_df[pca_horse_target_cols])
    if len(pca_jockey_target_cols)>0:
        scaler_jockey.fit(train_df[pca_jockey_target_cols])
    if len(other_num_cols)>0:
        scaler_other.fit(train_df[other_num_cols])

    # PCA horse
    horse_train_arr = scaler_horse.transform(train_df[pca_horse_target_cols]) if len(pca_horse_target_cols)>0 else np.zeros((len(train_df),0))
    horse_valid_arr = scaler_horse.transform(valid_df[pca_horse_target_cols]) if len(pca_horse_target_cols)>0 else np.zeros((len(valid_df),0))
    horse_test_arr  = scaler_horse.transform(test_df[pca_horse_target_cols])  if len(pca_horse_target_cols)>0 else np.zeros((len(test_df),0))

    if pca_dim_horse>0 and pca_dim_horse<=horse_train_arr.shape[1]:
        pca_model_horse = PCA(n_components=pca_dim_horse).fit(horse_train_arr)
        horse_train_pca = pca_model_horse.transform(horse_train_arr)
        horse_valid_pca = pca_model_horse.transform(horse_valid_arr)
        horse_test_pca  = pca_model_horse.transform(horse_test_arr)
    else:
        pca_model_horse = None
        horse_train_pca = horse_train_arr
        horse_valid_pca = horse_valid_arr
        horse_test_pca  = horse_test_arr

    # PCA jockey
    jockey_train_arr = scaler_jockey.transform(train_df[pca_jockey_target_cols]) if len(pca_jockey_target_cols)>0 else np.zeros((len(train_df),0))
    jockey_valid_arr = scaler_jockey.transform(valid_df[pca_jockey_target_cols]) if len(pca_jockey_target_cols)>0 else np.zeros((len(valid_df),0))
    jockey_test_arr  = scaler_jockey.transform(test_df[pca_jockey_target_cols])  if len(pca_jockey_target_cols)>0 else np.zeros((len(test_df),0))

    if pca_dim_jockey>0 and pca_dim_jockey<=jockey_train_arr.shape[1]:
        pca_model_jockey = PCA(n_components=pca_dim_jockey).fit(jockey_train_arr)
        jockey_train_pca = pca_model_jockey.transform(jockey_train_arr)
        jockey_valid_pca = pca_model_jockey.transform(jockey_valid_arr)
        jockey_test_pca  = pca_model_jockey.transform(jockey_test_arr)
    else:
        pca_model_jockey = None
        jockey_train_pca = jockey_train_arr
        jockey_valid_pca = jockey_valid_arr
        jockey_test_pca  = jockey_test_arr

    # other
    other_train_arr = scaler_other.transform(train_df[other_num_cols]) if len(other_num_cols)>0 else np.zeros((len(train_df),0))
    other_valid_arr = scaler_other.transform(valid_df[other_num_cols]) if len(other_num_cols)>0 else np.zeros((len(valid_df),0))
    other_test_arr  = scaler_other.transform(test_df[other_num_cols])  if len(other_num_cols)>0 else np.zeros((len(test_df),0))

    def concat_features(df_part, cat_cols, other_arr, horse_pca, jockey_pca):
        cat_val = df_part[cat_cols].values if len(cat_cols)>0 else np.zeros((len(df_part),0))
        return np.concatenate([cat_val, other_arr, horse_pca, jockey_pca], axis=1)

    X_train = concat_features(train_df, cat_cols, other_train_arr, horse_train_pca, jockey_train_pca)
    X_valid = concat_features(valid_df, cat_cols, other_valid_arr, horse_valid_pca, jockey_valid_pca)
    X_test  = concat_features(test_df,  cat_cols, other_test_arr,  horse_test_pca,  jockey_test_pca)

    # ==== ここがポイント: 単勝オッズの代わりに "0.8 / 単勝" をラベルにする ====
    EPS = 1e-8
    y_train_win   = train_df["is_win"].values
    y_valid_win   = valid_df["is_win"].values
    y_test_win    = test_df["is_win"].values

    def stack_multi_label(_df):
        return np.stack([
            _df["label_top1"].values,
            _df["label_top3"].values,
            _df["label_top5"].values,
            _df["label_pop1"].values,
            _df["label_pop3"].values,
            _df["label_pop5"].values,
        ], axis=-1)

    y_train_multi = stack_multi_label(train_df)
    y_valid_multi = stack_multi_label(valid_df)
    y_test_multi  = stack_multi_label(test_df)

    # >>> 予測したいのは "0.8 / 単勝"
    y_train_tansho = 0.8 / (train_df[pop_col].values + EPS)
    y_valid_tansho = 0.8 / (valid_df[pop_col].values + EPS)
    y_test_tansho  = 0.8 / (test_df[pop_col].values + EPS)

    sup_train = train_df["base_support"].values
    sup_valid = valid_df["base_support"].values
    sup_test  = test_df["base_support"].values

    def create_sequences(_df, X, y_win, y_mlt, y_tns, base_support):
        rids   = _df[id_col].values
        horses = _df["馬番"].values if "馬番" in _df.columns else np.arange(len(_df))

        groups = _df.groupby(id_col)
        max_seq_len = groups.size().max()
        feat_dim = X.shape[1]

        seq_list, win_list, mlt_list, tns_list, sup_list, mask_list = [], [], [], [], [], []
        rid_list, horse_list = [], []

        for rid_unique in _df[id_col].unique():
            idx = np.where(rids == rid_unique)[0]
            feat = X[idx]
            wn   = y_win[idx]
            ml   = y_mlt[idx]
            ts   = y_tns[idx]
            sup  = base_support[idx]
            length = len(idx)

            pad_len = max_seq_len - length
            if pad_len>0:
                feat = np.vstack([feat, np.zeros((pad_len, feat_dim))])
                wn   = np.concatenate([wn, np.zeros((pad_len,))])
                ml   = np.vstack([ml, np.zeros((pad_len, 6))])
                ts   = np.concatenate([ts, np.zeros((pad_len,))])
                sup  = np.concatenate([sup, np.zeros((pad_len,))])

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
            win_list.append(wn)
            mlt_list.append(ml)
            tns_list.append(ts)
            sup_list.append(sup)
            mask_list.append(mask)
            rid_list.append(rid_arr)
            horse_list.append(horse_arr)

        return (seq_list, win_list, mlt_list, tns_list, sup_list, mask_list,
                max_seq_len, rid_list, horse_list)

    tr_seq, tr_win, tr_mlt, tr_tns, tr_sup, tr_msk, len_tr, tr_rids, tr_horses = create_sequences(train_df, X_train, y_train_win, y_train_multi, y_train_tansho, sup_train)
    vl_seq, vl_win, vl_mlt, vl_tns, vl_sup, vl_msk, len_vl, vl_rids, vl_horses = create_sequences(valid_df, X_valid, y_valid_win, y_valid_multi, y_valid_tansho, sup_valid)
    ts_seq, ts_win, ts_mlt, ts_tns, ts_sup, ts_msk, len_ts, ts_rids, ts_horses = create_sequences(test_df,  X_test,  y_test_win,  y_test_multi,  y_test_tansho,  sup_test)

    max_seq_len = max(len_tr, len_vl, len_ts)

    cat_unique = {}
    for c in cat_cols:
        cat_unique[c] = len(train_df[c].unique()) + 1

    train_dataset = HorseRaceDatasetDiffBCE(tr_seq, tr_win, tr_mlt, tr_tns, tr_sup, tr_msk, tr_rids, tr_horses)
    valid_dataset = HorseRaceDatasetDiffBCE(vl_seq, vl_win, vl_mlt, vl_tns, vl_sup, vl_msk, vl_rids, vl_horses)
    test_dataset  = HorseRaceDatasetDiffBCE(ts_seq, ts_win, ts_mlt, ts_tns, ts_sup, ts_msk, ts_rids, ts_horses)

    actual_num_dim = other_train_arr.shape[1] + horse_train_pca.shape[1] + jockey_train_pca.shape[1]
    total_feature_dim = X_train.shape[1]

    return (train_dataset, valid_dataset, test_dataset,
            cat_cols, cat_unique, max_seq_len,
            actual_num_dim,
            pca_model_horse, pca_model_jockey, scaler_horse, scaler_jockey, scaler_other,
            df,  # 全体DF
            total_feature_dim)


########################################
# 4) 学習ルーチン (BCE + 差分 + 6分類 + "0.8/単勝")
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
    batch_size=256,
    lr=0.001,
    num_epochs=50,
    d_model=128,
    nhead=8,
    num_layers=6,
    dropout=0.15,
    weight_decay=1e-5,
    patience=10
):
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    def custom_loss_fn(outputs, label_win, label_multi, label_tansho, base_sups, masks):
        diff_out   = outputs[..., 0]      # shape (B,L)
        multi_out  = outputs[..., 1:7]    # (B,L,6)
        tansho_out = outputs[..., 7]      # (B,L)

        eps = 1e-7
        # (a) diff 用 BCE
        logit_bs = torch.logit(base_sups, eps=eps)
        pred_prob_win = torch.sigmoid(logit_bs + diff_out)
        bce_diff = - (label_win * torch.log(pred_prob_win+eps) + (1 - label_win)*torch.log(1-pred_prob_win+eps))

        # (b) 6分類タスク
        bce_multi = nn.functional.binary_cross_entropy_with_logits(multi_out, label_multi, reduction='none')

        # (c) "0.8/単勝" の MSE
        mse_tansho = (tansho_out - label_tansho)**2

        # パディング除外
        masks_3d = masks.unsqueeze(-1)  # shape=(B,L,1)
        bce_diff   = bce_diff * masks
        bce_multi  = bce_multi * masks_3d
        mse_tansho = mse_tansho * masks

        denom = masks.sum() + eps
        loss_diff   = bce_diff.sum() / denom
        loss_multi  = bce_multi.sum() / (denom*6)  
        loss_tansho = mse_tansho.sum() / denom

        loss = loss_diff + loss_multi + loss_tansho
        return loss, loss_diff.item(), loss_multi.item(), loss_tansho.item()

    def calc_metrics(outputs, label_win, label_multi, label_tansho, base_sups, masks):
        diff_out   = outputs[..., 0]
        multi_out  = outputs[..., 1:7]
        tansho_out = outputs[..., 7]

        eps=1e-7
        logit_bs = torch.logit(base_sups, eps=eps)
        pred_prob_win = torch.sigmoid(logit_bs + diff_out)
        bce_diff = - (label_win*torch.log(pred_prob_win+eps) + (1-label_win)*torch.log(1-pred_prob_win+eps))
        bce_diff = bce_diff * masks
        loss_diff_val = bce_diff.sum() / (masks.sum() + eps)

        multi_prob = torch.sigmoid(multi_out)  
        multi_pred = (multi_prob >= 0.5).float()
        correct = (multi_pred == label_multi).float()
        correct = correct * masks.unsqueeze(-1)
        acc_each_task = correct.sum(dim=(0,1)) / (masks.sum() + eps)  

        mse_tansho = (tansho_out - label_tansho)**2
        mse_tansho = mse_tansho * masks
        mse_val = mse_tansho.sum() / (masks.sum() + eps)
        rmse_val = mse_val.sqrt()

        return (loss_diff_val.item(),
                acc_each_task.cpu().numpy(),
                rmse_val.item())

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0

    print("=== Start Training (diff + multi + 0.8/tansho) ===")
    for epoch in range(num_epochs):
        # --- train ---
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        sum_loss_diff = 0.0
        sum_loss_multi= 0.0
        sum_loss_tansho=0.0

        for (seq, win, mlt, tns, sup, msk, _, _) in train_loader:
            seq = seq.to(device)
            win = win.to(device)
            mlt = mlt.to(device)
            tns = tns.to(device)
            sup = sup.to(device)
            msk = msk.to(device)

            optimizer.zero_grad()
            out = model(seq, src_key_padding_mask=msk)
            loss_total, ld, lm, lt = custom_loss_fn(out, win, mlt, tns, sup, msk)
            loss_total.backward()
            optimizer.step()

            train_loss_sum += loss_total.item()
            sum_loss_diff   += ld
            sum_loss_multi  += lm
            sum_loss_tansho += lt
            train_count += 1

        scheduler.step()
        avg_train_loss = train_loss_sum / max(train_count,1)
        avg_train_diff = sum_loss_diff / max(train_count,1)
        avg_train_mlt  = sum_loss_multi / max(train_count,1)
        avg_train_tns  = sum_loss_tansho / max(train_count,1)

        # --- valid ---
        model.eval()
        valid_loss_sum = 0.0
        valid_count = 0
        val_diff_sum = 0.0
        val_mlt_sum  = 0.0
        val_tns_sum  = 0.0

        sum_acc_each = np.zeros(6, dtype=np.float32)
        sum_rmse_val = 0.0
        n_batch_eval = 0

        with torch.no_grad():
            for (seq, win, mlt, tns, sup, msk, _, _) in valid_loader:
                seq = seq.to(device)
                win = win.to(device)
                mlt = mlt.to(device)
                tns = tns.to(device)
                sup = sup.to(device)
                msk = msk.to(device)

                out = model(seq, src_key_padding_mask=msk)
                loss_val, ld, lm, lt = custom_loss_fn(out, win, mlt, tns, sup, msk)

                valid_loss_sum += loss_val.item()
                val_diff_sum += ld
                val_mlt_sum  += lm
                val_tns_sum  += lt
                valid_count += 1

                diff_loss_v, acc_each, rmse_v = calc_metrics(out, win, mlt, tns, sup, msk)
                sum_acc_each += acc_each
                sum_rmse_val += rmse_v
                n_batch_eval += 1

        avg_valid_loss = valid_loss_sum / max(valid_count,1)
        avg_val_diff   = val_diff_sum / max(valid_count,1)
        avg_val_mlt    = val_mlt_sum  / max(valid_count,1)
        avg_val_tns    = val_tns_sum  / max(valid_count,1)

        avg_val_acc_each = sum_acc_each / max(n_batch_eval,1)
        avg_val_rmse     = sum_rmse_val / max(n_batch_eval,1)

        print(f"[Epoch {epoch+1}/{num_epochs}]  "
              f"TrainLoss={avg_train_loss:.4f} (diff={avg_train_diff:.4f}, mlt={avg_train_mlt:.4f}, tan={avg_train_tns:.4f}) | "
              f"ValidLoss={avg_valid_loss:.4f} (diff={avg_val_diff:.4f}, mlt={avg_val_mlt:.4f}, tan={avg_val_tns:.4f})")
        print("   Valid Acc (top1..pop5):", " ".join(f"{v:.3f}" for v in avg_val_acc_each))
        print("   Valid 0.8/tansho RMSE  :", f"{avg_val_rmse:.4f}")

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

    print("=== Training Finished ===")
    model.load_state_dict(best_model_wts)

    def predict_prob_tansho(loader, model):
        model.eval()
        all_pred_win = []
        all_pred_multi = []
        all_pred_tansho= []
        all_label_win  = []
        all_label_multi= []
        all_label_tansho=[]
        all_rids = []
        all_hnums= []

        with torch.no_grad():
            for (seq, win, mlt, tns, sup, msk, rids, hnums) in loader:
                seq = seq.to(device)
                win = win.to(device)
                mlt = mlt.to(device)
                tns = tns.to(device)
                sup = sup.to(device)
                msk = msk.to(device)

                out = model(seq, src_key_padding_mask=msk)
                diff_out   = out[..., 0]
                multi_out  = out[..., 1:7]
                tansho_out = out[..., 7]

                eps = 1e-7
                logit_bs = torch.logit(sup, eps=eps)
                pred_prob_win_ = torch.sigmoid(logit_bs + diff_out)
                pred_multi_    = torch.sigmoid(multi_out)

                # マスク除外
                msk_np = msk.cpu().numpy()
                pred_win_np   = pred_prob_win_.cpu().numpy()
                pred_multi_np = pred_multi_.cpu().numpy()
                pred_tns_np   = tansho_out.cpu().numpy()

                label_win_np  = win.cpu().numpy()
                label_mlt_np  = mlt.cpu().numpy()
                label_tns_np  = tns.cpu().numpy()

                rids_np = rids.cpu().numpy()
                hnums_np= hnums.cpu().numpy()

                B,L = msk_np.shape
                for b in range(B):
                    valid_len = msk_np[b].sum()
                    all_pred_win.append(pred_win_np[b,:valid_len])
                    all_pred_multi.append(pred_multi_np[b,:valid_len])
                    all_pred_tansho.append(pred_tns_np[b,:valid_len])

                    all_label_win.append(label_win_np[b,:valid_len])
                    all_label_multi.append(label_mlt_np[b,:valid_len])
                    all_label_tansho.append(label_tns_np[b,:valid_len])

                    all_rids.append(rids_np[b,:valid_len])
                    all_hnums.append(hnums_np[b,:valid_len])

        pw   = np.concatenate(all_pred_win,   axis=0)
        pm   = np.concatenate(all_pred_multi, axis=0)
        pt   = np.concatenate(all_pred_tansho,axis=0)

        lw   = np.concatenate(all_label_win,  axis=0)
        lm   = np.concatenate(all_label_multi,axis=0)
        lt   = np.concatenate(all_label_tansho,axis=0)

        rr   = np.concatenate(all_rids,   axis=0)
        hh   = np.concatenate(all_hnums,  axis=0)

        return pw, pm, pt, lw, lm, lt, rr, hh

    # テスト評価
    test_pw, test_pm, test_pt, test_lw, test_lm, test_lt, test_rid, test_hnum = predict_prob_tansho(test_loader, model)

    # 1) diff部
    eps=1e-9
    test_bce_diff = - ( test_lw*np.log(test_pw+eps) + (1-test_lw)*np.log(1-test_pw+eps) )
    test_loss_diff = test_bce_diff.mean()
    print(f"Test BCE Loss (diff) = {test_loss_diff:.4f}")

    # 2) 6分類 Accuracy
    test_pm_bin = (test_pm >= 0.5).astype(int)
    correct = (test_pm_bin == test_lm).astype(int)
    acc_6 = correct.mean(axis=0)
    print("Test 6-class Acc (top1,top3,top5,pop1,pop3,pop5):", " ".join(f"{v:.3f}" for v in acc_6))

    # 3) "0.8/単勝" RMSE
    mse_tansho_test = ((test_pt - test_lt)**2).mean()
    rmse_tansho_test= np.sqrt(mse_tansho_test)
    print(f"Test RMSE (0.8/単勝) = {rmse_tansho_test:.4f}")

    test_result_df = pd.DataFrame({
        "race_id": test_rid.astype(int),
        "馬番":    test_hnum.astype(int),
        "pred_win":   test_pw,
        "label_win":  test_lw,
        "pred_0.8_div_tansho": test_pt,
        "true_0.8_div_tansho": test_lt
    })
    for i, nm in enumerate(["top1","top3","top5","pop1","pop3","pop5"]):
        test_result_df["pred_"+nm] = test_pm[:, i]
        test_result_df["label_"+nm]= test_lm[:, i]

    test_result_df.to_csv(SAVE_PATH_PRED, index=False)
    print("Saved test predictions:", SAVE_PATH_PRED)

    # Full dataset
    full_dataset = ConcatDataset([train_dataset, valid_dataset, test_dataset])
    full_loader  = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    full_pw, full_pm, full_pt, full_lw, full_lm, full_lt, full_rid, full_hnum = predict_prob_tansho(full_loader, model)
    mse_full_tansho = ((full_pt - full_lt)**2).mean()
    rmse_full_tansho= np.sqrt(mse_full_tansho)
    print(f"Full RMSE (0.8/単勝) = {rmse_full_tansho:.4f}")

    full_result_df = pd.DataFrame({
        "race_id": full_rid.astype(int),
        "馬番":    full_hnum.astype(int),
        "pred_win":  full_pw,
        "label_win": full_lw,
        "pred_0.8_div_tansho": full_pt,
        "true_0.8_div_tansho": full_lt
    })
    for i, nm in enumerate(["top1","top3","top5","pop1","pop3","pop5"]):
        full_result_df["pred_"+nm] = full_pm[:, i]
        full_result_df["label_"+nm]= full_lm[:, i]

    full_result_df.to_csv(SAVE_PATH_FULL_PRED, index=False)
    print("Saved full predictions:", SAVE_PATH_FULL_PRED)

    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    with open(SAVE_PATH_MODEL, "wb") as f:
        pickle.dump(model.state_dict(), f)

    if pca_model_horse is not None:
        with open(SAVE_PATH_PCA_MODEL_HORSE, "wb") as f:
            pickle.dump(pca_model_horse, f)
    if pca_model_jockey is not None:
        with open(SAVE_PATH_PCA_MODEL_JOCKEY, "wb") as f:
            pickle.dump(pca_model_jockey, f)
    with open(SAVE_PATH_SCALER_HORSE, "wb") as f:
        pickle.dump(scaler_horse, f)
    with open(SAVE_PATH_SCALER_JOCKEY, "wb") as f:
        pickle.dump(scaler_jockey, f)
    with open(SAVE_PATH_SCALER_OTHER, "wb") as f:
        pickle.dump(scaler_other, f)

    print("Model & preprocessors saved.")
    return 0

if __name__ == "__main__":
    run_training_diff_bce(
        data_path=DATA_PATH,
        pred_path=PRED_PATH,
        rank_col="着順",
        pop_col="単勝",
        test_ratio=0.1,
        valid_ratio=0.1,
        batch_size=32,
        lr=0.0001,
        num_epochs=10,
        d_model=64,
        nhead=4,
        num_layers=4,
        dropout=0.1,
        patience=5
    )
