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
MODEL_SAVE_DIR = os.path.join(ROOT_PATH, f"models/transformer_期待値モデル/{DATE_STRING}")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

DATA_PATH = os.path.join(ROOT_PATH, "data/02_features/feature.csv")
PRED_PATH = os.path.join(ROOT_PATH, "result/predictions/transformer/20250109221743_full.csv")
FUKUSHO_PATH = os.path.join(ROOT_PATH, "data/01_processed/50_odds/odds_df.csv")

SAVE_PATH_PRED = os.path.join(ROOT_PATH, f"result/predictions/transformer_ev/{DATE_STRING}.csv")
SAVE_PATH_FULL_PRED = os.path.join(ROOT_PATH, f"result/predictions/transformer_ev/{DATE_STRING}_full.csv")
SAVE_PATH_MODEL = os.path.join(MODEL_SAVE_DIR, "model_ev.pickle")
SAVE_PATH_PCA_MODEL_HORSE = os.path.join(MODEL_SAVE_DIR, "pcamodel_horse_ev.pickle")
SAVE_PATH_PCA_MODEL_JOCKEY = os.path.join(MODEL_SAVE_DIR, "pcamodel_jockey_ev.pickle")
SAVE_PATH_SCALER_HORSE = os.path.join(MODEL_SAVE_DIR, "scaler_horse_ev.pickle")
SAVE_PATH_SCALER_JOCKEY = os.path.join(MODEL_SAVE_DIR, "scaler_jockey_ev.pickle")
SAVE_PATH_SCALER_OTHER = os.path.join(MODEL_SAVE_DIR, "scaler_other_ev.pickle")

# =====================================================
# Datasetクラス
# =====================================================
class HorseRaceDataset(Dataset):
    """
    sequences: (num_races, max_seq_len, feature_dim)
    labels:    (num_races, max_seq_len, 2)  2 -> [単勝期待値, 複勝期待値]
    masks:     (num_races, max_seq_len)
    race_ids:  (num_races, max_seq_len)
    horse_nums:(num_races, max_seq_len)
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
            #      カテゴリ特徴をある程度圧縮・表現することができる。
            emb_dim_real = min(cat_emb_dim, unique_count // 2 + 1)
            emb_dim_real = max(emb_dim_real, 4)
            self.emb_layers[c] = nn.Embedding(unique_count, emb_dim_real)

        # why: 数値特徴量にはLinearをかけて表現力を増やす
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

        cat_emb = torch.cat(embs, dim=-1)  # (batch, seq_len, sum_of_emb_dims)
        num_emb = self.num_linear(num_x)   # (batch, seq_len, num_dim)

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

        # why: sin, cosを交互に埋め込み、
        #      各位置ごとの周期的パターンでネットワークが位置を認識しやすい。
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        # why: 位置エンコードベクトルを足し込むだけでSelf-Attentionが位置を判断可能
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return x

class HorseTransformer(nn.Module):
    """
    単勝期待値、複勝期待値(2次元)を同時に予測するTransformerモデル
    Embedding + PositionalEncoding + TransformerEncoder -> 最後にLinearで2ターゲット出力
    """
    def __init__(self, cat_unique, cat_cols, max_seq_len, num_dim=50,
                 d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()

        # 特徴量埋め込み
        self.feature_embedder = FeatureEmbedder(
            cat_unique, cat_cols,
            cat_emb_dim=16,
            num_dim=num_dim,
            feature_dim=d_model  # Transformerの入力次元(d_model)に揃える
        )

        # 位置エンコーディング
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)

        # TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 最終出力層 -> 2つ(単勝期待値, 複勝期待値)を回帰で出す
        self.fc_out = nn.Linear(d_model, 2)

    def forward(self, src, src_key_padding_mask=None):
        """
        src_key_padding_mask=True の部分をアテンションから除外することで
        パディング部に対する誤学習を防ぐ
        """
        emb = self.feature_embedder(src)                  # (batch, seq_len, d_model)
        emb = self.pos_encoder(emb)                       # (batch, seq_len, d_model)
        out = self.transformer_encoder(emb, src_key_padding_mask=~src_key_padding_mask)
        logits = self.fc_out(out)                         # (batch, seq_len, 2)
        return logits

# =====================================================
# データ分割 + 前処理周りの関数
# =====================================================
def split_data(df, id_col="race_id", test_ratio=0.1, valid_ratio=0.1):
    """
    日付順でソートし、レースIDを分割。未来データの混入を防ぎつつ
    train->valid->testと時系列分割する
    """
    df = df.sort_values('date').reset_index(drop=True)
    race_ids = df[id_col].unique()
    dataset_len = len(race_ids)

    test_cut = int(dataset_len * (1 - test_ratio))
    valid_cut = int(test_cut * (1 - valid_ratio))

    train_ids = race_ids[:valid_cut]
    valid_ids = race_ids[valid_cut:test_cut]
    test_ids  = race_ids[test_cut:]

    train_df = df[df[id_col].isin(train_ids)].copy()
    valid_df = df[df[id_col].isin(valid_ids)].copy()
    test_df  = df[df[id_col].isin(test_ids)].copy()

    return train_df, valid_df, test_df

def prepare_data_for_ev(
    data_path,
    pred_path,
    fukusho_path,
    id_col="race_id",
    rank_col="着順",
    tansho_col="単勝",  # 払い戻し(総額orオッズ)
    fukusho_col="複勝", # 払い戻し(総額orオッズ)
    pca_dim_horse=50,
    pca_dim_jockey=50,
    test_ratio=0.1,
    valid_ratio=0.1
):
    """
    データを読み込み、以下を実施:
      1) 時系列split
      2) カテゴリ列 + 数値列の分割と加工
      3) horse/jockey専用PCA + その他数値スケーリング
      4) ラベル列: 単勝期待値/複勝期待値
      5) Sequence形式への変換 (パディング & マスク)
    """
    # -----------------------------------------------------
    # 1) CSV読み込み & split
    # -----------------------------------------------------
    df1 = pd.read_csv(data_path, encoding='utf-8-sig')
    df2 = pd.read_csv(pred_path, encoding='utf-8-sig')
    fukusho_df = pd.read_csv(fukusho_path, encoding='utf-8-sig')
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

    # データ分割
    train_df, valid_df, test_df = split_data(df, id_col=id_col,
                                             test_ratio=test_ratio,
                                             valid_ratio=valid_ratio)

    # -----------------------------------------------------
    # 2) カテゴリ列 + 数値列の加工 (leakage項目を排除など)
    #    ※ あくまで例: 不要列やID系は除外
    # -----------------------------------------------------
    # 例: 学習に使わないリーク情報や不要列(=レースの結果そのもの)は除く
    #     必要に応じて調整
    leakage_cols = [
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
    # 数値列・カテゴリ列を抽出
    cat_cols_all = df.select_dtypes(exclude=[np.number]).columns.tolist()
    num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()

    # 除外
    cat_cols = [c for c in cat_cols_all if (c not in leakage_cols) and (c not in [id_col])]
    num_cols = [c for c in num_cols_all if (c not in leakage_cols) and (c not in [id_col])]

    if rank_col in num_cols:
        num_cols.remove(rank_col)
    if tansho_col in num_cols:
        num_cols.remove(tansho_col)
    if fukusho_col in num_cols:
        num_cols.remove(fukusho_col)

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
        train_df[c] = train_df[c].cat.codes.replace(-1, 0)
        valid_df[c] = valid_df[c].cat.codes.replace(-1, 0)
        test_df[c] = test_df[c].cat.codes.replace(-1, 0)

    # -----------------------------------------------------
    # 3) horse/jockey系PCA + その他数値スケーリング
    #    (例: 競走馬芝成績 / 騎手芝成績 など)
    # -----------------------------------------------------
    pca_pattern_horse = r'^(競走馬芝|競走馬ダート|単年競走馬芝|単年競走馬ダート)'
    pca_pattern_jockey = r'^(騎手芝|騎手ダート|単年騎手芝|単年騎手ダート)'

    pca_horse_target_cols = [c for c in num_cols if re.match(pca_pattern_horse, c)]
    pca_jockey_target_cols= [c for c in num_cols if re.match(pca_pattern_jockey, c)]
    other_num_cols = [c for c in num_cols if (c not in pca_horse_target_cols)
                      and (c not in pca_jockey_target_cols)]

    # スケーリング
    scaler_horse = StandardScaler()
    scaler_jockey= StandardScaler()
    scaler_other = StandardScaler()

    # horse
    train_horse_scaled = scaler_horse.fit_transform(train_df[pca_horse_target_cols]) \
        if len(pca_horse_target_cols)>0 else np.zeros((len(train_df),0))
    valid_horse_scaled = scaler_horse.transform(valid_df[pca_horse_target_cols]) \
        if len(pca_horse_target_cols)>0 else np.zeros((len(valid_df),0))
    test_horse_scaled  = scaler_horse.transform(test_df[pca_horse_target_cols]) \
        if len(pca_horse_target_cols)>0 else np.zeros((len(test_df),0))

    pca_dim_horse = min(pca_dim_horse, train_horse_scaled.shape[1]) if len(pca_horse_target_cols)>0 else 0
    if pca_dim_horse > 0:
        pca_model_horse = PCA(n_components=pca_dim_horse)
        train_horse_pca = pca_model_horse.fit_transform(train_horse_scaled)
        valid_horse_pca = pca_model_horse.transform(valid_horse_scaled)
        test_horse_pca  = pca_model_horse.transform(test_horse_scaled)
    else:
        # no columns
        pca_model_horse = None
        train_horse_pca = np.zeros((len(train_df),0))
        valid_horse_pca = np.zeros((len(valid_df),0))
        test_horse_pca  = np.zeros((len(test_df),0))

    # jockey
    train_jockey_scaled = scaler_jockey.fit_transform(train_df[pca_jockey_target_cols]) \
        if len(pca_jockey_target_cols)>0 else np.zeros((len(train_df),0))
    valid_jockey_scaled = scaler_jockey.transform(valid_df[pca_jockey_target_cols]) \
        if len(pca_jockey_target_cols)>0 else np.zeros((len(valid_df),0))
    test_jockey_scaled  = scaler_jockey.transform(test_df[pca_jockey_target_cols]) \
        if len(pca_jockey_target_cols)>0 else np.zeros((len(test_df),0))

    pca_dim_jockey = min(pca_dim_jockey, train_jockey_scaled.shape[1]) if len(pca_jockey_target_cols)>0 else 0
    if pca_dim_jockey > 0:
        pca_model_jockey = PCA(n_components=pca_dim_jockey)
        train_jockey_pca = pca_model_jockey.fit_transform(train_jockey_scaled)
        valid_jockey_pca = pca_model_jockey.transform(valid_jockey_scaled)
        test_jockey_pca  = pca_model_jockey.transform(test_jockey_scaled)
    else:
        pca_model_jockey = None
        train_jockey_pca = np.zeros((len(train_df),0))
        valid_jockey_pca = np.zeros((len(valid_df),0))
        test_jockey_pca  = np.zeros((len(test_df),0))

    # other
    train_other_scaled = scaler_other.fit_transform(train_df[other_num_cols]) \
        if len(other_num_cols)>0 else np.zeros((len(train_df),0))
    valid_other_scaled = scaler_other.transform(valid_df[other_num_cols]) \
        if len(other_num_cols)>0 else np.zeros((len(valid_df),0))
    test_other_scaled  = scaler_other.transform(test_df[other_num_cols]) \
        if len(other_num_cols)>0 else np.zeros((len(test_df),0))

    # 結合
    def concat_features(df_part, cat_cols, other_scaled, horse_pca, jockey_pca):
        cat_values = df_part[cat_cols].values if len(cat_cols)>0 else np.zeros((len(df_part),0))
        return np.concatenate([cat_values, other_scaled, horse_pca, jockey_pca], axis=1)

    X_train = concat_features(train_df, cat_cols, train_other_scaled, train_horse_pca, train_jockey_pca)
    X_valid = concat_features(valid_df, cat_cols, valid_other_scaled, valid_horse_pca, valid_jockey_pca)
    X_test  = concat_features(test_df,  cat_cols, test_other_scaled,  test_horse_pca,  test_jockey_pca)

    # -----------------------------------------------------
    # 4) ラベル列(単勝期待値, 複勝期待値)を作成
    # -----------------------------------------------------
    # "100円賭けたときの期待値"を算出
    # 単勝 => 1着なら (単勝払い戻し - 100), それ以外は -100
    # 複勝 => 3着以内なら (複勝払い戻し - 100), それ以外は -100
    # ※ もし df[tansho_col] / df[fukusho_col] が「オッズ倍(2.5など)」の場合、
    #    実際の払い戻し額は odds*100円なので下記のように変換
    #    例: EV_tansho = (df[tansho_col] * 100 - 100) if 1着 else -100
    # ※ 既に df[tansho_col]/[fukusho_col] が「払い戻し金額(円)」ならそのまま -100
    def get_tansho_ev(row):
        if row[rank_col] == 1:
            # もし df["単勝"] がオッズ倍=2.5 なら row[tansho_col]*100 -100
            # もし df["単勝"] が払い戻し金額=250 なら row[tansho_col] -100
            # 下記はオッズ倍を想定
            return row[tansho_col]*100 - 100
        else:
            return -100

    def get_fukusho_ev(row):
        if row[rank_col] <= 3:
            # 同上の注意。複勝列がオッズ倍なら→ *100 -100
            return row[fukusho_col]*100 - 100
        else:
            return -100

    train_df = train_df.assign(
    EV_tansho=train_df.apply(get_tansho_ev, axis=1),
    EV_fukusho=train_df.apply(get_fukusho_ev, axis=1)
    )

    valid_df = valid_df.assign(
    EV_tansho=valid_df.apply(get_tansho_ev, axis=1),
    EV_fukusho=valid_df.apply(get_fukusho_ev, axis=1)
    )

    test_df = test_df.assign(
    EV_tansho=test_df.apply(get_tansho_ev, axis=1),
    EV_fukusho=test_df.apply(get_fukusho_ev, axis=1)
    )

    # -----------------------------------------------------
    # 5) Sequence化
    #    単勝&複勝(2次元)ラベルをパディングしてマスク
    # -----------------------------------------------------
    def create_sequences(_df, X):
        rids = _df[id_col].values
        horse_nums = _df["馬番"].values if "馬番" in _df.columns else np.arange(len(_df))

        ev_tansho = _df["EV_tansho"].values
        ev_fukusho= _df["EV_fukusho"].values

        groups = _df.groupby(id_col)
        max_seq_len = groups.size().max()
        feature_dim = X.shape[1]

        sequences, labels, masks = [], [], []
        race_ids_seq, horse_nums_seq = [], []

        for unique_rid in _df[id_col].unique():
            idx = np.where(rids == unique_rid)[0]
            feat = X[idx]
            seq_len = len(idx)

            # ラベル
            lab = np.stack([ev_tansho[idx], ev_fukusho[idx]], axis=-1)  # shape: (seq_len, 2)

            # race_id, 馬番
            rid_array = rids[idx]
            horse_array = horse_nums[idx]

            pad_len = max_seq_len - seq_len
            if pad_len > 0:
                feat = np.vstack([feat, np.zeros((pad_len, feature_dim))])
                pad_label = np.zeros((pad_len, 2), dtype=float)
                lab = np.concatenate([lab, pad_label], axis=0)

                # mask: True=実データ, False=pad
                mask = [True]*seq_len + [False]*pad_len

                rid_pad = np.full((pad_len,), fill_value=-1, dtype=rid_array.dtype)
                h_pad   = np.full((pad_len,), fill_value=-1, dtype=horse_array.dtype)
                rid_array   = np.concatenate([rid_array, rid_pad])
                horse_array = np.concatenate([horse_array, h_pad])
            else:
                mask = [True]*seq_len

            sequences.append(feat)
            labels.append(lab)
            masks.append(mask)
            race_ids_seq.append(rid_array)
            horse_nums_seq.append(horse_array)

        return sequences, labels, masks, max_seq_len, race_ids_seq, horse_nums_seq

    train_seq, train_lab, train_mask, max_seq_len_train, train_rids_seq, train_horses_seq = create_sequences(train_df, X_train)
    valid_seq, valid_lab, valid_mask, max_seq_len_valid, valid_rids_seq, valid_horses_seq = create_sequences(valid_df, X_valid)
    test_seq,  test_lab,  test_mask,  max_seq_len_test,  test_rids_seq,  test_horses_seq  = create_sequences(test_df,  X_test)

    max_seq_len = max(max_seq_len_train, max_seq_len_valid, max_seq_len_test)

    # パディング揃え
    def pad_sequences(sequences, labels, masks, rids_seq, horses_seq, seq_len_target):
        feature_dim = sequences[0].shape[1]
        new_seqs, new_labs, new_masks, new_rids, new_horses = [], [], [], [], []
        for feat, lab, m, r_arr, h_arr in zip(sequences, labels, masks, rids_seq, horses_seq):
            cur_len = len(feat)
            if cur_len < seq_len_target:
                pad_len = seq_len_target - cur_len
                feat = np.vstack([feat, np.zeros((pad_len, feature_dim))])
                pad_label = np.zeros((pad_len, 2), dtype=float)
                lab = np.concatenate([lab, pad_label], axis=0)

                m = m + [False]*pad_len

                rid_pad = np.full((pad_len,), fill_value=-1, dtype=r_arr.dtype)
                h_pad   = np.full((pad_len,), fill_value=-1, dtype=h_arr.dtype)
                r_arr   = np.concatenate([r_arr, rid_pad])
                h_arr   = np.concatenate([h_arr, h_pad])

            new_seqs.append(feat)
            new_labs.append(lab)
            new_masks.append(m)
            new_rids.append(r_arr)
            new_horses.append(h_arr)
        return new_seqs, new_labs, new_masks, new_rids, new_horses

    train_seq, train_lab, train_mask, train_rids_seq, train_horses_seq = \
        pad_sequences(train_seq, train_lab, train_mask, train_rids_seq, train_horses_seq, max_seq_len)
    valid_seq, valid_lab, valid_mask, valid_rids_seq, valid_horses_seq = \
        pad_sequences(valid_seq, valid_lab, valid_mask, valid_rids_seq, valid_horses_seq, max_seq_len)
    test_seq,  test_lab,  test_mask,  test_rids_seq,  test_horses_seq  = \
        pad_sequences(test_seq,  test_lab,  test_mask,  test_rids_seq,  test_horses_seq,  max_seq_len)

    # cat_unique
    cat_unique = {}
    for c in cat_cols:
        cat_unique[c] = len(train_df[c].unique()) + 1  # +1 はUNKぶん

    # 数値次元: (other + horsePCA + jockeyPCA)
    actual_num_dim = train_other_scaled.shape[1] + train_horse_pca.shape[1] + train_jockey_pca.shape[1]

    train_dataset = HorseRaceDataset(train_seq, train_lab, train_mask, train_rids_seq, train_horses_seq)
    valid_dataset = HorseRaceDataset(valid_seq, valid_lab, valid_mask, valid_rids_seq, valid_horses_seq)
    test_dataset  = HorseRaceDataset(test_seq,  test_lab,  test_mask,  test_rids_seq,  test_horses_seq)

    total_feature_dim = X_train.shape[1]  # カテゴリ + 数値(=other+pca)
    return (
        train_dataset, valid_dataset, test_dataset,
        cat_cols, cat_unique, max_seq_len,
        pca_dim_horse, pca_dim_jockey,  # PCA構成
        actual_num_dim,
        df,  # 何か後処理に使いたい場合
        cat_cols, num_cols,  # debugging
        pca_horse_target_cols, pca_jockey_target_cols, other_num_cols,
        scaler_horse, scaler_jockey, scaler_other,
        pca_model_horse, pca_model_jockey, total_feature_dim, id_col,
        train_df # train部分DF (何か使う場合)
    )

# =====================================================
# Transformerの学習 (期待値2タスク)
# =====================================================
def run_training_ev(
    data_path=DATA_PATH,
    pred_path = PRED_PATH,
    fukusho_path = FUKUSHO_PATH,
    id_col="race_id",
    rank_col="着順",
    tansho_col="単勝",
    fukusho_col="複勝",
    batch_size=256,
    lr=0.001,
    num_epochs=50,
    pca_dim_horse=50,
    pca_dim_jockey=50,
    test_ratio=0.1,
    valid_ratio=0.1,
    d_model=128,
    nhead=8,
    num_layers=6,
    dropout=0.1,
    weight_decay=1e-5,
    patience=10
):
    """
    1) prepare_data_for_ev() でデータセット作成
    2) HorseTransformer で (単勝期待値, 複勝期待値)を回帰学習
    3) テストで評価
    """
    # 1) データ用意
    (train_dataset, valid_dataset, test_dataset,
     cat_cols, cat_unique, max_seq_len,
     pca_dim_horse, pca_dim_jockey,
     actual_num_dim,
     df_all,
     _, _, 
     _, _, _,
     scaler_horse, scaler_jockey, scaler_other,
     pca_model_horse, pca_model_jockey,
     total_feature_dim, id_col,
     train_df) = prepare_data_for_ev(
        data_path=data_path,
        pred_path=pred_path,
        fukusho_path=fukusho_path,
        id_col=id_col,
        rank_col=rank_col,
        tansho_col=tansho_col,
        fukusho_col=fukusho_col,
        pca_dim_horse=pca_dim_horse,
        pca_dim_jockey=pca_dim_jockey,
        test_ratio=test_ratio,
        valid_ratio=valid_ratio
    )

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # 2) モデル構築
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HorseTransformer(
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # why: 回帰なので MSELoss
    criterion = nn.MSELoss(reduction='none')

    # 早期終了用
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0

    print("===== Training Start =====")
    for epoch in range(num_epochs):
        # --------------------------
        # Train
        # --------------------------
        model.train()
        total_loss_train = 0.0
        total_count_train= 0

        for sequences, labels, masks, _, _ in train_loader:
            sequences = sequences.to(device)
            labels    = labels.to(device)
            masks     = masks.to(device)

            optimizer.zero_grad()
            outputs = model(sequences, src_key_padding_mask=masks)  # (batch, seq_len, 2)
            # MSELoss
            loss_raw = criterion(outputs, labels)  # shape: (batch, seq_len, 2)

            # masks: True=実データ、False=pad.  -> ここではsrc_key_padding_mask=masks で
            # Transformer内部がパディングを除外するが、Loss側でも除外したい場合は
            # ちゃんと使う(本例では ~masks を使うとPad部がFalse->0になる)
            valid_mask_3d = masks.unsqueeze(-1).expand_as(loss_raw)  # shape: (batch, seq_len, 2)
            loss = (loss_raw * valid_mask_3d).sum() / valid_mask_3d.sum()
            loss.backward()
            optimizer.step()

            total_loss_train += loss.item()
            total_count_train += 1

        scheduler.step()
        avg_train_loss = total_loss_train / max(total_count_train, 1)

        # --------------------------
        # Validation
        # --------------------------
        model.eval()
        total_loss_val = 0.0
        total_count_val= 0
        with torch.no_grad():
            for sequences, labels, masks, _, _ in valid_loader:
                sequences = sequences.to(device)
                labels    = labels.to(device)
                masks     = masks.to(device)

                outputs = model(sequences, src_key_padding_mask=masks)
                loss_raw = criterion(outputs, labels)
                valid_mask_3d = masks.unsqueeze(-1).expand_as(loss_raw)
                loss_val = (loss_raw * valid_mask_3d).sum() / valid_mask_3d.sum()

                total_loss_val += loss_val.item()
                total_count_val += 1

        avg_valid_loss = total_loss_val / max(total_count_val, 1)

        print(f"Epoch [{epoch+1}/{num_epochs}]  "
              f"TrainLoss: {avg_train_loss:.4f}  ValidLoss: {avg_valid_loss:.4f}")

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

    # bestモデルを読み込み
    model.load_state_dict(best_model_wts)
    print("===== Training Finished =====")

    # 3) テスト評価
    def evaluate_mse(loader, model):
        model.eval()
        total_loss = 0.0
        count = 0
        preds_list = []
        labels_list= []
        masks_list = []
        rids_list  = []
        hnums_list = []

        with torch.no_grad():
            for sequences, labels, masks, rids, hnums in loader:
                sequences = sequences.to(device)
                labels    = labels.to(device)
                masks     = masks.to(device)

                outputs = model(sequences, src_key_padding_mask=masks)  # (batch, seq_len, 2)
                loss_raw = criterion(outputs, labels)
                valid_mask_3d = masks.unsqueeze(-1).expand_as(loss_raw)
                loss_val = (loss_raw * valid_mask_3d).sum() / valid_mask_3d.sum()

                total_loss += loss_val.item()
                count += 1

                # 予測値, 真値を "有効な部分" だけ取り出してリスト化
                mask_cpu = masks.cpu().numpy().astype(bool)
                outputs_cpu = outputs.cpu().numpy()
                labels_cpu  = labels.cpu().numpy()
                rids_cpu    = rids.cpu().numpy()
                hnums_cpu   = hnums.cpu().numpy()

                for b in range(outputs_cpu.shape[0]):
                    valid_len = mask_cpu[b].sum()
                    preds_list.append(outputs_cpu[b, :valid_len, :])   # shape: (valid_len, 2)
                    labels_list.append(labels_cpu[b, :valid_len, :])    # shape: (valid_len, 2)
                    rids_list.append(rids_cpu[b, :valid_len])
                    hnums_list.append(hnums_cpu[b, :valid_len])
                    masks_list.append(mask_cpu[b,:valid_len])

        avg_loss = total_loss / max(count,1)

        # 結合
        preds_concat = np.concatenate(preds_list, axis=0)
        labels_concat= np.concatenate(labels_list, axis=0)
        rids_concat  = np.concatenate(rids_list, axis=0)
        hnums_concat = np.concatenate(hnums_list, axis=0)

        return avg_loss, preds_concat, labels_concat, rids_concat, hnums_concat

    test_mse, test_preds, test_labels, test_rids, test_hnums = evaluate_mse(test_loader, model)
    print(f"Test MSE: {test_mse:.4f}")

    # 結果をDataFrame化して保存
    # test_preds[:, 0] = 単勝期待値予測, test_preds[:, 1] = 複勝期待値予測
    # test_labels[:,0] = 単勝期待値実測, test_labels[:,1] = 複勝期待値実測
    result_df = pd.DataFrame({
        "race_id": test_rids.astype(int),
        "馬番": test_hnums.astype(int),
        "pred_tansho_ev": test_preds[:,0],
        "pred_fukusho_ev":test_preds[:,1],
        "true_tansho_ev": test_labels[:,0],
        "true_fukusho_ev":test_labels[:,1],
    })
    result_df.to_csv(SAVE_PATH_PRED, index=False)

    # 追加で train+valid+test 全部まとめた予測が欲しい場合:
    # 例として、ConcatDatasetを使って一括評価
    full_dataset = ConcatDataset([train_dataset, valid_dataset, test_dataset])
    full_loader  = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    full_mse, full_preds, full_labels, full_rids, full_hnums = evaluate_mse(full_loader, model)
    print(f"Full MSE: {full_mse:.4f}")

    full_pred_df = pd.DataFrame({
        "race_id": full_rids.astype(int),
        "馬番": full_hnums.astype(int),
        "pred_tansho_ev": full_preds[:,0],
        "pred_fukusho_ev":full_preds[:,1],
        "true_tansho_ev": full_labels[:,0],
        "true_fukusho_ev":full_labels[:,1],
    })
    full_pred_df.to_csv(SAVE_PATH_FULL_PRED, index=False)

    # 4) モデル保存
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

    print("Model saved.")

    # 結果の可視化
    plot_ev_scatter(result_df)
    calc_positive_ev_gain(result_df)
    plot_error_distribution(result_df)
    plot_cumulative_return(result_df, is_tansho=True)   # 単勝
    plot_cumulative_return(result_df, is_tansho=False)  # 複勝

    return 0

def plot_ev_scatter(result_df):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))

    # 単勝散布図
    axes[0].scatter(result_df["true_tansho_ev"], result_df["pred_tansho_ev"], alpha=0.3)
    axes[0].set_title("Tansho EV: True vs Pred")
    axes[0].set_xlabel("True Tansho EV")
    axes[0].set_ylabel("Predicted Tansho EV")
    # 対角線
    min_val = min(result_df["true_tansho_ev"].min(), result_df["pred_tansho_ev"].min())
    max_val = max(result_df["true_tansho_ev"].max(), result_df["pred_tansho_ev"].max())
    axes[0].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

    # 複勝散布図
    axes[1].scatter(result_df["true_fukusho_ev"], result_df["pred_fukusho_ev"], alpha=0.3)
    axes[1].set_title("Fukusho EV: True vs Pred")
    axes[1].set_xlabel("True Fukusho EV")
    axes[1].set_ylabel("Predicted Fukusho EV")
    min_val = min(result_df["true_fukusho_ev"].min(), result_df["pred_fukusho_ev"].min())
    max_val = max(result_df["true_fukusho_ev"].max(), result_df["pred_fukusho_ev"].max())
    axes[1].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

    plt.tight_layout()
    plt.show()

def calc_positive_ev_gain(result_df):
    # 単勝で予測がプラスの馬
    df_tansho_positive = result_df[result_df["pred_tansho_ev"] > 0]
    total_tansho_real = df_tansho_positive["true_tansho_ev"].sum()

    # 複勝で予測がプラスの馬
    df_fukusho_positive = result_df[result_df["pred_fukusho_ev"] > 0]
    total_fukusho_real = df_fukusho_positive["true_fukusho_ev"].sum()

    print("【予測EVが+のレコードについて】")
    print(f"- 単勝: True EV 合計 = {total_tansho_real:.2f}")
    print(f"- 複勝: True EV 合計 = {total_fukusho_real:.2f}")

def plot_error_distribution(result_df):
    # 単勝誤差
    tansho_error = result_df["true_tansho_ev"] - result_df["pred_tansho_ev"]
    # 複勝誤差
    fukusho_error = result_df["true_fukusho_ev"] - result_df["pred_fukusho_ev"]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,5))

    axes[0].hist(tansho_error, bins=50, alpha=0.7, color='b')
    axes[0].set_title("Tansho EV Residuals (True - Pred)")
    axes[0].set_xlabel("Residual Value")
    axes[0].set_ylabel("Count")

    axes[1].hist(fukusho_error, bins=50, alpha=0.7, color='g')
    axes[1].set_title("Fukusho EV Residuals (True - Pred)")
    axes[1].set_xlabel("Residual Value")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.show()

def plot_cumulative_return(result_df, is_tansho=True):
    # 単勝 or 複勝
    pred_col = "pred_tansho_ev" if is_tansho else "pred_fukusho_ev"
    true_col = "true_tansho_ev" if is_tansho else "true_fukusho_ev"

    # 予測EVの降順に並べ替え
    df_sorted = result_df.sort_values(by=pred_col, ascending=False).reset_index(drop=True)
    # 実際のEVの累積和(買い目ごとに-100円している想定なので、df[true_col] そのものが収益)
    cumsum_true = df_sorted[true_col].cumsum()
    # 購入レース数に応じて何円使ったか = 100円ずつ × レース数
    num_races = np.arange(1, len(df_sorted) + 1)
    cost = 100 * num_races
    # 回収率 = 累積収益 / 累積コスト
    cumsum_return_rate = cumsum_true / cost

    plt.figure(figsize=(8,5))
    plt.plot(num_races, cumsum_return_rate, marker='o', markersize=2, linestyle='-')
    plt.axhline(y=1.0, color='red', linestyle='--')  # 回収率100%ライン
    title_name = "Tansho" if is_tansho else "Fukusho"
    plt.title(f"Cumulative Return Rate ({title_name})")
    plt.xlabel("Number of Bets (in descending order of predicted EV)")
    plt.ylabel("Cumulative Return Rate")
    plt.ylim(0, 2)  # 適宜調整
    plt.grid(True)
    plt.show()

# 実行例
if __name__ == "__main__":
    model, test_result, full_result = run_training_ev(
        data_path=DATA_PATH,
        id_col="race_id",
        rank_col="着順",
        tansho_col="単勝",
        fukusho_col="複勝",
        batch_size=128,
        lr=0.0005,
        num_epochs=30,
        pca_dim_horse=50,
        pca_dim_jockey=50,
        test_ratio=0.15,
        valid_ratio=0.1,
        d_model=128,
        nhead=4,
        num_layers=4,
        dropout=0.1,
        weight_decay=1e-5,
        patience=5
    )

    print("Done. Test shape:", test_result.shape)
    print("Done. Full shape:", full_result.shape)