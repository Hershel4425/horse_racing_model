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
    学習/推論に用いる独自Datasetクラス。

    Args:
        sequences (List[np.ndarray]): 各レースの特徴量時系列 (num_races, max_seq_len, feature_dim)
        labels_win (List[np.ndarray]): is_win(1 or 0) のラベル (num_races, max_seq_len)
        labels_multi (List[np.ndarray]): 6つの2値ラベル (num_races, max_seq_len, 6)
        labels_tansho (List[np.ndarray]): "0.8 / 単勝" で求めた数値ラベル (num_races, max_seq_len)
        base_supports (List[np.ndarray]): ベースサポート(0.8 / 単勝をさらに推定した差分のベース) (num_races, max_seq_len)
        masks (List[np.ndarray]): マスク (有効データ=True, パディング=False) (num_races, max_seq_len)
        race_ids (List[np.ndarray]): レースID (num_races, max_seq_len)
        horse_nums (List[np.ndarray]): 馬番 (num_races, max_seq_len)
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

    def __len__(self) -> int:
        """レース数を返す。"""
        return len(self.sequences)

    def __getitem__(self, idx: int):
        """idx番目のレースデータを返す。各返り値はtorch.Tensorにキャスト。"""
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
    """
    カテゴリ列をEmbeddingし、数値列にLinearをかけた後、結合して出力次元を合わせる。

    Args:
        cat_unique (dict): {カテゴリ列名: ユニーク数} の辞書
        cat_cols (List[str]): カテゴリ列の名前リスト
        cat_emb_dim (int): カテゴリ埋め込みの最大次元
        num_dim (int): 数値列の次元数
        feature_dim (int): 埋め込み後の出力次元数 (= d_model)
    """
    def __init__(self, cat_unique: dict, cat_cols: list, cat_emb_dim=16,
                 num_dim=50, feature_dim=128):
        super().__init__()
        self.cat_cols = cat_cols
        # 各カテゴリ列のEmbedding層を用意
        self.emb_layers = nn.ModuleDict()

        for c in cat_cols:
            unique_count = cat_unique[c]
            # Embedding次元は「cat_emb_dimか、(カテゴリ数//2 + 1)の最小値」などで調整
            emb_dim_real = min(cat_emb_dim, unique_count // 2 + 1)
            # あまりに小さくなりすぎないよう下限も設定
            emb_dim_real = max(emb_dim_real, 4)
            self.emb_layers[c] = nn.Embedding(num_embeddings=unique_count,
                                              embedding_dim=emb_dim_real)

        # 数値データの変換用 (線形層)
        self.num_linear = nn.Linear(num_dim, num_dim)

        # 最終的に (カテゴリ埋め込み + 数値線形) を結合して、feature_dim に落とす
        cat_out_dim = sum([self.emb_layers[c].embedding_dim for c in self.cat_cols])
        self.out_linear = nn.Linear(cat_out_dim + num_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (B, L, cat_len+num_dim)
           先頭 cat_len 列がカテゴリ、それ以降が数値
        """
        cat_len = len(self.cat_cols)
        # カテゴリ部分(整数)と数値部分(浮動小数)を分割
        cat_x = x[..., :cat_len].long()
        num_x = x[..., cat_len:]

        # カテゴリEmbeddingを列ごとに取得
        embs = []
        for i, c in enumerate(self.cat_cols):
            embs.append(self.emb_layers[c](cat_x[..., i]))

        # 結合
        cat_emb = torch.cat(embs, dim=-1)
        # 数値列にも1層Linear
        num_emb = self.num_linear(num_x)
        out = torch.cat([cat_emb, num_emb], dim=-1)
        # 出力を所望の次元(d_model)へ
        out = self.out_linear(out)
        return out


class PositionalEncoding(nn.Module):
    """
    Transformerで使用する位置エンコーディング(絶対位置エンコーディング)。
    以下の式を使って固定の三角関数テーブルを作る。

    PE(pos,2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos,2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model (int): 入力埋め込み次元
        max_len (int): 最大系列長
    """
    def __init__(self, d_model: int, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # 奇数次元の場合は末尾1次元だけ計算が合わないので調整
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        # register_bufferで学習パラメータにはしない
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (B,L,d_model)
        """
        seq_len = x.size(1)
        # xに直接足し込む（ブロードキャスト適用）
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return x


class HorseTransformerDiffBCE(nn.Module):
    """
    HorseRaceDatasetDiffBCE 用のTransformerモデル。

    出力: shape (B, L, 8)
      [0]: diff (ベースサポートに対する差分を出力)
      [1..6]: 6つの2値分類タスク用のスコア (top1, top3, top5, pop1, pop3, pop5)
      [7]:  "0.8 / 単勝" の回帰タスク用出力 (生値)

    Args:
        cat_unique (dict): {カテゴリ列名: ユニーク数} の辞書
        cat_cols (List[str]): カテゴリ列の名前
        max_seq_len (int): 最大系列長 (PositionalEncodingなどで使用)
        num_dim (int): 数値特徴量の次元数 (馬データPCA + 騎手PCA + その他スカラー)
        d_model (int): Transformer埋め込み次元
        nhead (int): Multi-head Attention のヘッド数
        num_layers (int): TransformerEncoderの層数
        dropout (float): ドロップアウト率
    """
    def __init__(self,
                 cat_unique: dict, 
                 cat_cols: list,
                 max_seq_len: int,
                 num_dim=50,
                 d_model=128,
                 nhead=4,
                 num_layers=4,
                 dropout=0.1):
        super().__init__()

        # 特徴量Embedder (カテゴリ+数値)
        self.feature_embedder = FeatureEmbedder(
            cat_unique=cat_unique,
            cat_cols=cat_cols,
            cat_emb_dim=16,
            num_dim=num_dim,
            feature_dim=d_model
        )

        # 位置エンコーディング
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)

        # TransformerEncoder本体 (batch_first=True で (B,L,d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 出力層 (8次元)
        self.fc_out = nn.Linear(d_model, 8)

    def forward(self,
                src: torch.Tensor,
                src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src (torch.Tensor): shape (B,L,cat_len+num_dim)
            src_key_padding_mask (torch.Tensor): shape (B,L)
               True=有効, False=パディング のマスク
               PyTorchのTransformerでは "True=パディング" がデフォルトだが、
               ここでは逆の論理になっているので注意して呼び出す。
        """
        # 特徴量を埋め込み
        emb = self.feature_embedder(src)      # (B,L,d_model)
        emb = self.pos_encoder(emb)          # (B,L,d_model)

        # TransformerEncoderにマスクを渡す場合は ~src_key_padding_mask (論理反転)が必要
        out = self.transformer_encoder(emb, src_key_padding_mask=~src_key_padding_mask)
        out = self.fc_out(out)  # (B,L,8)
        return out


########################################
# 3) データ準備用の例関数 (省略部分あり)
########################################
def split_data(df, id_col="race_id", test_ratio=0.1, valid_ratio=0.1):
    """
    時系列(開催日順)にソートして train/valid/test に分割する。

    Args:
        df (pd.DataFrame): 入力データ
        id_col (str): レースIDのカラム名
        test_ratio (float): テストセット割合
        valid_ratio (float): バリデーションセット割合

    Returns:
        (train_df, valid_df, test_df): 分割済みDataFrame
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
    rank_col="着順",   
    pop_col="単勝",    
    test_ratio=0.1,
    valid_ratio=0.1,
    pca_dim_horse=50,
    pca_dim_jockey=50
):
    """
    Diff+BCEタスク用のデータセットを作成する。

    - feature.csv と 予測スコアcsv (例: "P_top1","P_pop1"など) を結合し、
      is_winやlabel_top3などのラベルを追加。
    - horse/jockey系の特徴量をPCAする。
    - レースごとにpadしてDatasetを作成。

    Returns:
        (train_dataset, valid_dataset, test_dataset,
         cat_cols, cat_unique, max_seq_len,
         actual_num_dim,
         pca_model_horse, pca_model_jockey,
         scaler_horse, scaler_jockey, scaler_other,
         df,  # 全体DF
         total_feature_dim)
    """
    df_feature = pd.read_csv(data_path, encoding='utf-8-sig')
    df_pred    = pd.read_csv(pred_path, encoding='utf-8-sig')

    # feature.csvと予測スコアをマージ
    df = pd.merge(
        df_feature,
        df_pred[["race_id","馬番","P_top1","P_pop1","P_top3","P_pop3","P_top5","P_pop5"]],
        on=["race_id","馬番"],
        how="inner"
    )

    # 1) is_win (1着or0)
    df["is_win"] = (df[rank_col] == 1).astype(int)

    # 2) 6タスク (top1, top3, top5, pop1, pop3, pop5) → 2値 (0 or 1)
    df["label_top1"] = (df[rank_col] <= 1).astype(int)
    df["label_top3"] = (df[rank_col] <= 3).astype(int)
    df["label_top5"] = (df[rank_col] <= 5).astype(int)

    # 人気をpop_col=単勝で近似(ただし"人気"列があればそれを優先可)
    df["label_pop1"] = (df["人気"] <= 1).astype(int)
    df["label_pop3"] = (df["人気"] <= 3).astype(int)
    df["label_pop5"] = (df["人気"] <= 5).astype(int)

    # 3) base_support (0.8/単勝)
    df["base_support"] = 0.8 / (df[pop_col] + 1e-10)

    # ---- train/valid/test に分割 ----
    train_df, valid_df, test_df = split_data(df, id_col=id_col,
                                             test_ratio=test_ratio,
                                             valid_ratio=valid_ratio)

    # カテゴリ列/数値列を抽出
    cat_cols_all = train_df.select_dtypes(exclude=[np.number]).columns.tolist()
    num_cols_all = train_df.select_dtypes(include=[np.number]).columns.tolist()

    # 学習時には情報リークになる列を除外 (着順、単勝、人気など)
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
            'label_top1', 'label_top3', 'label_top5', 'label_pop1', 'label_pop3', 'label_pop3'
        ]
    cat_cols = [c for c in cat_cols_all if c not in leakage_cols and c not in [id_col]]
    num_cols = [c for c in num_cols_all if c not in leakage_cols and c not in [id_col]]

    # カテゴリ列のNaN→"missing"、数値列のNaN→0
    for c in cat_cols:
        for d in [train_df, valid_df, test_df]:
            d[c] = d[c].fillna("missing").astype(str)
    for n in num_cols:
        for d in [train_df, valid_df, test_df]:
            d[n] = d[n].fillna(0)

    # カテゴリ列をcategory→codes化
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

    # Horse/Jockey用PCA対象列を判定 (例: '競走馬芝xxx', '騎手芝xxx' など)
    pca_pattern_horse = r'^(競走馬芝|競走馬ダート)'
    pca_pattern_jockey= r'^(騎手芝|騎手ダート)'

    pca_horse_target_cols = [c for c in num_cols if re.match(pca_pattern_horse, c)]
    pca_jockey_target_cols= [c for c in num_cols if re.match(pca_pattern_jockey, c)]
    other_num_cols = [c for c in num_cols if c not in pca_horse_target_cols + pca_jockey_target_cols]

    # StandardScaler用意
    scaler_horse = StandardScaler()
    scaler_jockey= StandardScaler()
    scaler_other = StandardScaler()

    # horse/jockey/otherそれぞれ学習データでfit
    if len(pca_horse_target_cols)>0:
        scaler_horse.fit(train_df[pca_horse_target_cols])
    if len(pca_jockey_target_cols)>0:
        scaler_jockey.fit(train_df[pca_jockey_target_cols])
    if len(other_num_cols)>0:
        scaler_other.fit(train_df[other_num_cols])

    # 変換
    horse_train_arr = scaler_horse.transform(train_df[pca_horse_target_cols]) if len(pca_horse_target_cols)>0 else np.zeros((len(train_df),0))
    horse_valid_arr = scaler_horse.transform(valid_df[pca_horse_target_cols]) if len(pca_horse_target_cols)>0 else np.zeros((len(valid_df),0))
    horse_test_arr  = scaler_horse.transform(test_df[pca_horse_target_cols])  if len(pca_horse_target_cols)>0 else np.zeros((len(test_df),0))

    # PCA(馬)
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

    # PCA(騎手)
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

    # 結合関数
    def concat_features(df_part, cat_cols, other_arr, horse_pca, jockey_pca):
        cat_val = df_part[cat_cols].values if len(cat_cols)>0 else np.zeros((len(df_part),0))
        return np.concatenate([cat_val, other_arr, horse_pca, jockey_pca], axis=1)

    X_train = concat_features(train_df, cat_cols, other_train_arr, horse_train_pca, jockey_train_pca)
    X_valid = concat_features(valid_df, cat_cols, other_valid_arr, horse_valid_pca, jockey_valid_pca)
    X_test  = concat_features(test_df,  cat_cols, other_test_arr,  horse_test_pca,  jockey_test_pca)

    # ラベル類
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

    # "0.8/単勝"
    y_train_tansho = 0.8 / (train_df[pop_col].values + EPS)
    y_valid_tansho = 0.8 / (valid_df[pop_col].values + EPS)
    y_test_tansho  = 0.8 / (test_df[pop_col].values + EPS)

    sup_train = train_df["base_support"].values
    sup_valid = valid_df["base_support"].values
    sup_test  = test_df["base_support"].values

    # レースIDごとに系列化(同じレース内で馬のエントリが時系列のように並ぶ)
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
                # zero padding
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

    # Dataset化
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
    d_model=64,
    nhead=8,
    num_layers=6,
    dropout=0.15,
    weight_decay=1e-5,
    patience=10
):
    """
    Diff + BCE の学習を行うメイン関数。

    - Transformerモデルの構築
    - (diff_out, multi_out, tansho_out) 用にカスタムロスを定義
    - train/validの早期終了
    - テストデータで評価→結果csvを出力
    - モデル保存、前処理(Scaler/PCA)保存
    """
    # ============ 1) prepare_dataでDatasetなど取得 ============
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

    # GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============ 2) モデル準備 ============
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

    # optimizer, scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # ============ 3) 損失関数定義 ============
    def custom_loss_fn(outputs: torch.Tensor,
                       label_win: torch.Tensor,
                       label_multi: torch.Tensor,
                       label_tansho: torch.Tensor,
                       base_sups: torch.Tensor,
                       masks: torch.Tensor):
        """
        outputs: shape (B,L,8)
           [0]: diff
           [1..6]: 6分類タスク
           [7]: tan
        label_win: shape (B,L)
        label_multi: shape (B,L,6)
        label_tansho: shape (B,L)
        base_sups: shape (B,L)
        masks: shape (B,L) True=有効, False=PAD
        """
        diff_out   = outputs[..., 0]      # (B,L)
        multi_out  = outputs[..., 1:7]    # (B,L,6)
        tansho_out = outputs[..., 7]      # (B,L)

        eps = 1e-7
        # (a) diff 用 BCE
        #    logit(base_support) + diff_out → 予測確率を求める
        logit_bs = torch.logit(base_sups, eps=eps)  # base_supsは(0.8/単勝)
        pred_prob_win = torch.sigmoid(logit_bs + diff_out)
        # BCEを手書き（reduction='none'を使うため）
        bce_diff = - (label_win * torch.log(pred_prob_win+eps) 
                      + (1 - label_win)*torch.log(1-pred_prob_win+eps))

        # (b) 6分類タスク (top1, top3, top5, pop1, pop3, pop5)
        bce_multi = nn.functional.binary_cross_entropy_with_logits(multi_out,
                                                                   label_multi,
                                                                   reduction='none')

        # (c) "0.8/単勝" の MSE
        mse_tansho = (tansho_out - label_tansho)**2

        # --- パディング部を除外する ---
        masks_3d = masks.unsqueeze(-1)  # (B,L,1)
        bce_diff   = bce_diff * masks
        bce_multi  = bce_multi * masks_3d
        mse_tansho = mse_tansho * masks

        denom = masks.sum() + eps
        # 各ロスを平均
        loss_diff   = bce_diff.sum() / denom
        loss_multi  = bce_multi.sum() / (denom*6)  # 6タスク分で割る
        loss_tansho = mse_tansho.sum() / denom

        # 合計を最終ロス
        loss = loss_diff + loss_multi + loss_tansho
        return loss, loss_diff.item(), loss_multi.item(), loss_tansho.item()

    def calc_metrics(outputs: torch.Tensor,
                     label_win: torch.Tensor,
                     label_multi: torch.Tensor,
                     label_tansho: torch.Tensor,
                     base_sups: torch.Tensor,
                     masks: torch.Tensor):
        """
        検証用の指標計算 (diff部分のBCE, multiのaccuracy, tanのRMSE)
        """
        diff_out   = outputs[..., 0]
        multi_out  = outputs[..., 1:7]
        tansho_out = outputs[..., 7]

        eps=1e-7
        # diff part BCE
        logit_bs = torch.logit(base_sups, eps=eps)
        pred_prob_win = torch.sigmoid(logit_bs + diff_out)
        bce_diff = - (label_win*torch.log(pred_prob_win+eps) + (1-label_win)*torch.log(1-pred_prob_win+eps))
        bce_diff = bce_diff * masks
        loss_diff_val = bce_diff.sum() / (masks.sum() + eps)

        # multi accuracy
        multi_prob = torch.sigmoid(multi_out)  
        multi_pred = (multi_prob >= 0.5).float()  # threshold=0.5
        correct = (multi_pred == label_multi).float()
        correct = correct * masks.unsqueeze(-1)
        acc_each_task = correct.sum(dim=(0,1)) / (masks.sum() + eps)  # 各タスク別

        # tan RMSE
        mse_tansho = (tansho_out - label_tansho)**2
        mse_tansho = mse_tansho * masks
        mse_val = mse_tansho.sum() / (masks.sum() + eps)
        rmse_val = mse_val.sqrt()

        return (loss_diff_val.item(),
                acc_each_task.cpu().numpy(),
                rmse_val.item())

    # ============ 4) 学習ループ ============
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

                # metrics
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

    print("=== Training Finished ===")
    model.load_state_dict(best_model_wts)

    # ============ 5) テストデータで評価 & 予測結果出力 ============
    def predict_prob_tansho(loader, model):
        """
        モデル推論の共通関数。
        - diff_out / multi_out / tan_out の予測値、マスク除外後のリスト
        - 同時にラベルや race_id, 馬番 も取得

        Returns:
            pw   (np.ndarray): shape (全件,)
            pm   (np.ndarray): shape (全件,6)
            pt   (np.ndarray): shape (全件,)
            lw, lm, lt, rr, hh (同様に)
            diff_out (np.ndarray): shape (全件,)
        """
        model.eval()
        all_pred_win = []    # diffで求めた最終確率
        all_pred_multi = []
        all_pred_tansho= []
        all_diff_out   = []

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
                diff_out   = out[..., 0]           # (B,L)
                multi_out  = out[..., 1:7]         # (B,L,6)
                tansho_out = out[..., 7]          # (B,L)

                # diffからwin確率に変換
                eps = 1e-7
                logit_bs = torch.logit(sup, eps=eps)
                pred_prob_win_ = torch.sigmoid(logit_bs + diff_out)
                pred_multi_    = torch.sigmoid(multi_out)

                # マスク除外
                msk_np = msk.cpu().numpy()

                pred_win_np   = pred_prob_win_.cpu().numpy()
                pred_multi_np = pred_multi_.cpu().numpy()
                pred_tns_np   = tansho_out.cpu().numpy()
                diff_np       = diff_out.cpu().numpy()

                label_win_np  = win.cpu().numpy()
                label_mlt_np  = mlt.cpu().numpy()
                label_tns_np  = tns.cpu().numpy()

                rids_np = rids.cpu().numpy()
                hnums_np= hnums.cpu().numpy()

                B,L = msk_np.shape
                for b in range(B):
                    valid_len = int(msk_np[b].sum())  # Trueの数
                    all_pred_win.append(pred_win_np[b,:valid_len])
                    all_pred_multi.append(pred_multi_np[b,:valid_len])
                    all_pred_tansho.append(pred_tns_np[b,:valid_len])
                    all_diff_out.append(diff_np[b,:valid_len])

                    all_label_win.append(label_win_np[b,:valid_len])
                    all_label_multi.append(label_mlt_np[b,:valid_len])
                    all_label_tansho.append(label_tns_np[b,:valid_len])

                    all_rids.append(rids_np[b,:valid_len])
                    all_hnums.append(hnums_np[b,:valid_len])

        pw   = np.concatenate(all_pred_win,   axis=0)
        pm   = np.concatenate(all_pred_multi, axis=0)
        pt   = np.concatenate(all_pred_tansho,axis=0)
        df_  = np.concatenate(all_diff_out,   axis=0)

        lw   = np.concatenate(all_label_win,  axis=0)
        lm   = np.concatenate(all_label_multi,axis=0)
        lt   = np.concatenate(all_label_tansho,axis=0)

        rr   = np.concatenate(all_rids,   axis=0)
        hh   = np.concatenate(all_hnums,  axis=0)

        return pw, pm, pt, lw, lm, lt, rr, hh, df_

    # テストデータに対する推論
    test_pw, test_pm, test_pt, test_lw, test_lm, test_lt, test_rid, test_hnum, test_diff = predict_prob_tansho(test_loader, model)

    # 1) diff部のBCEロス計算
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

    # ============ 6) CSV出力用: test_result_df, full_result_df ============
    # まずはテスト用の結果を作りたいので、元の df_all から、race_id & 馬番 で結合(merge)して
    # 真の単勝や着順、人気を取得する。
    # 注意: df_all には複数行同一(race_id,馬番)がない前提(なければdrop_duplicatesするなど)
    # ここでは念のため race_id,馬番 で一意になることを想定。

    df_all_test_part = df_all[df_all[id_col].isin(test_rid)].copy()
    df_all_test_part = df_all_test_part.drop_duplicates(subset=[id_col, "馬番"])
    # マージ用にキーを揃えるために DataFrame を作成
    test_pred_df = pd.DataFrame({
        "race_id": test_rid.astype(int),
        "馬番"   : test_hnum.astype(int),
        "pred_win":  test_pw,              # 最終 win 確率
        "diff"    : test_diff,             # diff (logit空間)
        "pred_tansho": test_pt,            # 予測の 0.8/単勝
        # multiタスク: (top1,top3,top5,pop1,pop3,pop5)
        "pred_top1": test_pm[:, 0],
        "pred_top3": test_pm[:, 1],
        "pred_top5": test_pm[:, 2],
        "pred_pop1": test_pm[:, 3],
        "pred_pop3": test_pm[:, 4],
        "pred_pop5": test_pm[:, 5],
    })

    # マージ
    test_result_merged_df = pd.merge(
        df_all_test_part,
        test_pred_df,
        on=["race_id","馬番"],
        how="inner"
    )

    # 指定のカラム構成にする
    #   race_id,馬番,真の単勝,真の着順,真の人気,diff,予測の0.8/単勝,予測のtop1,...pop5
    #   ※ 真の単勝=pop_col, 真の着順=rank_col, 真の人気="人気"
    test_result_final_df = test_result_merged_df[[
        "race_id",
        "馬番",
        pop_col,        # 真の単勝
        rank_col,       # 真の着順
        "人気",         # 真の人気
        "diff",         # 予測diff
        "pred_tansho",  # 予測の0.8/単勝
        "pred_top1",
        "pred_top3",
        "pred_top5",
        "pred_pop1",
        "pred_pop3",
        "pred_pop5",
    ]].copy()

    # 列名変更
    test_result_final_df.rename(columns={
        pop_col: "真の単勝",
        rank_col: "真の着順",
        "人気": "真の人気",
        "pred_tansho": "予測の0.8/単勝",
        "pred_top1":   "予測のtop1",
        "pred_top3":   "予測のtop3",
        "pred_top5":   "予測のtop5",
        "pred_pop1":   "予測のpop1",
        "pred_pop3":   "予測のpop3",
        "pred_pop5":   "予測のpop5",
    }, inplace=True)

    # CSV保存 (テストデータ)
    test_result_final_df.to_csv(SAVE_PATH_PRED, index=False)
    print("Saved test predictions:", SAVE_PATH_PRED)

    # 全データ (train+valid+test) 用にも同様に行う
    full_dataset = ConcatDataset([train_dataset, valid_dataset, test_dataset])
    full_loader  = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    full_pw, full_pm, full_pt, full_lw, full_lm, full_lt, full_rid, full_hnum, full_diff = predict_prob_tansho(full_loader, model)

    df_all_full_part = df_all.drop_duplicates(subset=[id_col, "馬番"]).copy()
    full_pred_df = pd.DataFrame({
        "race_id": full_rid.astype(int),
        "馬番"   : full_hnum.astype(int),
        "pred_win":  full_pw,
        "diff"    : full_diff,
        "pred_tansho": full_pt,
        "pred_top1": full_pm[:, 0],
        "pred_top3": full_pm[:, 1],
        "pred_top5": full_pm[:, 2],
        "pred_pop1": full_pm[:, 3],
        "pred_pop3": full_pm[:, 4],
        "pred_pop5": full_pm[:, 5],
    })
    full_result_merged_df = pd.merge(
        df_all_full_part,
        full_pred_df,
        on=["race_id","馬番"],
        how="inner"
    )
    full_result_final_df = full_result_merged_df[[
        "race_id",
        "馬番",
        pop_col,
        rank_col,
        "人気",
        "diff",
        "pred_tansho",
        "pred_top1",
        "pred_top3",
        "pred_top5",
        "pred_pop1",
        "pred_pop3",
        "pred_pop5",
    ]].copy()
    full_result_final_df.rename(columns={
        pop_col: "真の単勝",
        rank_col: "真の着順",
        "人気": "真の人気",
        "pred_tansho": "予測の0.8/単勝",
        "pred_top1":   "予測のtop1",
        "pred_top3":   "予測のtop3",
        "pred_top5":   "予測のtop5",
        "pred_pop1":   "予測のpop1",
        "pred_pop3":   "予測のpop3",
        "pred_pop5":   "予測のpop5",
    }, inplace=True)

    # 保存
    full_result_final_df.to_csv(SAVE_PATH_FULL_PRED, index=False)
    print("Saved full predictions:", SAVE_PATH_FULL_PRED)

    # 0.8/単勝 のRMSE (full)
    mse_full_tansho = ((full_pt - full_lt)**2).mean()
    rmse_full_tansho= np.sqrt(mse_full_tansho)
    print(f"Full RMSE (0.8/単勝) = {rmse_full_tansho:.4f}")

    # ============ 7) 簡単な可視化 (テストデータで) ============
    print("=== Visualization / Metrics on test data ===")
    # ここでは test_result_final_df を使って可視化する。
    # まず「1着率」や「回収率」を計算するために is_win(真の着順==1) を作っておく。
    test_result_final_df["is_win"] = (test_result_final_df["真の着順"] == 1).astype(int)

    # ------------------------
    # (a) 真の人気が1の馬
    # ------------------------
    cond_ninki1 = (test_result_final_df["真の人気"] == 1)
    n_ninki1 = cond_ninki1.sum()
    if n_ninki1 > 0:
        win_rate_ninki1 = test_result_final_df.loc[cond_ninki1, "is_win"].mean()
        # 回収率: 1着なら 真の単勝 × 100円
        #         totalReturn / ( n_ninki1 * 100 )
        total_return_ninki1 = (test_result_final_df.loc[cond_ninki1, "is_win"] * 
                               test_result_final_df.loc[cond_ninki1, "真の単勝"] * 100).sum()
        roi_ninki1 = total_return_ninki1 / (n_ninki1 * 100)
        print(f"真の人気=1馬: 件数={n_ninki1}, 1着率={win_rate_ninki1:.3f}, 回収率={roi_ninki1:.3f}")
    else:
        print("真の人気=1の馬がありません。")

    # ------------------------
    # (b) 予測のtop1(>=0.5)の馬
    # ------------------------
    cond_pred_top1 = (test_result_final_df["予測のtop1"] >= 0.5)
    n_pred_top1 = cond_pred_top1.sum()
    if n_pred_top1 > 0:
        win_rate_pred_top1 = test_result_final_df.loc[cond_pred_top1, "is_win"].mean()
        total_return_pred_top1 = (test_result_final_df.loc[cond_pred_top1, "is_win"] *
                                  test_result_final_df.loc[cond_pred_top1, "真の単勝"] * 100).sum()
        roi_pred_top1 = total_return_pred_top1 / (n_pred_top1 * 100)
        print(f"予測top1(>=0.5)馬: 件数={n_pred_top1}, 1着率={win_rate_pred_top1:.3f}, 回収率={roi_pred_top1:.3f}")
    else:
        print("予測top1(>=0.5)の馬がありません。")

    # ------------------------
    # (c) diff > 0 の馬
    #  diffは logit空間の差分
    # ------------------------
    cond_diff_pos = (test_result_final_df["diff"] > 0)
    n_diff_pos = cond_diff_pos.sum()
    if n_diff_pos > 0:
        win_rate_diff_pos = test_result_final_df.loc[cond_diff_pos, "is_win"].mean()
        total_return_diff_pos = (test_result_final_df.loc[cond_diff_pos, "is_win"] *
                                 test_result_final_df.loc[cond_diff_pos, "真の単勝"] * 100).sum()
        roi_diff_pos = total_return_diff_pos / (n_diff_pos * 100)
        print(f"diff>0馬: 件数={n_diff_pos}, 1着率={win_rate_diff_pos:.3f}, 回収率={roi_diff_pos:.3f}")
    else:
        print("diff>0の馬がありません。")

    # ------------------------
    # (d) 予測のpop1(>=0.5)の馬
    # ------------------------
    cond_pred_pop1 = (test_result_final_df["予測のpop1"] >= 0.5)
    n_pred_pop1 = cond_pred_pop1.sum()
    if n_pred_pop1 > 0:
        win_rate_pred_pop1 = test_result_final_df.loc[cond_pred_pop1, "is_win"].mean()
        total_return_pred_pop1 = (test_result_final_df.loc[cond_pred_pop1, "is_win"] *
                                  test_result_final_df.loc[cond_pred_pop1, "真の単勝"] * 100).sum()
        roi_pred_pop1 = total_return_pred_pop1 / (n_pred_pop1 * 100)
        print(f"予測pop1(>=0.5)馬: 件数={n_pred_pop1}, 1着率={win_rate_pred_pop1:.3f}, 回収率={roi_pred_pop1:.3f}")
    else:
        print("予測pop1(>=0.5)の馬がありません。")

    # ------------------------
    # (e) diffの分布ヒストグラム
    # ------------------------
    plt.figure(figsize=(6,4))
    plt.hist(test_result_final_df["diff"], bins=50, color='blue', alpha=0.7)
    plt.title("Histogram of diff (logit space)")
    plt.xlabel("diff")
    plt.ylabel("count")
    plt.show()

    # ============ 8) モデルや前処理の保存 ============
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


# 実行スクリプト例
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
