import os  # OS操作用
# pytorchでmpsを使用する際、mps非対応の機能があってもCPUへフォールバックする設定
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import re  # 正規表現処理用
import pandas as pd  # データフレーム操作用
import numpy as np  # 数値計算用
import math  # 数学関数用
import torch  # PyTorch本体
import torch.nn as nn  # PyTorchのニューラルネットワーク関連モジュール
from sklearn.preprocessing import StandardScaler  # 標準化用
from torch.utils.data import Dataset, DataLoader  # データセット・データローダー
from sklearn.decomposition import PCA  # 次元削減用
import matplotlib.pyplot as plt  # プロット用
import copy  # オブジェクトのディープコピー用
import pickle
from tqdm import tqdm

import random  # 乱数制御用

# 乱数固定(再現性確保)
random.seed(42)  # Python組み込み乱数シード
np.random.seed(42)  # NumPy乱数シード
torch.manual_seed(42)  # PyTorch CPU乱数シード
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)  # CUDA使用時の乱数シード

# ファイルパス設定
ROOT_PATH = "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/153_競馬_v3"
DATA_PATH = os.path.join(ROOT_PATH, "data/02_features/feature.csv")  # 特徴量CSVのパス
SAVE_PATH_PRED = os.path.join(ROOT_PATH, "result/predictions/test.csv") 
SAVE_PATH_MODEL = os.path.join(ROOT_PATH, "models/test/model") 
SAVE_PATH_PCA_MODEL = os.path.join(ROOT_PATH, "models/test/pcamodel") 
SAVE_PATH_SCALER_ALL= os.path.join(ROOT_PATH, "models/test/scaler_all") 
SAVE_PATH_SCALER_OTHER = os.path.join(ROOT_PATH, "models/test/scaler_other") 

# データセットクラス定義
class HorseRaceDataset(Dataset):
    def __init__(self, sequences, labels, masks):
        # sequences: 特徴量系列
        # labels: ターゲット(トップ1,3,5)
        # masks: 有効データ領域を示すマスク
        self.sequences = sequences
        self.labels = labels
        self.masks = masks

    def __len__(self):
        # データ数を返す
        return len(self.sequences)

    def __getitem__(self, idx):
        # idx番目のデータを取得
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)  # 特徴量テンソル化
        lab = torch.tensor(self.labels[idx], dtype=torch.float32)  # ラベルテンソル化
        m = torch.tensor(self.masks[idx], dtype=torch.bool)  # マスクテンソル化
        return seq, lab, m

# 特徴量を埋め込み表現に変換するクラス(カテゴリ埋め込み + 数値特徴線形変換)
class FeatureEmbedder(nn.Module):
    def __init__(self, cat_unique, cat_cols, cat_emb_dim=16, num_dim=50, feature_dim=None):
        super().__init__()
        # cat_unique: カテゴリ列ごとのユニーク数
        # cat_cols: カテゴリ列の名前一覧
        # cat_emb_dim: カテゴリ埋め込みの基本次元数
        # num_dim: 数値特徴の次元数
        # feature_dim: 埋め込み後の出力次元(d_model相当)

        self.cat_cols = cat_cols
        self.emb_layers = nn.ModuleDict()
        # 各カテゴリ変数に対し埋め込み層を定義
        for c in cat_cols:
            unique_count = cat_unique[c]
            emb_dim = min(cat_emb_dim, unique_count // 2 + 1)  # 埋め込み次元をある程度制限
            emb_dim = max(emb_dim, 4)  # 最低4次元
            self.emb_layers[c] = nn.Embedding(unique_count, emb_dim)
        self.num_linear = nn.Linear(num_dim, num_dim)  # 数値特徴を線形変換
        cat_out_dim = sum([self.emb_layers[c].embedding_dim for c in self.cat_cols])  # 全カテゴリ埋め込み合計次元
        self.out_linear = nn.Linear(cat_out_dim + num_dim, feature_dim)  # カテゴリ+数値特徴合わせてfeature_dimへ変換

    def forward(self, x):
        # x: (batch, seq_len, feature_dim)の想定
        cat_len = len(self.cat_cols)  # カテゴリ列数
        cat_x = x[..., :cat_len].long()  # カテゴリ部分をint化
        num_x = x[..., cat_len:]  # 数値部分
        embs = []
        # 各カテゴリ列に対して埋め込み適用
        for i, c in enumerate(self.cat_cols):
            embs.append(self.emb_layers[c](cat_x[..., i]))
        cat_emb = torch.cat(embs, dim=-1)  # 全カテゴリ埋め込みを結合
        num_emb = self.num_linear(num_x)  # 数値部分を線形変換
        out = torch.cat([cat_emb, num_emb], dim=-1)  # カテゴリ+数値を結合
        out = self.out_linear(out)  # feature_dimに圧縮
        return out

# 位置エンコーディング
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        # d_model: 次元数
        # max_len: 最大系列長

        # Positional Encoding行列生成
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # 位置情報(0,1,2,...)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 周期性定義用
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数次元: sin
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])  # 奇数次元: cos(次元数奇数対応)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # 勾配不要のbuffer登録

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        # xにpeを加算して位置情報付与
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return x

# Transformerモデル本体
class HorseTransformer(nn.Module):
    def __init__(self, cat_unique, cat_cols, max_seq_len, num_dim=50, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        # cat_unique: 各カテゴリ列ユニーク値
        # cat_cols: カテゴリ列名リスト
        # max_seq_len: シーケンス最大長
        # num_dim: 数値特徴次元
        # d_model: エンコーダモデル次元
        # nhead: マルチヘッド数
        # num_layers: トランスフォーマーレイヤー数
        # dropout: ドロップアウト率

        self.feature_embedder = FeatureEmbedder(cat_unique, cat_cols, cat_emb_dim=16, num_dim=num_dim, feature_dim=d_model)  # 特徴埋め込み
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)  # 位置エンコード
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)  # エンコーダ層定義
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  # エンコーダスタック
        self.fc_out = nn.Linear(d_model, 3)  # 最終出力層(トップ1,3,5用)

    def forward(self, src, src_key_padding_mask=None):
        # src: (batch, seq_len, features)
        emb = self.feature_embedder(src)  # 特徴量埋め込み
        emb = self.pos_encoder(emb)  # 位置エンコード付与
        out = self.transformer_encoder(emb, src_key_padding_mask=src_key_padding_mask)  # トランスフォーマーエンコード
        logits = self.fc_out(out)  # 出力層
        return logits

def split_data(df, id_col="race_id", target_col="着順", test_ratio=0.1, valid_ratio=0.1):
    # データを日付順でソートしてtrain/valid/testに分割
    df = df.sort_values('date').reset_index(drop=True)  # 日付でソート
    race_ids = df[id_col].unique()  # race_idの一覧
    dataset_len = len(race_ids)
    test_cut = int(dataset_len * (1 - test_ratio))  # テスト分割位置
    valid_cut = int(test_cut * (1 - valid_ratio))   # バリデーション分割位置
    train_ids = race_ids[:valid_cut]  # トレイン用ID
    valid_ids = race_ids[valid_cut:test_cut]  # バリデーション用ID
    test_ids = race_ids[test_cut:]  # テスト用ID

    train_df = df[df[id_col].isin(train_ids)].copy()  # トレインデータ
    valid_df = df[df[id_col].isin(valid_ids)].copy()  # バリデーションデータ
    test_df = df[df[id_col].isin(test_ids)].copy()    # テストデータ

    return train_df, valid_df, test_df

def prepare_data(
    data_path,
    target_col="着順",
    id_col="race_id",
    leakage_cols=None,
    pca_dim=50,
    test_ratio=0.1,
    valid_ratio=0.1
):
    # データ読み込み、前処理、train/valid/test分割、PCAなどを行う関数
    if leakage_cols is None:
        # 利用しない特徴列（リーク回避用）
        leakage_cols = ['斤量','タイム','着差','人気','単勝','上がり3F','馬体重','horse_id','jockey_id',
                        'trainer_id',
                        '騎手', # ルメール過学習と思われる挙動がひどいため追加
                        '順位点',
                        '入線','1着タイム差','先位タイム差','5着着差','増減','1C通過順位','2C通過順位',
                        '3C通過順位','4C通過順位','賞金','前半ペース','後半ペース','ペース',
                        '上がり3F順位',
                        '100m','200m','300m','400m','500m','600m','700m','800m','900m','1000m',
                        '1100m','1200m','1300m','1400m','1500m','1600m','1700m','1800m','1900m','2000m',
                        '2100m','2200m','2300m','2400m','2500m','2600m','2700m','2800m','2900m','3000m',
                        '3100m','3200m','3300m','3400m','3500m','3600m',
                        'horse_ability']

    df = pd.read_csv(data_path, encoding="utf_8_sig")  # データ読み込み

    train_df, valid_df, test_df = split_data(df, id_col=id_col, target_col=target_col, test_ratio=test_ratio, valid_ratio=valid_ratio)

    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()  # 数値以外 = カテゴリ列
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()   # 数値列

    # ターゲット・ID・リーク列は除外
    cat_cols = [c for c in cat_cols if c not in leakage_cols and c not in [target_col, id_col]]
    num_cols = [c for c in num_cols if c not in leakage_cols and c not in [target_col, id_col]]

    print('カテゴリ特徴量数：', len(cat_cols))
    print('数値特徴量数：', len(num_cols))

    # 欠損埋め：カテゴリは"missing"、数値は0
    for c in cat_cols:
        train_df[c] = train_df[c].fillna("missing").astype(str)
        valid_df[c] = valid_df[c].fillna("missing").astype(str)
        test_df[c] = test_df[c].fillna("missing").astype(str)
    for n in num_cols:
        train_df[n] = train_df[n].fillna(0)
        valid_df[n] = valid_df[n].fillna(0)
        test_df[n] = test_df[n].fillna(0)

    # カテゴリエンコード: train基準でカテゴリを固定
    for c in cat_cols:
        train_df[c] = train_df[c].astype('category')
        valid_df[c] = valid_df[c].astype('category')
        test_df[c] = test_df[c].astype('category')

        train_cat = train_df[c].cat.categories  # トレインで得たカテゴリ
        valid_df[c] = pd.Categorical(valid_df[c], categories=train_cat)
        test_df[c] = pd.Categorical(test_df[c], categories=train_cat)

        train_df[c] = train_df[c].cat.codes
        valid_df[c] = valid_df[c].cat.codes
        test_df[c] = test_df[c].cat.codes

    # PCA対象列のパターン定義
    pca_pattern = r'^(競走馬芝|競走馬ダート|単年競走馬芝|単年競走馬ダート|騎手芝|騎手ダート)'
    pca_target_cols = [c for c in num_cols if re.match(pca_pattern, c)]
    other_num_cols = [c for c in num_cols if c not in pca_target_cols]

    # スケーラーとPCAはtrainでfitし、valid/testはtransformのみ
    scaler_all = StandardScaler()
    pca_target_features_train = scaler_all.fit_transform(train_df[pca_target_cols].values)
    pca_dim = min(pca_dim, pca_target_features_train.shape[1])
    pca_model = PCA(n_components=pca_dim)
    pca_features_train = pca_model.fit_transform(pca_target_features_train)

    pca_target_features_valid = scaler_all.transform(valid_df[pca_target_cols].values)
    pca_features_valid = pca_model.transform(pca_target_features_valid)
    pca_target_features_test = scaler_all.transform(test_df[pca_target_cols].values)
    pca_features_test = pca_model.transform(pca_target_features_test)

    scaler_other = StandardScaler()
    other_features_train = scaler_other.fit_transform(train_df[other_num_cols].values)
    other_features_valid = scaler_other.transform(valid_df[other_num_cols].values)
    other_features_test = scaler_other.transform(test_df[other_num_cols].values)

    cat_features_train = train_df[cat_cols].values
    cat_features_valid = valid_df[cat_cols].values
    cat_features_test = test_df[cat_cols].values

    X_train = np.concatenate([cat_features_train, other_features_train, pca_features_train], axis=1)
    X_valid = np.concatenate([cat_features_valid, other_features_valid, pca_features_valid], axis=1)
    X_test = np.concatenate([cat_features_test, other_features_test, pca_features_test], axis=1)

    actual_num_dim = other_features_train.shape[1] + pca_dim

    # 系列化関数定義(レース単位で系列化し、パディング実施)
    def create_sequences(_df, X):
        original_ranks = _df[target_col].values
        race_ids = _df[id_col].values
        groups = _df.groupby(id_col)
        max_seq_len = groups.size().max()
        feature_dim = X.shape[1]

        y_all = original_ranks
        sequences = []
        labels = []
        masks = []
        for rid in _df[id_col].unique():
            idx = np.where(race_ids == rid)[0]
            feat = X[idx]
            ranks = y_all[idx]
            top1 = (ranks == 1).astype(int)
            top3 = (ranks <= 3).astype(int)
            top5 = (ranks <= 5).astype(int)
            tar = np.stack([top1, top3, top5], axis=-1)
            seq_len = len(idx)
            pad_len = max_seq_len - seq_len
            if pad_len > 0:
                # パディング
                feat = np.vstack([feat, np.zeros((pad_len, feature_dim))])
                pad_label = np.zeros((pad_len, 3), dtype=int)
                mask = [1]*seq_len + [0]*pad_len
                tar = np.concatenate([tar, pad_label], axis=0)
            else:
                mask = [1]*max_seq_len

            sequences.append(feat)
            labels.append(tar)
            masks.append(mask)
        return sequences, labels, masks, max_seq_len

    # train/valid/testそれぞれで系列化
    train_sequences, train_labels, train_masks, max_seq_len_train = create_sequences(train_df, X_train)
    valid_sequences, valid_labels, valid_masks, max_seq_len_valid = create_sequences(valid_df, X_valid)
    test_sequences, test_labels, test_masks, max_seq_len_test = create_sequences(test_df, X_test)

    # 全セットにおける最大長
    max_seq_len = max(max_seq_len_train, max_seq_len_valid, max_seq_len_test)

    # パディング再調整関数(全セットで統一最大長に合わせる)
    def pad_sequences(sequences, labels, masks, seq_len_target):
        feature_dim = sequences[0].shape[1]
        new_seqs = []
        new_labs = []
        new_masks = []
        for feat, tar, m in zip(sequences, labels, masks):
            cur_len = len(feat)
            if cur_len < seq_len_target:
                pad_len = seq_len_target - cur_len
                feat = np.vstack([feat, np.zeros((pad_len, feature_dim))])
                pad_label = np.zeros((pad_len, 3), dtype=int)
                tar = np.concatenate([tar, pad_label], axis=0)
                m = m + [0]*pad_len
            new_seqs.append(feat)
            new_labs.append(tar)
            new_masks.append(m)
        return new_seqs, new_labs, new_masks

    # 全データセットを最大長に合わせて再パディング
    train_sequences, train_labels, train_masks = pad_sequences(train_sequences, train_labels, train_masks, max_seq_len)
    valid_sequences, valid_labels, valid_masks = pad_sequences(valid_sequences, valid_labels, valid_masks, max_seq_len)
    test_sequences, test_labels, test_masks = pad_sequences(test_sequences, test_labels, test_masks, max_seq_len)

    # カテゴリユニーク数を記録
    cat_unique = {}
    for c in cat_cols:
        cat_unique[c] = len(train_df[c].unique())

    # データセット化
    train_dataset = HorseRaceDataset(train_sequences, train_labels, train_masks)
    valid_dataset = HorseRaceDataset(valid_sequences, valid_labels, valid_masks)
    test_dataset = HorseRaceDataset(test_sequences, test_labels, test_masks)

    # 関数戻り値
    return (train_dataset, valid_dataset, test_dataset,
            cat_cols, cat_unique, max_seq_len, pca_dim, 3, actual_num_dim,
            df, cat_cols, num_cols, pca_target_cols, other_num_cols,
            scaler_all, scaler_other, pca_model, X_train.shape[1], id_col, target_col,
            train_df)

def run_train(
    data_path=DATA_PATH,
    target_col="着順",
    id_col="race_id",
    batch_size=256,
    lr=0.001,
    num_epochs=100,
    pca_dim=50,
    test_ratio=0.2,
    valid_ratio=0.1,
    d_model=128,
    nhead=8,
    num_layers=6,
    dropout=0.15,
    weight_decay = 1e-5,
    patience=10
):
    # モデル学習実行関数
    # データ準備
    (train_dataset, valid_dataset, test_dataset,
     cat_cols, cat_unique, max_seq_len, pca_dim, num_outputs, actual_num_dim,
     df, cat_cols_all, num_cols_all, pca_target_cols, other_num_cols,
     scaler_all, scaler_other, pca_model, feature_dim, id_col, target_col,
     train_df) = prepare_data(
        data_path, target_col=target_col, id_col=id_col, pca_dim=pca_dim,
        test_ratio=test_ratio, valid_ratio=valid_ratio
    )

    # DataLoader作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # トレインデータローダー
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False) # バリデーションローダー
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)   # テストローダー

    # デバイス選択(MPS>CUDA>CPU)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # モデル、オプティマイザ、損失関数定義
    model = HorseTransformer(cat_unique, cat_cols, max_seq_len,
                             num_dim=actual_num_dim, d_model=d_model, nhead=nhead,
                             num_layers=num_layers, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.BCEWithLogitsLoss(reduction='none')  # マスク対応のためreduction=none

    def evaluate_with_detail(loader, model, criterion):
        model.eval()
        total_loss = 0.0
        total_count = 0.0

        loss_sum_top1 = 0.0
        loss_sum_top3 = 0.0
        loss_sum_top5 = 0.0
        valid_count_top1 = 0.0
        valid_count_top3 = 0.0
        valid_count_top5 = 0.0

        with torch.no_grad():
            for sequences, labels, masks in loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
                src_key_padding_mask = ~masks

                outputs = model(sequences, src_key_padding_mask=src_key_padding_mask)
                loss_raw = criterion(outputs, labels)  # shape: (batch, seq_len, 3)
                valid_mask = masks.unsqueeze(-1).expand_as(loss_raw)  # (batch, seq_len, 3)

                # 全体のロス
                loss_batch = (loss_raw * valid_mask).sum() / valid_mask.sum()
                total_loss += loss_batch.item()
                total_count += 1

                # 各ターゲット別にロスを集計
                top1_loss_sum = (loss_raw[..., 0] * valid_mask[..., 0]).sum().item()
                top3_loss_sum = (loss_raw[..., 1] * valid_mask[..., 1]).sum().item()
                top5_loss_sum = (loss_raw[..., 2] * valid_mask[..., 2]).sum().item()

                valid_count_top1_batch = valid_mask[..., 0].sum().item()
                valid_count_top3_batch = valid_mask[..., 1].sum().item()
                valid_count_top5_batch = valid_mask[..., 2].sum().item()

                loss_sum_top1 += top1_loss_sum
                loss_sum_top3 += top3_loss_sum
                loss_sum_top5 += top5_loss_sum
                valid_count_top1 += valid_count_top1_batch
                valid_count_top3 += valid_count_top3_batch
                valid_count_top5 += valid_count_top5_batch

        avg_loss = total_loss / total_count if total_count > 0 else 0
        avg_loss_top1 = loss_sum_top1 / valid_count_top1 if valid_count_top1 > 0 else 0
        avg_loss_top3 = loss_sum_top3 / valid_count_top3 if valid_count_top3 > 0 else 0
        avg_loss_top5 = loss_sum_top5 / valid_count_top5 if valid_count_top5 > 0 else 0

        return avg_loss, avg_loss_top1, avg_loss_top3, avg_loss_top5


    best_model_wts = copy.deepcopy(model.state_dict())  # ベストモデルの重み記録用
    best_loss = float('inf')
    epochs_no_improve = 0

    train_losses = []
    valid_losses = []

    # 学習ループ
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_count = 0
        
        for sequences, labels, masks in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            src_key_padding_mask = ~masks

            optimizer.zero_grad()
            outputs = model(sequences, src_key_padding_mask=src_key_padding_mask)
            loss_raw = criterion(outputs, labels)
            valid_mask = masks.unsqueeze(-1).expand_as(loss_raw)
            loss = (loss_raw * valid_mask).sum() / valid_mask.sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_count += 1
        
        # スケジューラ更新
        scheduler.step()

        # epoch終了後、全バッチ合計から平均loglossを求める
        avg_train_loss = total_loss / total_count if total_count > 0 else 0
        # ここで valid の各ターゲット logloss も集計
        valid_loss_overall, valid_loss_top1, valid_loss_top3, valid_loss_top5 = evaluate_with_detail(valid_loader, model, criterion)

        train_losses.append(avg_train_loss)
        valid_losses.append(valid_loss_overall)


        print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f} "
          f"Valid Loss (overall): {valid_loss_overall:.4f} "
          f"Valid Top1: {valid_loss_top1:.4f} "
          f"Valid Top3: {valid_loss_top3:.4f} "
          f"Valid Top5: {valid_loss_top5:.4f}")


        # Early Stopping監視
        if valid_loss_overall < best_loss:
            best_loss = valid_loss_overall
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            # 改善止まったため早期終了
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_model_wts)  # ベストモデルロード

    def test_evaluate(loader):
        # テスト時にトップ1,3,5のログロス計算
        model.eval()
        criterion_eval = nn.BCEWithLogitsLoss(reduction='none')
        total_loss_top1 = 0
        total_loss_top3 = 0
        total_loss_top5 = 0
        total_count = 0
        with torch.no_grad():
            for sequences, labels, masks in loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
                src_key_padding_mask = ~masks
                outputs = model(sequences, src_key_padding_mask=src_key_padding_mask)
                loss_raw = criterion_eval(outputs, labels)
                valid_mask = masks.unsqueeze(-1).expand_as(loss_raw)
                loss_top1 = (loss_raw[..., 0] * valid_mask[..., 0]).sum() / valid_mask[..., 0].sum()
                loss_top3 = (loss_raw[..., 1] * valid_mask[..., 1]).sum() / valid_mask[..., 1].sum()
                loss_top5 = (loss_raw[..., 2] * valid_mask[..., 2]).sum() / valid_mask[..., 2].sum()

                total_loss_top1 += loss_top1.item()
                total_loss_top3 += loss_top3.item()
                total_loss_top5 += loss_top5.item()
                total_count += 1

        avg_loss_top1 = total_loss_top1 / total_count if total_count > 0 else 0
        avg_loss_top3 = total_loss_top3 / total_count if total_count > 0 else 0
        avg_loss_top5 = total_loss_top5 / total_count if total_count > 0 else 0
        return avg_loss_top1, avg_loss_top3, avg_loss_top5

    top1_logloss, top3_logloss, top5_logloss = test_evaluate(test_loader)
    print(f"Test Top1 Logloss: {top1_logloss:.4f}")
    print(f"Test Top3 Logloss: {top3_logloss:.4f}")
    print(f"Test Top5 Logloss: {top5_logloss:.4f}")

    # 学習曲線プロット
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    # 全race_idについて予測
    model.eval()
    results = []
    with torch.no_grad():
        for rid in tqdm(range(202406050801, 202406050801 + 12)):
            race_df = df[df[id_col] == rid].copy()
            if len(race_df) == 0:
                continue
            for c in cat_cols_all:
                race_df[c] = race_df[c].fillna("missing").astype('category')
                race_df[c] = pd.Categorical(race_df[c], categories=train_df[c].astype('category').cat.categories)
                race_df[c] = race_df[c].cat.codes
            for n in other_num_cols:
                race_df[n] = race_df[n].fillna(0)
            for p in pca_target_cols:
                race_df[p] = race_df[p].fillna(0)
            cat_features = race_df[cat_cols_all].values
            other_feats_scaled = scaler_other.transform(race_df[other_num_cols].values)
            pca_feats_scaled = scaler_all.transform(race_df[pca_target_cols].values)
            pca_transformed = pca_model.transform(pca_feats_scaled)

            X_race = np.concatenate([cat_features, other_feats_scaled, pca_transformed], axis=1)
            seq_len = X_race.shape[0]
            pad_len = max_seq_len - seq_len
            if pad_len > 0:
                X_race = np.vstack([X_race, np.zeros((pad_len, X_race.shape[1]))])
            race_seq = torch.tensor(X_race, dtype=torch.float32).unsqueeze(0).to(device)

            mask = torch.ones(seq_len, dtype=torch.bool)
            if pad_len > 0:
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.bool)])
            mask = mask.unsqueeze(0).to(device)
            src_key_padding_mask = ~mask
            outputs = model(race_seq, src_key_padding_mask=src_key_padding_mask)
            probs = torch.sigmoid(outputs)
            probs = probs[:, :seq_len, :]

            if '馬番' in race_df.columns:
                race_df = race_df.assign(
                    P_top1=probs[0, :, 0].cpu().numpy(),
                    P_top3=probs[0, :, 1].cpu().numpy(),
                    P_top5=probs[0, :, 2].cpu().numpy()
                )
                results.append(race_df[['race_id', '馬番', 'P_top1', 'P_top3', 'P_top5']])
            else:
                out_df = pd.DataFrame({
                    'race_id': [rid]*seq_len,
                    '馬番': list(range(1, seq_len+1)),
                    'P_top1': probs[0, :, 0].cpu().numpy(),
                    'P_top3': probs[0, :, 1].cpu().numpy(),
                    'P_top5': probs[0, :, 2].cpu().numpy()
                })
                results.append(out_df)

    final_df = pd.concat(results, ignore_index=True)
    final_df.to_csv(SAVE_PATH_PRED, index=False)

    # pickleでモデル保存
    with open(SAVE_PATH_MODEL, "wb") as f:
        pickle.dump(model.state_dict(), f)
    with open(SAVE_PATH_PCA_MODEL, "wb") as f:
        pickle.dump(pca_model, f)
    with open(SAVE_PATH_SCALER_ALL, "wb") as f:
        pickle.dump(scaler_all, f)
    with open(SAVE_PATH_SCALER_OTHER, "wb") as f:
        pickle.dump(scaler_other, f)

    return model