import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA


ROOT_PATH = "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/153_競馬_v3/data"
DATA_PATH = os.path.join(ROOT_PATH, "02_features/feature.csv")

def prepare_data(
    data_path,
    target_col="着順",
    id_col="race_id",
    leakage_cols=None,
    pca_dim=50,
    test_ratio=0.1,
):
    if leakage_cols is None:
        leakage_cols = ['着差','人気','単勝','上がり3F','馬体重','horse_id','jockey_id',
                        'trainer_id',
                        '入線','1着タイム差','先位タイム差','5着着差','増減','1C通過順位','2C通過順位',
                        '3C通過順位','4C通過順位','賞金','前半ペース','後半ペース',
                        '100m','200m','300m','400m','500m','600m','700m','800m','900m','1000m',
                        '1100m','1200m','1300m','1400m','1500m','1600m','1700m','1800m','1900m','2000m',
                        '2100m','2200m','2300m','2400m','2500m','2600m','2700m','2800m','2900m','3000m',
                        '3100m','3200m','3300m','3400m','3500m','3600m',
                        'horse_ability']
    df = pd.read_csv(data_path, encoding="utf_8_sig")
    print('データロード完了')
    # 時系列列でソート
    df = df.sort_values('date').reset_index(drop=True)

    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 不要列除外
    cat_cols = [c for c in cat_cols if c not in leakage_cols and c not in [target_col, id_col]]
    num_cols = [c for c in num_cols if c not in leakage_cols and c not in [target_col, id_col]]

    # 欠損埋め
    for c in cat_cols:
        df[c] = df[c].fillna("missing").astype(str)
    for n in num_cols:
        df[n] = df[n].fillna(0)

    # カテゴリコード化
    for c in cat_cols:
        df[c] = df[c].astype('category').cat.codes
        
    
    print('数値列、カテゴリ列処理完了')

    # PCA対象列の抽出（特定パターンにマッチする列のみ）
    pca_pattern = r'^(競走馬芝|競走馬ダート|単年競走馬芝|単年競走馬ダート|騎手芝|騎手ダート)'
    pca_target_cols = [c for c in num_cols if re.match(pca_pattern, c)]
    other_num_cols = [c for c in num_cols if c not in pca_target_cols]

    # スケーリング
    scaler = StandardScaler()
    pca_target_features = scaler.fit_transform(df[pca_target_cols].values)

    # PCA適用
    pca_dim = min(pca_dim, pca_target_features.shape[1])
    pca = PCA(n_components=pca_dim)
    pca_features = pca.fit_transform(pca_target_features)

    # PCA後特徴量とその他数値特徴量の結合
    other_features = scaler.fit_transform(df[other_num_cols].values)
    cat_features = df[cat_cols].values
    X = np.concatenate([cat_features, other_features, pca_features], axis=1)

    # num_dimを計算 (other_features + pca_featuresの次元)
    actual_num_dim = other_features.shape[1] + pca_dim


    print('PCA処理完了')

    # シーケンス化
    y_all = df[target_col].values - 1
    race_ids = df[id_col].values

    groups = df.groupby(id_col)
    max_seq_len = groups.size().max()
    feature_dim = X.shape[1]

    sequences = []
    labels = []
    masks = []
    print('シーケンス化開始')
    for rid in tqdm(df[id_col].unique()):
        idx = np.where(race_ids == rid)[0]
        feat = X[idx]
        tar = y_all[idx]
        seq_len = len(idx)
        pad_len = max_seq_len - seq_len
        if pad_len > 0:
            feat = np.vstack([feat, np.zeros((pad_len, feature_dim))])
            mask = [1]*seq_len + [0]*pad_len
            tar = np.concatenate([tar, np.full((pad_len,), -1)])
        else:
            mask = [1]*max_seq_len

        sequences.append(feat)
        labels.append(tar)
        masks.append(mask)

    print('シーケンス化完了')

    # 時系列データのため、上位何割かをテスト用に確保
    split_idx = int(len(sequences) * (1 - test_ratio))
    train_sequences = sequences[:split_idx]
    train_labels = labels[:split_idx]
    train_masks = masks[:split_idx]

    test_sequences = sequences[split_idx:]
    test_labels = labels[split_idx:]
    test_masks = masks[split_idx:]

    cat_unique = {}
    for c in cat_cols:
        cat_unique[c] = df[c].astype('category').cat.categories.size

    return (train_sequences, train_labels, train_masks,
            test_sequences, test_labels, test_masks,
            cat_cols, cat_unique, max_seq_len, pca_dim, int(df[target_col].max()), actual_num_dim)


class HorseRaceDataset(Dataset):
    def __init__(self, sequences, labels, masks):
        self.sequences = sequences
        self.labels = labels
        self.masks = masks

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        lab = torch.tensor(self.labels[idx], dtype=torch.long)
        m = torch.tensor(self.masks[idx], dtype=torch.bool)
        return seq, lab, m


class FeatureEmbedder(nn.Module):
    def __init__(self, cat_unique, cat_cols, cat_emb_dim=16, num_dim=50, feature_dim=None):
        super().__init__()
        self.cat_cols = cat_cols
        self.emb_layers = nn.ModuleDict()
        for c in cat_cols:
            unique_count = cat_unique[c]
            emb_dim = min(cat_emb_dim, unique_count//2+1)
            emb_dim = max(emb_dim, 4)
            self.emb_layers[c] = nn.Embedding(unique_count, emb_dim)
        self.num_linear = nn.Linear(num_dim, num_dim)
        # 総次元数
        cat_out_dim = sum([self.emb_layers[c].embedding_dim for c in self.cat_cols])
        self.out_linear = nn.Linear(cat_out_dim+num_dim, feature_dim)

    def forward(self, x):
        # x: (batch, seq_len, cat_cols+num_dim)
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
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        # (max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model) batch_first=True
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return x


class HorseTransformer(nn.Module):
    def __init__(self, cat_unique, cat_cols, max_seq_len, num_dim=50, d_model=128, nhead=8, num_layers=4, num_classes=20, dropout=0.1):
        super().__init__()
        self.feature_embedder = FeatureEmbedder(cat_unique, cat_cols, cat_emb_dim=16, num_dim=num_dim, feature_dim=d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, src, src_key_padding_mask=None, return_attn=False):
        # src: (batch, seq_len, features)
        emb = self.feature_embedder(src) # (batch, seq_len, d_model)
        emb = self.pos_encoder(emb)
        out = self.transformer_encoder(emb, src_key_padding_mask=src_key_padding_mask) 
        logits = self.fc_out(out) # (batch, seq_len, num_classes)
        return logits


def correlation_loss(pred, target):
    """
    Spearman相関を求めるには順位付けが必要ですが、順位変換は非微分的です。
    ここでは簡略化のため、期待順位(exp_rank)とtarget間のPearson相関を用いています。
    """
    pred_score = pred.softmax(dim=-1)
    # 期待順位
    ranks = torch.arange(pred.size(-1), device=pred.device).float()
    exp_rank = (pred_score * ranks.unsqueeze(0)).sum(dim=-1)
    target_f = target.float()

    exp_mean = exp_rank.mean()
    tar_mean = target_f.mean()

    cov = ((exp_rank - exp_mean)*(target_f - tar_mean)).mean()
    std_x = exp_rank.std()
    std_y = target_f.std()
    corr = cov/(std_x*std_y + 1e-8)
    loss = 1 - corr
    return loss

def run_train(
    data_path = DATA_PATH,
    target_col="着順",
    id_col="race_id",
    batch_size=16,
    lr=0.001,
    num_epochs=10,
    pca_dim=50,
    test_ratio=0.2,
    d_model=128,
    nhead=8,
    num_layers=4,
    dropout=0.1
):
    (train_sequences, train_labels, train_masks,
     test_sequences, test_labels, test_masks,
     cat_cols, cat_unique, max_seq_len, pca_dim, num_classes, actual_num_dim) = prepare_data(
         data_path, target_col=target_col, id_col=id_col, pca_dim=pca_dim,
         test_ratio=test_ratio
    )

    train_dataset = HorseRaceDataset(train_sequences, train_labels, train_masks)
    test_dataset = HorseRaceDataset(test_sequences, test_labels, test_masks)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print('データセット化完了')

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    # num_dimをactual_num_dimに修正
    model = HorseTransformer(cat_unique, cat_cols, max_seq_len,
                             num_dim=actual_num_dim, d_model=d_model, nhead=nhead,
                             num_layers=num_layers, num_classes=num_classes, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('学習開始')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for sequences, labels, masks in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            src_key_padding_mask = ~masks

            optimizer.zero_grad()
            outputs = model(sequences, src_key_padding_mask=src_key_padding_mask)
            outputs = outputs.reshape(-1, num_classes)
            labels_flat = labels.reshape(-1)
            valid = labels_flat >= 0
            if valid.sum() == 0:
                continue
            loss = correlation_loss(outputs[valid], labels_flat[valid])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sequences, labels, masks in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            src_key_padding_mask = ~masks

            outputs = model(sequences, src_key_padding_mask=src_key_padding_mask)
            pred = outputs.argmax(dim=-1)
            valid = labels >= 0
            correct += (pred[valid] == labels[valid]).sum().item()
            total += valid.sum().item()

    accuracy = correct / total if total > 0 else 0
    print(f"Test Accuracy: {accuracy:.4f}")

    return model

# 実行例:
# model = run_train(DATA_PATH, num_epochs=5)
