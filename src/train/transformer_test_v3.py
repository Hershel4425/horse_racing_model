import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import re
import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
DATA_PATH = os.path.join(ROOT_PATH, "data/02_features/feature.csv")
SAVE_PATH_PRED = os.path.join(ROOT_PATH, "result/predictions/test.csv")
SAVE_PATH_MODEL = os.path.join(ROOT_PATH, "models/test/model")
SAVE_PATH_PCA_MODEL = os.path.join(ROOT_PATH, "models/test/pcamodel")
SAVE_PATH_SCALER_ALL = os.path.join(ROOT_PATH, "models/test/scaler_all")
SAVE_PATH_SCALER_OTHER = os.path.join(ROOT_PATH, "models/test/scaler_other")

# =====================
# Datasetクラス
# =====================
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
        # x: (batch, seq_len, feature_dim)
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
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return x


class HorseTransformer(nn.Module):
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
        self.fc_out = nn.Linear(d_model, 6)  # 6ターゲット

    def forward(self, src, src_key_padding_mask=None):
        emb = self.feature_embedder(src)
        emb = self.pos_encoder(emb)
        out = self.transformer_encoder(emb, src_key_padding_mask=src_key_padding_mask)
        logits = self.fc_out(out)  # (batch, seq_len, 6)
        return logits


def split_data(df, id_col="race_id", target_col="着順", test_ratio=0.1, valid_ratio=0.1):
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
    pca_dim=50,
    test_ratio=0.1,
    valid_ratio=0.1
):
    if leakage_cols is None:
        leakage_cols = [
            '斤量','タイム','着差','単勝','上がり3F','馬体重','horse_id','jockey_id',
            'trainer_id','騎手','順位点','入線','1着タイム差','先位タイム差','5着着差','増減',
            '1C通過順位','2C通過順位','3C通過順位','4C通過順位','賞金','前半ペース','後半ペース','ペース',
            '上がり3F順位','100m','200m','300m','400m','500m','600m','700m','800m','900m','1000m',
            '1100m','1200m','1300m','1400m','1500m','1600m','1700m','1800m','1900m','2000m',
            '2100m','2200m','2300m','2400m','2500m','2600m','2700m','2800m','2900m','3000m',
            '3100m','3200m','3300m','3400m','3500m','3600m','horse_ability'
        ]

    df = pd.read_csv(data_path, encoding="utf_8_sig")
    train_df, valid_df, test_df = split_data(df, id_col=id_col, target_col=target_col,
                                             test_ratio=test_ratio, valid_ratio=valid_ratio)

    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in leakage_cols and c not in [target_col, id_col, pop_col]]
    num_cols = [c for c in num_cols if c not in leakage_cols and c not in [target_col, id_col, pop_col]]

    print('カテゴリ特徴量数：', len(cat_cols))
    print('数値特徴量数：', len(num_cols))

    for c in cat_cols:
        for d in [train_df, valid_df, test_df]:
            d[c] = d[c].fillna("missing").astype(str)
    for n in num_cols:
        for d in [train_df, valid_df, test_df]:
            d[n] = d[n].fillna(0)

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

    pca_pattern = r'^(競走馬芝|競走馬ダート|単年競走馬芝|単年競走馬ダート|騎手芝|騎手ダート)'
    pca_target_cols = [c for c in num_cols if re.match(pca_pattern, c)]
    other_num_cols = [c for c in num_cols if c not in pca_target_cols]

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

        # ラベルを作成
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

            # ラベル
            tar = np.stack([top1[idx], top3[idx], top5[idx],
                            pop1[idx], pop3[idx], pop5[idx]], axis=-1)  # (seq_len, 6)

            # race_id(同じ値だがseq_len分並べる)、馬番
            rid_array = rids[idx]
            horse_array = horse_nums[idx]

            pad_len = max_seq_len - seq_len
            if pad_len > 0:
                # sequences
                feat = np.vstack([feat, np.zeros((pad_len, feature_dim))])
                # labels
                pad_label = np.zeros((pad_len, 6), dtype=int)
                tar = np.concatenate([tar, pad_label], axis=0)
                # masks
                mask = [1]*seq_len + [0]*pad_len
                # race_id, 馬番
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
    valid_seq, valid_lab, valid_mask, max_seq_len_valid, valid_rids_seq, valid_horses_seq = create_sequences(valid_df, X_valid)
    test_seq, test_lab, test_mask, max_seq_len_test, test_rids_seq, test_horses_seq = create_sequences(test_df, X_test)

    max_seq_len = max(max_seq_len_train, max_seq_len_valid, max_seq_len_test)

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
    valid_seq, valid_lab, valid_mask, valid_rids_seq, valid_horses_seq = pad_sequences(
        valid_seq, valid_lab, valid_mask, valid_rids_seq, valid_horses_seq, max_seq_len
    )
    test_seq, test_lab, test_mask, test_rids_seq, test_horses_seq = pad_sequences(
        test_seq, test_lab, test_mask, test_rids_seq, test_horses_seq, max_seq_len
    )

    cat_unique = {}
    for c in cat_cols:
        cat_unique[c] = len(train_df[c].unique())

    # Dataset化
    train_dataset = HorseRaceDataset(train_seq, train_lab, train_mask, train_rids_seq, train_horses_seq)
    valid_dataset = HorseRaceDataset(valid_seq, valid_lab, valid_mask, valid_rids_seq, valid_horses_seq)
    test_dataset  = HorseRaceDataset(test_seq,  test_lab,  test_mask,  test_rids_seq,  test_horses_seq)

    return (train_dataset, valid_dataset, test_dataset,
            cat_cols, cat_unique, max_seq_len, pca_dim, 6,
            actual_num_dim, df, cat_cols, num_cols,
            pca_target_cols, other_num_cols, scaler_all, scaler_other,
            pca_model, X_train.shape[1], id_col, target_col, train_df)


def run_train(
    data_path=DATA_PATH,
    target_col="着順",
    pop_col="人気",
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
    weight_decay=1e-5,
    patience=10
):
    (train_dataset, valid_dataset, test_dataset,
     cat_cols, cat_unique, max_seq_len, pca_dim, num_outputs,
     actual_num_dim, df, cat_cols_all, num_cols_all,
     pca_target_cols, other_num_cols,
     scaler_all, scaler_other, pca_model, feature_dim,
     id_col, target_col, train_df) = prepare_data(
        data_path=data_path,
        target_col=target_col,
        pop_col=pop_col,
        id_col=id_col,
        pca_dim=pca_dim,
        test_ratio=test_ratio,
        valid_ratio=valid_ratio
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # デバイス選択
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = HorseTransformer(
        cat_unique, cat_cols, max_seq_len,
        num_dim=actual_num_dim, d_model=d_model,
        nhead=nhead, num_layers=num_layers, dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.BCEWithLogitsLoss(reduction='none')  # マスク対応

    def evaluate_with_detail(loader, model, criterion):
        model.eval()
        total_loss = 0.0
        total_count = 0.0
        import numpy as np
        loss_sum = np.zeros(6, dtype=np.float64)
        valid_count = np.zeros(6, dtype=np.float64)

        with torch.no_grad():
            for sequences, labels, masks, _, _ in loader:
                sequences = sequences.to(device)
                labels    = labels.to(device)
                masks     = masks.to(device)
                outputs = model(sequences, src_key_padding_mask=~masks)
                loss_raw = criterion(outputs, labels)
                valid_mask = masks.unsqueeze(-1).expand_as(loss_raw)

                loss_batch = (loss_raw * valid_mask).sum() / valid_mask.sum()
                total_loss += loss_batch.item()
                total_count += 1

                for i in range(6):
                    loss_sum[i] += (loss_raw[..., i] * valid_mask[..., i]).sum().item()
                    valid_count[i] += valid_mask[..., i].sum().item()

        avg_loss = total_loss / total_count if total_count > 0 else 0
        avg_loss_each = loss_sum / np.maximum(valid_count, 1e-15)
        return avg_loss, avg_loss_each

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_count = 0.0

        for sequences, labels, masks, _, _ in train_loader:
            sequences = sequences.to(device)
            labels    = labels.to(device)
            masks     = masks.to(device)

            optimizer.zero_grad()
            outputs = model(sequences, src_key_padding_mask=~masks)
            loss_raw = criterion(outputs, labels)
            valid_mask = masks.unsqueeze(-1).expand_as(loss_raw)
            loss = (loss_raw * valid_mask).sum() / valid_mask.sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_count += 1

        scheduler.step()
        avg_train_loss = total_loss / total_count if total_count > 0 else 0
        valid_loss_overall, valid_loss_each = evaluate_with_detail(valid_loader, model, criterion)
        train_losses.append(avg_train_loss)
        valid_losses.append(valid_loss_overall)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"TrainLoss: {avg_train_loss:.4f}  "
              f"ValidLoss(overall): {valid_loss_overall:.4f}  "
              f"ValidLoss(each6): {[f'{x:.4f}' for x in valid_loss_each]}")

        if valid_loss_overall < best_loss:
            best_loss = valid_loss_overall
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_model_wts)

    # ===== テストセット評価 =====
    def test_evaluate(loader):
        model.eval()
        criterion_eval = nn.BCEWithLogitsLoss(reduction='none')
        import numpy as np
        loss_sum6 = np.zeros(6, dtype=np.float64)
        valid_count6 = np.zeros(6, dtype=np.float64)

        with torch.no_grad():
            for sequences, labels, masks, _, _ in loader:
                sequences = sequences.to(device)
                labels    = labels.to(device)
                masks     = masks.to(device)
                outputs = model(sequences, src_key_padding_mask=~masks)
                loss_raw = criterion_eval(outputs, labels)
                valid_mask = masks.unsqueeze(-1).expand_as(loss_raw)
                for i in range(6):
                    loss_sum6[i] += (loss_raw[..., i] * valid_mask[..., i]).sum().item()
                    valid_count6[i] += valid_mask[..., i].sum().item()

        avg_loss_each = loss_sum6 / np.maximum(valid_count6, 1e-15)
        return avg_loss_each

    test_loss_each6 = test_evaluate(test_loader)
    print("Test BCE Logloss each of 6 targets:", [f"{v:.4f}" for v in test_loss_each6])

    # ===== 学習曲線 =====
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    # ===== 推論 & pred_df 生成 =====
    model.eval()
    all_probs_list = []
    all_trues_list = []
    all_rids_list  = []
    all_horses_list = []

    with torch.no_grad():
        for sequences, labels, masks, rids, hnums in test_loader:
            sequences = sequences.to(device)
            labels    = labels.to(device)
            masks     = masks.to(device)

            outputs = model(sequences, src_key_padding_mask=~masks)  # (batch, seq_len, 6)
            probs = torch.sigmoid(outputs)

            # 有効部分のみ抽出
            valid_mask = masks.unsqueeze(-1).expand_as(probs)
            num_targets = probs.size(-1)

            # shape: [N, 6]
            probs_valid  = probs[valid_mask].view(-1, num_targets).cpu().numpy()
            labels_valid = labels[valid_mask].view(-1, num_targets).cpu().numpy()

            # race_id, 馬番 も同様に有効部分だけ
            # rids, hnums は CPU tensor として使うならそのままnumpy変換してOK (to(device)しなくてよい)
            # ただし形が (batch, seq_len) なので flatten して valid_mask に合わせる
            rids_np  = rids.numpy()  # (batch, seq_len)
            hnums_np = hnums.numpy() # (batch, seq_len)
            # valid_mask[..., 0] shape: (batch, seq_len)
            valid_mask_2d = valid_mask[..., 0].cpu().numpy().astype(bool)

            rids_valid   = rids_np[valid_mask_2d].reshape(-1)
            hnums_valid  = hnums_np[valid_mask_2d].reshape(-1)

            all_probs_list.append(probs_valid)
            all_trues_list.append(labels_valid)
            all_rids_list.append(rids_valid)
            all_horses_list.append(hnums_valid)

    all_probs = np.concatenate(all_probs_list, axis=0)   # (合計サンプル数, 6)
    all_trues = np.concatenate(all_trues_list, axis=0)   # (合計サンプル数, 6)
    all_rids  = np.concatenate(all_rids_list,  axis=0)   # (合計サンプル数,)
    all_horses= np.concatenate(all_horses_list, axis=0)  # (合計サンプル数,)

    # DataFrame化 (race_id, 馬番, 各確率, ラベル)
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

    # ===== キャリブレーション曲線 =====
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    target_names = ["Top1", "Top3", "Top5", "Pop1", "Pop3", "Pop5"]
    for i in range(6):
        prob_true, prob_pred = calibration_curve(all_trues[:, i], all_probs[:, i], n_bins=10)
        ax = axes[i]
        ax.plot(prob_pred, prob_true, marker='o', label='Calibration')
        ax.plot([0,1],[0,1], '--', color='gray', label='Perfect')

        ax2 = ax.twinx()
        ax2.hist(all_probs[:, i], bins=10, range=(0,1), alpha=0.3, color='gray')

        ax.set_title(f'Calibration Curve ({target_names[i]})')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('True Probability')
        ax.legend()

    plt.suptitle("Epoch End Calibration Curves", fontsize=16)
    plt.tight_layout()
    plt.show()

    # モデル保存
    with open(SAVE_PATH_MODEL, "wb") as f:
        pickle.dump(model.state_dict(), f)
    with open(SAVE_PATH_PCA_MODEL, "wb") as f:
        pickle.dump(pca_model, f)
    with open(SAVE_PATH_SCALER_ALL, "wb") as f:
        pickle.dump(scaler_all, f)
    with open(SAVE_PATH_SCALER_OTHER, "wb") as f:
        pickle.dump(scaler_other, f)

    return model
