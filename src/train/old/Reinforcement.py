import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

import pandas as pd

from tqdm import tqdm

## データの読み込み
root_path = "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2"
score_data = (
    root_path + "/50_machine_learning/output_data/pred/20240418160918_result_df.csv"
)
huku_data = root_path + "/10_scraping/odds/huku_odds.csv"
wide_data = root_path + "/10_scraping/odds/wide_odds.csv"

score_df = pd.read_csv(score_data)
huku_df = pd.read_csv(huku_data, encoding="utf-8-sig")
wide_df = pd.read_csv(wide_data, encoding="utf-8-sig")

# オッズが'---.-'となっているデータを削除する
huku_df = huku_df.loc[huku_df["最低オッズ"] != "---.-"]
wide_df = wide_df.loc[wide_df["倍率下限"] != "---.-"]

# 型変換
huku_df["最低オッズ"] = huku_df["最低オッズ"].astype("float")
huku_df["馬番"] = huku_df["馬番"].astype("int")
wide_df["馬番1"] = wide_df["馬番1"].astype("int")
wide_df["馬番2"] = wide_df["馬番2"].astype("int")
wide_df["倍率下限"] = wide_df["倍率下限"].astype("float")

# 3つのデータフレームのrace_idを取得します
race_ids_1 = set(score_df["race_id"])
race_ids_2 = set(huku_df["race_id"])
race_ids_3 = set(wide_df["race_id"])

# 3つのデータフレームに共通するrace_idを取得します
common_race_ids = race_ids_1.intersection(race_ids_2, race_ids_3)

# 各データフレームから共通するrace_idのデータを取り出します
score_df = score_df[score_df["race_id"].isin(common_race_ids)]
huku_df = huku_df[huku_df["race_id"].isin(common_race_ids)]
wide_df = wide_df[wide_df["race_id"].isin(common_race_ids)]

# race_idの一意な値を取得
unique_race_ids = pd.unique(score_df["race_id"])
num_races = len(unique_race_ids)

# テンソル1: score
print("tensor1")
score_tensor = torch.zeros(num_races, 18)
for i, race_id in tqdm(enumerate(unique_race_ids)):
    race_scores = score_df[score_df["race_id"] == race_id]
    for _, row in race_scores.iterrows():
        score_tensor[i, row["馬番"] - 1] = row["score"]

# テンソル2: 単勝
print("tensor2")
odds_tensor = torch.zeros(num_races, 18)
for i, race_id in tqdm(enumerate(unique_race_ids)):
    race_odds = score_df[score_df["race_id"] == race_id]
    for _, row in race_odds.iterrows():
        odds_tensor[i, row["馬番"] - 1] = row["単勝"]

# テンソル3: 最低オッズ
print("tensor3")
min_odds_tensor = torch.zeros(num_races, 18)
for i, race_id in tqdm(enumerate(unique_race_ids)):
    race_min_odds = huku_df[huku_df["race_id"] == race_id]
    for _, row in race_min_odds.iterrows():
        min_odds_tensor[i, row["馬番"] - 1] = row["最低オッズ"]

# テンソル4: 単勝 (着順が1の場合)
print("tensor4")
win_odds_tensor = torch.zeros(num_races, 18)
for i, race_id in tqdm(enumerate(unique_race_ids)):
    race_data = score_df[score_df["race_id"] == race_id]
    for _, row in race_data.iterrows():
        if row["着順"] == 1:
            win_odds_tensor[i, row["馬番"] - 1] = row["単勝"]

# テンソル5: 最低オッズ (着順が3以下の場合)
print("tensor5")
top3_min_odds_tensor = torch.zeros(num_races, 18)
for i, race_id in tqdm(enumerate(unique_race_ids)):
    race_data = pd.merge(
        score_df[score_df["race_id"] == race_id],
        huku_df[huku_df["race_id"] == race_id],
        on="馬番",
    )
    for _, row in race_data.iterrows():
        if row["着順"] <= 3:
            top3_min_odds_tensor[i, row["馬番"] - 1] = row["最低オッズ"]

# テンソル6: 倍率下限 (ワイド)
print("tensor6")
wide_lower_tensor = torch.zeros(num_races, 18 * 18)
for i, race_id in tqdm(enumerate(unique_race_ids)):
    race_wide = wide_df[wide_df["race_id"] == race_id]
    for _, row in race_wide.iterrows():
        index = (row["馬番1"] - 1) * 18 + (row["馬番2"] - 1)
        wide_lower_tensor[i, index] = row["倍率下限"]

# テンソル7: 倍率下限 (ワイド、着順が3以下の場合)
print("tensor7")
top3_wide_lower_tensor = torch.zeros(num_races, 18 * 18)
for i, race_id in tqdm(enumerate(unique_race_ids)):
    race_score = score_df[score_df["race_id"] == race_id]
    race_wide = wide_df[wide_df["race_id"] == race_id]

    for _, wide_row in race_wide.iterrows():
        horse1 = wide_row["馬番1"]
        horse2 = wide_row["馬番2"]

        score_horse1 = race_score[race_score["馬番"] == horse1]
        score_horse2 = race_score[race_score["馬番"] == horse2]

        if not score_horse1.empty and not score_horse2.empty:
            rank1 = score_horse1["着順"].values[0]
            rank2 = score_horse2["着順"].values[0]

            if rank1 <= 3 and rank2 <= 3:
                index = (horse1 - 1) * 18 + (horse2 - 1)
                top3_wide_lower_tensor[i, index] = wide_row["倍率下限"]

# 学習用データ
train_score_tensor = score_tensor[:-100]
train_odds_tensor = odds_tensor[:-100]
train_min_odds_tensor = min_odds_tensor[:-100]
train_win_odds_tensor = win_odds_tensor[:-100]
train_top3_min_odds_tensor = top3_min_odds_tensor[:-100]
train_wide_lower_tensor = wide_lower_tensor[:-100]
train_top3_wide_lower_tensor = top3_wide_lower_tensor[:-100]


# 検証用データ
test_score_tensor = score_tensor[-100:]
test_odds_tensor = odds_tensor[-100:]
test_min_odds_tensor = min_odds_tensor[-100:]
test_win_odds_tensor = win_odds_tensor[-100:]
test_top3_min_odds_tensor = top3_min_odds_tensor[-100:]
test_wide_lower_tensor = wide_lower_tensor[-100:]
test_top3_wide_lower_tensor = top3_wide_lower_tensor[-100:]


pre_odds = torch.cat(
    (train_odds_tensor, train_min_odds_tensor, train_wide_lower_tensor), dim=1
)
post_odds = torch.cat(
    (train_win_odds_tensor, train_top3_min_odds_tensor, train_top3_wide_lower_tensor),
    dim=1,
)


class RacingAI(nn.Module):
    def __init__(self, input_size, output_size):
        super(RacingAI, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, output_size)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        x = torch.clamp(x, min=-10, max=10)
        x = torch.softmax(x, dim=-1)
        return x


def calculate_reward(purchase_action, post_odds, current_balance):
    purchase_action = torch.tensor(purchase_action).float()
    total_purchase = purchase_action.sum()

    if total_purchase == 0:
        return 0.0
    else:
        returns = purchase_action * post_odds
        total_return = returns.sum()
        reward = (total_return / total_purchase) - 1
        reward = torch.clamp(reward, min=-1, max=1)
        return reward.item()


def train(
    model,
    optimizer,
    scheduler,
    dataloader,
    val_victory_scores,
    val_pre_odds,
    val_post_odds,
    epochs,
    patience=5,
):
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        for batch in dataloader:
            victory_scores, pre_odds, post_odds = batch
            state = torch.cat((victory_scores, pre_odds), dim=1)
            probs = model(state)
            actions = torch.multinomial(probs, num_samples=1).squeeze()
            purchase_actions = actions.detach().numpy()
            rewards = torch.tensor(
                [
                    calculate_reward(
                        purchase_action.item(), post_odd, current_balance=100.0
                    )
                    for purchase_action, post_odd in zip(purchase_actions, post_odds)
                ]
            )
            loss = (-rewards * probs.log()[range(len(actions)), actions]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_state = torch.cat((val_victory_scores, val_pre_odds), dim=1)
            val_probs = model(val_state)
            val_actions = val_probs.argmax(dim=1)
            val_purchase_actions = val_actions.detach().numpy()
            val_rewards = torch.tensor(
                [
                    calculate_reward(
                        purchase_action.item(), post_odd, current_balance=100.0
                    )
                    for purchase_action, post_odd in zip(
                        val_purchase_actions, val_post_odds
                    )
                ]
            )
            val_loss = (
                (-val_probs.log()[range(len(val_actions)), val_actions] * val_rewards)
                .mean()
                .item()
            )
            val_accuracy = (
                (val_actions == val_probs.argmax(dim=1)).float().mean().item()
            )

        val_losses.append(val_loss)

        print(
            f"Epoch: {epoch}, Loss: {loss.item():.2f}, Val Loss: {val_loss:.2f}, Val Reward: {val_rewards.mean().item():.2f}, Val Accuracy: {val_accuracy:.2f}"
        )

        if val_loss < best_val_loss and val_accuracy > best_val_accuracy:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        scheduler.step()

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    model.load_state_dict(best_model_state)
    return model


def train_with_kfold(
    model_class,
    input_size,
    output_size,
    victory_scores,
    pre_odds,
    post_odds,
    epochs,
    batch_size,
    n_splits=5,
):
    kfold = KFold(n_splits=n_splits)
    models = []

    val_index = int(len(victory_scores) * 0.8)
    val_victory_scores = victory_scores[val_index:]
    val_pre_odds = pre_odds[val_index:]
    val_post_odds = post_odds[val_index:]

    victory_scores = victory_scores[:val_index]
    pre_odds = pre_odds[:val_index]
    post_odds = post_odds[:val_index]

    for train_index, _ in kfold.split(victory_scores):
        train_victory_scores, train_pre_odds, train_post_odds = (
            victory_scores[train_index],
            pre_odds[train_index],
            post_odds[train_index],
        )
        train_dataset = TensorDataset(
            train_victory_scores, train_pre_odds, train_post_odds
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        model = model_class(input_size, output_size)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        model = train(
            model,
            optimizer,
            scheduler,
            train_dataloader,
            val_victory_scores,
            val_pre_odds,
            val_post_odds,
            epochs,
        )
        models.append(model)

    return models


def ensemble_predict(models, victory_scores, pre_odds):
    state = torch.cat((victory_scores, pre_odds), dim=1)
    probs = torch.stack([model(state) for model in models])
    weights = torch.tensor([0.4, 0.3, 0.2, 0.1])
    weighted_probs = torch.sum(probs * weights.view(-1, 1, 1), dim=0)
    return weighted_probs


input_size = 18 + 18 + 18 + 18 * 18
output_size = 18 + 18 + 18 * 18
batch_size = 32
epochs = 100

models = train_with_kfold(
    RacingAI,
    input_size,
    output_size,
    train_score_tensor,
    pre_odds,
    post_odds,
    epochs,
    batch_size,
)
