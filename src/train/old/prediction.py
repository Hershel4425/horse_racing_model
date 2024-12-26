import pickle
from IPython.display import display

import numpy as np

from .preprocessing import process_null
from .result_saving import make_url_from_date
from .stacking import stacking


# ヘルパー関数
def load_file_from_path(path):
    """ファイルをロードする"""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_pretrained_models(config, col, date, mode):
    """学習済みモデルをロードする

    Args:
        config (dict): configファイル
        col (str): 予測対象の列名
        date (str): 学習済みモデルの日付
        mode (str): モデルか精度か特徴量かを指定する

    Returns:
        return_list (list): 学習済みモデルなどのリスト
    """
    model_url = make_url_from_date(date, config, "model")
    mode_to_suffix = {
        "model": "lightgbm_model.pkl",
        "accuracy": "accuracy.pickle",
        "feature": "feature.pickle",
    }
    # mode_to_suffix = {'model': 'tabnet_model.pkl', 'accuracy': 'accuracy.pickle'}

    return [
        load_file_from_path(f"{model_url}/{col}_{i}_{mode_to_suffix[mode]}")
        for i in range(7)
    ]


# へルパー関数
def process_target_dict(
    col,
    config,
    feature,
    model_list_dict,
    feature_list_dict,
    accuracy_list_dict,
    pred_target_list,
):
    """予測対象のリストを作成する、学習済みモデルを用いる場合は、学習済みモデルをロードする

    Args:
        col (str): 予測対象の列名
        config (dict): configファイル
        feature (dict): configファイルの一部を抜き出したdict
        model_list_dict (dict): 修正するモデルのリストのdict
        feature_list_dict (dict): 修正するモデルの特徴量のリストのdict
        accuracy_list_dict (dict): 修正するモデルの精度のリストのdict
        pred_target_list (list): 予測対象のリスト
    """
    pred_target_list.append(col)

    # 学習フラグが立っておらず、学習済みモデルを用いる時
    if (not feature["train_flag"]) & (not feature["use_model_pred"]["flag"]):
        date = feature["use_model_pred"]["date"]
        model_list_dict[col] = load_pretrained_models(config, col, date, "model")
        accuracy_list_dict[col] = load_pretrained_models(config, col, date, "accuracy")
        try:
            feature_list_dict[col] = load_pretrained_models(
                config, col, date, "feature"
            )
        except Exception:
            feature_list_dict[col] = []
        print(f"{col}の学習済みモデルをロードしました")
    # 学習フラグが立っている時は何もしない
    elif feature["train_flag"]:
        pass
    # それ以外の時はエラーを出す
    else:
        raise ValueError(f'{feature["name"]}のtrain_flagが不正です')


def make_pred_target_list(
    model_list_dict, accuracy_list_dict, feature_list_dict, config
):
    """予測対象のリストを作成する、学習済みモデルを用いる場合は、学習済みモデルをロードする

    Args:
        model_list_dict (dict): 学習を行ったモデルのリストのdict
        accuracy_list_dict (dict): 学習を行ったモデルの精度のリストのdict
        feature_list_dict (dict): 学習を行ったモデルの特徴量のリストのdict
        config (dict): configファイル

    Returns:
        pred_target_list (list): 予測対象のリスト
        model_list_dict (dict): 予測に用いるモデルのリストのdict
        accuracy_list_dict (dict): 予測に用いるモデルの精度のリストのdict
    """
    pred_target_list = []

    for feature in config["feature"]:
        # 予測フラグが立っている時
        if feature["pred_flag"]:
            # 馬番別特徴量の時は、馬番別の特徴量を処理する
            if feature["column_num"] == 18:
                for i in range(18):
                    col = f'{i + 1}_{feature["name"]}'
                    process_target_dict(
                        col,
                        config,
                        feature,
                        model_list_dict,
                        feature_list_dict,
                        accuracy_list_dict,
                        pred_target_list,
                    )
            # 1つだけの時はその特徴量を処理する
            elif feature["column_num"] == 1:
                col = feature["name"]
                process_target_dict(
                    col,
                    config,
                    feature,
                    model_list_dict,
                    feature_list_dict,
                    accuracy_list_dict,
                    pred_target_list,
                )
            # それ以外の時はエラーを出す
            else:
                raise ValueError(f'{feature["name"]}のcolumn_numが不正です')

    return pred_target_list, model_list_dict, accuracy_list_dict


def predict(
    train_data,
    model_list,
    feature_list,
    feature,
    target_col,
    config,
    stacking_flag=False,
):
    """予測を行う

    Args:
        train_data (dataframe): 学習データ
        model_list (list): 学習済みモデル
        feature_list (list): 寄与の高い特徴量
        feature (dict): 学習方法を記載した辞書
        target_col (str): 学習対象の列名
        config (dict): configファイル
        stacking_flag (boolean): stacking関数内のコードかどうかのフラグ

    Returns:
        preds: _description_
    """
    preds = []

    # trainとtestに分けない
    if stacking_flag:
        x_test = train_data
    else:
        x_test = train_data.tail(config["test_num"])

    ##### 予測
    try:
        for i, model in enumerate(model_list):
            print(f"{i + 1}番目のモデルで予測を行います...")

            # 特徴量を取り出す
            top_feature = feature_list[i]
            x_test_top = x_test[top_feature]

            if feature["train_method"] == "lightgbm":
                pred = model.predict(x_test_top, num_iteration=model.best_iteration)

            # tabnetの時は、x_testをnumpyに変換する
            elif feature["train_method"] == "tabnet":
                # multiclassのstacking時は、predictで回答のみを返す
                # mulitclassのpredict時は、predict_probaで各クラスの確率を返す
                if feature["train_type"] == "multiclass":
                    pred = model.predict_proba(x_test.values)
                else:
                    pred = model.predict(x_test.values)

            preds.append(pred)

    except Exception as e:
        print(e)
        raise ValueError(f"{i + 1}番目のモデルで予測に失敗しました")

    return preds


def predicts(
    df,
    feature_df,
    label_df,
    model_list_dict,
    accuracy_list_dict,
    feature_list_dict,
    pred_target_list,
    config,
):
    """予測を行う

    Args:
        df (_type_): inputのデータフレーム
        feature_df (_type_): 特徴量を記載したデータフレーム
        label_df (_type_): 結果確認用データフレーム
        model_list_dict (_type_): _description_
        accuracy_list_dict (_type_): _description_
        feature_list_dict (_type_): _description_
        pred_target_list (_type_): _description_
        config (_type_): _description_

    Returns:
        return_df: _description_
    """
    # 結果確認用データフレームをコピー
    result_df = feature_df.copy()

    # tager_columnsの数だけ学習を行う
    for i, target_col in enumerate(pred_target_list):
        print(f"{target_col}の予測を実行しています...")

        # 後のif文処理のため、target_colに対応したconfigのfeatureを取り出す
        if "_" in target_col:
            target_name = target_col.split("_")[1]
        else:
            target_name = target_col
        config_feature = [
            feature for feature in config["feature"] if feature["name"] == target_name
        ][0]

        # tabnetで学習する時は、欠損値を処理する
        if config_feature["train_method"] == "tabnet":
            df = process_null(df, config)

        # targetとtrainに分ける
        features = [col for col in df.columns if col not in pred_target_list]
        train_data = df[features]

        # モデルを取り出す
        model_list = model_list_dict[target_col]
        accuracy_list = accuracy_list_dict[target_col]
        feature_list = feature_list_dict[target_col]

        # 一つずつ予測を行う
        preds = predict(
            train_data,
            model_list,
            feature_list,
            config_feature,
            target_col,
            config,
            stacking_flag=False,
        )

        # 精度を元に重み付け
        # 誤差の逆数を重みとして計算
        weights = [1 / rmse for rmse in accuracy_list]
        # 重みを正規化して合計が1になるようにする
        normalized_weights = [w / sum(weights) for w in weights]
        # 重み付けされた予測の合計を計算
        ensemble_pred = sum(
            pred * weight for pred, weight in zip(preds, normalized_weights)
        )
        # 予測がregressionの時は、flattenする
        if config_feature["train_type"] == "regression":
            ensemble_pred = ensemble_pred.flatten().tolist()
        else:
            pass

        test_race_list = list(label_df["race_id"].unique())[-config["test_num"] :]

        # 予測df
        df_test = feature_df.loc[feature_df["race_id"].isin(test_race_list)]
        display(f"必要行取り出し後: {df_test.shape}")

        # 予測結果をデータフレームに結合
        for i in range(config["test_num"]):
            tatesu = df_test.loc[df_test["race_id"] == test_race_list[i]].shape[0]

            if target_col in ["1着馬番", "2着馬番", "3着馬番"]:
                a = result_df.loc[result_df["race_id"] == test_race_list[i]].shape[0]
                b = ensemble_pred[i][:tatesu].shape[0]
                if a != b:
                    print(a)
                    print(b)
                    display(result_df.loc[result_df["race_id"] == test_race_list[i]])
                    display(ensemble_pred[i][:tatesu])
                    print(i)
                result_df.loc[
                    result_df["race_id"] == test_race_list[i], target_col + "確率"
                ] = ensemble_pred[i][:tatesu]

                if target_col == "1着馬番":
                    first_max_preds = [
                        list(pred).index(max(list(pred))) + 1 for pred in ensemble_pred
                    ]
                    result_df.loc[
                        result_df["race_id"] == test_race_list[i], "予測1着"
                    ] = first_max_preds[i]
            elif "着差" in target_col:
                result_df.loc[
                    result_df["race_id"] == test_race_list[i], "予測_" + target_col
                ] = ensemble_pred[i]
            elif "通過順位" in target_col:
                pred = ensemble_pred[i] * tatesu
                rounded_output = np.round(pred).astype(int)
                clipped_output = np.clip(rounded_output, 1, 18)
                result_df.loc[
                    result_df["race_id"] == test_race_list[i], "予測_" + target_col
                ] = clipped_output
            else:
                result_df.loc[
                    result_df["race_id"] == test_race_list[i], "予測_" + target_col
                ] = ensemble_pred[i]

        # 予測したrace_idのみ取り出す
        result_df = result_df.loc[result_df["race_id"].isin(test_race_list)]

        # スタッキングを行うstacking(df, target_col, model_list, accuracy_list, feature_list, config_feature, config)
        df = stacking(
            df,
            target_col,
            model_list,
            accuracy_list,
            feature_list,
            config_feature,
            config,
        )

    return result_df
