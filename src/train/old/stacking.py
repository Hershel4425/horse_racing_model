
from .preprocessing import process_null


def predict(
    train_data, model_list, feature_list, config_feature, config, stacking_flag=True
):
    """予測を行う

    Args:
        train_data (dataframe): 学習データ
        model_list (list): 学習済みモデル
        feature_list (list): 寄与の高い特徴量
        config_feature (dict): 学習方法を記載した辞書
        config (dict): configファイル
        stacking_flag (boolean): stacking関数内のコードかどうかのフラグ

    Returns:
        preds: _description_
    """
    preds = []

    ##### 予測
    try:
        for i, model in enumerate(model_list):
            x_test = train_data[feature_list[i]]
            print(f"{i + 1}番目のモデルで予測を行います...")
            if config_feature["train_method"] == "lightgbm":
                pred = model.predict(x_test, num_iteration=model.best_iteration)

            # tabnetの時は、x_testをnumpyに変換する
            elif config_feature["train_method"] == "tabnet":
                # multiclassのstacking時は、predictで回答のみを返す
                # mulitclassのpredict時は、predict_probaで各クラスの確率を返す
                if (
                    config_feature["train_type"] == "multiclass"
                    and stacking_flag
                ):
                    pred = model.predict_proba(x_test.values)
                else:
                    pred = model.predict(x_test.values)

            preds.append(pred)

    except Exception as e:
        print(e)
        raise ValueError(f"{i + 1}番目のモデルで予測に失敗しました")

    return preds


def stacking(
    df, target_col, model_list, accuracy_list, feature_list, config_feature, config
):
    """スタッキングを実施し、データフレームに予測値を結合する関数

    Args:
        df (datagrame): inputのデータフレーム
        target_col (str): 目的変数の列名
        model_list (list): 学習したモデルのリスト
        accuracy_list (list): 学習したモデルの精度のリスト
        features_list (list): 特徴量のリストのリスト
        config_feature (dict): 学習方法を記載した辞書
        config (dict): configファイル

    Returns:
        df: 処理したデータフレーム
    """

    # tabnetで学習する時は、欠損値を処理する
    if config_feature["train_method"] == "tabnet":
        not_null_df = process_null(df, config)
    else:
        not_null_df = df

    if config_feature["stacking_flag"]:
        print(target_col + "のスタッキング用予測を行います...")
        preds = []

        # 予測を作成する
        preds = predict(
            not_null_df,
            model_list,
            feature_list,
            config_feature,
            config,
            stacking_flag=True,
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
        print("予測の長さ：", len(ensemble_pred))

        # クラス予測の時、最も確率の高いクラスを予測結果とする
        if config_feature["train_type"] == "multiclass":
            pass
        elif config_feature["train_type"] == "regression":
            ensemble_pred = ensemble_pred.flatten().tolist()
        else:
            raise ValueError(f'{config_feature["name"]}のtrain_typeが不正です')

        # 予測値を結合する
        print(f"target_col:{target_col}")
        print("スタッキング用予測を結合します...")
        print("結合前のdfのshape: " + str(df.shape))
        if config_feature["train_type"] == "multiclass":
            columns = [
                f"stack_{i+1}_{target_col}" for i in range(len(ensemble_pred[0]))
            ]
            df.loc[:, columns] = ensemble_pred
        # adhoc対応
        elif "C通過順位" in target_col:
            df.loc[:, f"stack_{target_col}"] = ensemble_pred
        else:
            df.loc[:, f"stack_{target_col}"] = ensemble_pred
        print("結合後のdfのshape: " + str(df.shape))

    return df
