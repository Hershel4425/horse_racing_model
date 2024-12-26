# データ取得に関係するプログラム

import datetime
import os
import pickle
import time
import traceback
import logging

import bs4
import chromedriver_binary  # driverのpath指定を省略するために必要  # noqa: F401
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from tqdm import tqdm


INTERVAL_TIME = 0.5  # 遷移間隔（秒）
PARSER = "html5lib"  # beautifulsoupのparser

ROOT_PATH = "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2"

RACE_ID_FOLDER_PATH = "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2/10_scraping/race_id"
RACE_ID_FILE_NAME = "past_race_id.txt"


class IdFormatError(Exception):
    """idの形式がnetkeibaの仕様に合っていない時に出すエラー

    Args:
        Exception (exception): 吐き出すエラー
    """

    pass


class IdLengthError(Exception):
    """結果ページもしくは予想ページにあるhorse_idやjockey_idが同量取得できていない時に出すエラー

    Args:
        Exception (excption): 吐き出すエラー
    """

    pass


def id_format_check(id, id_type):
    """idの形式がnetkeibaの仕様に合っているかチェックする関数

    Args:
        id (int): チェック対象のid
        id_type (str): race_idかhorse_idかなどidの種類
    """
    id_str = str(id)

    if id_type == "race_id":
        # 桁数チェック
        if len(id_str) != 12:
            raise IdFormatError(f"race_idの桁数が異なる: {id}")
        # フォーマットチェック
        if id_str[:2] != "20":
            raise IdFormatError(f"race_idの上2桁が20でない: {id}")
    elif id_type == "horse_id":
        # 桁数チェック
        if len(id_str) != 10:
            raise IdFormatError(f"horse_idの桁数が異なる: {id}")
    elif id_type == "jockey_id":
        # 桁数チェック
        if len(id_str) > 6:
            raise IdFormatError(f"jockey_idの桁数が異なる: {id}")
    elif id_type == "trainer_id":
        # 桁数チェック
        if len(id_str) > 6:
            raise IdFormatError(f"trainer_idの桁数が異なる: {id}")
    else:
        raise IdFormatError(f"想定されていないidタイプ: {id_type}")


def get_driver():
    """ドライバーを準備する関数

    Returns:
        driver (driver, optional): ブラウザのドライバ. Defaults to get_driver().
    """
    # ヘッドレスモードでブラウザを起動
    options = Options()
    options.add_argument("--headless")

    # ブラウザーを起動
    driver = webdriver.Chrome(options=options)

    return driver


def scrape_source_from_page(url, driver=None):
    """対象ページのソース取得

    Args:
        url (str): 対象ページのurl
        driver (driver, optional): ブラウザのドライバ. Defaults to None.

    Returns:
        page_source (source): 対象ページのソース
    """
    if driver is None:
        driver = get_driver()

    try:
        driver.get(url)
        page_source = driver.page_source

        return page_source

    except Exception:
        print(f"Exception\n{traceback.format_exc()}")
        print("urlをdriverに入力してソースを得るところでエラー")
        print(f"page_souce取得時にエラーが発生したurl: {url}")
        # driver.quit()
        driver = get_driver()  # ドライバーを再起動
        print("ドライバーを再起動しました")
        return retry_scrape(url, driver)  # リトライ機制の実装


def retry_scrape(url, driver, max_attempts=3):
    """リトライ機制を持つスクレイピング関数

    Args:
        url (str): 対象ページのurl
        driver (driver): ブラウザのドライバ
        max_attempts (int): 最大試行回数. Defaults to 3.

    Returns:
        page_source (source): 対象ページのソースまたはNone
    """
    attempt = 0
    while attempt < max_attempts:
        try:
            driver.get(url)
            print("リトライ成功")
            return driver.page_source
        except Exception:
            print(f"リトライ {attempt + 1} / {max_attempts}")
            attempt += 1

    print("最大リトライ回数に達しました")
    return None


def scrape_soup_from_past_race_id(race_id, driver=None):
    """過去のrace_idから該当レースページのsoupを取得する関数

    Args:
        race_id (int): soupを取得したいrace_id
        driver (driver, optional): ブラウザのドライバ. Defaults to None.

    Returns:
        soup (soup) : 取得したページのsoup
    """
    if driver is None:
        driver = get_driver()

    # 罪にならないようにsleepをつける
    time.sleep(INTERVAL_TIME)
    wide_url = (
        "https://race.netkeiba.com/odds/index.html?type=b5&race_id="
        + str(race_id)
        + "&housiki=c99"
    )
    huku_url = (
        "https://race.netkeiba.com/odds/index.html?type=b1&race_id="
        + str(race_id)
        + "&rf=shutuba_submenu"
    )
    try:
        wide_page_source = scrape_source_from_page(wide_url, driver)
        wide_soup = bs4.BeautifulSoup(wide_page_source, features=PARSER)
    except Exception:
        print(f"Exception\n{traceback.format_exc()}")
        print("wide_urlをdriverに入力してソースを得るところでエラー")
        print(f"soup取得時にエラーが発生したrace_id: {race_id}")
    try:
        huku_page_source = scrape_source_from_page(huku_url, driver)
        huku_soup = bs4.BeautifulSoup(huku_page_source, features=PARSER)
    except Exception:
        print(f"Exception\n{traceback.format_exc()}")
        print("huku_wide_urlをdriverに入力してソースを得るところでエラー")
        print(f"soup取得時にエラーが発生したrace_id: {race_id}")

    return wide_soup, huku_soup


def scrape_odds(race_id, driver=None):
    """過去のrace_idからオッズを取得する"""

    try:
        wide_soup, huku_soup = scrape_soup_from_past_race_id(race_id, driver)

        # ワイドのオッズを取得
        odds_list = []
        # 各行をループ処理
        for row in wide_soup.find_all(
            "tr", id=lambda x: x and x.startswith("ninki-data_")
        ):
            # 人気順位を取得
            popularity = row.find(
                "span", id=lambda x: x and x.startswith("Ninki-")
            ).text

            # 馬番1と馬番2を取得
            horse_numbers = row.find_all("span", class_="UmaBan")
            horse_number1 = horse_numbers[0].text
            horse_number2 = horse_numbers[1].text

            # 倍率下限と倍率上限を取得
            odds_min_element = row.find(
                "span", id=lambda x: x and x.startswith("odds-")
            )
            odds_min = odds_min_element.text if odds_min_element else ""

            odds_max_element = row.find(
                "span", id=lambda x: x and x.startswith("odds-min-")
            )
            odds_max = odds_max_element.text if odds_max_element else odds_min

            # 倍率一覧に追加
            odds_list.append(
                [race_id, popularity, horse_number1, horse_number2, odds_min, odds_max]
            )

        # DataFrameを作成
        wide_df = pd.DataFrame(
            odds_list,
            columns=["race_id", "人気", "馬番1", "馬番2", "倍率下限", "倍率上限"],
        )

        # 複勝オッズを格納するリスト
        odds_list = []

        # 複勝オッズのテーブルを取得
        fuku_table = huku_soup.find("div", id="odds_fuku_block").find("table")
        # 各行をループ処理
        for row in fuku_table.find_all("tr")[1:]:  # ヘッダー行をスキップ
            # 馬番を取得
            horse_number = row.find_all("td", class_="W31")[1].text.strip()

            # オッズを取得
            odds_range = row.find("span", class_="Odds").text.split(" - ")
            odds_min = odds_range[0]
            odds_max = odds_range[1] if len(odds_range) > 1 else odds_min

            # リストに追加
            odds_list.append([race_id, horse_number, odds_min, odds_max])

        # DataFrameを作成
        huku_df = pd.DataFrame(
            odds_list, columns=["race_id", "馬番", "最低オッズ", "最高オッズ"]
        )

        return wide_df, huku_df

    except Exception as e:
        logging.error(
            f"Error occurred while scraping odds for race_id: {race_id}. Error: {str(e)}"
        )
        return pd.DataFrame(), pd.DataFrame()


def scrape_multiple_race_result(race_id_list, driver=None):
    """過去のrace_idのリストからレース結果を取得する関数

    Args:
        race_id_list (list): race_idが格納されたリスト
        driver (driver, optional): ブラウザのドライバ. Defaults to get_driver().

    Returns:
        wide_df (dataframe): ワイドのオッズデータ
        huku_df (dataframe): 複勝のオッズデータ
    """

    if driver is None:
        driver = get_driver()

    wide_odds_df_list = list()
    huku_odds_df_list = list()
    # race_idごとにスクレイピングしてリストに格納
    for i, race_id in enumerate(tqdm(race_id_list)):
        wide_odds_df, huku_odds_df = scrape_odds(race_id, driver)
        wide_odds_df_list.append(wide_odds_df)
        huku_odds_df_list.append(huku_odds_df)

        # driverのインスタンスが切れないように、一定回数ごとに再起動
        if i % 100 == 0:
            driver = get_driver()

    # 結合
    try:
        wide_df = pd.concat(wide_odds_df_list, axis=0)
        huku_df = pd.concat(huku_odds_df_list, axis=0)
    except Exception:
        print("Exception\n" + traceback.format_exc())
        print(
            "このエラー分のみが出ている場合、スクレピングしたDataFrameの結合でエラーの可能性あり"
        )

    return wide_df, huku_df


def select_scrape_race_id():
    """スクレイピングすべきrace_idを選択する関数"""
    # レース情報のスクレイピング結果を取得
    with open(os.path.join(RACE_ID_FOLDER_PATH, RACE_ID_FILE_NAME), "rb") as f:
        past_race_id_list = pickle.load(f)
        new_race_id_list = [item for sublist in past_race_id_list for item in sublist]

    # スクレイピング済みのrace_idを、odds_dfから取得
    root_path = "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2/10_scraping/odds"
    huku_path = os.path.join(root_path, "huku_odds.csv")
    id_df = pd.read_csv(huku_path)
    scraped_race_id_list = list(
        id_df.loc[id_df["最低オッズ"] != "---.-"]["race_id"].unique()
    )

    # スクレイプ済みのrace_idを取り除く
    race_id_list = [
        item for item in new_race_id_list if item not in scraped_race_id_list
    ]

    print("スクレイピング対象のrace_id数:", len(race_id_list))

    return race_id_list


def run_odds_scrape():
    """スクレイピんぐを実行する関数"""
    # レースIDを取得
    race_id_list = select_scrape_race_id()
    # ドライバーを取得
    driver = get_driver()
    # レースIDリストからオッズを取得
    wide_df, huku_df = scrape_multiple_race_result(race_id_list, driver)

    # 過去のwide_dfとhuku_dfを取得
    root_path = "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2/10_scraping/odds"
    wide_path = os.path.join(root_path, "wide_odds.csv")
    huku_path = os.path.join(root_path, "huku_odds.csv")
    if os.path.exists(wide_path):
        past_wide_df = pd.read_csv(wide_path)
        wide_df = pd.concat([past_wide_df, wide_df], axis=0)
    if os.path.exists(huku_path):
        past_huku_df = pd.read_csv(huku_path)
        huku_df = pd.concat([past_huku_df, huku_df], axis=0)

    # ファイルに保存
    date = datetime.date.today().strftime("%Y%m%d")

    back_up_wide_path = os.path.join(root_path, f"backup/{date}_wide_odds_backup.csv")
    back_up_huku_path = os.path.join(root_path, f"backup/{date}_huku_odds_backup.csv")
    wide_df.to_csv(wide_path, index=False)
    huku_df.to_csv(huku_path, index=False)
    wide_df.to_csv(back_up_wide_path, index=False)
    huku_df.to_csv(back_up_huku_path, index=False)
