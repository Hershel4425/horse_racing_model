# データ取得に関係するプログラム

import os
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
    fuku_url = (
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
        fuku_page_source = scrape_source_from_page(fuku_url, driver)
        fuku_soup = bs4.BeautifulSoup(fuku_page_source, features=PARSER)
    except Exception:
        print(f"Exception\n{traceback.format_exc()}")
        print("fuku_wide_urlをdriverに入力してソースを得るところでエラー")
        print(f"soup取得時にエラーが発生したrace_id: {race_id}")

    return wide_soup, fuku_soup


def scrape_odds(race_id, driver=None):
    """過去のrace_idからオッズを取得する"""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            wide_soup, fuku_soup = scrape_soup_from_past_race_id(race_id, driver)
            
            # ワイドのオッズを取得
            odds_list = []
            for row in wide_soup.find_all("tr", id=lambda x: x and x.startswith("ninki-data_")):
                popularity = row.find("span", id=lambda x: x and x.startswith("Ninki-")).text
                horse_numbers = row.find_all("span", class_="UmaBan")
                horse_number1 = horse_numbers[0].text
                horse_number2 = horse_numbers[1].text
                odds_min_element = row.find("span", id=lambda x: x and x.startswith("odds-"))
                odds_min = odds_min_element.text if odds_min_element else ""
                odds_max_element = row.find("span", id=lambda x: x and x.startswith("odds-min-"))
                odds_max = odds_max_element.text if odds_max_element else odds_min
                odds_list.append([race_id, popularity, horse_number1, horse_number2, odds_min, odds_max])
            
            if odds_list == []:
                raise Exception("Wide odds list is empty.")
            
            wide_df = pd.DataFrame(odds_list, columns=["race_id", "人気", "馬番1", "馬番2", "倍率下限", "倍率上限"])
            print('wide_df作成完了')
            
            # 複勝オッズを格納するリスト
            odds_list = []
            fuku_table = fuku_soup.find("div", id="odds_fuku_block").find("table")
            for row in fuku_table.find_all("tr")[1:]:
                horse_number = row.find_all("td", class_="W31")[1].text.strip()
                if row.find("span", class_="Odds") is None:
                    odds_min = "---.-"
                    odds_max = "---.-"
                else:
                    # もしhorse_infoが「取消」であれば、オッズが取得できないのでスキップ
                    # horse_infoが存在するかチェックし、存在する場合のみテキストを取得
                    horse_info = row.find("span", class_="horse_info")
                    if horse_info is not None and horse_info.text == "取消":
                        continue
                    odds_range = row.find("span", class_="Odds").text.split(" - ")
                    odds_min = odds_range[0]
                    odds_max = odds_range[1] if len(odds_range) > 1 else odds_min
                odds_list.append([race_id, horse_number, odds_min, odds_max])
            
            if odds_list == []:
                raise Exception("Fuku odds list is empty.")
            
            fuku_df = pd.DataFrame(odds_list, columns=["race_id", "馬番", "最低オッズ", "最高オッズ"])
            print('fuku_df作成完了')
            
            # 単勝オッズを格納するリスト
            odds_list = []
            tan_table = fuku_soup.find("div", id="odds_tan_block").find("table")
            for row in tan_table.find_all("tr")[1:]:
                horse_number = row.find_all("td", class_="W31")[1].text.strip()
                if row.find("span", class_="Odds") is None:
                    odds_tan = "---.-"
                else:
                    # もしhorse_infoが「取消」であれば、オッズが取得できないのでスキップ
                    # horse_infoが存在するかチェックし、存在する場合のみテキストを取得
                    horse_info = row.find("span", class_="horse_info")
                    if horse_info is not None and horse_info.text == "取消":
                        continue
                    odds_range = row.find("span", class_="Odds").text.split(" - ")
                    odds_tan = odds_range[0]
                odds_list.append([race_id, horse_number, odds_tan])
            
            if odds_list == []:
                raise Exception("Tan odds list is empty.")
            
            tan_df = pd.DataFrame(odds_list, columns=["race_id", "馬番", "単勝オッズ"])
            print('tan_df作成完了')
            
            return wide_df, fuku_df, tan_df
        
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1}/{max_retries} failed. Error: {str(e)}")
            if attempt < max_retries - 1:
                logging.info("Retrying scraping...")
            else:
                logging.error(f"Failed to scrape odds for race_id: {race_id} after {max_retries} attempts.")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def scrape_multiple_race_result(race_id_list, driver=None):
    """race_idリストからオッズを取得する関数

    Args:
        race_id_list (list): race_idが格納されたリスト
        driver (driver, optional): ブラウザのドライバ. Defaults to get_driver().

    Returns:
        wide_df (dataframe): ワイドのオッズデータ
        fuku_df (dataframe): 複勝のオッズデータ
    """

    if driver is None:
        driver = get_driver()

    wide_odds_df_list = list()
    fuku_odds_df_list = list()
    tan_odds_df_list = list()
    # race_idごとにスクレイピングしてリストに格納
    for i, race_id in enumerate(tqdm(race_id_list)):
        wide_odds_df, fuku_odds_df, tan_odds_df = scrape_odds(race_id, driver)
        wide_odds_df_list.append(wide_odds_df)
        fuku_odds_df_list.append(fuku_odds_df)
        tan_odds_df_list.append(tan_odds_df)

        # driverのインスタンスが切れないように、一定回数ごとに再起動
        if i % 100 == 0:
            driver = get_driver()

    # 結合
    try:
        wide_df = pd.concat(wide_odds_df_list, axis=0)
        fuku_df = pd.concat(fuku_odds_df_list, axis=0)
        tan_df = pd.concat(tan_odds_df_list, axis=0)
    except Exception:
        print("Exception\n" + traceback.format_exc())
        print(
            "このエラー分のみが出ている場合、スクレピングしたDataFrameの結合でエラーの可能性あり"
        )

    return wide_df, fuku_df, tan_df


def run_odds_scrape(race_id):
    """スクレイピングを実行する関数"""
    # ドライバーを取得
    driver = get_driver()
    # レースIDリストからオッズを取得
    wide_df, fuku_df, tan_df = scrape_multiple_race_result([race_id], driver)

    # 過去のwide_dfとfuku_dfを取得
    root_path = "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/152_競馬_v2/40_automation"
    wide_path = os.path.join(root_path, f"{race_id}_wide_odds.csv")
    fuku_path = os.path.join(root_path, f"{race_id}_fuku_odds.csv")
    tan_path = os.path.join(root_path, f"{race_id}_tan_odds.csv")

    # ファイルに保存
    wide_df.to_csv(wide_path, index=False)
    fuku_df.to_csv(fuku_path, index=False)
    tan_df.to_csv(tan_path, index=False)

def run_odds_scrape_with_retry(race_id):
    """スクレイピングを実行し、エラーが発生した場合は5分後に再実行する関数"""
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            run_odds_scrape(race_id)
            print(f"race_id: {race_id} のスクレイピングが正常に完了しました")
            break
        except Exception as e:
            retry_count += 1
            logging.error(f"race_id: {race_id} のスクレイピング中にエラーが発生しました: {str(e)}")
            logging.error(traceback.format_exc())

            if retry_count < max_retries:
                logging.info(f"5分後に再試行します (試行回数: {retry_count}/{max_retries})")
                time.sleep(300)  # 5分待機
            else:
                logging.error("最大試行回数に達しました。スクレイピングを中止します")