# データ取得に関係するプログラム

import datetime
import math
import os
import re
import time
import traceback
# import random

import bs4
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import chromedriver_binary  # driverのpath指定を省略するために必要  # noqa: F401
from tqdm import tqdm
from io import StringIO


# 先頭あたりに
from functools import wraps
_last_access = 0

def wait_interval(sec=10):
    def deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            global _last_access
            gap = sec - (time.time() - _last_access)
            if gap > 0:
                time.sleep(gap)
            res = func(*args, **kwargs)
            _last_access = time.time()
            return res
        return wrapper
    return deco



PARSER = "html5lib"  # beautifulsoupのparser

# 全体のルートパス
ROOT_PATH = "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/153_競馬_v3/data"

# 過去のレースデータをdataframe
RACE_INFO_DF_PATH = ROOT_PATH + "/00_raw/10_race_info/race_info_df.csv"
RACE_RESULT_DF_PATH = ROOT_PATH + "/00_raw/11_race_result/race_result_df.csv"
PACE_DF_PATH = ROOT_PATH + "/00_raw/30_pace/pace_df.csv"
ODDS_DF_PATH = ROOT_PATH + "/00_raw/50_odds/odds_df.csv"

# バックアプとして保存するdataframe
DATE_STRING = datetime.date.today().strftime("%Y%m%d")
BACKUP_RACE_INFO_DF_PATH = (
    ROOT_PATH + f"/00_raw/10_race_info/backup/{DATE_STRING}_race_info_df.csv"
)
BACKUP_RACE_RESULT_DF_PATH = (
    ROOT_PATH + f"/00_raw/11_race_result/backup/{DATE_STRING}_race_result_df.csv"
)
BACKUP_PACE_DF_PATH = ROOT_PATH + f"/00_raw/30_pace/backup/{DATE_STRING}_pace_df.csv"
BACKUP_ODDS_DF_PATH = ROOT_PATH + f"/00_raw/50_odds/backup/{DATE_STRING}_odds_df.csv"

# 未来のデータ
RACE_FORECAST_DF_PATH = (
    ROOT_PATH
    + f"/00_raw/12_future_data/race_forecast/{DATE_STRING}_race_forecast_df.csv"
)
FUTURE_RACE_INFO_DF_PATH = (
    ROOT_PATH
    + f"/00_raw/12_future_data/future_race_info/{DATE_STRING}_future_race_info_df.csv"
)

# 過去成績データ
HORSE_PAST_PERFORMANCE_DF_PATH = (
    ROOT_PATH + "/00_raw/20_horse_past_performance/horse_past_performance_df.csv"
)
BACKUP_HORSE_PAST_PERFORMANCE_DF_PATH = (
    ROOT_PATH
    + f"/00_raw/20_horse_past_performance/backup/{DATE_STRING}_horse_past_performance_df.csv"
)

# 血統データ
PEDIGREE_DF_PATH = os.path.join(ROOT_PATH, "00_raw/40_pedigree/pedigree_df.csv")
BACKUP_PEDIGREE_DF_PATH = os.path.join(
    ROOT_PATH, f"00_raw/40_pedigree/backup/{DATE_STRING}_pedigree_df.csv"
)



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


def id_length_check(horse_id_list, jockey_id_list, trainer_id_list=None):
    """idの長さが揃っているかチェックする関数

    Args:
        horse_id_list (list): horse_idのリスト
        jockey_id_list (list): jockey_idのリスト
        trainer_id_list (list, optional): trainer_idのリスト. Defaults to None.
    """
    # trainer_id_listがあるかないかで場合分けして長さチェック
    if trainer_id_list is None:
        if len(horse_id_list) != len(jockey_id_list):
            raise IdLengthError("horse_id_listとjockey_id_listの長さが合っていない")
    else:
        if len(horse_id_list) != len(jockey_id_list):
            raise IdLengthError("horse_id_listとjockey_id_listの長さが合っていない")
        elif len(horse_id_list) != len(trainer_id_list):
            raise IdLengthError("horse_id_listとtrainer_id_listの長さが合っていない")


def get_driver():
    """ドライバーを準備する関数

    Returns:
        driver (driver, optional): ブラウザのドライバ. Defaults to get_driver().
    """
    # ヘッドレスモードでブラウザを起動
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--incognito")  # シークレットモードを有効にする
    # 読み込みを自分で待つので軽めの 'eager' に
    options.page_load_strategy = "eager"

    # ブラウザーを起動
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(90)   # ←★ここを追加（好きな秒に）
    

    return driver


@wait_interval(7)
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


def scrape_race_id_from_date(date, driver=None):
    """日付からrace_idをスクレイピングする

    Args:
        date (datetime): スクレイピングしたい日付
        driver (driver, optional): ブラウザのドライバ. Defaults to None.

    Returns:
        race_id_list (list): そのページにあるrace_idの一覧が含まれたリスト
    """
    if driver is None:
        driver = get_driver()

    # race_idを格納するリスト
    race_id_list = list()
    # # 罪にならないようにsleepをつける
    # time.sleep(random.uniform(3, 10))
    # 開催日ページurl取得
    date = f"{date.year:04}{date.month:02}{date.day:02}"
    url = "https://race.netkeiba.com/top/race_list.html?kaisai_date=" + date
    try:
        page_source = scrape_source_from_page(url, driver)
        soup = bs4.BeautifulSoup(page_source, features=PARSER)
        # ↓レースidがある部分を探す
        race_list = soup.find("div", attrs={"class": "RaceList_Box clearfix"})
        a_tag_list = race_list.find_all("a")
        href_list = [a_tag.get("href") for a_tag in a_tag_list]
        for href in href_list:
            for race_id in re.findall("[0-9]{12}", href):  # idを取得
                race_id = int(race_id)
                id_format_check(race_id, "race_id")
                race_id_list.append(race_id)
        # 順序を維持したまま重複を削除
        race_id_list = list(dict.fromkeys(race_id_list))

        return race_id_list

    except IdFormatError:
        print(f"Exception\n{traceback.format_exc()}")
        print(f"エラーrace_id：{race_id}")
        print(f"race_id取得時にエラーが発生した日付：{date}")

    except Exception:
        pass

        return []


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

    # # 罪にならないようにsleepをつける
    # time.sleep(random.uniform(3, 10))
    url = "https://race.netkeiba.com/race/result.html?race_id=" + str(race_id)
    try:
        page_source = scrape_source_from_page(url, driver)
        soup = bs4.BeautifulSoup(page_source, features=PARSER)
    except Exception:
        print(f"Exception\n{traceback.format_exc()}")
        print("urlをdriverに入力してソースを得るところでエラー")
        print(f"soup取得時にエラーが発生したrace_id: {race_id}")

    return soup


def scrape_soup_from_future_race_id(race_id, driver=None):
    """未来のrace_idから対象ページのsoupを取得する関数

    Args:
        race_id (int): soupを取得したいrace_id
        driver (driver, optional): ブラウザのドライバ. Defaults to None.

    Returns:
        soup (soup) : 取得したページのsoup
    """
    if driver is None:
        driver = get_driver()

    # # 罪にならないようにsleepをつける
    # time.sleep(random.uniform(3, 10))
    url = "https://race.netkeiba.com/race/shutuba.html?race_id=" + str(race_id)
    try:
        page_source = scrape_source_from_page(url, driver)
        soup = bs4.BeautifulSoup(page_source, features=PARSER)
    except Exception:
        print(f"Exception\n{traceback.format_exc()}")
        print("urlをdriverに入力してソースを得るところでエラー")
        print(f"soup取得時にエラーが発生したrace_id: {race_id}")

    return soup


def scrape_idlist_from_race_result(soup):
    horse_id_list = []
    jockey_id_list = []
    trainer_id_list = []

    try:
        # "全着順"テーブル内の馬ごとの行を取得
        rows = soup.select("table[summary='全着順'] tbody tr.HorseList")
        for row in rows:
            # 馬ID取得
            # Horse_Infoカラム内の aタグ（馬名リンク）からhorse idを抽出
            horse_a = row.select_one("td.Horse_Info a")
            if horse_a:
                # https://db.netkeiba.com/horse/2021102320 のようなリンクからIDを抽出
                horse_id_match = re.search(r"/horse/(\d+)", horse_a.get("href", ""))
                if horse_id_match:
                    h_id = int(horse_id_match.group(1))
                    id_format_check(h_id, "horse_id")
                    horse_id_list.append(h_id)

            # 騎手ID取得
            # td.Jockey 内の aタグからjockey idを抽出
            jockey_a = row.select_one("td.Jockey a")
            if jockey_a:
                # https://db.netkeiba.com/jockey/result/recent/05203/ のようなリンクからID抽出
                jockey_id_match = re.search(r"/jockey/result/recent/(\d+)/", jockey_a.get("href", ""))
                if jockey_id_match:
                    j_id = int(jockey_id_match.group(1))
                    id_format_check(j_id, "jockey_id")
                    jockey_id_list.append(j_id)

            # 調教師ID取得
            # td.Trainer 内の aタグからtrainer idを抽出
            trainer_a = row.select_one("td.Trainer a")
            if trainer_a:
                # https://db.netkeiba.com/trainer/result/recent/01084/ のようなリンクからID抽出
                trainer_id_match = re.search(r"/trainer/result/recent/(\d+)/", trainer_a.get("href", ""))
                if trainer_id_match:
                    t_id = int(trainer_id_match.group(1))
                    id_format_check(t_id, "trainer_id")
                    trainer_id_list.append(t_id)

        # ID数が揃っているかチェック
        id_length_check(horse_id_list, jockey_id_list, trainer_id_list)

    except Exception:
        print(f"Exception\n{traceback.format_exc()}")
        print("馬や騎手、調教師のidを取得するところでエラー")

    return horse_id_list, jockey_id_list, trainer_id_list


def scrape_idlist_from_race_forecast(soup):
    horse_id_list = []
    jockey_id_list = []
    trainer_id_list = []

    try:
        rows = soup.select("tbody > tr.HorseList")
        for row in rows:
            # 馬ID
            horse_a = row.select_one("td.HorseInfo a")
            if horse_a:
                horse_id = re.search(r"/horse/(\d+)", horse_a.get("href"))
                if horse_id:
                    h_id = int(horse_id.group(1))
                    id_format_check(h_id, "horse_id")
                    horse_id_list.append(h_id)

            # 騎手ID
            jockey_a = row.select_one("td.Jockey a")
            if jockey_a:
                jockey_id = re.search(r"/jockey/result/recent/(\d+)/", jockey_a.get("href"))
                if jockey_id:
                    j_id = int(jockey_id.group(1))
                    id_format_check(j_id, "jockey_id")
                    jockey_id_list.append(j_id)

            # 調教師ID
            trainer_a = row.select_one("td.Trainer a")
            if trainer_a:
                trainer_id = re.search(r"/trainer/result/recent/(\d+)/", trainer_a.get("href"))
                if trainer_id:
                    t_id = int(trainer_id.group(1))
                    id_format_check(t_id, "trainer_id")
                    trainer_id_list.append(t_id)

        id_length_check(horse_id_list, jockey_id_list, trainer_id_list)
    except Exception:
        print(f"Exception\n{traceback.format_exc()}")
        print("馬や騎手のidを取得するところでエラー")

    return horse_id_list, jockey_id_list, trainer_id_list


def scrape_race_result(race_id, driver=None):
    if driver is None:
        driver = get_driver()

    """ race_idからレース結果を取得する関数

    Args:
        race_id (int): soupを取得したいrace_id
        driver (driver, optional): ブラウザのドライバ. Defaults to get_driver().

    Returns:
        race_result_df (dataframe) : レース結果を保存したDF
        odds_df (dataframe) : 配当情報を保存したDF
        pace_df (dataframe) : 基準位置通過タイムを保存したDF
        race_info_df (dataframe) : レース情報を保存したDF
    """
    url = "https://race.netkeiba.com/race/result.html?race_id=" + str(race_id)

    # レース結果の取得
    try:
        # # 罪にならないようにsleepをつける
        # time.sleep(random.uniform(3, 10))

        # read_htmlが使えないのでseleniumで取得する
        driver.get(url)
        page_source = driver.page_source

        # pandas.read_htmlを使用してテーブルを取得
        html_io = StringIO(page_source)
        race_result_df_list = pd.read_html(html_io, flavor='lxml')

        race_result_df = race_result_df_list[0]
        odds_df = pd.concat(
            [race_result_df_list[1], race_result_df_list[2]], ignore_index=True
        )
        try:
            pace_df = race_result_df_list[5]
        except Exception:
            pace_df = None  # pace_dfは競走中止がいる時Noneになるため

        # horce_idやjoceky_idの取得
        try:
            soup = scrape_soup_from_past_race_id(race_id, driver)
            horse_id_list, joceky_id_list, trainer_id_list = (
                scrape_idlist_from_race_result(soup)
            )

            # レース結果にhorse_idやjockey_idを追加
            race_result_df["horse_id"] = horse_id_list
            race_result_df["jockey_id"] = joceky_id_list
            race_result_df["trainer_id"] = trainer_id_list
        except Exception:
            print(f"Exception\n{traceback.format_exc()}")
            print("dfにidを入れるところでエラー")
            print(f"horse_idやjockey_idが取得できなかったrace_id: {race_id}")
            race_result_df["horse_id"] = np.nan
            race_result_df["jockey_id"] = np.nan
            race_result_df["trainer_id"] = np.nan

        try:
            race_info_df = scrape_race_info(soup, race_id)
        except Exception:
            print(f"Exception\n{traceback.format_exc()}")
            print("race_info_df取得時にエラー")
            print(f"レース情報が取得できなかったrace_id: {race_id}")

        # race_idの追加
        df_list = [race_result_df, odds_df, pace_df]
        for df in df_list:
            if df is not None:  # pace_dfは競走中止がいる時Noneになるため
                df["race_id"] = race_id
                df = df.set_index("race_id")
                df = df.reset_index()

        # race_result_dfの列名が過去と異なるため修正
        if '着 順' in race_result_df.columns:
            race_result_df = race_result_df.rename(columns = {'着 順':'着順'})
        if '馬 番' in race_result_df.columns:
            race_result_df = race_result_df.rename(columns = {'馬 番':'馬番'})
        if '人 気' in race_result_df.columns:
            race_result_df = race_result_df.rename(columns = {'人 気':'人気'})
        if  '単勝 オッズ' in race_result_df.columns:
            race_result_df = race_result_df.rename(columns = {'単勝 オッズ':'単勝オッズ'})
        if '馬体重 (増減)' in race_result_df.columns:
            race_result_df = race_result_df.rename(columns = {'馬体重 (増減)':'馬体重(増減)'})
        if 'コーナー 通過順' in race_result_df.columns:
            race_result_df = race_result_df.rename(columns = {'コーナー 通過順':'コーナー通過順'})

        return race_result_df, odds_df, pace_df, race_info_df
    except Exception:
        print(f"Exception\n{traceback.format_exc()}")
        print("レース結果スクレイピング時点かindex設定でエラー")
        print(f"結果が取得できなかったrace_id: {race_id}")

        return None, None, None, None


def scrape_race_forecast(race_id, driver=None):
    if driver is None:
        driver = get_driver()
    """ 未来のrace_idからレース結果を取得する関数

    Args:
        race_id (int): soupを取得したいrace_id
        driver (driver, optional): ブラウザのドライバ. Defaults to get_driver().

    Returns:
        race_result_df (dataframe) : レース結果を保存したDF
        race_info_df (dataframe) : レース情報を保存したDF
    """
    url = "https://race.netkeiba.com/race/shutuba.html?race_id=" + str(race_id)

    # レース結果の取得
    try:
        # read_htmlが使えないのでseleniumで取得する
        driver.get(url)
        page_source = driver.page_source

        # pandas.read_htmlを使用してテーブルを取得
        html_io = StringIO(page_source)
        race_result_df_list = pd.read_html(html_io, flavor='lxml')
        # レース予想ページは複数columnのdataframeになっているため取得後整形
        if "予想 オッズ" in race_result_df_list[0].columns:
            race_result_df = race_result_df_list[0][
                ["枠", "馬 番", "馬名", "性齢", "斤量", "騎手", "厩舎", "予想 オッズ"]
            ]
        # 前日発売があるレースは予想オッズで無くなる
        else:
            race_result_df = race_result_df_list[0][
                ["枠", "馬 番", "馬名", "性齢", "斤量", "騎手", "厩舎", "オッズ 更新"]
            ]
        race_result_df.columns = ["枠", "馬番", "馬名", "性齢", "斤量", "騎手", "厩舎", "単勝オッズ"]
        # race_idの追加
        race_result_df.loc[:, "race_id"] = race_id
        race_result_df = race_result_df.set_index("race_id")
        race_result_df = race_result_df.reset_index()

        # horce_idやjoceky_idの取得
        try:
            soup = scrape_soup_from_future_race_id(race_id, driver)
            horse_id_list, joceky_id_list, trainer_id_list = (
                scrape_idlist_from_race_forecast(soup)
            )

            # レース結果にhorse_idやjockey_idを追加
            race_result_df["horse_id"] = horse_id_list
            race_result_df["jockey_id"] = joceky_id_list
            race_result_df["trainer_id"] = trainer_id_list
        except Exception:
            print("Exception\n" + traceback.format_exc())
            print("dfにidを入れるところでエラー")
            print("horse_idやjockey_idが取得できなかったrace_id:", race_id)
            race_result_df["horse_id"] = np.nan
            race_result_df["jockey_id"] = np.nan
            race_result_df["trainer_id"] = np.nan

        try:
            race_info_df = scrape_race_info(soup, race_id)
        except Exception:
            print("Exception\n" + traceback.format_exc())
            print("race_info_df取得時にエラー")
            print("レース情報が取得できなかったrace_id:", race_id)

        return race_result_df, race_info_df
    except Exception:
        print("Exception\n" + traceback.format_exc())
        print("レース結果スクレイピング時点でエラー")
        print("結果が取得できなかったrace_id:", race_id)

        return None, None


def convert_multiline_string_to_list(text):
    """複数行の文字列をリストに変換する関数

    Args:
        text (str): リストに変換したい文字列

    Returns:
        (list): リストに変換された文字列
    """
    text = text.replace("\n", "")
    return text.strip()


def scrape_race_info(soup, race_id):
    """レース情報を取得する関数

    Args:
        soup (BeautifulSoup): レース結果ページのsoupオブジェクト
        race_id (int): 取得したいレースのID

    Returns:
        race_info_df (DataFrame): レース情報を格納したデータフレーム
    """

    race_info = {
        "race_id": race_id,
        "ラウンド": None,
        "レース名": None,
        "日付": None,
        "発走時刻": None,
        "距離条件": None,
        "天気": None,
        "馬場": None,
        "競馬場": None,
        "回数": None,
        "日数": None,
        "条件": None,
        "グレード": None,
        "重賞": None,
        "分類": None,
        "特殊条件": None,
        "立て数": None,
        "賞金": None,
    }

    # レース情報のスクレイピング
    race_name_box = soup.find(class_="RaceList_NameBox")
    if race_name_box:
        race_name_element = race_name_box.find(class_="RaceName")
        if race_name_element:
            race_info["レース名"] = convert_multiline_string_to_list(
                race_name_element.text
            )

        # 重賞の場合、グレードを取得
        for i in range(1, 4):
            grade_element = race_name_box.find(
                class_=f"Icon_GradeType Icon_GradeType{i}"
            )
            if grade_element:
                race_info["重賞"] = f"G{i}"
                break

        round_element = race_name_box.find(class_="RaceNum")
        if round_element:
            race_info["ラウンド"] = convert_multiline_string_to_list(round_element.text)

        date_element = soup.find("dd", attrs={"class", "Active"}).find("a")["title"]
        if date_element:
            race_info["日付"] = convert_multiline_string_to_list(date_element)

        race_data_01_element = race_name_box.find(class_="RaceData01")
        if race_data_01_element:
            data_list = race_data_01_element.text.split("/")
            if len(data_list) >= 4:
                race_info.update(
                    {
                        "発走時刻": convert_multiline_string_to_list(data_list[0]),
                        "距離条件": convert_multiline_string_to_list(data_list[1]),
                        "天気": convert_multiline_string_to_list(data_list[2]),
                        "馬場": convert_multiline_string_to_list(data_list[3]),
                    }
                )
            elif len(data_list) == 2:
                race_info.update(
                    {
                        "発走時刻": convert_multiline_string_to_list(data_list[0]),
                        "距離条件": convert_multiline_string_to_list(data_list[1]),
                    }
                )

        race_data_02_element = race_name_box.find(class_="RaceData02")
        if race_data_02_element:
            elements = race_data_02_element.find_all("span")
            if len(elements) >= 9:
                race_info.update(
                    {
                        "競馬場": convert_multiline_string_to_list(elements[1].text),
                        "回数": convert_multiline_string_to_list(elements[0].text),
                        "日数": convert_multiline_string_to_list(elements[2].text),
                        "条件": convert_multiline_string_to_list(elements[3].text),
                        "グレード": convert_multiline_string_to_list(elements[4].text),
                        "分類": convert_multiline_string_to_list(elements[5].text),
                        "特殊条件": convert_multiline_string_to_list(elements[6].text),
                        "立て数": convert_multiline_string_to_list(elements[7].text),
                        "賞金": convert_multiline_string_to_list(elements[8].text),
                    }
                )

    race_info_df = pd.DataFrame(race_info.values(), index=race_info.keys()).T

    return race_info_df


def scrape_horse_past_performance(horse_id_list, driver=None):
    """馬の過去成績データを取得する関数

    Args:
        horse_id_list (list): 馬のIDが格納されたリスト

    Returns:
        horse_past_performance_df (DataFrame): 競走馬の過去レース結果が格納されたデータフレーム
    """
    driver = driver or get_driver()   # 既存を使う
    horse_past_performance = dict()

    for horse_id in tqdm(horse_id_list):
        try:
            # # スクレイピングによる問題を防ぐためにスリープを設定
            # time.sleep(random.uniform(3, 10))
            url = f"https://db.netkeiba.com/horse/{horse_id}"

            # read_htmlが使えないのでseleniumで取得する
            driver.get(url)
            page_source = driver.page_source

            # pandas.read_htmlを使用してテーブルを取得
            html_io = StringIO(page_source)
            race_result_df_list = pd.read_html(html_io, flavor='lxml')

            df = race_result_df_list[3]
            if df.columns[0] == "受賞歴":
                df = race_result_df_list[4]

            df = df[["日付", "開催", "R", "レース名", "ペース", "賞金"]]

            df.index = [horse_id] * len(df)  # 馬のIDをインデックスに設定
            horse_past_performance[horse_id] = df

        except IndexError:
            print(f"データベースを取得できないhorse_id: {horse_id}")
            print(f"Exception\n{traceback.format_exc()}")
            continue
        except Exception:
            print(f"データベースを取得できないhorse_id: {horse_id}")
            print(f"Exception\n{traceback.format_exc()}")
            continue

    horse_past_performance_df = pd.concat(
        [
            horse_past_performance[key][
                ["日付", "開催", "R", "レース名", "ペース", "賞金"]
            ]
            for key in horse_past_performance
        ]
    )
    horse_past_performance_df.reset_index(inplace=True)
    horse_past_performance_df.rename(columns={"index": "horse_id"}, inplace=True)

    driver.quit()

    return horse_past_performance_df


def scrape_pedigree(horse_id_list):
    """血統データを取得する関数

    Args:
        horse_id_list (list): horse_idを格納したリスト

    Returns:
        pedigree_df (dataframe): 血統が格納されたdataframe
    """
    pedigree_dict = dict()
    for horse_id in tqdm(horse_id_list):
        try:
            # # 罪にならないようにsleepをつける
            # time.sleep(random.uniform(3, 10))
            url = f"https://db.netkeiba.com/horse/ped/{horse_id}"

            # read_htmlが使えないのでseleniumで取得する
            driver = get_driver()
            driver.get(url)
            page_source = driver.page_source

            # pandas.read_htmlを使用してテーブルを取得
            html_io = StringIO(page_source)
            race_result_df_list = pd.read_html(html_io, flavor='lxml')

            df = race_result_df_list[0]

            # 重複を削除して1列のSeries型データに直す
            generations = dict()
            for i in reversed(range(5)):
                generations[i] = df[i]
                df = df.drop([i], axis=1)
                df = df.drop_duplicates()
            ped = pd.concat([generations[i] for i in range(5)]).rename(horse_id)

            pedigree_dict[horse_id] = ped.reset_index(drop=True)
        except IndexError:
            print("Exception\n" + traceback.format_exc())
            print("血統データを取得できないhorse_id:", horse_id)
            continue
        except Exception:
            print("Exception\n" + traceback.format_exc())
            print("血統データを取得できないhorse_id:", horse_id)
            continue

    # 列名をpedigree_0, ..., pedigree_61にする
    pedigree_df = pd.concat(
        [pedigree_dict[key] for key in pedigree_dict], axis=1
    ).T.add_prefix("pedigree_")
    pedigree_df.reset_index(inplace=True)
    pedigree_df.rename(columns={"index": "horse_id"}, inplace=True)

    return pedigree_df


def scrape_race_id_list(year, driver=None):
    """年を入力することでその年のレースid一覧を取得する関数

    Args:
        year (int): スクレイピングしたい年の数値
        driver (driver, optional): ブラウザのドライバ. Defaults to get_driver().

    Returns:
        past_race_id_list (list): 現在の日付より過去のrace_id_list
        future_race_id_list (list): 現在の日付より未来のrace_id_list
    """

    if driver is None:
        driver = get_driver()

    # 1月1日から順にスクレイピングをしていく
    date = datetime.date(year, 1, 1)
    past_race_id_list = list()
    future_race_id_list = list()
    for i in tqdm(range(365)):
        # 過去未来判定
        if date <= datetime.date.today():
            past_race_id_list += scrape_race_id_from_date(date, driver)
        elif date <= datetime.date.today() + relativedelta(days=14):
            future_race_id_list += scrape_race_id_from_date(date, driver)
        date = date + relativedelta(days=1)

    return past_race_id_list, future_race_id_list


def scrape_multiple_race_result(race_id_list, driver=None):
    """過去のrace_idのリストからレース結果を取得する関数

    Args:
        race_id_list (list): race_idが格納されたリスト
        driver (driver, optional): ブラウザのドライバ. Defaults to get_driver().

    Returns:
        race_result_df (dataframe) : レース結果を保存したDFを複数結合したDF
        odds_df (dataframe) : 配当情報を保存したDFを複数結合したDF
        pace_df (dataframe) : 基準位置通過タイムを保存したDFを複数結合したDF
        race_info_df (dataframe) : レース情報を保存したDFを複数結合したDF
    """

    if driver is None:
        driver = get_driver()

    race_result_df_list = list()
    odds_df_list = list()
    pace_df_list = list()
    race_info_df_list = list()
    # race_idごとにスクレイピングしてリストに格納
    for i, race_id in enumerate(tqdm(race_id_list)):
        race_result_df, odds_df, pace_df, race_info_df = scrape_race_result(
            race_id, driver
        )
        race_result_df_list.append(race_result_df)
        odds_df_list.append(odds_df)
        pace_df_list.append(pace_df)
        race_info_df_list.append(race_info_df)

        # driverのインスタンスが切れないように、一定回数ごとに再起動
        if i % 100 == 0:
            driver = get_driver()

    # 結合
    try:
        race_result_df = pd.concat(race_result_df_list)
        odds_df = pd.concat(odds_df_list)
        pace_df = pd.concat(pace_df_list)
        race_info_df = pd.concat(race_info_df_list)
    except Exception:
        print("Exception\n" + traceback.format_exc())
        print(
            "このエラー分のみが出ている場合、スクレピングしたDataFrameの結合でエラーの可能性あり"
        )

    return race_result_df, odds_df, pace_df, race_info_df


def scrape_multiple_race_forecast(race_id_list, driver=None):
    """未来のrace_idのリストからレース結果を取得する関数

    Args:
        race_id_list (list): race_idが格納されたリスト
        driver (driver, optional): ブラウザのドライバ. Defaults to get_driver().

    Returns:
        race_result_df (dataframe) : レース結果を保存したDFを複数結合したDF
        race_info_df (dataframe) : レース情報を保存したDFを複数結合したDF
    """
    if driver is None:
        driver = get_driver()

    race_result_df_list = list()
    race_info_df_list = list()
    # race_idごとにスクレイピングしてリストに格納
    for race_id in tqdm(race_id_list):
        race_result_df, race_info_df = scrape_race_forecast(race_id, driver)
        race_result_df_list.append(race_result_df)
        race_info_df_list.append(race_info_df)

    # 結合
    try:
        race_result_df = pd.concat(race_result_df_list)
        race_info_df = pd.concat(race_info_df_list)
    except Exception:
        print("Exception\n" + traceback.format_exc())
        print(
            "このエラー分のみが出ている場合、スクレピングしたDataFrameの結合でエラーの可能性あり"
        )

    return race_result_df, race_info_df


def rename_odds_columns(odds_df):
    """odds_dfの列名を書き換える関数

    Args:
        odds_df (dataframe): 払戻情報が保存されたDF

    Returns:
        odds_df (daragrame): 払戻情報が保存されたDF
    """
    odds_df = odds_df.rename(
        columns={"0": "券種", "1": "馬番", "2": "払戻金額", "3": "人気"}
    )
    odds_df = odds_df.rename(columns={0: "券種", 1: "馬番", 2: "払戻金額", 3: "人気"})

    return odds_df


def merge_and_save_df(new_df, existing_df_path, backup_df_path, encoding="utf_8_sig"):
    """既存のデータフレームと新しいデータフレームを結合し、保存する関数"""
    if os.path.exists(existing_df_path):
        existing_df = pd.read_csv(existing_df_path, encoding=encoding)
        merged_df = pd.concat([new_df, existing_df]).drop_duplicates()
    else:
        merged_df = new_df

    merged_df.to_csv(existing_df_path, encoding=encoding, index=False)
    merged_df.to_csv(backup_df_path, encoding=encoding, index=False)


def run_scrape():
    """スクレイピングを一通り実行し、race_idを保存し、レースデータをcsv形式で保存する関数

    Args:

    Returns:

    """
    ##### race_idをスクレイピング
    # 現在の年数を調べる
    today_year = int(datetime.date.today().year)

    # 読み込み済みのrace_idを調べる
    # CSV読み込み
    df = pd.read_csv(RACE_RESULT_DF_PATH)

    # uniqueなrace_idの取得
    unique_race_ids = df['race_id'].unique()

    # race_idを上4桁の年でグルーピング
    year_dict = {}
    for rid in unique_race_ids:
        year = str(rid)[:4]
        if year not in year_dict:
            year_dict[year] = []
        year_dict[year].append(rid)

    # 年号順にソート
    sorted_years = sorted(year_dict.keys())

    # リストのリスト作成
    past_race_id_list = [year_dict[y] for y in sorted_years]
    last_record_year = len(past_race_id_list) + 2007

    # race_idのスクレイピングを実行する
    future_race_id_list = []
    new_race_id_list = past_race_id_list.copy()

    for year in range(last_record_year, today_year + 1):
        scrape_past_race_id_list, scrape_future_race_id_list = scrape_race_id_list(year)
        print(
            f"{year}年のスクレイピング完了: 過去のID数={len(scrape_past_race_id_list)}, 未来のID数={len(scrape_future_race_id_list)}"
        )

        if year <= last_record_year:
            print(f"{year}年のレースIDは既に取得済みです。")
            new_race_id_list[year - 2008] = scrape_past_race_id_list
        else:
            print(f"{year}年のレースIDを追加しました。")
            new_race_id_list.append(scrape_past_race_id_list)

        future_race_id_list += scrape_future_race_id_list

    # scrapingするraceidの一覧をリスト化する
    new_race_list = []
    for i, year_race_ids in enumerate(new_race_id_list):
        if i < len(past_race_id_list) and isinstance(past_race_id_list[i], list):
            new_race_ids = list(set(year_race_ids) - set(past_race_id_list[i]))
            print(f"{i + 2008}年の新しいレコード数: {len(new_race_ids)}")
            new_race_list.extend(new_race_ids)
        else:
            print(f"{i + 2008}年の新しいレコード数: {len(year_race_ids)}")
            new_race_list.extend(year_race_ids)

    # レース結果のスクレイピングを実行する
    new_race_result_df, new_odds_df, new_pace_df, new_race_info_df = (
        scrape_multiple_race_result(new_race_list)
    )
    new_odds_df = rename_odds_columns(new_odds_df)

    # 競走馬idを取り出す
    horse_id_list = [
        int(horse_id)
        for horse_id in list(new_race_result_df["horse_id"].unique())
        if not math.isnan(horse_id)
    ]

    # 未来データのスクレイピングを実行する
    race_forecast_df, future_race_info_df = scrape_multiple_race_forecast(
        future_race_id_list, driver=get_driver()
    )

    # 未来データからのちの血統処理のため、未来データに存在する馬のidを取得する
    future_horse_id_list = [
        int(horse_id)
        for horse_id in list(race_forecast_df["horse_id"].unique())
        if not math.isnan(horse_id)
    ]
    # horse_id_listに未来データの馬のidを重複させずに追加する
    horse_id_list = list(set(horse_id_list + future_horse_id_list))

    # 過去のデータを読み込んで過去のレースにいない馬のidを取得する
    if os.path.exists(RACE_RESULT_DF_PATH):
        past_race_result_df = pd.read_csv(RACE_RESULT_DF_PATH, encoding="utf_8_sig")
        # 後の血統読み込み処理のため、過去のレース結果に存在しない馬のidを取得する
        new_horse_id_list = list(
            set(horse_id_list)
            - set(
                [
                    int(horse_id)
                    for horse_id in list(past_race_result_df["horse_id"].unique())
                    if not math.isnan(horse_id)
                ]
            )
        )
    else:
        # 後の血統読み込み処理のため、過去のレース結果に存在しない馬のidをner_horse_id_listに格納する
        new_horse_id_list = horse_id_list

    ###### 過去のレース結果との結合処理
    merge_and_save_df(
        new_race_result_df, RACE_RESULT_DF_PATH, BACKUP_RACE_RESULT_DF_PATH
    )
    merge_and_save_df(new_odds_df, ODDS_DF_PATH, BACKUP_ODDS_DF_PATH)
    merge_and_save_df(new_pace_df, PACE_DF_PATH, BACKUP_PACE_DF_PATH)
    merge_and_save_df(new_race_info_df, RACE_INFO_DF_PATH, BACKUP_RACE_INFO_DF_PATH)

    # 未来データはそのまま保存
    race_forecast_df.to_csv(RACE_FORECAST_DF_PATH, encoding="utf_8_sig", index=False)
    future_race_info_df.to_csv(
        FUTURE_RACE_INFO_DF_PATH, encoding="utf_8_sig", index=False
    )

    # horse_idから競走馬過去成績をスクレイピングする
    new_horse_past_performance_df = scrape_horse_past_performance(horse_id_list)
    # 過去のスクレイピングデータと結合する
    # 馬の過去成績データの結合して保存する
    merge_and_save_df(
        new_horse_past_performance_df,
        HORSE_PAST_PERFORMANCE_DF_PATH,
        BACKUP_HORSE_PAST_PERFORMANCE_DF_PATH,
    )

    # 過去にスクレイピング済みの馬を読み込む
    if os.path.exists(PEDIGREE_DF_PATH):
        try:
            pedigree_df = pd.read_csv(PEDIGREE_DF_PATH)
            # 'horse_id' カラムが存在することを確認してください
            if 'horse_id' in pedigree_df.columns:
                pedigree_horse_id_list = pedigree_df['horse_id'].unique().tolist()
            else:
                print(f"'horse_id' カラムが {PEDIGREE_DF_PATH} に存在しません。")
                pedigree_horse_id_list = []
        except Exception as e:
            print(f"{PEDIGREE_DF_PATH} の読み込み中にエラーが発生しました: {e}")
            pedigree_horse_id_list = []
    else:
        pedigree_horse_id_list = []


    # new_horse_id_listのうち、pedigree_horse_id_listに存在しないhorse_idをリストにまとめる
    horse_ids_to_scrape = list(set(new_horse_id_list) - set(pedigree_horse_id_list))

    # horse_idから競走馬血統をスクレイピングする
    try:
        new_horse_pedigree_df = scrape_pedigree(horse_ids_to_scrape)

        # ROOT_PATH + "/00_raw/pedigree/pedigree_df.csv" が存在する時、このデータフレームを読み込み、上記のpedigree_dfと結合して保存
        # 馬の血統データの結合と保存
        merge_and_save_df(new_horse_pedigree_df, PEDIGREE_DF_PATH, BACKUP_PEDIGREE_DF_PATH)
    except Exception:
        print("Exception\n" + traceback.format_exc())
        print("競走馬血統データのスクレイピングでエラー")