import datetime as dt
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import chromedriver_binary  # driverのpath指定を省略するために必要  # noqa: F401
from selenium.webdriver.common.by import By
from datetime import datetime, timedelta
import time
import re
import os
from typing import List, Dict, Tuple
import logging
import jpholiday # ← これを追加するのを忘れないでね！


# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定数定義
# 全体のルートパス
ROOT_PATH = "/Users/okamuratakeshi/Documents/100_プログラム_趣味/150_野望/153_競馬_v3/data"
DATE_STRING = dt.date.today().strftime("%Y%m%d")
# ファイルパス
RACE_RESULT_DF_PATH = ROOT_PATH + "/00_raw/11_race_result/race_result_df.csv"
RACE_INFO_DF_PATH = ROOT_PATH + "/00_raw/10_race_info/race_info_df.csv"
# 未来のデータ
RACE_FORECAST_DF_PATH = (
    ROOT_PATH
    + f"/00_raw/12_future_data/race_forecast/{DATE_STRING}_race_forecast_df.csv"
)
FUTURE_RACE_INFO_DF_PATH = (
    ROOT_PATH
    + f"/00_raw/12_future_data/future_race_info/{DATE_STRING}_future_race_info_df.csv"
)
HORSE_PAST_DF_PATH = ROOT_PATH + "/00_raw/20_horse_past_performance/horse_past_performance_df.csv"
HORSE_PEDIGREE_DF_PATH = ROOT_PATH + "00_raw/40_pedigree/pedigree_df.csv"

# URLテンプレート
RACE_LIST_URL = "https://race.netkeiba.com/top/race_list.html?kaisai_date={date}"
RACE_RESULT_URL = "https://race.netkeiba.com/race/result.html?race_id={race_id}"
RACE_SHUTUBA_URL = "https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
HORSE_DB_URL = "https://db.netkeiba.com/horse/{horse_id}"
HORSE_PEDIGREE_URL = "https://db.netkeiba.com/horse/ped/{horse_id}"

class NetKeibaScraperConfig:
    """スクレイピング設定クラス"""
    def __init__(self, 
                 min_interval: float = 1.0,
                 max_interval: float = 3.0,
                 retry_count: int = 3,
                 timeout: int = 30):
        self.min_interval = min_interval  # 最小待機時間（秒）
        self.max_interval = max_interval  # 最大待機時間（秒）
        self.retry_count = retry_count    # リトライ回数
        self.timeout = timeout            # タイムアウト時間（秒）

class IDValidator:
    """ID形式検証クラス"""
    
    @staticmethod
    def validate_race_id(race_id: str) -> bool:
        """レースIDの形式を検証
        
        Args:
            race_id: レースID（12桁: 年4桁+競馬場2桁+開催回2桁+開催日2桁+レース番号2桁）
            
        Returns:
            bool: 有効な形式の場合True
        """
        if not isinstance(race_id, str) or len(race_id) != 12:
            return False
        
        if not race_id.isdigit():
            return False
            
        # 年の妥当性チェック（1990年〜2050年）
        year = int(race_id[:4])
        if year < 1990 or year > 2050:
            return False
            
        # 競馬場コードの妥当性（01〜10が中央競馬場）
        jyo_code = int(race_id[4:6])
        if jyo_code < 1 or jyo_code > 10:
            return False
            
        # レース番号の妥当性（01〜12）
        race_num = int(race_id[10:12])
        if race_num < 1 or race_num > 12:
            return False
            
        return True
    
    @staticmethod
    def validate_horse_id(horse_id: str) -> bool:
        """馬IDの形式を検証
        
        Args:
            horse_id: 馬ID（10桁: 生年4桁+6桁）
            
        Returns:
            bool: 有効な形式の場合True
        """
        if not isinstance(horse_id, str) or len(horse_id) != 10:
            return False
            
        if not horse_id.isdigit():
            return False
            
        # 生年の妥当性チェック（1980年〜現在年）
        birth_year = int(horse_id[:4])
        current_year = datetime.now().year
        if birth_year < 1980 or birth_year > current_year:
            return False
            
        return True

class NetKeibaScraper:
    """NetKeibaスクレイピングメインクラス"""
    
    def __init__(self, config: NetKeibaScraperConfig = None):
        self.config = config or NetKeibaScraperConfig()
        self.driver = None
        self.validator = IDValidator()
        
    def get_driver(self) -> webdriver.Chrome:
        """Chromeドライバーを準備する
        
        Returns:
            webdriver.Chrome: Chromeドライバー
        """
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--incognito")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.page_load_strategy = "eager"
        
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(self.config.timeout)
        
        return driver
    
    def _wait_interval(self):
        """スクレイピング間隔を待機"""
        wait_time = np.random.uniform(self.config.min_interval, self.config.max_interval)
        time.sleep(wait_time)
    
    def _safe_request(self, url: str, retry_count: int = None) -> bool:
        """安全にページをリクエスト
        
        Args:
            url: リクエストURL
            retry_count: リトライ回数
            
        Returns:
            bool: 成功した場合True
        """
        retry_count = retry_count or self.config.retry_count
        
        for i in range(retry_count):
            try:
                self.driver.get(url)
                self._wait_interval()
                return True
            except Exception as e:
                logger.warning(f"Request failed (attempt {i+1}/{retry_count}): {e}")
                if i < retry_count - 1:
                    time.sleep(5)
                    
        return False
    
    def check_existing_data(self) -> int:
        """既存データから最新の取得済み年を確認
        
        Returns:
            int: 最新の取得済み年（データがない場合は今年）
        """
        if not os.path.exists(RACE_RESULT_DF_PATH):
            return datetime.now().year
            
        try:
            df = pd.read_csv(RACE_RESULT_DF_PATH)
            if 'race_id' not in df.columns or len(df) == 0:
                return datetime.now().year - 1
                
            # race_idから年を抽出
            years = df['race_id'].astype(str).str[:4].astype(int)
            return years.max()
            
        except Exception as e:
            logger.error(f"Error reading existing data: {e}")
            return datetime.now().year - 1

    def collect_race_ids(
        self,
        start_year: int,
        end_year: int | None = None
        ) -> Tuple[List[str], List[str]]:
        """指定期間のレースIDを収集し、過去と未来に分けて返す。

        Args:
            start_year: 開始年
            end_year: 終了年（Noneの場合は現在年）

        Returns:
            Tuple[past_ids, future_ids]
        """
        if end_year is None:
            end_year = datetime.now().year

        today = datetime.now().date()
        past_ids: set[str] = set()
        future_ids: set[str] = set()

        current_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)

        # ２週間先まで未来を覗く
        future_limit = datetime.now() + timedelta(days=14)
        if future_limit > end_date:
            end_date = future_limit

        # 日付ごとにレースIDを収集
        while current_date <= end_date:
            # current_date の曜日を取得 (0:月曜日, 5:土曜日, 6:日曜日)
            weekday = current_date.weekday()
            # current_date が土日または祝日かチェック
            is_race_day = (weekday >= 5) or jpholiday.is_holiday(current_date.date())

            # 土日祝じゃなかったら、次の日に進む
            if not is_race_day:
                logger.debug(f"Skipping {current_date.strftime('%Y%m%d')} (Weekday/Not Holiday)")
                current_date += timedelta(days=1)
                continue # ループの先頭に戻る
            date_str = current_date.strftime("%Y%m%d")
            url = RACE_LIST_URL.format(date=date_str)
            logger.info(f"Collecting race IDs for {date_str}")

            if self._safe_request(url):
                try:
                    # タブが存在する場合の処理
                    # 複数の日付がタブで表示される場合があるため、各タブをチェック
                    tab_elements = self.driver.find_elements(
                        By.CSS_SELECTOR, "#date_list_sub li"
                    )
                    tabs = tab_elements if tab_elements else [None]

                    for tab in tabs:
                        # タブがある場合は各タブをクリックして取得
                        if tab:
                            self.driver.execute_script(
                                "arguments[0].click();", tab.find_element(By.TAG_NAME, "a")
                            )
                            time.sleep(1) # タブ切り替えの待機

                            # レースIDを抽出
                            ids = self._extract_race_ids_from_list_page()

                        else:
                            # タブがない場合は直接抽出
                            ids = self._extract_race_ids_from_list_page()

                    # ここで過去／未来に振り分け
                    target_set = (
                        past_ids if current_date.date() <= today else future_ids
                    )
                    target_set.update(ids)

                except Exception as e:
                    logger.warning(f"Error collecting race IDs for {date_str}: {e}")

            current_date += timedelta(days=1)

        logger.info(
            f"Total collected race IDs → past: {len(past_ids)}, future: {len(future_ids)}"
        )
        return list(past_ids), list(future_ids)

    
    def _extract_race_ids_from_list_page(self) -> List[str]:
        """レース一覧ページからレースIDを抽出
        
        Returns:
            List[str]: レースIDのリスト
        """
        race_ids = []
        
        try:
            # JavaScriptの実行完了を待つ
            self.driver.execute_script("return document.readyState") == "complete"
            
            # 複数の方法でレースIDを探す
            # 方法1: RaceList_DataItemから取得
            race_items = self.driver.find_elements(By.CLASS_NAME, "RaceList_DataItem")
            
            for item in race_items:
                try:
                    # 各アイテム内のリンクを取得
                    links = item.find_elements(By.TAG_NAME, "a")
                    
                    for link in links:
                        href = link.get_attribute('href')
                        if href and 'race_id=' in href:
                            match = re.search(r'race_id=(\d{12})', href)
                            if match:
                                race_id = match.group(1)
                                if self.validator.validate_race_id(race_id):
                                    race_ids.append(race_id)
                                    
                except Exception as e:
                    logger.debug(f"Error processing race item: {e}")
                    continue
            
            # 方法2: 直接hrefを検索（バックアップ）
            if not race_ids:
                all_links = self.driver.find_elements(By.XPATH, "//a[contains(@href, 'race_id=')]")
                
                for link in all_links:
                    href = link.get_attribute('href')
                    if href:
                        match = re.search(r'race_id=(\d{12})', href)
                        if match:
                            race_id = match.group(1)
                            if self.validator.validate_race_id(race_id):
                                race_ids.append(race_id)
            
            # 重複を除去（順序を保持）
            race_ids = list(dict.fromkeys(race_ids))
            
            if race_ids:
                logger.info(f"Successfully extracted {len(race_ids)} race IDs")
            else:
                logger.warning("No race IDs found on the page")
                
        except Exception as e:
            logger.error(f"Critical error extracting race IDs: {e}")
            
        return race_ids
    
    def identify_new_race_ids(self, past_ids: List[str],future_ids: List[str]) -> Tuple[List[str], List[str]]:
        """新規レースIDを特定（過去・未来を分離）
        
        Args:
            past_ids: 過去レースIDリスト
            future_ids: 未来レースIDリスト
            
        Returns:
            Tuple[List[str], List[str]]: (過去の新規レースID, 未来の新規レースID)
        """
        # 既存のレースIDを読み込み
        existing_ids = set()
        if os.path.exists(RACE_RESULT_DF_PATH):
            df = pd.read_csv(RACE_RESULT_DF_PATH)
            existing_ids = set(df['race_id'].astype(str))
            
        # 新規IDを特定
        new_past = [rid for rid in past_ids if rid not in existing_ids]
        new_future = [rid for rid in future_ids if rid not in existing_ids]

        return new_past, new_future
    

    def scrape_race_results(self, race_ids: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """過去レース結果をスクレイピング
        
        Args:
            race_ids: レースIDリスト
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (レース結果データフレーム, レース情報データフレーム)
        """
        all_results = []
        all_race_info = []
        
        for race_id in race_ids:
            if not self.validator.validate_race_id(race_id):
                logger.warning(f"Invalid race ID: {race_id}")
                continue
            
            url = RACE_RESULT_URL.format(race_id=race_id)
            
            if self._safe_request(url):
                # レース結果を取得
                race_data = self._extract_race_result_data(race_id)
                all_results.extend(race_data)
                
                # レース情報を取得
                race_info = self._extract_race_info(race_id)
                race_info['race_id'] = race_id  # race_idを追加
                all_race_info.append(race_info)
        
        # 空のデータフレームを返す場合の処理
        if not all_results:
            logger.info("No race results found")
            return pd.DataFrame(), pd.DataFrame()
    
        return pd.DataFrame(all_results), pd.DataFrame(all_race_info)

    def scrape_future_races(self, race_ids: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """未来レース（出馬表）をスクレイピング
        
        Args:
            race_ids: レースIDリスト
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (出馬表データフレーム, レース情報データフレーム)
        """
        all_shutuba_data = []
        all_race_info = []
        
        for race_id in race_ids:
            if not self.validator.validate_race_id(race_id):
                logger.warning(f"Invalid race ID: {race_id}")
                continue
            
            url = RACE_SHUTUBA_URL.format(race_id=race_id)
            
            if self._safe_request(url):
                # 出馬表データを取得
                shutuba_data = self._extract_shutuba_data(race_id)
                all_shutuba_data.extend(shutuba_data)
                
                # レース情報を取得
                race_info = self._extract_race_info(race_id)
                race_info['race_id'] = race_id  # race_idを追加
                all_race_info.append(race_info)
        
        # 空のデータフレームを返す場合の処理
        if not all_shutuba_data:
            logger.info("No future races found")
            return pd.DataFrame(), pd.DataFrame()
        
        return pd.DataFrame(all_shutuba_data), pd.DataFrame(all_race_info)
    
    
    def _extract_race_result_data(self, race_id: str) -> List[Dict]:
        """レース結果ページからデータを抽出
        
        Args:
            race_id: レースID
            
        Returns:
            List[Dict]: レース結果データのリスト
        """
        results = []
        
        try:
            # テーブルデータを取得（pandas.read_html相当の処理）
            tables = self.driver.find_elements(By.TAG_NAME, "table")
            
            # 全着順テーブルを探す
            result_table = None
            for table in tables:
                summary = table.get_attribute("summary")
                if summary and "全着順" in summary:
                    result_table = table
                    break
            
            if not result_table:
                logger.error(f"Result table not found for race_id: {race_id}")
                return results
            
            # 各行からデータを抽出
            rows = result_table.find_elements(By.CSS_SELECTOR, "tbody tr.HorseList")
            
            for row in rows:
                try:
                    result_data = {
                        'race_id': race_id,
                        '着順': self._safe_get_text(row, "td.Result_Num"),
                        '枠': self._safe_get_text(row, "td.Num.Waku1, td.Num.Waku2, td.Num.Waku3, td.Num.Waku4, td.Num.Waku5, td.Num.Waku6, td.Num.Waku7, td.Num.Waku8"),
                        '馬番': self._safe_get_text(row, "td.Num.Txt_C"),
                        '馬名': self._safe_get_text(row, "td.Horse_Info span.Horse_Name"),
                        '性齢': self._safe_get_text(row, "td.Horse_Info.Txt_C span"),
                        '斤量': self._safe_get_text(row, "td.Jockey_Info span.JockeyWeight"),
                        '騎手': self._safe_get_text(row, "td.Jockey a"),
                        'タイム': self._safe_get_text(row, "td.Time span.RaceTime"),
                        '着差': self._safe_get_text_by_index(row, "td.Time span.RaceTime", 1),
                        '人気': self._safe_get_text(row, "td.Odds.Txt_C span"),
                        '単勝オッズ': self._safe_get_text(row, "td.Odds.Txt_R span"),
                        '後3F': self._safe_get_text(row, "td.Time.BgBlue02, td.Time.BgOrange, td.Time.BgYellow, td.Time:not(.BgBlue02):not(.BgOrange):not(.BgYellow)"),
                        'コーナー通過順': self._safe_get_text(row, "td.PassageRate"),
                        '厩舎': self._safe_get_text(row, "td.Trainer a"),
                        '馬体重(増減)': self._safe_get_text(row, "td.Weight")
                    }
                    
                    # IDを抽出
                    horse_link = row.find_element(By.CSS_SELECTOR, "td.Horse_Info a")
                    if horse_link:
                        href = horse_link.get_attribute("href")
                        horse_id_match = re.search(r"/horse/(\d+)", href)
                        if horse_id_match:
                            result_data['horse_id'] = horse_id_match.group(1)
                    
                    jockey_link = row.find_element(By.CSS_SELECTOR, "td.Jockey a")
                    if jockey_link:
                        href = jockey_link.get_attribute("href")
                        jockey_id_match = re.search(r"/jockey/result/recent/(\d+)/", href)
                        if jockey_id_match:
                            result_data['jockey_id'] = jockey_id_match.group(1)
                    
                    trainer_link = row.find_element(By.CSS_SELECTOR, "td.Trainer a")
                    if trainer_link:
                        href = trainer_link.get_attribute("href")
                        trainer_id_match = re.search(r"/trainer/result/recent/(\d+)/", href)
                        if trainer_id_match:
                            result_data['trainer_id'] = trainer_id_match.group(1)
                    
                    results.append(result_data)
                    
                except Exception as e:
                    logger.warning(f"Error extracting row data: {e}")
                    continue
            
            logger.info(f"Extracted {len(results)} race results for race_id: {race_id}")
            
        except Exception as e:
            logger.error(f"Error extracting race result for {race_id}: {e}")
        
        return results

    def _extract_shutuba_data(self, race_id: str) -> List[Dict]:
        """
        出馬表（shutuba.html）からデータを抽出して返す。
        _extract_race_result_data と同じく Selenium だけで処理し、
        ヘッダー名の揺れに強い実装にしたわ。
        """
        shutuba_data: List[Dict] = []

        try:
            # 各馬 1 行 = <tr class="HorseList">
            rows = self.driver.find_elements(By.CSS_SELECTOR, "tbody tr.HorseList")
            for row in rows:
                try:
                    # --- 基本情報 --------------------------------------------------
                    data = {
                        "race_id": race_id,
                        "枠":  self._safe_get_text(row, "td.Waku"),                 # 枠
                        "馬番": self._safe_get_text(row, "td.Umaban"),               # 馬番
                        "馬名": self._safe_get_text(row, "td.HorseInfo span.HorseName"),
                        "性齢": self._safe_get_text(row, "td.Barei"),               # 性齢(牡3など)
                        # 斤量はクラス名が無いので Barei の次セル（=6列目）で取得
                        "斤量": self._safe_get_text_by_index(row, "td", 5),
                        "騎手": self._safe_get_text(row, "td.Jockey a"),
                        "厩舎": self._safe_get_text(row, "td.Trainer a"),
                        "馬体重(増減)": self._safe_get_text(row, "td.Weight"),
                        # --- オッズ & 人気 ------------------------------------------
                        # オッズ本体は td.Popular（Popular_Ninki ではない）
                        "単勝オッズ": self._safe_get_text(row, "td.Popular span[id^='odds-'], td.Popular"),
                        "人気": self._safe_get_text(row, "td.Popular_Ninki span, td.Popular_Ninki"),
                    }

                    # --- 各 ID -----------------------------------------------------
                    # 馬 ID
                    horse_link = row.find_element(By.CSS_SELECTOR, "td.HorseInfo a")
                    if horse_link:
                        m = re.search(r"/horse/(\d+)", horse_link.get_attribute("href"))
                        if m:
                            data["horse_id"] = m.group(1)

                    # 騎手 ID
                    jockey_link = row.find_element(By.CSS_SELECTOR, "td.Jockey a")
                    if jockey_link:
                        m = re.search(r"/jockey/result/recent/(\d+)/", jockey_link.get_attribute("href"))
                        if m:
                            data["jockey_id"] = m.group(1)

                    # 調教師 ID
                    trainer_link = row.find_element(By.CSS_SELECTOR, "td.Trainer a")
                    if trainer_link:
                        m = re.search(r"/trainer/result/recent/(\d+)/", trainer_link.get_attribute("href"))
                        if m:
                            data["trainer_id"] = m.group(1)

                    shutuba_data.append(data)

                except Exception as e_row:
                    logger.warning(f"row parse error: {e_row}")
                    continue

            logger.info(f"Extracted {len(shutuba_data)} shutuba rows for race_id={race_id}")

        except Exception as e:
            logger.error(f"extract shutuba error ({race_id}): {e}")

        return shutuba_data
    

    def _extract_race_info(self, race_id: str) -> Dict:
        """レース情報を抽出
        
        Args:
            race_id: レースID
            
        Returns:
            Dict: レース情報
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
        
        try:
            # レース名ボックスを取得
            race_name_box = self.driver.find_element(By.CLASS_NAME, "RaceList_NameBox")
            
            # レース名
            try:
                race_name = race_name_box.find_element(By.CLASS_NAME, "RaceName")
                race_info["レース名"] = race_name.text.strip()
            except Exception:
                pass
            
            # グレード（重賞）
            for i in range(1, 4):
                try:
                    grade = race_name_box.find_element(By.CLASS_NAME, f"Icon_GradeType{i}")
                    if grade:
                        race_info["重賞"] = f"G{i}"
                        break
                except Exception:
                    continue
            
            # ラウンド
            try:
                round_elem = race_name_box.find_element(By.CLASS_NAME, "RaceNum")
                race_info["ラウンド"] = round_elem.text.strip()
            except Exception:
                pass
            
            # 日付
            try:
                date_elem = self.driver.find_element(By.CSS_SELECTOR, "dd.Active a")
                race_info["日付"] = date_elem.get_attribute("title")
            except Exception:
                pass
            
            # RaceData01の情報
            try:
                race_data_01 = race_name_box.find_element(By.CLASS_NAME, "RaceData01")
                data_list = race_data_01.text.split("/")
                
                if len(data_list) >= 4:
                    race_info["発走時刻"] = data_list[0].strip()
                    race_info["距離条件"] = data_list[1].strip()
                    race_info["天気"] = data_list[2].strip().replace("天候:", "")
                    race_info["馬場"] = data_list[3].strip()
                elif len(data_list) == 2:
                    race_info["発走時刻"] = data_list[0].strip()
                    race_info["距離条件"] = data_list[1].strip()
            except Exception:
                pass
            
            # RaceData02の情報
            try:
                race_data_02 = race_name_box.find_element(By.CLASS_NAME, "RaceData02")
                spans = race_data_02.find_elements(By.TAG_NAME, "span")
                
                if len(spans) >= 9:
                    race_info["回数"] = spans[0].text.strip()
                    race_info["競馬場"] = spans[1].text.strip()
                    race_info["日数"] = spans[2].text.strip()
                    race_info["条件"] = spans[3].text.strip()
                    race_info["グレード"] = spans[4].text.strip()
                    race_info["分類"] = spans[5].text.strip()
                    race_info["特殊条件"] = spans[6].text.strip()
                    race_info["立て数"] = spans[7].text.strip()
                    race_info["賞金"] = spans[8].text.strip()
            except Exception:
                pass
            
        except Exception as e:
            logger.error(f"Error extracting race info: {e}")
        
        return race_info


    def _safe_get_text(self, element, selector: str, default: str = "") -> str:
        """要素から安全にテキストを取得
        
        Args:
            element: 親要素
            selector: CSSセレクタ
            default: デフォルト値
            
        Returns:
            str: 取得したテキスト
        """
        try:
            target = element.find_element(By.CSS_SELECTOR, selector)
            return target.text.strip()
        except Exception:
            return default

    def _safe_get_text_by_index(self, element, selector: str, index: int, default: str = "") -> str:
        """要素から安全にテキストを取得（インデックス指定）
        
        Args:
            element: 親要素
            selector: CSSセレクタ
            index: インデックス
            default: デフォルト値
            
        Returns:
            str: 取得したテキスト
        """
        try:
            targets = element.find_elements(By.CSS_SELECTOR, selector)
            if len(targets) > index:
                return targets[index].text.strip()
            return default
        except Exception:
            return default
    
    
    def extract_horse_ids(self, race_results_df: pd.DataFrame, 
                        shutuba_df: pd.DataFrame) -> List[str]:
        """レースデータから馬IDを抽出
        
        Args:
            race_results_df: レース結果データフレーム
            shutuba_df: 出馬表データフレーム
            
        Returns:
            List[str]: 馬IDのリスト
        """
        horse_ids = set()
        
        # レース結果から馬IDを抽出
        if 'horse_id' in race_results_df.columns:
            horse_ids.update(race_results_df['horse_id'].astype(str).unique())
            
        # 出馬表から馬IDを抽出
        if 'horse_id' in shutuba_df.columns:
            horse_ids.update(shutuba_df['horse_id'].astype(str).unique())
            
        # 有効なIDのみを返す
        valid_ids = [id for id in horse_ids if self.validator.validate_horse_id(id)]
        
        return valid_ids
    
    def scrape_horse_data(self, horse_ids: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """馬の過去成績と血統情報をスクレイピング
        
        Args:
            horse_ids: 馬IDリスト
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (過去成績DF, 血統情報DF)
        """
        past_performance = []
        pedigree_data = []
        
        for horse_id in horse_ids:
            if not self.validator.validate_horse_id(horse_id):
                logger.warning(f"Invalid horse ID: {horse_id}")
                continue
                
            # 過去成績を取得
            past_url = HORSE_DB_URL.format(horse_id=horse_id)
            if self._safe_request(past_url):
                past_data = self._extract_horse_past_data(horse_id)
                if past_data:
                    past_performance.extend(past_data)
                    
            # 血統情報を取得
            pedigree_url = HORSE_PEDIGREE_URL.format(horse_id=horse_id)
            if self._safe_request(pedigree_url):
                ped_data = self._extract_horse_pedigree_data(horse_id)
                if ped_data:
                    pedigree_data.append(ped_data)
                    
        return pd.DataFrame(past_performance), pd.DataFrame(pedigree_data)
    
    def _extract_horse_past_data(self, horse_id: str) -> List[Dict]:
        """
        馬の過去成績をスクレイピング（HTMLベースに最適化）
        
        Args:
            horse_id: 馬ID
            
        Returns:
            List[Dict]: 過去成績データのリスト
        """
        results = []
        
        try:
            # 全着順テーブルを探す
            table = self.driver.find_element(By.CSS_SELECTOR, "table.db_h_race_results")
            
            if not table:
                logger.info(f"No race results found for horse {horse_id}")
                return results
            
            # 各行からデータを抽出
            rows = table.find_elements(By.CSS_SELECTOR, "tbody tr")
            
            for row in rows:
                try:
                    # 各セルのデータを取得
                    cells = row.find_elements(By.TAG_NAME, "td")
                    
                    if len(cells) < 20:  # 必要な列数がない場合はスキップ
                        continue
                    
                    # レースIDを取得
                    race_link = cells[4].find_element(By.TAG_NAME, "a")
                    race_href = race_link.get_attribute("href")
                    race_id_match = re.search(r'/race/(\d+)/', race_href)
                    race_id = race_id_match.group(1) if race_id_match else ""
                    
                    result_data = {
                        'horse_id': horse_id,
                        'race_id': race_id,
                        '日付': cells[0].text.strip(),
                        '開催': cells[1].text.strip(),
                        '天気': cells[2].text.strip(),
                        'R': cells[3].text.strip(),
                        'レース名': cells[4].text.strip(),
                        '頭数': cells[6].text.strip(),
                        '枠番': cells[7].text.strip(),
                        '馬番': cells[8].text.strip(),
                        'オッズ': cells[9].text.strip(),
                        '人気': cells[10].text.strip(),
                        '着順': cells[11].text.strip(),
                        '騎手': cells[12].text.strip(),
                        '斤量': cells[13].text.strip(),
                        '距離': cells[14].text.strip(),
                        '馬場': cells[15].text.strip(),
                        'タイム': cells[17].text.strip(),
                        '着差': cells[18].text.strip(),
                        '通過': cells[20].text.strip(),
                        'ペース': cells[21].text.strip(),
                        '上り': cells[22].text.strip(),
                        '馬体重': cells[23].text.strip(),
                        '賞金': cells[27].text.strip() if len(cells) > 27 else ""
                    }
                    
                    results.append(result_data)
                    
                except Exception as e:
                    logger.debug(f"Error extracting row data: {e}")
                    continue
            
            logger.info(f"Extracted {len(results)} race results for horse {horse_id}")
            
        except Exception as e:
            logger.error(f"Error extracting horse past data for {horse_id}: {e}")
        
        return results

    def _extract_horse_pedigree_data(self, horse_id: str) -> Dict:
        """
        馬の血統情報をスクレイピング（HTMLベースに最適化）
        
        Args:
            horse_id: 馬ID
            
        Returns:
            Dict: 血統情報
        """
        pedigree_data = {'horse_id': horse_id}
        
        try:
            # 血統表を取得
            blood_table = self.driver.find_element(By.CSS_SELECTOR, "table.blood_table")
            
            if not blood_table:
                logger.warning(f"No pedigree table found for horse {horse_id}")
                return pedigree_data
            
            # 全てのセルを取得
            cells = blood_table.find_elements(By.CSS_SELECTOR, "td")
            
            # セルから血統情報を抽出（インデックスベース）
            pedigree_names = []
            for cell in cells:
                # リンクがある場合は馬名を取得
                links = cell.find_elements(By.TAG_NAME, "a")
                if links:
                    # 最初のリンクのテキストを取得（馬名）
                    horse_name = links[0].text.strip()
                    # 英名や補足情報を除去
                    horse_name = horse_name.split('\n')[0].split('(')[0].strip()
                    if horse_name:
                        pedigree_names.append(horse_name)
            
            # 血統データを辞書に格納
            for i, name in enumerate(pedigree_names):
                pedigree_data[f'pedigree_{i}'] = name
            
            logger.info(f"Extracted {len(pedigree_names)} pedigree entries for horse {horse_id}")
            
        except Exception as e:
            logger.error(f"Error extracting pedigree data for {horse_id}: {e}")
        
        return pedigree_data
    
    def save_data(self, race_results_df: pd.DataFrame, 
                race_info_df: pd.DataFrame,
                future_races_df: pd.DataFrame,
                future_race_info_df: pd.DataFrame,
                horse_past_df: pd.DataFrame, 
                horse_pedigree_df: pd.DataFrame):
        """データを保存（既存データとマージ）
        
        Args:
            race_results_df: レース結果データフレーム
            race_info_df: レース情報データフレーム
            future_races_df: 未来レースデータフレーム
            future_race_info_df: 未来レース情報データフレーム
            horse_past_df: 馬の過去成績データフレーム
            horse_pedigree_df: 馬の血統情報データフレーム
        """
        # レース結果を保存
        if not race_results_df.empty:
            if os.path.exists(RACE_RESULT_DF_PATH):
                existing_df = pd.read_csv(RACE_RESULT_DF_PATH)
                race_results_df = pd.concat([existing_df, race_results_df], ignore_index=True)
                race_results_df.drop_duplicates(subset=['race_id', 'horse_id'], inplace=True)
            race_results_df.to_csv(RACE_RESULT_DF_PATH, index=False)
        
        # レース情報を保存
        if not race_info_df.empty:
            if os.path.exists(RACE_INFO_DF_PATH):
                existing_df = pd.read_csv(RACE_INFO_DF_PATH)
                race_info_df = pd.concat([existing_df, race_info_df], ignore_index=True)
                race_info_df.drop_duplicates(subset=['race_id'], inplace=True)
            race_info_df.to_csv(RACE_INFO_DF_PATH, index=False)
        
        # 未来レースを保存（上書き）
        if not future_races_df.empty:
            future_races_df.to_csv(RACE_FORECAST_DF_PATH, index=False)
        
        # 未来レース情報を保存（上書き）
        if not future_race_info_df.empty:
            future_race_info_df.to_csv(FUTURE_RACE_INFO_DF_PATH, index=False)
        
        # 馬の過去成績を保存
        if not horse_past_df.empty:
            if os.path.exists(HORSE_PAST_DF_PATH):
                existing_df = pd.read_csv(HORSE_PAST_DF_PATH)
                horse_past_df = pd.concat([existing_df, horse_past_df], ignore_index=True)
                horse_past_df.drop_duplicates(subset=['horse_id', 'race_id'], inplace=True)
            horse_past_df.to_csv(HORSE_PAST_DF_PATH, index=False)
        
        # 血統情報を保存
        if not horse_pedigree_df.empty:
            if os.path.exists(HORSE_PEDIGREE_DF_PATH):
                existing_df = pd.read_csv(HORSE_PEDIGREE_DF_PATH)
                horse_pedigree_df = pd.concat([existing_df, horse_pedigree_df], ignore_index=True)
                horse_pedigree_df.drop_duplicates(subset=['horse_id'], inplace=True)
            horse_pedigree_df.to_csv(HORSE_PEDIGREE_DF_PATH, index=False)

    def run(self):
        """メイン処理を実行"""
        try:
            self.driver = self.get_driver()
            
            # 1. 既存データの確認
            logger.info("Checking existing data...")
            last_year = self.check_existing_data()
            
            # 2. レースID収集
            logger.info(f"Collecting race IDs from {last_year} to present...")
            past_ids_all, future_ids_all = self.collect_race_ids(last_year)
            
            # 3. 新規ID特定
            logger.info("Identifying new race IDs...")
            past_ids, future_ids = self.identify_new_race_ids(past_ids_all, future_ids_all)
            logger.info(f"Found {len(past_ids)} past races and {len(future_ids)} future races")
            
            # 4. 過去レースデータ収集
            race_results_df = pd.DataFrame()
            race_info_df = pd.DataFrame()
            if past_ids:
                logger.info("Scraping past race results...")
                race_results_df, race_info_df = self.scrape_race_results(past_ids)
            else:
                logger.info("No new past races to scrape")
            
            # 5. 未来レース情報取得
            future_races_df = pd.DataFrame()
            future_race_info_df = pd.DataFrame()
            if future_ids:
                logger.info("Scraping future race data...")
                future_races_df, future_race_info_df = self.scrape_future_races(future_ids)
            else:
                logger.info("No future races to scrape")
            
            # 6. 馬ID抽出
            logger.info("Extracting horse IDs...")
            horse_ids = self.extract_horse_ids(race_results_df, future_races_df)
            logger.info(f"Found {len(horse_ids)} unique horses")
            
            # 7. 馬データ収集
            horse_past_df = pd.DataFrame()
            horse_pedigree_df = pd.DataFrame()
            if horse_ids:
                logger.info("Scraping horse data...")
                horse_past_df, horse_pedigree_df = self.scrape_horse_data(horse_ids)
            else:
                logger.info("No horse data to scrape")
            
            # 8. データ保存
            logger.info("Saving data...")
            self.save_data(race_results_df, race_info_df, 
                        future_races_df, future_race_info_df,
                        horse_past_df, horse_pedigree_df)
            
            logger.info("Scraping completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during scraping: {e}")
            raise
            
        finally:
            if self.driver:
                self.driver.quit()

def main():
    """メイン関数"""
    # カスタム設定を作成
    config = NetKeibaScraperConfig(
        min_interval=3.5,  # 最小3.5秒待機
        max_interval=10.0,  # 最大10秒待機
        retry_count=3,     # 3回リトライ
        timeout=30         # 30秒でタイムアウト
    )
    
    # スクレイパーを実行
    scraper = NetKeibaScraper(config)
    scraper.run()

if __name__ == "__main__":
    main()