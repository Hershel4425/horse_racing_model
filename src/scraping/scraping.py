import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime, timedelta
import time
import re
import os
from typing import List, Dict, Tuple, Optional
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定数定義
RACE_RESULT_DF_PATH = "race_results.csv"
FUTURE_RACE_DF_PATH = "future_races.csv"
HORSE_PAST_DF_PATH = "horse_past_performance.csv"
HORSE_PEDIGREE_DF_PATH = "horse_pedigree.csv"

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
            int: 最新の取得済み年（データがない場合は前年）
        """
        if not os.path.exists(RACE_RESULT_DF_PATH):
            return datetime.now().year - 1
            
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
    
    def collect_race_ids(self, start_year: int, end_year: int = None) -> List[str]:
        """指定期間のレースIDを収集
        
        Args:
            start_year: 開始年
            end_year: 終了年（Noneの場合は現在年）
            
        Returns:
            List[str]: レースIDのリスト
        """
        if end_year is None:
            end_year = datetime.now().year
            
        all_race_ids = []
        current_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        
        # 未来のレースも含める（2週間先まで）
        future_date = datetime.now() + timedelta(days=14)
        if future_date > end_date:
            end_date = future_date
            
        # 日付ごとにレースIDを収集
        while current_date <= end_date:
            date_str = current_date.strftime("%Y%m%d")
            url = RACE_LIST_URL.format(date=date_str)
            
            logger.info(f"Collecting race IDs for {date_str}")
            
            if self._safe_request(url):
                try:
                    # タブが存在する場合の処理
                    # 複数の日付がタブで表示される場合があるため、各タブをチェック
                    tab_elements = self.driver.find_elements(By.CSS_SELECTOR, "#date_list_sub li")
                    
                    if tab_elements:
                        # タブがある場合は各タブをクリックして取得
                        for tab in tab_elements:
                            try:
                                # タブをクリック
                                tab_link = tab.find_element(By.TAG_NAME, "a")
                                self.driver.execute_script("arguments[0].click();", tab_link)
                                time.sleep(1)  # タブ切り替えの待機
                                
                                # レースIDを抽出
                                ids = self._extract_race_ids_from_list_page()
                                all_race_ids.extend(ids)
                                
                            except Exception as e:
                                logger.warning(f"Error processing tab: {e}")
                                continue
                    else:
                        # タブがない場合は直接抽出
                        ids = self._extract_race_ids_from_list_page()
                        all_race_ids.extend(ids)
                        
                except Exception as e:
                    logger.error(f"Error collecting race IDs for {date_str}: {e}")
                    
            current_date += timedelta(days=1)
            
        # 重複を除去
        all_race_ids = list(dict.fromkeys(all_race_ids))
        
        logger.info(f"Total collected race IDs: {len(all_race_ids)}")
        
        return all_race_ids
    
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
    
    def identify_new_race_ids(self, all_race_ids: List[str]) -> Tuple[List[str], List[str]]:
        """新規レースIDを特定（過去・未来を分離）
        
        Args:
            all_race_ids: 全レースIDリスト
            
        Returns:
            Tuple[List[str], List[str]]: (過去の新規レースID, 未来の新規レースID)
        """
        # 既存のレースIDを読み込み
        existing_ids = set()
        if os.path.exists(RACE_RESULT_DF_PATH):
            df = pd.read_csv(RACE_RESULT_DF_PATH)
            existing_ids = set(df['race_id'].astype(str))
            
        # 新規IDを特定
        new_ids = [id for id in all_race_ids if id not in existing_ids]
        
        # 現在日時で過去・未来を分離
        today = datetime.now().strftime("%Y%m%d")
        past_ids = []
        future_ids = []
        
        for race_id in new_ids:
            race_date = race_id[:4] + race_id[6:10]  # 年月日を抽出
            if race_date <= today:
                past_ids.append(race_id)
            else:
                future_ids.append(race_id)
                
        return past_ids, future_ids
    

    def scrape_race_results(self, race_ids: List[str]) -> pd.DataFrame:
        """過去レース結果をスクレイピング
        
        Args:
            race_ids: レースIDリスト
            
        Returns:
            pd.DataFrame: レース結果データフレーム
        """
        all_results = []
        
        for race_id in race_ids:
            if not self.validator.validate_race_id(race_id):
                logger.warning(f"Invalid race ID: {race_id}")
                continue
            
            url = RACE_RESULT_URL.format(race_id=race_id)
            
            if self._safe_request(url):
                # レース結果を取得
                race_data = self._extract_race_result_data(race_id)
                
                # レース情報を取得
                race_info = self._extract_race_info(race_id)
                
                # レース情報を各結果に追加
                for data in race_data:
                    data.update(race_info)
                
                all_results.extend(race_data)
        
        return pd.DataFrame(all_results)

    def scrape_future_races(self, race_ids: List[str]) -> pd.DataFrame:
        """未来レース（出馬表）をスクレイピング
        
        Args:
            race_ids: レースIDリスト
            
        Returns:
            pd.DataFrame: 出馬表データフレーム
        """
        all_shutuba_data = []
        
        for race_id in race_ids:
            if not self.validator.validate_race_id(race_id):
                logger.warning(f"Invalid race ID: {race_id}")
                continue
            
            url = RACE_SHUTUBA_URL.format(race_id=race_id)
            
            if self._safe_request(url):
                # 出馬表データを取得
                shutuba_data = self._extract_shutuba_data(race_id)
                
                # レース情報を取得
                race_info = self._extract_race_info(race_id)
                
                # レース情報を各データに追加
                for data in shutuba_data:
                    data.update(race_info)
                
                all_shutuba_data.extend(shutuba_data)
        
        return pd.DataFrame(all_shutuba_data)
    
    
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
        """出馬表ページからデータを抽出
        
        Args:
            race_id: レースID
            
        Returns:
            List[Dict]: 出馬表データのリスト
        """
        shutuba_data = []
        
        try:
            # pandas.read_html相当の処理
            # HTMLをStringIOに変換してpandasで読み込む
            page_source = self.driver.page_source
            from io import StringIO
            html_io = StringIO(page_source)
            
            try:
                dfs = pd.read_html(html_io, flavor='lxml')
                
                if not dfs:
                    logger.error(f"No tables found for race_id: {race_id}")
                    return shutuba_data
                
                # 出馬表のデータフレームを取得
                df = dfs[0]
                
                # カラム名の候補を定義（オッズ列の変動に対応）
                possible_odds_columns = ['予想 オッズ', 'オッズ 更新', '予想オッズ', 'オッズ更新', 'オッズ']
                odds_column = None
                
                # 実際に存在するオッズ列を探す
                for col in possible_odds_columns:
                    if col in df.columns:
                        odds_column = col
                        break
                
                # 必要なカラムを選択（オッズ列がない場合も対応）
                required_columns = ["枠", "馬 番", "馬名", "性齢", "斤量", "騎手", "厩舎"]
                if odds_column:
                    required_columns.append(odds_column)
                
                # カラムが存在することを確認
                available_columns = [col for col in required_columns if col in df.columns]
                
                if len(available_columns) < 7:  # 最低限必要なカラム数
                    logger.warning(f"Not enough columns found. Available: {df.columns.tolist()}")
                    # 別の方法で抽出を試みる
                    return self._extract_shutuba_data_by_selenium(race_id)
                
                # データフレームから必要な列を抽出
                race_df = df[available_columns].copy()
                
                # カラム名を統一
                rename_dict = {
                    "馬 番": "馬番",
                    "予想 オッズ": "単勝オッズ",
                    "オッズ 更新": "単勝オッズ",
                    "予想オッズ": "単勝オッズ",
                    "オッズ更新": "単勝オッズ",
                    "オッズ": "単勝オッズ"
                }
                
                race_df.rename(columns=rename_dict, inplace=True)
                
                # race_idを追加
                race_df['race_id'] = race_id
                
                # Seleniumを使ってIDを取得
                rows = self.driver.find_elements(By.CSS_SELECTOR, "tbody tr.HorseList")
                
                horse_ids = []
                jockey_ids = []
                trainer_ids = []
                
                for row in rows:
                    # 馬ID
                    try:
                        horse_link = row.find_element(By.CSS_SELECTOR, "td.HorseInfo a")
                        href = horse_link.get_attribute("href")
                        horse_id_match = re.search(r"/horse/(\d+)", href)
                        if horse_id_match:
                            horse_ids.append(horse_id_match.group(1))
                        else:
                            horse_ids.append(None)
                    except:
                        horse_ids.append(None)
                    
                    # 騎手ID
                    try:
                        jockey_link = row.find_element(By.CSS_SELECTOR, "td.Jockey a")
                        href = jockey_link.get_attribute("href")
                        jockey_id_match = re.search(r"/jockey/result/recent/(\d+)/", href)
                        if jockey_id_match:
                            jockey_ids.append(jockey_id_match.group(1))
                        else:
                            jockey_ids.append(None)
                    except:
                        jockey_ids.append(None)
                    
                    # 調教師ID
                    try:
                        trainer_link = row.find_element(By.CSS_SELECTOR, "td.Trainer a")
                        href = trainer_link.get_attribute("href")
                        trainer_id_match = re.search(r"/trainer/result/recent/(\d+)/", href)
                        if trainer_id_match:
                            trainer_ids.append(trainer_id_match.group(1))
                        else:
                            trainer_ids.append(None)
                    except:
                        trainer_ids.append(None)
                
                # IDをデータフレームに追加
                if len(horse_ids) == len(race_df):
                    race_df['horse_id'] = horse_ids
                    race_df['jockey_id'] = jockey_ids
                    race_df['trainer_id'] = trainer_ids
                
                # DataFrameをDictのリストに変換
                shutuba_data = race_df.to_dict('records')
                
            except Exception as e:
                logger.error(f"Error with pandas.read_html: {e}")
                # フォールバック: Seleniumで直接抽出
                return self._extract_shutuba_data_by_selenium(race_id)
            
        except Exception as e:
            logger.error(f"Error extracting shutuba data for {race_id}: {e}")
        
        return shutuba_data

    def _extract_shutuba_data_by_selenium(self, race_id: str) -> List[Dict]:
        """Seleniumを使用して出馬表データを直接抽出（フォールバック）
        
        Args:
            race_id: レースID
            
        Returns:
            List[Dict]: 出馬表データのリスト
        """
        shutuba_data = []
        
        try:
            rows = self.driver.find_elements(By.CSS_SELECTOR, "tbody tr.HorseList")
            
            for row in rows:
                try:
                    data = {
                        'race_id': race_id,
                        '枠': self._safe_get_text(row, "td:nth-child(1)"),
                        '馬番': self._safe_get_text(row, "td:nth-child(2)"),
                        '馬名': self._safe_get_text(row, "td.HorseInfo"),
                        '性齢': self._safe_get_text(row, "td:nth-child(4)"),
                        '斤量': self._safe_get_text(row, "td:nth-child(5)"),
                        '騎手': self._safe_get_text(row, "td.Jockey"),
                        '厩舎': self._safe_get_text(row, "td.Trainer"),
                        '単勝オッズ': self._safe_get_text(row, "td:nth-child(8)")  # オッズ列の位置は可変
                    }
                    
                    # IDを抽出（上記と同じロジック）
                    # ... (省略)
                    
                    shutuba_data.append(data)
                    
                except Exception as e:
                    logger.warning(f"Error extracting row data: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in selenium extraction: {e}")
        
        return shutuba_data

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
        except:
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
        except:
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
        """馬の過去成績を抽出
        
        Args:
            horse_id: 馬ID
            
        Returns:
            List[Dict]: 過去成績データのリスト
        """
        # TODO: 実際のページ構造に応じて実装
        return []
    
    def _extract_horse_pedigree_data(self, horse_id: str) -> Dict:
        """馬の血統情報を抽出
        
        Args:
            horse_id: 馬ID
            
        Returns:
            Dict: 血統情報データ
        """
        # TODO: 実際のページ構造に応じて実装
        return {}
    
    def save_data(self, race_results_df: pd.DataFrame, 
                  future_races_df: pd.DataFrame,
                  horse_past_df: pd.DataFrame, 
                  horse_pedigree_df: pd.DataFrame):
        """データを保存（既存データとマージ）
        
        Args:
            race_results_df: レース結果データフレーム
            future_races_df: 未来レースデータフレーム
            horse_past_df: 馬の過去成績データフレーム
            horse_pedigree_df: 馬の血統情報データフレーム
        """
        # レース結果を保存
        if os.path.exists(RACE_RESULT_DF_PATH):
            existing_df = pd.read_csv(RACE_RESULT_DF_PATH)
            race_results_df = pd.concat([existing_df, race_results_df], ignore_index=True)
            race_results_df.drop_duplicates(subset=['race_id', 'horse_id'], inplace=True)
        race_results_df.to_csv(RACE_RESULT_DF_PATH, index=False)
        
        # 未来レースを保存（上書き）
        future_races_df.to_csv(FUTURE_RACE_DF_PATH, index=False)
        
        # 馬の過去成績を保存
        if os.path.exists(HORSE_PAST_DF_PATH):
            existing_df = pd.read_csv(HORSE_PAST_DF_PATH)
            horse_past_df = pd.concat([existing_df, horse_past_df], ignore_index=True)
            horse_past_df.drop_duplicates(subset=['horse_id', 'race_id'], inplace=True)
        horse_past_df.to_csv(HORSE_PAST_DF_PATH, index=False)
        
        # 血統情報を保存
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
            all_race_ids = self.collect_race_ids(last_year)
            
            # 3. 新規ID特定
            logger.info("Identifying new race IDs...")
            past_ids, future_ids = self.identify_new_race_ids(all_race_ids)
            logger.info(f"Found {len(past_ids)} past races and {len(future_ids)} future races")
            
            # 4. 過去レースデータ収集
            logger.info("Scraping past race results...")
            race_results_df = self.scrape_race_results(past_ids)
            
            # 5. 未来レース情報取得
            logger.info("Scraping future race data...")
            future_races_df = self.scrape_future_races(future_ids)
            
            # 6. 馬ID抽出
            logger.info("Extracting horse IDs...")
            horse_ids = self.extract_horse_ids(race_results_df, future_races_df)
            logger.info(f"Found {len(horse_ids)} unique horses")
            
            # 7. 馬データ収集
            logger.info("Scraping horse data...")
            horse_past_df, horse_pedigree_df = self.scrape_horse_data(horse_ids)
            
            # 8. データ保存
            logger.info("Saving data...")
            self.save_data(race_results_df, future_races_df, 
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
        min_interval=1.5,  # 最小1.5秒待機
        max_interval=3.0,  # 最大3秒待機
        retry_count=3,     # 3回リトライ
        timeout=30         # 30秒でタイムアウト
    )
    
    # スクレイパーを実行
    scraper = NetKeibaScraper(config)
    scraper.run()

if __name__ == "__main__":
    main()