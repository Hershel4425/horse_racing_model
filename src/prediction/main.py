import schedule
import time
from scraping.odds_scraper import run_odds_scrape_with_retry
from prediction.predictor import run_prediction
from betting.bet_generator import generate_bets
from icloud_integration.icloud_uploader import upload_to_icloud


def process_race(race_id):
    """レースIDに対する一連の処理を行う関数"""
    try:
        # スクレイピングの実行
        run_odds_scrape_with_retry(race_id)

        # 予測の実行
        run_prediction(race_id)

        # 賭け金の生成
        generate_bets(race_id)

        # iCloudへのアップロード
        upload_to_icloud(race_id)

        print(f"race_id: {race_id} の処理が完了しました")
    except Exception as e:
        print(f"race_id: {race_id} の処理中にエラーが発生しました: {str(e)}")


def run_scheduled_tasks(start_race_id, schedule_time_start):
    # スケジュールの設定
    for i in range(1, 13):
        race_id = start_race_id + i
        minutes = (int(schedule_time_start.split(":")[1]) + (i - 1) * 30) % 60
        hours = int(schedule_time_start.split(":")[0]) + (int(schedule_time_start.split(":")[1]) + (i - 1) * 1) // 60
        schedule_time = f"{hours:02d}:{minutes:02d}"
        schedule.every().sunday.at(schedule_time).do(process_race, race_id)

    # メインループ
    while True:
        schedule.run_pending()
        time.sleep(1)

def run_immediate_tasks(start_race_id):
    # 即時実行
    for i in range(1, 13):
        race_id = start_race_id + i
        process_race(race_id)

# 実行方法を選択
print("実行方法を選択してください:")
print("1. スケジュールに従って実行")
print("2. 即時実行")
choice = input("選択肢の番号を入力してください: ")

start_race_id = int(input("開始レースIDを入力してください: "))

if choice == "1":
    schedule_time_start = input("開始時刻を入力してください (HH:MM): ")
    run_scheduled_tasks(start_race_id, schedule_time_start)
elif choice == "2":
    run_immediate_tasks(start_race_id)
else:
    print("無効な選択肢です。プログラムを終了します。")
