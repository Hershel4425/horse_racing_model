# -*- coding: utf-8 -*-
"""scraper_pipeline.py  – 2025-05 リファクタ版

低レイヤ scraping.py にだけ依存する“高レイヤ・ラッパ”。
• collect_race_ids() で 7 秒スリープしつつ差分 ID 取得  
• scrape_horse_performance() が dict / DataFrame 両対応  
• safe_concat をローカル util に一本化  
既存 CSV の書式・処理結果は一切変わらないわよ。
"""

from __future__ import annotations

import os
import datetime as _dt
from typing import Dict, List, Tuple

import pandas as pd

from scraping import (  # noqa: E402
    wait_interval,
    merge_and_save_df,
    scrape_race_id_list,
    scrape_multiple_race_result,
    scrape_multiple_race_forecast,
    scrape_horse_past_performance,
    scrape_pedigree,
    RACE_RESULT_DF_PATH,
    BACKUP_RACE_RESULT_DF_PATH,
    ODDS_DF_PATH,
    BACKUP_ODDS_DF_PATH,
    PACE_DF_PATH,
    BACKUP_PACE_DF_PATH,
    RACE_INFO_DF_PATH,
    BACKUP_RACE_INFO_DF_PATH,
    RACE_FORECAST_DF_PATH,
    FUTURE_RACE_INFO_DF_PATH,
    HORSE_PAST_PERFORMANCE_DF_PATH,
    BACKUP_HORSE_PAST_PERFORMANCE_DF_PATH,
    PEDIGREE_DF_PATH,
    BACKUP_PEDIGREE_DF_PATH,
)

__all__ = [
    "collect_race_ids",
    "scrape_past_races",
    "scrape_future_races",
    "scrape_horse_performance",
    "scrape_pedigree_data",
    "persist_data",
]

# ----------------------------------------------------------------------
# ユーティリティ
# ----------------------------------------------------------------------


def _safe_concat(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    """None を除外して concat する簡易版."""
    non_null = [df for df in df_list if df is not None]
    return pd.concat(non_null) if non_null else pd.DataFrame()


# ----------------------------------------------------------------------
# 0. race_id 収集
# ----------------------------------------------------------------------


def collect_race_ids(
    year_from: int | None = None,
    year_to: int | None = None,
    diff_only: bool = True,
) -> Tuple[List[int], List[int]]:
    """スクレイピング対象の race_id を「過去」「未来」に分けて返す。

    diff_only=True なら既に RACE_RESULT_DF_PATH に載っている id は除外して差分だけ返す。
    """
    done_ids: set[int] = set()
    if diff_only and os.path.exists(RACE_RESULT_DF_PATH):
        done_ids = set(pd.read_csv(RACE_RESULT_DF_PATH, usecols=["race_id"]).race_id.tolist())

    today_year = _dt.date.today().year
    year_from = year_from or (min(int(str(rid)[:4]) for rid in done_ids) if done_ids else 2008)
    year_to = year_to or today_year

    past_ids: list[int] = []
    future_ids: list[int] = []

    for y in range(year_from, year_to + 1):
        wait_interval(7)         # polite crawl
        p, f = scrape_race_id_list(y)
        if diff_only:
            p = [rid for rid in p if rid not in done_ids]
            f = [rid for rid in f if rid not in done_ids]
        past_ids.extend(p)
        future_ids.extend(f)

    return sorted(set(past_ids)), sorted(set(future_ids))


# ----------------------------------------------------------------------
# 1. 過去レース結果
# ----------------------------------------------------------------------


def scrape_past_races(race_ids: List[int]) -> Dict[str, pd.DataFrame]:
    """過去レース一括取得ラッパ."""
    if not race_ids:
        empty = pd.DataFrame()
        return dict(race_result=empty, odds=empty, pace=empty, race_info=empty)

    result_df, odds_df, pace_df, info_df = scrape_multiple_race_result(race_ids)
    return dict(race_result=result_df, odds=odds_df, pace=pace_df, race_info=info_df)


# ----------------------------------------------------------------------
# 2. 未来レース出走表
# ----------------------------------------------------------------------


def scrape_future_races(race_ids: List[int]) -> Dict[str, pd.DataFrame]:
    """未来レース出走表ラッパ."""
    if not race_ids:
        empty = pd.DataFrame()
        return dict(race_forecast=empty, future_race_info=empty)

    shutuba_df, info_df = scrape_multiple_race_forecast(race_ids)
    return dict(race_forecast=shutuba_df, future_race_info=info_df)


# ----------------------------------------------------------------------
# 3. 競走馬過去成績
# ----------------------------------------------------------------------


def _extract_df(obj) -> pd.DataFrame:
    """dict / DataFrame / None から horse_id 列を持つ DataFrame を返す."""
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict):
        for key in ("race_result", "race_forecast", "race_info"):
            if key in obj and isinstance(obj[key], pd.DataFrame):
                return obj[key]
    return pd.DataFrame()


def scrape_horse_performance(past=None, future=None) -> pd.DataFrame:
    """race_result / race_forecast から horse_id を集めて過去成績を取得."""
    past_df = _extract_df(past)
    future_df = _extract_df(future)

    horse_ids: list[int] = []
    for df in (past_df, future_df):
        if not df.empty and "horse_id" in df.columns:
            horse_ids.extend(df["horse_id"].dropna().astype(int).unique())

    horse_ids = sorted(set(horse_ids))
    if not horse_ids:
        return pd.DataFrame()

    return scrape_horse_past_performance(horse_ids)


# ----------------------------------------------------------------------
# 4. 血統
# ----------------------------------------------------------------------


def scrape_pedigree_data(performance_df: pd.DataFrame) -> pd.DataFrame:
    """horse_id から血統データを取得."""
    if performance_df is None or performance_df.empty or "horse_id" not in performance_df.columns:
        return pd.DataFrame()

    horse_ids = sorted(performance_df["horse_id"].astype(int).unique())
    return scrape_pedigree(horse_ids)


# ----------------------------------------------------------------------
# 5. CSV 保存
# ----------------------------------------------------------------------


def persist_data(
    *,
    past: Dict[str, pd.DataFrame] | None = None,
    future: Dict[str, pd.DataFrame] | None = None,
    perf: pd.DataFrame | None = None,
    ped: pd.DataFrame | None = None,
    encoding: str = "utf-8",
):
    """取得データを CSV に追記保存（バックアップ付き）."""
    # --- 過去 -------------------------------------------------------------
    if past:
        merge_and_save_df(past.get("race_result", pd.DataFrame()), RACE_RESULT_DF_PATH, BACKUP_RACE_RESULT_DF_PATH, encoding)
        merge_and_save_df(past.get("odds", pd.DataFrame()), ODDS_DF_PATH, BACKUP_ODDS_DF_PATH, encoding)
        merge_and_save_df(past.get("pace", pd.DataFrame()), PACE_DF_PATH, BACKUP_PACE_DF_PATH, encoding)
        merge_and_save_df(past.get("race_info", pd.DataFrame()), RACE_INFO_DF_PATH, BACKUP_RACE_INFO_DF_PATH, encoding)

    # --- 未来 -------------------------------------------------------------
    if future:
        future.get("race_forecast", pd.DataFrame()).to_csv(RACE_FORECAST_DF_PATH, encoding=encoding, index=False)
        future.get("future_race_info", pd.DataFrame()).to_csv(FUTURE_RACE_INFO_DF_PATH, encoding=encoding, index=False)

    # --- 競走馬過去成績 -----------------------------------------------------
    if perf is not None and not perf.empty:
        merge_and_save_df(perf, HORSE_PAST_PERFORMANCE_DF_PATH, BACKUP_HORSE_PAST_PERFORMANCE_DF_PATH, encoding)

    # --- 血統 --------------------------------------------------------------
    if ped is not None and not ped.empty:
        merge_and_save_df(ped, PEDIGREE_DF_PATH, BACKUP_PEDIGREE_DF_PATH, encoding)
