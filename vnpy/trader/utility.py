### 第四版增强型1分钟Bar
"""
General utility functions.
"""

import json
import logging
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Callable, Dict, Tuple, Union, Optional
from decimal import Decimal
from math import floor, ceil

import numpy as np
import talib

from .object import BarData, TickData
from .constant import Exchange, Interval
from .locale import _

if sys.version_info >= (3, 9):
    from zoneinfo import ZoneInfo, available_timezones              # noqa
else:
    from backports.zoneinfo import ZoneInfo, available_timezones    # noqa


log_formatter: logging.Formatter = logging.Formatter("[%(asctime)s] %(message)s")


def extract_vt_symbol(vt_symbol: str) -> Tuple[str, Exchange]:
    """
    :return: (symbol, exchange)
    """
    symbol, exchange_str = vt_symbol.rsplit(".", 1)
    return symbol, Exchange(exchange_str)


def generate_vt_symbol(symbol: str, exchange: Exchange) -> str:
    """
    return vt_symbol
    """
    return f"{symbol}.{exchange.value}"


def _get_trader_dir(temp_name: str) -> Tuple[Path, Path]:
    """
    Get path where trader is running in.
    """
    cwd: Path = Path.cwd()
    temp_path: Path = cwd.joinpath(temp_name)

    # If .vntrader folder exists in current working directory,
    # then use it as trader running path.
    if temp_path.exists():
        return cwd, temp_path

    # Otherwise use home path of system.
    home_path: Path = Path.home()
    temp_path: Path = home_path.joinpath(temp_name)

    # Create .vntrader folder under home path if not exist.
    if not temp_path.exists():
        temp_path.mkdir()

    return home_path, temp_path


TRADER_DIR, TEMP_DIR = _get_trader_dir(".vntrader")
sys.path.append(str(TRADER_DIR))


def get_file_path(filename: str) -> Path:
    """
    Get path for temp file with filename.
    """
    return TEMP_DIR.joinpath(filename)


def get_folder_path(folder_name: str) -> Path:
    """
    Get path for temp folder with folder name.
    """
    folder_path: Path = TEMP_DIR.joinpath(folder_name)
    if not folder_path.exists():
        folder_path.mkdir()
    return folder_path


def get_icon_path(filepath: str, ico_name: str) -> str:
    """
    Get path for icon file with ico name.
    """
    ui_path: Path = Path(filepath).parent
    icon_path: Path = ui_path.joinpath("ico", ico_name)
    return str(icon_path)


def load_json(filename: str) -> dict:
    """
    Load data from json file in temp path.
    """
    filepath: Path = get_file_path(filename)

    if filepath.exists():
        with open(filepath, mode="r", encoding="UTF-8") as f:
            data: dict = json.load(f)
        return data
    else:
        save_json(filename, {})
        return {}


def save_json(filename: str, data: dict) -> None:
    """
    Save data into json file in temp path.
    """
    filepath: Path = get_file_path(filename)
    with open(filepath, mode="w+", encoding="UTF-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )


def round_to(value: float, target: float) -> float:
    """
    Round price to price tick value.
    """
    value: Decimal = Decimal(str(value))
    target: Decimal = Decimal(str(target))
    rounded: float = float(int(round(value / target)) * target)
    return rounded


def floor_to(value: float, target: float) -> float:
    """
    Similar to math.floor function, but to target float number.
    """
    value: Decimal = Decimal(str(value))
    target: Decimal = Decimal(str(target))
    result: float = float(int(floor(value / target)) * target)
    return result


def ceil_to(value: float, target: float) -> float:
    """
    Similar to math.ceil function, but to target float number.
    """
    value: Decimal = Decimal(str(value))
    target: Decimal = Decimal(str(target))
    result: float = float(int(ceil(value / target)) * target)
    return result


def get_digits(value: float) -> int:
    """
    Get number of digits after decimal point.
    """
    value_str: str = str(value)

    if "e-" in value_str:
        _, buf = value_str.split("e-")
        return int(buf)
    elif "." in value_str:
        _, buf = value_str.split(".")
        return len(buf)
    else:
        return 0


class BarGenerator:
    """
    For:
    1. generating 1 minute bar data from tick data
    2. generating x minute bar/x hour bar data from 1 minute data
    Notice:
    1. for x minute bar, x must be able to divide 60: 2, 3, 5, 6, 10, 15, 20, 30
    2. for x hour bar, x can be any number
    """

    def __init__(
        self,
        on_bar: Callable,
        window: int = 0,
        on_window_bar: Callable = None,
        interval: Interval = Interval.MINUTE,
        daily_end: time = None
    ) -> None:
        """Constructor"""
        self.bar: BarData = None
        self.on_bar: Callable = on_bar

        self.interval: Interval = interval
        self.interval_count: int = 0

        self.hour_bar: BarData = None
        self.daily_bar: BarData = None

        self.window: int = window
        self.window_bar: BarData = None
        self.on_window_bar: Callable = on_window_bar

        self.last_tick: TickData = None

        self.daily_end: time = daily_end
        if self.interval == Interval.DAILY and not self.daily_end:
            raise RuntimeError(_("合成日K线必须传入每日收盘时间"))

    def update_tick(self, tick: TickData) -> None:
        """
        Update new tick data into generator.
        """
        # try:
        #     print(f"[DEBUG] update_tick: tick.datetime={tick.datetime}, last_price={tick.last_price}")
        # except Exception as e:
        #     print(f"[DEBUG] Exception in printing tick: {e}")

        new_minute: bool = False

        # Filter tick data with 0 last price
        if not tick.last_price:
            return

        if not self.bar:
            new_minute = True
        elif (
            (self.bar.datetime.minute != tick.datetime.minute)
            or (self.bar.datetime.hour != tick.datetime.hour)
        ):
            # print(f"[DEBUG] Minute boundary reached. Current bar datetime: {self.bar.datetime}")
            # 新增2025/03/01：在一分钟结束前，先完成自定义指标计算
            self._finalize_custom_metrics(self.bar)
            # 新增2025/03/01
            # print(f"[DEBUG] Finalized bar metrics: {self.bar.__dict__}")

            self.bar.datetime = self.bar.datetime.replace(
                second=0, microsecond=0
            )
            self.on_bar(self.bar)

            new_minute = True

        if new_minute:
            self.bar = BarData(
                symbol=tick.symbol,
                exchange=tick.exchange,
                interval=Interval.MINUTE,
                datetime=tick.datetime,
                gateway_name=tick.gateway_name,
                open_price=tick.last_price,
                high_price=tick.last_price,
                low_price=tick.last_price,
                close_price=tick.last_price,
                open_interest=tick.open_interest
            )
            # 新增2025/03/01：初始化自定义指标累加器extra_metrics

            ### 新增2026/03/30：TWAP初始权重不再为1，而是使用第一个tick的时间减去分钟初始值
            # TWAP 按 previous-tick / piecewise-constant 方法计算：
            # 1. bar 起点到首笔 tick：若存在上一笔 tick，则用上一笔价格补齐；
            # 2. 相邻两笔 tick 之间：使用前一笔 tick 的价格；
            # 3. 最后一笔 tick 到 bar 终点：在 _finalize_custom_metrics 中补齐。
            minute_start: datetime = tick.datetime.replace(second=0, microsecond=0)
            opening_delta: float = 0
            opening_price: float = 0
            if self.last_tick and self.last_tick.datetime <= tick.datetime:
                opening_delta = (tick.datetime - minute_start).total_seconds()
                if opening_delta > 0:
                    opening_price = self.last_tick.last_price
                else:
                    opening_delta = 0
            ### 新增2026/03/30

            # self.bar.extra_metrics = {
            #     'bid_price_1_sum': tick.bid_price_1,   # 累计买一价
            #     'ask_price_1_sum': tick.ask_price_1,   # 累计卖一价
            #     'bid_volume_1_sum': tick.bid_volume_1, # 累计买一量
            #     'ask_volume_1_sum': tick.ask_volume_1, # 累计卖一量
            #     # 'spread_ratio_1_sum': 0 if (tick.ask_price_1==0) and (tick.bid_price_1==0) else (tick.ask_price_1 - tick.bid_price_1) / ((tick.ask_price_1 + tick.bid_price_1) / 2),
            #     'bid_price_2_sum': tick.bid_price_2,   # 累计买二价
            #     'ask_price_2_sum': tick.ask_price_2,   # 累计卖二价
            #     'bid_volume_2_sum': tick.bid_volume_2, # 累计买二量
            #     'ask_volume_2_sum': tick.ask_volume_2, # 累计卖二量
            #     # 'spread_ratio_2_sum': 0 if (tick.ask_price_2==0) and (tick.bid_price_2==0) else (tick.ask_price_2 - tick.bid_price_2) / ((tick.ask_price_2 + tick.bid_price_2) / 2) if tick.ask_price_2 != 0 else 0,
            #     'bid_price_3_sum': tick.bid_price_3,   # 累计买三价
            #     'ask_price_3_sum': tick.ask_price_3,   # 累计卖三价
            #     'bid_volume_3_sum': tick.bid_volume_3, # 累计买三量
            #     'ask_volume_3_sum': tick.ask_volume_3, # 累计卖三量
            #     # 'spread_ratio_3_sum': 0 if (tick.ask_price_3==0) and (tick.bid_price_3==0) else (tick.ask_price_3 - tick.bid_price_3) / ((tick.ask_price_3 + tick.bid_price_3) / 2) if tick.ask_price_3 != 0 else 0,
            #     'bid_price_4_sum': tick.bid_price_4,   # 累计买四价
            #     'ask_price_4_sum': tick.ask_price_4,   # 累计卖四价
            #     'bid_volume_4_sum': tick.bid_volume_4, # 累计买四量
            #     'ask_volume_4_sum': tick.ask_volume_4, # 累计卖四量
            #     # 'spread_ratio_4_sum': 0 if (tick.ask_price_4==0) and (tick.bid_price_4==0) else (tick.ask_price_4 - tick.bid_price_4) / ((tick.ask_price_4 + tick.bid_price_4) / 2) if tick.ask_price_4 != 0 else 0,
            #     'bid_price_5_sum': tick.bid_price_5,   # 累计买五价
            #     'ask_price_5_sum': tick.ask_price_5,   # 累计卖五价
            #     'bid_volume_5_sum': tick.bid_volume_5, # 累计买五量
            #     'ask_volume_5_sum': tick.ask_volume_5, # 累计卖五量
            #     # 'spread_ratio_5_sum': 0 if (tick.ask_price_5==0) and (tick.bid_price_5==0) else (tick.ask_price_5 - tick.bid_price_5) / ((tick.ask_price_5 + tick.bid_price_5) / 2) if tick.ask_price_5 != 0 else 0,
            #     'tick_count': 1,                     # tick 数量初始化为1
            #     # 'price_time_sum': tick.last_price * 1,   # 第一笔 tick 权重为1
            #     # 'time_weight_sum': 1                     # 累计权重初始化为1
            #     ### 新增2026/03/30: 修改初始tick的权重和价格
            #     'price_time_sum': opening_price * opening_delta,
            #     'time_weight_sum': opening_delta
            #     ### 新增2026/03/30
            # }
            # 新增2025/03/01
            # print(f"[DEBUG] New bar created with extra_metrics: {self.bar.extra_metrics}")

            # 新增2026/04/28：增加bid_price_i_count和ask_price_i_count
            metrics = {
                'tick_count': 1,                     # tick 数量初始化为1
                # 'price_time_sum': tick.last_price * 1,   # 第一笔 tick 权重为1
                # 'time_weight_sum': 1                     # 累计权重初始化为1
                ### 新增2026/03/30: 修改初始tick的权重和价格
                'price_time_sum': opening_price * opening_delta,
                'time_weight_sum': opening_delta
                ### 新增2026/03/30
            }
            for i in range(1, 6):
                metrics[f'bid_price_{i}_sum'] = 0.0
                metrics[f'bid_price_{i}_count'] = 0
                metrics[f'ask_price_{i}_sum'] = 0.0
                metrics[f'ask_price_{i}_count'] = 0
                metrics[f'bid_volume_{i}_sum'] = 0.0
                metrics[f'ask_volume_{i}_sum'] = 0.0

                bid_price = getattr(tick, f'bid_price_{i}')
                ask_price = getattr(tick, f'ask_price_{i}')
                bid_volume = getattr(tick, f'bid_volume_{i}')
                ask_volume = getattr(tick, f'ask_volume_{i}')

                # 线上口径：盘口价格非0即有效；线下 NaN 已在主循环转 0。
                if bid_price > 0:
                    metrics[f'bid_price_{i}_sum'] += bid_price
                    metrics[f'bid_price_{i}_count'] += 1
                    metrics[f'bid_volume_{i}_sum'] += bid_volume

                if ask_price > 0:
                    metrics[f'ask_price_{i}_sum'] += ask_price
                    metrics[f'ask_price_{i}_count'] += 1
                    metrics[f'ask_volume_{i}_sum'] += ask_volume

            self.bar.extra_metrics = metrics
            # 新增2026/04/28

        else:
            self.bar.high_price = max(self.bar.high_price, tick.last_price)
            if tick.high_price > self.last_tick.high_price:
                self.bar.high_price = max(self.bar.high_price, tick.high_price)

            self.bar.low_price = min(self.bar.low_price, tick.last_price)
            if tick.low_price < self.last_tick.low_price:
                self.bar.low_price = min(self.bar.low_price, tick.low_price)

            self.bar.close_price = tick.last_price
            self.bar.open_interest = tick.open_interest
            self.bar.datetime = tick.datetime

            # # 新增2025/03/01：更新自定义指标累加器
            # metrics = self.bar.extra_metrics
            # metrics['bid_price_1_sum'] += tick.bid_price_1
            # metrics['ask_price_1_sum'] += tick.ask_price_1
            # metrics['bid_volume_1_sum'] += tick.bid_volume_1
            # metrics['ask_volume_1_sum'] += tick.ask_volume_1
            # # if (tick.bid_price_1 == 0) and (tick.ask_price_1 == 0):
            # #     metrics['spread_ratio_1_sum'] += 0
            # # else:
            # #     metrics['spread_ratio_1_sum'] += (tick.ask_price_1 - tick.bid_price_1) / ((tick.ask_price_1 + tick.bid_price_1) / 2)
            # metrics['bid_price_2_sum'] += tick.bid_price_2
            # metrics['ask_price_2_sum'] += tick.ask_price_2
            # metrics['bid_volume_2_sum'] += tick.bid_volume_2
            # metrics['ask_volume_2_sum'] += tick.ask_volume_2
            # # if (tick.bid_price_2 == 0) and (tick.ask_price_2 == 0):
            # #     metrics['spread_ratio_2_sum'] += 0
            # # else:
            # #     metrics['spread_ratio_2_sum'] += (tick.ask_price_2 - tick.bid_price_2) / ((tick.ask_price_2 + tick.bid_price_2) / 2)
            # metrics['bid_price_3_sum'] += tick.bid_price_3
            # metrics['ask_price_3_sum'] += tick.ask_price_3
            # metrics['bid_volume_3_sum'] += tick.bid_volume_3
            # metrics['ask_volume_3_sum'] += tick.ask_volume_3
            # # if (tick.bid_price_3 == 0) and (tick.ask_price_3 == 0):
            # #     metrics['spread_ratio_3_sum'] += 0
            # # else:
            # #     metrics['spread_ratio_3_sum'] += (tick.ask_price_3 - tick.bid_price_3) / ((tick.ask_price_3 + tick.bid_price_3) / 2)
            # metrics['bid_price_4_sum'] += tick.bid_price_4
            # metrics['ask_price_4_sum'] += tick.ask_price_4
            # metrics['bid_volume_4_sum'] += tick.bid_volume_4
            # metrics['ask_volume_4_sum'] += tick.ask_volume_4
            # # if (tick.bid_price_4 == 0) and (tick.ask_price_4 == 0):
            # #     metrics['spread_ratio_4_sum'] += 0
            # # else:
            # #     metrics['spread_ratio_4_sum'] += (tick.ask_price_4 - tick.bid_price_4) / ((tick.ask_price_4 + tick.bid_price_4) / 2)
            # metrics['bid_price_5_sum'] += tick.bid_price_5
            # metrics['ask_price_5_sum'] += tick.ask_price_5
            # metrics['bid_volume_5_sum'] += tick.bid_volume_5
            # metrics['ask_volume_5_sum'] += tick.ask_volume_5
            # # if (tick.bid_price_5 == 0) and (tick.ask_price_5 == 0):
            # #     metrics['spread_ratio_5_sum'] += 0
            # # else:
            # #     metrics['spread_ratio_5_sum'] += (tick.ask_price_5 - tick.bid_price_5) / ((tick.ask_price_5 + tick.bid_price_5) / 2)    
            # # # 计算当前 tick 与上一 tick 的时间间隔（秒）
            # # if self.last_tick is not None:
            # #     delta = (tick.datetime - self.last_tick.datetime).total_seconds()
            # #     # 如果时间间隔小于0.5秒，设为0.5
            # #     if delta < 0.5:
            # #         delta = 0.5
            # # else:
            # #     delta = 1  # 如果没有上一个 tick，则设为1秒

            # # # 权重 = delta / 0.5
            # # weight = delta / 0.5
            # # metrics['price_time_sum'] += tick.last_price * weight
            # # metrics['time_weight_sum'] += weight
            # # metrics['tick_count'] += 1
            # # #新增2025/03/01
            # # # print(f"[DEBUG] Updated metrics: {metrics}")

            # 新增2026/04/28：metrics更新
            for i in range(1, 6):
                bid_price = getattr(tick, f'bid_price_{i}')
                ask_price = getattr(tick, f'ask_price_{i}')
                bid_volume = getattr(tick, f'bid_volume_{i}')
                ask_volume = getattr(tick, f'ask_volume_{i}')

                # 只在对应盘口价格有效时累计价格、count、挂量。
                # 不保存 volume_count，后续平均挂量直接使用对应价格 count 作分母。
                if bid_price > 0:
                    self.bar.extra_metrics[f'bid_price_{i}_sum'] += bid_price
                    self.bar.extra_metrics[f'bid_price_{i}_count'] += 1
                    self.bar.extra_metrics[f'bid_volume_{i}_sum'] += bid_volume

                if ask_price > 0:
                    self.bar.extra_metrics[f'ask_price_{i}_sum'] += ask_price
                    self.bar.extra_metrics[f'ask_price_{i}_count'] += 1
                    self.bar.extra_metrics[f'ask_volume_{i}_sum'] += ask_volume
            # 新增2026/04/28

            # 计算当前 tick 与上一 tick 的实际时间间隔（秒）
            if self.last_tick is not None:
                delta = (tick.datetime - self.last_tick.datetime).total_seconds()
                # 如果出现负间隔或过小间隔，可视情况过滤或设为0
                if delta <= 0:
                    delta = 0
            else:
                # 第一笔 tick：我们没有 delta，可置为 0（不影响后续累加）
                delta = 0

            # # 直接将时间差作为权重
            # self.bar.extra_metrics['price_time_sum'] += tick.last_price * delta
            # self.bar.extra_metrics['time_weight_sum'] += delta

            ### 新增2026/03/30：当前时段使用上一笔tick的价格
            # TWAP 按 previous-tick / piecewise-constant 方法更新：
            # [上一笔 tick 时点, 当前 tick 时点) 这段时间，使用上一笔价格。
            self.bar.extra_metrics['price_time_sum'] += self.last_tick.last_price * delta
            self.bar.extra_metrics['time_weight_sum'] += delta
            ### 新增2026/03/30

            # tick_count 仍然记录 tick 数量
            self.bar.extra_metrics['tick_count'] += 1

        if self.last_tick:
            volume_change: float = tick.volume - self.last_tick.volume
            self.bar.volume += max(volume_change, 0)

            turnover_change: float = tick.turnover - self.last_tick.turnover
            self.bar.turnover += max(turnover_change, 0)

        self.last_tick = tick

    #新增2025/03/01：新增_finalize_custom_metrics函数
    def _finalize_custom_metrics(self, bar: BarData) -> None:
        """
        当一分钟K线结束时，将 extra_metrics 中的自定义充分统计量写入 bar 对象。

        输出字段：
        - sum_bid_price_i / cnt_bid_price_i
        - sum_ask_price_i / cnt_ask_price_i
        - sum_bid_volume_i / sum_ask_volume_i
        - tick_count
        - twap_num = price_time_sum
        - twap_den = time_weight_sum

        不直接写 avg_bid_price_i / avg_ask_price_i / twap。
        后续派生：
        - avg_bid_price_i = sum_bid_price_i / cnt_bid_price_i
        - avg_ask_price_i = sum_ask_price_i / cnt_ask_price_i
        - avg_bid_volume_i = sum_bid_volume_i / cnt_bid_price_i
        - avg_ask_volume_i = sum_ask_volume_i / cnt_ask_price_i
        - twap = twap_num / twap_den
        """
        metrics = bar.extra_metrics
        count = metrics.get('tick_count', 0)
        # print(f"[DEBUG] Finalizing metrics: {metrics}, tick_count={count}")

        ### 新增2026/03/30：TWAP收尾
        # TWAP 收尾：将最后一笔 tick 的价格补到当前 bar 结束时刻。
        minute_start: datetime = bar.datetime.replace(second=0, microsecond=0)
        minute_end: datetime = minute_start + timedelta(minutes=1)
        tail_delta: float = (minute_end - self.last_tick.datetime).total_seconds()
        if tail_delta > 0:
            metrics['price_time_sum'] += self.last_tick.last_price * tail_delta
            metrics['time_weight_sum'] += tail_delta
        ### 新增2026/03/30

        if count > 0:
            # # 新增2026/04/20：avg_bid_price和avg_ask_price分别除以各自的non-zero计数
            # bar.avg_bid_price_1 = round(metrics['bid_price_1_sum'] / metrics['bid_price_1_count'], 4) if metrics['bid_price_1_count'] > 0 else 0
            # bar.avg_ask_price_1 = round(metrics['ask_price_1_sum'] / metrics['ask_price_1_count'], 4) if metrics['ask_price_1_count'] > 0 else 0
            # bar.sum_bid_volume_1 = metrics['bid_volume_1_sum']
            # bar.sum_ask_volume_1 = metrics['ask_volume_1_sum']
            # # bar.avg_spread_ratio_1 = round(metrics['spread_ratio_1_sum'] / count, 10)
            # bar.avg_bid_price_2 = round(metrics['bid_price_2_sum'] / metrics['bid_price_2_count'], 4) if metrics['bid_price_2_count'] > 0 else 0
            # bar.avg_ask_price_2 = round(metrics['ask_price_2_sum'] / metrics['ask_price_2_count'], 4) if metrics['ask_price_2_count'] > 0 else 0
            # bar.sum_bid_volume_2 = metrics['bid_volume_2_sum']
            # bar.sum_ask_volume_2 = metrics['ask_volume_2_sum']
            # # bar.avg_spread_ratio_2 = round(metrics['spread_ratio_2_sum'] / count, 10)
            # bar.avg_bid_price_3 = round(metrics['bid_price_3_sum'] / metrics['bid_price_3_count'], 4) if metrics['bid_price_3_count'] > 0 else 0
            # bar.avg_ask_price_3 = round(metrics['ask_price_3_sum'] / metrics['ask_price_3_count'], 4) if metrics['ask_price_3_count'] > 0 else 0
            # bar.sum_bid_volume_3 = metrics['bid_volume_3_sum']
            # bar.sum_ask_volume_3 = metrics['ask_volume_3_sum']
            # # bar.avg_spread_ratio_3 = round(metrics['spread_ratio_3_sum'] / count, 10)
            # bar.avg_bid_price_4 = round(metrics['bid_price_4_sum'] / metrics['bid_price_4_count'], 4) if metrics['bid_price_4_count'] > 0 else 0
            # bar.avg_ask_price_4 = round(metrics['ask_price_4_sum'] / metrics['ask_price_4_count'], 4) if metrics['ask_price_4_count'] > 0 else 0
            # bar.sum_bid_volume_4 = metrics['bid_volume_4_sum']
            # bar.sum_ask_volume_4 = metrics['ask_volume_4_sum']
            # # bar.avg_spread_ratio_4 = round(metrics['spread_ratio_4_sum'] / count, 10)
            # bar.avg_bid_price_5 = round(metrics['bid_price_5_sum'] / metrics['bid_price_5_count'], 4) if metrics['bid_price_5_count'] > 0 else 0
            # bar.avg_ask_price_5 = round(metrics['ask_price_5_sum'] / metrics['ask_price_5_count'], 4) if metrics['ask_price_5_count'] > 0 else 0
            # bar.sum_bid_volume_5 = metrics['bid_volume_5_sum']
            # bar.sum_ask_volume_5 = metrics['ask_volume_5_sum']
            # # bar.avg_spread_ratio_5 = round(metrics['spread_ratio_5_sum'] / count, 10)
            # bar.tick_count = count
            # # 新增2026/04/20：avg_bid_price和avg_ask_price分别除以各自的non-zero计数
            # # # 计算 TWAP：用 price_time_sum / time_weight_sum
            # # bar.twap = round(metrics['price_time_sum'] / metrics['time_weight_sum'], 4)
            # ### 新增2026/03/30：用close_price为极端情况兜底
            # time_weight_sum: float = metrics['time_weight_sum']
            # if time_weight_sum > 0:
            #     bar.twap = round(metrics['price_time_sum'] / time_weight_sum, 4)
            # else:
            #     bar.twap = round(bar.close_price, 4)
            # ### 新增2026/03/30

            # 新增2026/04/28
            for i in range(1, 6):
                setattr(bar, f'sum_bid_price_{i}', metrics[f'bid_price_{i}_sum'])
                setattr(bar, f'cnt_bid_price_{i}', metrics[f'bid_price_{i}_count'])
                setattr(bar, f'sum_ask_price_{i}', metrics[f'ask_price_{i}_sum'])
                setattr(bar, f'cnt_ask_price_{i}', metrics[f'ask_price_{i}_count'])
                setattr(bar, f'sum_bid_volume_{i}', metrics[f'bid_volume_{i}_sum'])
                setattr(bar, f'sum_ask_volume_{i}', metrics[f'ask_volume_{i}_sum'])

            bar.tick_count = count
            bar.twap_num = metrics['price_time_sum']
            bar.twap_den = metrics['time_weight_sum']
            # 新增2026/04/28

        else:
        #     bar.avg_bid_price_1 = None
        #     bar.avg_ask_price_1 = None
        #     bar.sum_bid_volume_1 = 0
        #     bar.sum_ask_volume_1 = 0
        #     # bar.avg_spread_ratio_1 = None
        #     bar.avg_bid_price_2 = None
        #     bar.avg_ask_price_2 = None
        #     bar.sum_bid_volume_2 = 0
        #     bar.sum_ask_volume_2 = 0
        #     # bar.avg_spread_ratio_2 = None
        #     bar.avg_bid_price_3 = None
        #     bar.avg_ask_price_3 = None
        #     bar.sum_bid_volume_3 = 0
        #     bar.sum_ask_volume_3 = 0
        #     # bar.avg_spread_ratio_3 = None
        #     bar.avg_bid_price_4 = None
        #     bar.avg_ask_price_4 = None
        #     bar.sum_bid_volume_4 = 0
        #     bar.sum_ask_volume_4 = 0
        #     # bar.avg_spread_ratio_4 = None
        #     bar.avg_bid_price_5 = None
        #     bar.avg_ask_price_5 = None
        #     bar.sum_bid_volume_5 = 0
        #     bar.sum_ask_volume_5 = 0
        #     # bar.avg_spread_ratio_5 = None
        #     bar.tick_count = 0
        #     bar.twap = None
        # # print(f"[DEBUG] Finalized bar: {bar.__dict__}")
            # 新增2026/04/28
            for i in range(1, 6):
                setattr(bar, f'sum_bid_price_{i}', 0.0)
                setattr(bar, f'cnt_bid_price_{i}', 0)
                setattr(bar, f'sum_ask_price_{i}', 0.0)
                setattr(bar, f'cnt_ask_price_{i}', 0)
                setattr(bar, f'sum_bid_volume_{i}', 0.0)
                setattr(bar, f'sum_ask_volume_{i}', 0.0)

            bar.tick_count = 0
            bar.twap_num = 0.0
            bar.twap_den = 0.0
            # 新增2026/04/28
    #新增2025/03/01

    def update_bar(self, bar: BarData) -> None:
        """
        Update 1 minute bar into generator
        """
        if self.interval == Interval.MINUTE:
            self.update_bar_minute_window(bar)
        elif self.interval == Interval.HOUR:
            self.update_bar_hour_window(bar)
        else:
            self.update_bar_daily_window(bar)

    def update_bar_minute_window(self, bar: BarData) -> None:
        """"""
        # If not inited, create window bar object
        if not self.window_bar:
            dt: datetime = bar.datetime.replace(second=0, microsecond=0)
            ###原始代码
            self.window_bar = BarData(
                symbol=bar.symbol,
                exchange=bar.exchange,
                datetime=dt,
                gateway_name=bar.gateway_name,
                open_price=bar.open_price,
                high_price=bar.high_price,
                low_price=bar.low_price
            )
            ###原始代码
        #     #新增2025/03/01
        #     self.window_bar = BarData(
        #         symbol=bar.symbol,
        #         exchange=bar.exchange,
        #         datetime=dt,
        #         gateway_name=bar.gateway_name,
        #         open_price=bar.open_price,
        #         high_price=bar.high_price,
        #         low_price=bar.low_price,
        #         volume=bar.volume,
        #         turnover=bar.turnover,
        #         # 初始化自定义指标为当前1分钟bar的值
        #         avg_bid_price_1 = bar.avg_bid_price_1,
        #         avg_ask_price_1 = bar.avg_ask_price_1,
        #         sum_bid_volume_1 = bar.sum_bid_volume_1,
        #         sum_ask_volume_1 = bar.sum_ask_volume_1,
        #         avg_spread_ratio = bar.avg_spread_ratio,
        #         tick_count = bar.tick_count,
        #         twap = bar.twap
        #     )
        #     #新增2025/03/01
        # # Otherwise, update high/low price into window bar
        else:
            self.window_bar.high_price = max(
                self.window_bar.high_price,
                bar.high_price
            )
            self.window_bar.low_price = min(
                self.window_bar.low_price,
                bar.low_price
            )

        # Update close price/volume/turnover into window bar
        self.window_bar.close_price = bar.close_price
        self.window_bar.volume += bar.volume
        self.window_bar.turnover += bar.turnover
        self.window_bar.open_interest = bar.open_interest

        # #新增2025/03/01
        # # 对自定义指标进行聚合：假设使用简单的累计和再求平均（对于平均型指标）
        # # 计算加权平均时，需要知道累计的1分钟 bar 数量（我们用 self.interval_count 累计）
        # self.interval_count += 1
        # # 这里假设各平均值用简单的加权平均
        # # 将累计的1分钟 bar 的指标累加：
        # # 对于 avg_bid_price_1、avg_ask_price_1：我们可以将各分钟的平均值累加再除以总数
        # if self.window_bar.avg_bid_price_1 is None:
        #     self.window_bar.avg_bid_price_1 = bar.avg_bid_price_1
        # else:
        #     self.window_bar.avg_bid_price_1 = (self.window_bar.avg_bid_price_1 * (self.interval_count - 1) + bar.avg_bid_price_1) / self.interval_count

        # if self.window_bar.avg_ask_price_1 is None:
        #     self.window_bar.avg_ask_price_1 = bar.avg_ask_price_1
        # else:
        #     self.window_bar.avg_ask_price_1 = (self.window_bar.avg_ask_price_1 * (self.interval_count - 1) + bar.avg_ask_price_1) / self.interval_count

        # # 对于 sum_bid_volume_1 和 sum_ask_volume_1，直接累加
        # self.window_bar.sum_bid_volume_1 += bar.sum_bid_volume_1
        # self.window_bar.sum_ask_volume_1 += bar.sum_ask_volume_1

        # # 对于 avg_spread_ratio，累加后平均
        # if self.window_bar.avg_spread_ratio is None:
        #     self.window_bar.avg_spread_ratio = bar.avg_spread_ratio
        # else:
        #     self.window_bar.avg_spread_ratio = (self.window_bar.avg_spread_ratio * (self.interval_count - 1) + bar.avg_spread_ratio) / self.interval_count

        # # Tick 数量累加
        # self.window_bar.tick_count += bar.tick_count

        # # 对于 TWAP，累加1分钟 bar 的 TWAP，再加权平均
        # if self.window_bar.twap is None:
        #     self.window_bar.twap = bar.twap
        # else:
        #     self.window_bar.twap = (self.window_bar.twap * (self.interval_count - 1) + bar.twap) / self.interval_count
        # #新增2025/03/01

        # Check if window bar completed
        if not (bar.datetime.minute + 1) % self.window:
            # print(f"[DEBUG] 3-minute window complete at {bar.datetime}")
            self.on_window_bar(self.window_bar)
            self.window_bar = None
            # #新增2025/03/01
            # self.interval_count = 0
            # #新增2025/03/01

    def update_bar_hour_window(self, bar: BarData) -> None:
        """"""
        # If not inited, create window bar object
        if not self.hour_bar:
            dt: datetime = bar.datetime.replace(minute=0, second=0, microsecond=0)
            self.hour_bar = BarData(
                symbol=bar.symbol,
                exchange=bar.exchange,
                datetime=dt,
                gateway_name=bar.gateway_name,
                open_price=bar.open_price,
                high_price=bar.high_price,
                low_price=bar.low_price,
                close_price=bar.close_price,
                volume=bar.volume,
                turnover=bar.turnover,
                open_interest=bar.open_interest
            )
            return

        finished_bar: BarData = None

        # If minute is 59, update minute bar into window bar and push
        if bar.datetime.minute == 59:
            self.hour_bar.high_price = max(
                self.hour_bar.high_price,
                bar.high_price
            )
            self.hour_bar.low_price = min(
                self.hour_bar.low_price,
                bar.low_price
            )

            self.hour_bar.close_price = bar.close_price
            self.hour_bar.volume += bar.volume
            self.hour_bar.turnover += bar.turnover
            self.hour_bar.open_interest = bar.open_interest

            finished_bar = self.hour_bar
            self.hour_bar = None

        # If minute bar of new hour, then push existing window bar
        elif bar.datetime.hour != self.hour_bar.datetime.hour:
            finished_bar = self.hour_bar

            dt: datetime = bar.datetime.replace(minute=0, second=0, microsecond=0)
            self.hour_bar = BarData(
                symbol=bar.symbol,
                exchange=bar.exchange,
                datetime=dt,
                gateway_name=bar.gateway_name,
                open_price=bar.open_price,
                high_price=bar.high_price,
                low_price=bar.low_price,
                close_price=bar.close_price,
                volume=bar.volume,
                turnover=bar.turnover,
                open_interest=bar.open_interest
            )
        # Otherwise only update minute bar
        else:
            self.hour_bar.high_price = max(
                self.hour_bar.high_price,
                bar.high_price
            )
            self.hour_bar.low_price = min(
                self.hour_bar.low_price,
                bar.low_price
            )

            self.hour_bar.close_price = bar.close_price
            self.hour_bar.volume += bar.volume
            self.hour_bar.turnover += bar.turnover
            self.hour_bar.open_interest = bar.open_interest

        # Push finished window bar
        if finished_bar:
            self.on_hour_bar(finished_bar)

    def on_hour_bar(self, bar: BarData) -> None:
        """"""
        if self.window == 1:
            self.on_window_bar(bar)
        else:
            if not self.window_bar:
                self.window_bar = BarData(
                    symbol=bar.symbol,
                    exchange=bar.exchange,
                    datetime=bar.datetime,
                    gateway_name=bar.gateway_name,
                    open_price=bar.open_price,
                    high_price=bar.high_price,
                    low_price=bar.low_price
                )
            else:
                self.window_bar.high_price = max(
                    self.window_bar.high_price,
                    bar.high_price
                )
                self.window_bar.low_price = min(
                    self.window_bar.low_price,
                    bar.low_price
                )

            self.window_bar.close_price = bar.close_price
            self.window_bar.volume += bar.volume
            self.window_bar.turnover += bar.turnover
            self.window_bar.open_interest = bar.open_interest

            self.interval_count += 1
            if not self.interval_count % self.window:
                self.interval_count = 0
                self.on_window_bar(self.window_bar)
                self.window_bar = None

    def update_bar_daily_window(self, bar: BarData) -> None:
        """"""
        # If not inited, create daily bar object
        if not self.daily_bar:
            self.daily_bar = BarData(
                symbol=bar.symbol,
                exchange=bar.exchange,
                datetime=bar.datetime,
                gateway_name=bar.gateway_name,
                open_price=bar.open_price,
                high_price=bar.high_price,
                low_price=bar.low_price
            )
        # Otherwise, update high/low price into daily bar
        else:
            self.daily_bar.high_price = max(
                self.daily_bar.high_price,
                bar.high_price
            )
            self.daily_bar.low_price = min(
                self.daily_bar.low_price,
                bar.low_price
            )

        # Update close price/volume/turnover into daily bar
        self.daily_bar.close_price = bar.close_price
        self.daily_bar.volume += bar.volume
        self.daily_bar.turnover += bar.turnover
        self.daily_bar.open_interest = bar.open_interest

        # Check if daily bar completed
        if bar.datetime.time() == self.daily_end:
            self.daily_bar.datetime = bar.datetime.replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0
            )
            self.on_window_bar(self.daily_bar)

            self.daily_bar = None

    def generate(self) -> Optional[BarData]:
        """
        Generate the bar data and call callback immediately.
        """
        bar: BarData = self.bar

        if self.bar:
            ###新增2026/01/19：自定义指标收尾
            if getattr(bar, "extra_metrics", None):
                self._finalize_custom_metrics(bar)
            ###新增2026/01/19：自定义指标收尾
            bar.datetime = bar.datetime.replace(second=0, microsecond=0)
            # print(f"[DEBUG] generate: calling on_bar with bar: {bar.__dict__}")
            self.on_bar(bar)

        self.bar = None
        return bar


class ArrayManager(object):
    """
    For:
    1. time series container of bar data
    2. calculating technical indicator value
    """

    def __init__(self, size: int = 100) -> None:
        """Constructor"""
        self.count: int = 0
        self.size: int = size
        self.inited: bool = False

        self.open_array: np.ndarray = np.zeros(size)
        self.high_array: np.ndarray = np.zeros(size)
        self.low_array: np.ndarray = np.zeros(size)
        self.close_array: np.ndarray = np.zeros(size)
        self.volume_array: np.ndarray = np.zeros(size)
        self.turnover_array: np.ndarray = np.zeros(size)
        self.open_interest_array: np.ndarray = np.zeros(size)

    def update_bar(self, bar: BarData) -> None:
        """
        Update new bar data into array manager.
        """
        self.count += 1
        if not self.inited and self.count >= self.size:
            self.inited = True

        self.open_array[:-1] = self.open_array[1:]
        self.high_array[:-1] = self.high_array[1:]
        self.low_array[:-1] = self.low_array[1:]
        self.close_array[:-1] = self.close_array[1:]
        self.volume_array[:-1] = self.volume_array[1:]
        self.turnover_array[:-1] = self.turnover_array[1:]
        self.open_interest_array[:-1] = self.open_interest_array[1:]

        self.open_array[-1] = bar.open_price
        self.high_array[-1] = bar.high_price
        self.low_array[-1] = bar.low_price
        self.close_array[-1] = bar.close_price
        self.volume_array[-1] = bar.volume
        self.turnover_array[-1] = bar.turnover
        self.open_interest_array[-1] = bar.open_interest

    @property
    def open(self) -> np.ndarray:
        """
        Get open price time series.
        """
        return self.open_array

    @property
    def high(self) -> np.ndarray:
        """
        Get high price time series.
        """
        return self.high_array

    @property
    def low(self) -> np.ndarray:
        """
        Get low price time series.
        """
        return self.low_array

    @property
    def close(self) -> np.ndarray:
        """
        Get close price time series.
        """
        return self.close_array

    @property
    def volume(self) -> np.ndarray:
        """
        Get trading volume time series.
        """
        return self.volume_array

    @property
    def turnover(self) -> np.ndarray:
        """
        Get trading turnover time series.
        """
        return self.turnover_array

    @property
    def open_interest(self) -> np.ndarray:
        """
        Get trading volume time series.
        """
        return self.open_interest_array

    def sma(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        Simple moving average.
        """
        result: np.ndarray = talib.SMA(self.close, n)
        if array:
            return result
        return result[-1]

    def ema(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        Exponential moving average.
        """
        result: np.ndarray = talib.EMA(self.close, n)
        if array:
            return result
        return result[-1]

    def kama(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        KAMA.
        """
        result: np.ndarray = talib.KAMA(self.close, n)
        if array:
            return result
        return result[-1]

    def wma(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        WMA.
        """
        result: np.ndarray = talib.WMA(self.close, n)
        if array:
            return result
        return result[-1]

    def apo(
        self,
        fast_period: int,
        slow_period: int,
        matype: int = 0,
        array: bool = False
    ) -> Union[float, np.ndarray]:
        """
        APO.
        """
        result: np.ndarray = talib.APO(self.close, fast_period, slow_period, matype)
        if array:
            return result
        return result[-1]

    def cmo(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        CMO.
        """
        result: np.ndarray = talib.CMO(self.close, n)
        if array:
            return result
        return result[-1]

    def mom(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        MOM.
        """
        result: np.ndarray = talib.MOM(self.close, n)
        if array:
            return result
        return result[-1]

    def ppo(
        self,
        fast_period: int,
        slow_period: int,
        matype: int = 0,
        array: bool = False
    ) -> Union[float, np.ndarray]:
        """
        PPO.
        """
        result: np.ndarray = talib.PPO(self.close, fast_period, slow_period, matype)
        if array:
            return result
        return result[-1]

    def roc(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        ROC.
        """
        result: np.ndarray = talib.ROC(self.close, n)
        if array:
            return result
        return result[-1]

    def rocr(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        ROCR.
        """
        result: np.ndarray = talib.ROCR(self.close, n)
        if array:
            return result
        return result[-1]

    def rocp(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        ROCP.
        """
        result: np.ndarray = talib.ROCP(self.close, n)
        if array:
            return result
        return result[-1]

    def rocr_100(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        ROCR100.
        """
        result: np.ndarray = talib.ROCR100(self.close, n)
        if array:
            return result
        return result[-1]

    def trix(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        TRIX.
        """
        result: np.ndarray = talib.TRIX(self.close, n)
        if array:
            return result
        return result[-1]

    def std(self, n: int, nbdev: int = 1, array: bool = False) -> Union[float, np.ndarray]:
        """
        Standard deviation.
        """
        result: np.ndarray = talib.STDDEV(self.close, n, nbdev)
        if array:
            return result
        return result[-1]

    def obv(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        OBV.
        """
        result: np.ndarray = talib.OBV(self.close, self.volume)
        if array:
            return result
        return result[-1]

    def cci(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        Commodity Channel Index (CCI).
        """
        result: np.ndarray = talib.CCI(self.high, self.low, self.close, n)
        if array:
            return result
        return result[-1]

    def atr(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        Average True Range (ATR).
        """
        result: np.ndarray = talib.ATR(self.high, self.low, self.close, n)
        if array:
            return result
        return result[-1]

    def natr(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        NATR.
        """
        result: np.ndarray = talib.NATR(self.high, self.low, self.close, n)
        if array:
            return result
        return result[-1]

    def rsi(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        Relative Strenght Index (RSI).
        """
        result: np.ndarray = talib.RSI(self.close, n)
        if array:
            return result
        return result[-1]

    def macd(
        self,
        fast_period: int,
        slow_period: int,
        signal_period: int,
        array: bool = False
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[float, float, float]
    ]:
        """
        MACD.
        """
        macd, signal, hist = talib.MACD(
            self.close, fast_period, slow_period, signal_period
        )
        if array:
            return macd, signal, hist
        return macd[-1], signal[-1], hist[-1]

    def adx(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        ADX.
        """
        result: np.ndarray = talib.ADX(self.high, self.low, self.close, n)
        if array:
            return result
        return result[-1]

    def adxr(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        ADXR.
        """
        result: np.ndarray = talib.ADXR(self.high, self.low, self.close, n)
        if array:
            return result
        return result[-1]

    def dx(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        DX.
        """
        result: np.ndarray = talib.DX(self.high, self.low, self.close, n)
        if array:
            return result
        return result[-1]

    def minus_di(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        MINUS_DI.
        """
        result: np.ndarray = talib.MINUS_DI(self.high, self.low, self.close, n)
        if array:
            return result
        return result[-1]

    def plus_di(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        PLUS_DI.
        """
        result: np.ndarray = talib.PLUS_DI(self.high, self.low, self.close, n)
        if array:
            return result
        return result[-1]

    def willr(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        WILLR.
        """
        result: np.ndarray = talib.WILLR(self.high, self.low, self.close, n)
        if array:
            return result
        return result[-1]

    def ultosc(
        self,
        time_period1: int = 7,
        time_period2: int = 14,
        time_period3: int = 28,
        array: bool = False
    ) -> Union[float, np.ndarray]:
        """
        Ultimate Oscillator.
        """
        result: np.ndarray = talib.ULTOSC(self.high, self.low, self.close, time_period1, time_period2, time_period3)
        if array:
            return result
        return result[-1]

    def trange(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        TRANGE.
        """
        result: np.ndarray = talib.TRANGE(self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def boll(
        self,
        n: int,
        dev: float,
        array: bool = False
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[float, float]
    ]:
        """
        Bollinger Channel.
        """
        mid: Union[float, np.ndarray] = self.sma(n, array)
        std: Union[float, np.ndarray] = self.std(n, 1, array)

        up: Union[float, np.ndarray] = mid + std * dev
        down: Union[float, np.ndarray] = mid - std * dev

        return up, down

    def keltner(
        self,
        n: int,
        dev: float,
        array: bool = False
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[float, float]
    ]:
        """
        Keltner Channel.
        """
        mid: Union[float, np.ndarray] = self.sma(n, array)
        atr: Union[float, np.ndarray] = self.atr(n, array)

        up: Union[float, np.ndarray] = mid + atr * dev
        down: Union[float, np.ndarray] = mid - atr * dev

        return up, down

    def donchian(
        self, n: int, array: bool = False
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[float, float]
    ]:
        """
        Donchian Channel.
        """
        up: np.ndarray = talib.MAX(self.high, n)
        down: np.ndarray = talib.MIN(self.low, n)

        if array:
            return up, down
        return up[-1], down[-1]

    def aroon(
        self,
        n: int,
        array: bool = False
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[float, float]
    ]:
        """
        Aroon indicator.
        """
        aroon_down, aroon_up = talib.AROON(self.high, self.low, n)

        if array:
            return aroon_up, aroon_down
        return aroon_up[-1], aroon_down[-1]

    def aroonosc(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        Aroon Oscillator.
        """
        result: np.ndarray = talib.AROONOSC(self.high, self.low, n)

        if array:
            return result
        return result[-1]

    def minus_dm(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        MINUS_DM.
        """
        result: np.ndarray = talib.MINUS_DM(self.high, self.low, n)

        if array:
            return result
        return result[-1]

    def plus_dm(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        PLUS_DM.
        """
        result: np.ndarray = talib.PLUS_DM(self.high, self.low, n)

        if array:
            return result
        return result[-1]

    def mfi(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        Money Flow Index.
        """
        result: np.ndarray = talib.MFI(self.high, self.low, self.close, self.volume, n)
        if array:
            return result
        return result[-1]

    def ad(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        AD.
        """
        result: np.ndarray = talib.AD(self.high, self.low, self.close, self.volume)
        if array:
            return result
        return result[-1]

    def adosc(
        self,
        fast_period: int,
        slow_period: int,
        array: bool = False
    ) -> Union[float, np.ndarray]:
        """
        ADOSC.
        """
        result: np.ndarray = talib.ADOSC(self.high, self.low, self.close, self.volume, fast_period, slow_period)
        if array:
            return result
        return result[-1]

    def bop(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        BOP.
        """
        result: np.ndarray = talib.BOP(self.open, self.high, self.low, self.close)

        if array:
            return result
        return result[-1]

    def stoch(
        self,
        fastk_period: int,
        slowk_period: int,
        slowk_matype: int,
        slowd_period: int,
        slowd_matype: int,
        array: bool = False
    ) -> Union[
        Tuple[float, float],
        Tuple[np.ndarray, np.ndarray]
    ]:
        """
        Stochastic Indicator
        """
        k, d = talib.STOCH(
            self.high,
            self.low,
            self.close,
            fastk_period,
            slowk_period,
            slowk_matype,
            slowd_period,
            slowd_matype
        )
        if array:
            return k, d
        return k[-1], d[-1]

    def sar(self, acceleration: float, maximum: float, array: bool = False) -> Union[float, np.ndarray]:
        """
        SAR.
        """
        result: np.ndarray = talib.SAR(self.high, self.low, acceleration, maximum)
        if array:
            return result
        return result[-1]


def virtual(func: Callable) -> Callable:
    """
    mark a function as "virtual", which means that this function can be override.
    any base class should use this or @abstractmethod to decorate all functions
    that can be (re)implemented by subclasses.
    """
    return func


file_handlers: Dict[str, logging.FileHandler] = {}


def _get_file_logger_handler(filename: str) -> logging.FileHandler:
    handler: logging.FileHandler = file_handlers.get(filename, None)
    if handler is None:
        handler = logging.FileHandler(filename)
        file_handlers[filename] = handler  # Am i need a lock?
    return handler


def get_file_logger(filename: str) -> logging.Logger:
    """
    return a logger that writes records into a file.
    """
    logger: logging.Logger = logging.getLogger(filename)
    handler: logging.FileHandler = _get_file_logger_handler(filename)  # get singleton handler.
    handler.setFormatter(log_formatter)
    logger.addHandler(handler)  # each handler will be added only once.
    return logger
