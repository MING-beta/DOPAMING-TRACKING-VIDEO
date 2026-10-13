"""아이스하키 중계 영상에서 특정 등번호의 출연 구간을 뽑는다.

파이프라인
  1. YOLO26-pose로 선수 박스와 관절을 찾고, BoT-SORT(ReID + 카메라 움직임 보정)로 ID를 유지한다.
  2. 스코어버그 영역과 너무 작은 박스는 버린다.
  3. 어깨-엉덩이 관절로 등판을 자르고, 어깨-팔꿈치-손목으로 팔뚝 숫자도 자른다.
  4. 빨강+흰색이면 이름 또는 등번호(0.35부터) 중 하나만 맞아도 확정한다.
  5. 이름과 번호가 같이 보인 장면의 외형을 기준으로 저장하고, 그 뒤에만 짧게 재연결한다.
  6. 확정된 트랙이 링크 위에 있던 시각을 모아 2초 이내 간격은 하나의 구간으로 합친다.
  7. 상단 스코어보드의 경기 시계로 시프트를 예측하고, 피리어드별·경기 전체 플레이타임을 더한다.

실행
  py -3 track_player.py --video hockey_game.mp4 --number 26
  py -3 track_player.py --video hockey_game.mp4 --number 26 --start 1800 --end 2400

같은 번호가 양 팀에 있으면 먼저 확정된 유니폼 색으로 상대 팀을 걸러 낸다.
색을 미리 알려 주려면 --team-bgr 230,230,230 (하양, BGR)처럼 넘긴다.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import traceback
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


VIDEO_PATH = "hockey_game.mp4"
TARGET_NUMBER = "26"
START_TIME_SEC = 0
END_TIME_SEC = None

# 추적 샘플. OCR은 프레임마다 돌리지 않고 OCR_INTERVAL_SEC마다만 돌린다.
# 2fps는 스케이팅 속도에서 ID가 자주 끊겨, 추적은 5fps로 두고 OCR만 띄운다.
SAMPLE_FPS = 5.0
OCR_INTERVAL_SEC = 0.45
IMGSZ = 1280
DET_CONF = 0.10
OCR_DET_CONF = 0.35
TRACK_DET_CONF = 0.25
CONFIRM_HITS = 3
MIN_OCR_CONF = 0.60
MERGE_GAP_SEC = 2.0
RECONNECT_SEC = 2.0
BLUR_MIN_VARIANCE = 25.0
TEAM_COLOR_MAX_DIST = 0.48
TEAM_HINT_BGR: tuple[int, int, int] | None = None
TARGET_NAME = "EAN"
REQUIRE_RED_WHITE = True
# 56px 등판의 26은 0.35 근처에 머문다. 빨강+흰색이면 이 점수부터 받는다.
TARGET_MIN_CONF = 0.35
# 비스듬한 등도 등판 OCR 후보에 넣는다. 오인이 늘면 --no-oblique 로 끈다.
ALLOW_OBLIQUE = True
# 클로즈업 기준과 0.93 이상 같을 때만 번호 없이 확정한다.
# 0.84대는 같은 팀 다른 선수도 넘어서, 그 값은 재연결에만 쓴다.
PROTO_MATCH = 0.93
STITCH_MATCH = 0.75
# 번호가 읽힌 원거리 등판과 이 이상 비슷하면 같은 선수로 본다.
WIDE_MATCH = 0.88
# 글자가 끊긴 뒤 위치와 외형이 이어지면 이 간격까지 번호 없이 붙인다.
FOLLOW_HOP_SEC = 4.0
FOLLOW_PASSES = 8

ENABLE_PARSEQ = True
ENABLE_SAM2 = True
ENABLE_REID = True
# 팔뚝 OCR은 30:00–40:00에서 917회 모두 26을 내지 못했다. 인식만 생략하고 호출 간격은 유지한다.
ENABLE_SLEEVE_OCR = False
SAVE_DEBUG_CROPS = False

SCOREBUG_TOP_RATIO = 0.14
# 1920x1080 중계의 "1st | 19:09" 알약. 다른 해상도는 같은 비율로 자른다.
SCOREBUG_CLOCK_BOX = (844, 16, 1110, 105)
SCOREBUG_CLOCK_REF = (1920, 1080)
PERIOD_LENGTH_SEC = 20 * 60
SHIFT_GAP_SEC = 75.0
SHIFT_MIN_SEC = 40.0
SHIFT_OPEN_MAX_SEC = 60.0
SHIFT_WHISTLE_MAX_SEC = 90.0
# 재투입 최소 휴식(약 1분 30초)보다 짧은 공백은 같은 투입으로 잇는다.
SHIFT_BRIDGE_SEC = 75.0
# 3피리어드 이후, 종료 1분 안에 보이면 남은 시계는 버저까지 뛴 것으로 본다.
FINAL_MINUTE_SEC = 60.0
GAME_END_PERIOD = 3
CLOCK_HOLD_SEC = 1.6
CLOCK_MATCH_SEC = 3.0
MIN_TRACK_HEIGHT_RATIO = 0.040
MIN_OCR_HEIGHT_RATIO = 0.060
MAX_OCR_PER_FRAME = 5

POSE_WEIGHTS = ("yolo26m-pose.pt", "yolo26s-pose.pt", "yolo11m-pose.pt")

# COCO 관절 인덱스
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
FACE_KEYPOINTS = (0, 1, 2)


def format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def format_period_label(period: int) -> str:
    labels = {1: "1st", 2: "2nd", 3: "3rd"}
    if period in labels:
        return labels[period]
    if period >= 4:
        return "OT"
    return f"{period}th"


def format_play_length(seconds: float) -> str:
    """1분은 1, 1분 30초는 1:30처럼 적는다."""
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    if secs == 0:
        return str(minutes)
    return f"{minutes}:{secs:02d}"


def parse_bgr(text: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in text.split(",")]
    if len(parts) != 3 or any(part < 0 or part > 255 for part in parts):
        raise argparse.ArgumentTypeError("BGR은 R,G,B가 아니라 B,G,R 세 값입니다. 예: 230,230,230")
    return parts[0], parts[1], parts[2]


def merge_intervals(
    times: list[float],
    sample_dt: float,
    merge_gap: float,
    end_limit: float,
) -> list[tuple[float, float]]:
    """관측 시각을 구간으로 합친다. 관측 사이 공백이 merge_gap 이하면 한 구간이다."""
    if not times:
        return []
    ordered = sorted({round(t, 3) for t in times})
    start = prev = ordered[0]
    spans: list[tuple[float, float]] = []
    for t in ordered[1:]:
        if t - prev <= merge_gap:
            prev = t
            continue
        spans.append((round(start, 3), round(min(end_limit, prev + sample_dt), 3)))
        start = prev = t
    spans.append((round(start, 3), round(min(end_limit, prev + sample_dt), 3)))
    return [(a, b) for a, b in spans if b - a >= 1.0]


@dataclass
class ClockReading:
    video_t: float
    period: int
    clock_sec: float
    running: bool


_PERIOD_RE = re.compile(r"([1-4])(?:st|nd|rd|th)", re.IGNORECASE)
_CLOCK_RE = re.compile(r"(\d{1,2}):(\d{2})")
_DIGITS_RE = re.compile(r"[1-4](?:st|nd|rd|th)\|?(\d{3,5})", re.IGNORECASE)


def _clock_from_digits(blob: str) -> float | None:
    if len(blob) == 4:
        minutes, seconds = int(blob[:2]), int(blob[2:])
    elif len(blob) == 3:
        minutes, seconds = int(blob[0]), int(blob[1:])
    else:
        return None
    if minutes > 20 or seconds > 59:
        return None
    return float(minutes * 60 + seconds)


def scorebug_options(text: str) -> list[tuple[int, float]]:
    """한 문자열에서 가능한 피리어드·남은 초를 모두 꺼낸다. 다섯 자리는 끝자리나 끼어든 숫자로 본다."""
    compact = re.sub(r"\s+", "", text)
    period_match = _PERIOD_RE.search(compact)
    period = int(period_match.group(1)) if period_match else None
    if period is None and re.search(r"ot", compact, re.IGNORECASE):
        period = 4
    if period is None:
        return []
    found: list[tuple[int, float]] = []

    def add(clock: float | None) -> None:
        if clock is None or clock < 0 or clock > PERIOD_LENGTH_SEC:
            return
        pair = (period, clock)
        if pair not in found:
            found.append(pair)

    clock_match = _CLOCK_RE.search(compact)
    if clock_match:
        add(_clock_from_digits(f"{int(clock_match.group(1)):02d}{clock_match.group(2)}"))
        return found
    digits_match = _DIGITS_RE.search(compact)
    if digits_match is None:
        return []
    digits = digits_match.group(1)
    if len(digits) == 5:
        add(_clock_from_digits(digits[:4]))
        for index in range(5):
            add(_clock_from_digits(digits[:index] + digits[index + 1 :]))
    else:
        add(_clock_from_digits(digits))
    return found


def parse_scorebug_text(text: str) -> tuple[int, float] | None:
    """'1st|19:09', '2nd|17450'처럼 읽힌 문자열에서 피리어드와 남은 초를 꺼낸다."""
    options = scorebug_options(text)
    return options[0] if options else None


def consider_clock(
    history: list[ClockReading],
    video_t: float,
    period: int,
    clock_sec: float,
    max_drop: float | None = None,
) -> ClockReading | None:
    """1초 안팎으로 줄어드는 값만 경기 시계로 받는다. 20:00 근처로 되돌아가면 다음 피리어드다."""
    if clock_sec < 0 or clock_sec > PERIOD_LENGTH_SEC:
        return None
    if not history:
        return ClockReading(video_t, period, clock_sec, False)
    last = history[-1]
    video_dt = video_t - last.video_t
    if video_dt < 0.2:
        return None
    if (
        period == last.period
        and last.clock_sec <= 5 * 60
        and clock_sec >= 15 * 60
    ):
        period = last.period + 1
    if period < last.period:
        return None
    if period > last.period:
        if clock_sec <= last.clock_sec + 5:
            return None
        return ClockReading(video_t, period, clock_sec, False)
    dropped = last.clock_sec - clock_sec
    limit = video_dt + 2.0 if max_drop is None else max(video_dt + 2.0, max_drop)
    if dropped < -1.0 or dropped > limit:
        return None
    running = dropped >= 0.5
    return ClockReading(video_t, period, clock_sec, running)


def _clock_blobs(texts: list[str]) -> list[str]:
    blobs = [text for text in texts if text]
    if len(texts) > 1:
        blobs.append("".join(texts))
    return blobs


def _option_lists(texts: list[str]) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """콜론이 있는 시각과, 숫자가 끼어든 시각을 나눠 꺼낸다."""
    anchored: list[tuple[int, float]] = []
    found: list[tuple[int, float]] = []
    for text in _clock_blobs(texts):
        compact = re.sub(r"\s+", "", text)
        options = scorebug_options(text)
        colon = _CLOCK_RE.search(compact) is not None
        for parsed in options:
            if parsed not in found:
                found.append(parsed)
            if colon and parsed not in anchored:
                anchored.append(parsed)
    return anchored, found


def _recent_rate(history: list[ClockReading]) -> float:
    """최근 시계가 영상 시간과 같이 줄면 1, 멈춰 있으면 0."""
    if len(history) < 2:
        return 1.0
    total_dt = 0.0
    total_drop = 0.0
    for older, newer in zip(history[-8:], history[-7:]):
        dt = newer.video_t - older.video_t
        if dt < 0.2 or newer.period != older.period:
            continue
        total_dt += dt
        total_drop += older.clock_sec - newer.clock_sec
    if total_dt < 0.5:
        return 1.0
    return 0.0 if total_drop / total_dt < 0.35 else 1.0


def _rank_clock(
    options: list[tuple[int, float]],
    history: list[ClockReading],
    video_t: float,
    max_drop: float | None,
) -> ClockReading | None:
    """같은 피리어드에서, 영상 시간만큼 줄어든 해석을 고른다."""
    best: ClockReading | None = None
    best_key: tuple | None = None
    video_dt = 0.0 if not history else max(0.0, video_t - history[-1].video_t)
    expected = _recent_rate(history) * video_dt
    for period, clock_sec in options:
        reading = consider_clock(history, video_t, period, clock_sec, max_drop=max_drop)
        if reading is None:
            continue
        dropped = 0.0 if not history else history[-1].clock_sec - clock_sec
        same_period = not history or reading.period == history[-1].period
        key = (0 if same_period else 1, round(abs(dropped - expected), 3), abs(dropped))
        if best_key is None or key < best_key:
            best = reading
            best_key = key
    return best


def _note_clock_anchor(
    anchors: list[tuple[float, int, float]],
    video_t: float,
    options: list[tuple[int, float]],
) -> None:
    if not options:
        return
    if len(options) == 1 or not anchors:
        chosen = options[0]
    else:
        prev_t, prev_period, prev_clock = anchors[-1]
        same = [item for item in options if item[0] == prev_period]
        pool = same or options
        elapsed = max(0.2, video_t - prev_t)
        chosen = min(pool, key=lambda item: abs((prev_clock - item[1]) - elapsed))
    anchors.append((video_t, chosen[0], chosen[1]))
    del anchors[:-8]


def _recover_clock(
    anchors: list[tuple[float, int, float]],
    history: list[ClockReading],
    video_t: float,
) -> ClockReading | None:
    """콜론으로 읽힌 시계가 세 번 이어지면, 어긋난 이전 값을 그 시계로 되돌린다."""
    if len(anchors) < 3 or not history or abs(anchors[-1][0] - video_t) > 0.05:
        return None
    tail = anchors[-3:]
    period = tail[-1][1]
    if any(item[1] != period for item in tail) or period < history[-1].period:
        return None
    for older, newer in zip(tail, tail[1:]):
        dt = newer[0] - older[0]
        dropped = older[2] - newer[2]
        if dt < 0.2 or dt > 3.0 or dropped < -0.6 or dropped > dt + 1.5:
            return None
    clock = tail[-1][2]
    if period == history[-1].period and abs(history[-1].clock_sec - clock) <= 2.0:
        return None
    while history and history[-1].period == period:
        age = video_t - history[-1].video_t
        if age > 120 or history[-1].clock_sec - clock <= age + 8.0:
            break
        history.pop()
    rate = (tail[0][2] - clock) / max(0.2, video_t - tail[0][0])
    return ClockReading(video_t, period, clock, rate > 0.4)


def choose_scorebug(
    texts: list[str],
    history: list[ClockReading],
    video_t: float,
    anchors: list[tuple[float, int, float]] | None = None,
) -> ClockReading | None:
    anchored, found = _option_lists(texts)
    if anchors is not None:
        _note_clock_anchor(anchors, video_t, anchored)
    picked = _rank_clock(anchored, history, video_t, max_drop=4.0) if anchored else None
    if picked is None:
        picked = _rank_clock(found, history, video_t, max_drop=None)
    if picked is None and anchors is not None:
        picked = _recover_clock(anchors, history, video_t)
    return picked


def clock_panel_visible(bgr: np.ndarray) -> bool:
    if bgr.size == 0:
        return False
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    dark_blue = (value < 90) & (sat > 40) & (hue >= 90) & (hue <= 140)
    return float(dark_blue.mean()) >= 0.25


def scorebug_clock_crop(frame: np.ndarray) -> np.ndarray | None:
    height, width = frame.shape[:2]
    if height < 100 or width < 200:
        return None
    x1, y1, x2, y2 = SCOREBUG_CLOCK_BOX
    ref_w, ref_h = SCOREBUG_CLOCK_REF
    left = max(0, int(round(x1 * width / ref_w)))
    top = max(0, int(round(y1 * height / ref_h)))
    right = min(width, int(round(x2 * width / ref_w)))
    bottom = min(height, int(round(y2 * height / ref_h)))
    if right - left < 20 or bottom - top < 12:
        return None
    crop = frame[top:bottom, left:right]
    if not clock_panel_visible(crop):
        return None
    return crop


def stoppage_clocks(readings: list[ClockReading]) -> dict[int, list[float]]:
    """같은 초가 1.6초 이상 머물면 휘슬로 본다. 그 시각이 교체를 예측하는 경계다."""
    stops: dict[int, list[float]] = {}
    index = 0
    while index < len(readings):
        nxt = index + 1
        anchor = readings[index]
        while (
            nxt < len(readings)
            and readings[nxt].period == anchor.period
            and abs(readings[nxt].clock_sec - anchor.clock_sec) <= 0.6
        ):
            nxt += 1
        if readings[nxt - 1].video_t - anchor.video_t >= CLOCK_HOLD_SEC:
            stops.setdefault(anchor.period, []).append(anchor.clock_sec)
        index = nxt
    return stops


def map_sightings(
    readings: list[ClockReading],
    video_times: list[float],
    tolerance: float = CLOCK_MATCH_SEC,
) -> list[tuple[int, float, float]]:
    """감지 시각을 (피리어드, 남은 초, 영상 시각)으로 붙인다. 읽힌 시계 사이는 비율로 메운다."""
    if not readings:
        return []
    times = [item.video_t for item in readings]
    mapped: list[tuple[int, float, float]] = []
    for video_t in video_times:
        index = bisect_left(times, video_t)
        best: ClockReading | None = None
        best_dt = tolerance
        for probe in (index - 1, index):
            if probe < 0 or probe >= len(readings):
                continue
            delta = abs(readings[probe].video_t - video_t)
            if delta <= best_dt:
                best_dt = delta
                best = readings[probe]
        if best is not None:
            mapped.append((best.period, best.clock_sec, video_t))
            continue
        if index <= 0 or index >= len(readings):
            continue
        left, right = readings[index - 1], readings[index]
        span = right.video_t - left.video_t
        if left.period != right.period or span <= 0 or span > 180:
            continue
        dropped = left.clock_sec - right.clock_sec
        if dropped < -1.0 or dropped > span + 3.0:
            continue
        clock = left.clock_sec - dropped * ((video_t - left.video_t) / span)
        mapped.append((left.period, clock, video_t))
    return mapped


def _nearest_stop(stops: list[float], clock_sec: float, before: bool) -> float | None:
    if before:
        candidates = [item for item in stops if clock_sec <= item <= clock_sec + SHIFT_WHISTLE_MAX_SEC]
        return min(candidates) if candidates else None
    candidates = [item for item in stops if clock_sec - SHIFT_WHISTLE_MAX_SEC <= item <= clock_sec]
    return max(candidates) if candidates else None


def fit_shift(
    sight_hi: float,
    sight_lo: float,
    stop_before: float | None,
    stop_after: float | None,
) -> tuple[float, float]:
    """휘슬이 없으면 40~60초로 두고, 시계 정지까지 이어지면 90초까지 늘린다."""
    cap = SHIFT_WHISTLE_MAX_SEC if stop_before is not None or stop_after is not None else SHIFT_OPEN_MAX_SEC
    start = sight_hi if stop_before is None else stop_before
    end = sight_lo if stop_after is None else stop_after
    start = min(PERIOD_LENGTH_SEC, max(start, sight_hi))
    end = max(0.0, min(end, sight_lo))
    length = start - end
    if length > cap:
        extra = length - cap
        back = start - sight_hi
        forward = sight_lo - end
        if forward >= back:
            trim = min(forward, extra)
            end += trim
            extra -= trim
            start -= min(back, extra)
        else:
            trim = min(back, extra)
            start -= trim
            extra -= trim
            end += min(forward, extra)
        if start - end > cap:
            end = start - cap
    length = start - end
    if length < SHIFT_MIN_SEC:
        need = SHIFT_MIN_SEC - length
        back_room = 0.0 if stop_before is not None and abs(start - stop_before) < 0.5 else PERIOD_LENGTH_SEC - start
        forward_room = 0.0 if stop_after is not None and abs(end - stop_after) < 0.5 else end
        grow_back = min(need / 2.0, back_room)
        grow_forward = min(need - grow_back, forward_room)
        remain = need - grow_back - grow_forward
        if remain > 0:
            grow_back += min(remain, back_room - grow_back)
            remain = need - grow_back - grow_forward
            grow_forward += min(remain, forward_room - grow_forward)
        start += grow_back
        end -= grow_forward
    if end > sight_lo:
        end = sight_lo
    if start < sight_hi:
        start = sight_hi
    start = min(PERIOD_LENGTH_SEC, max(start, end))
    end = max(0.0, end)
    return start, end


def clip_for_teammates(
    start: float,
    end: float,
    sight_hi: float,
    sight_lo: float,
    mate_clocks: list[float],
) -> tuple[float, float]:
    """늘린 구간에 같은 팀 다른 번호가 두 번 이상 보이면 그 방향의 추정은 버린다."""
    before = [clock for clock in mate_clocks if sight_hi < clock <= start]
    after = [clock for clock in mate_clocks if end <= clock < sight_lo]
    if len(before) >= 2:
        start = sight_hi
    if len(after) >= 2:
        end = sight_lo
    return start, end


def union_length(spans: list[tuple[float, float]]) -> float:
    intervals = sorted((min(start, end), max(start, end)) for start, end in spans)
    if not intervals:
        return 0.0
    merged: list[list[float]] = [list(intervals[0])]
    for low, high in intervals[1:]:
        if low <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    return sum(high - low for low, high in merged)


def bridge_shifts(shifts: list[dict], stops: list[float], mate_clocks: list[float] | None = None) -> list[dict]:
    """맞닿은 시프트 사이의 안 보인 시간도 넣는다. 휘슬이나 같은 팀 다른 번호가 있으면 교체로 끊는다."""
    if not shifts:
        return []
    mates = mate_clocks or []
    ordered = sorted(shifts, key=lambda item: item["start_clock_sec"], reverse=True)
    merged: list[dict] = [dict(ordered[0])]
    for shift in ordered[1:]:
        previous = merged[-1]
        gap = previous["sight_lo"] - shift["sight_hi"]
        whistle = any(shift["sight_hi"] < stop < previous["sight_lo"] for stop in stops)
        mate_hits = sum(1 for clock in mates if shift["sight_hi"] < clock < previous["sight_lo"])
        if gap <= SHIFT_BRIDGE_SEC and not whistle and mate_hits < 2:
            previous["end_clock_sec"] = round(shift["end_clock_sec"], 3)
            previous["end_clock"] = shift["end_clock"]
            previous["duration_sec"] = round(previous["start_clock_sec"] - previous["end_clock_sec"], 3)
            previous["sightings"] = previous["sightings"] + shift["sightings"]
            previous["sight_lo"] = min(previous["sight_lo"], shift["sight_lo"])
            continue
        merged.append(dict(shift))
    return merged


def play_to_buzzer(shifts: list[dict], period: int, last_period: int) -> None:
    """마지막 피리어드에서 종료 1분 안에 보였으면 남은 시간을 0초까지 넣는다."""
    if period < GAME_END_PERIOD or period != last_period:
        return
    for shift in shifts:
        if shift["sight_lo"] > FINAL_MINUTE_SEC or shift["end_clock_sec"] <= 0:
            continue
        shift["end_clock_sec"] = 0.0
        shift["end_clock"] = format_time(0)
        shift["duration_sec"] = round(shift["start_clock_sec"], 3)


def build_playtime(
    readings: list[ClockReading],
    video_times: list[float],
    teammate_times: list[float] | None = None,
) -> dict:
    """확정된 출연 시각을 시프트로 묶고, 시계가 간 시간만 피리어드별로 더한다."""
    current_period = readings[-1].period if readings else None
    if not readings:
        return {"current_period": None, "periods": [], "total_sec": 0.0}
    stops = stoppage_clocks(readings)
    grouped: dict[int, list[tuple[float, float]]] = {}
    for period, clock_sec, video_t in map_sightings(readings, video_times):
        grouped.setdefault(period, []).append((clock_sec, video_t))
    mates_by_period: dict[int, list[float]] = {}
    for period, clock_sec, _video_t in map_sightings(readings, teammate_times or []):
        mates_by_period.setdefault(period, []).append(clock_sec)
    seen = sorted({item.period for item in readings})
    period_rows: list[dict] = []
    game_total = 0.0
    for period in range(seen[0], seen[-1] + 1):
        points = sorted(grouped.get(period, []), key=lambda item: item[1])
        held = stops.get(period, [])
        mates = mates_by_period.get(period, [])
        clusters: list[list[tuple[float, float]]] = []
        for clock_sec, video_t in points:
            if not clusters:
                clusters.append([(clock_sec, video_t)])
                continue
            prev_clock, prev_video = clusters[-1][-1]
            video_gap = video_t - prev_video
            clock_gap = prev_clock - clock_sec
            split = clock_gap > SHIFT_GAP_SEC or clock_gap < -5.0
            if not split and video_gap > MERGE_GAP_SEC:
                split = any(clock_sec < stop < prev_clock for stop in held)
                split = split or sum(1 for mate in mates if clock_sec < mate < prev_clock) >= 2
            if split:
                clusters.append([(clock_sec, video_t)])
            else:
                clusters[-1].append((clock_sec, video_t))
        shifts: list[dict] = []
        for cluster in clusters:
            clocks = [item[0] for item in cluster]
            sight_hi = max(clocks)
            sight_lo = min(clocks)
            start, end = fit_shift(
                sight_hi,
                sight_lo,
                _nearest_stop(held, sight_hi, before=True),
                _nearest_stop(held, sight_lo, before=False),
            )
            start, end = clip_for_teammates(start, end, sight_hi, sight_lo, mates)
            if start - end <= 0:
                start = sight_hi
                end = max(0.0, sight_hi - 1.0)
            duration = round(start - end, 3)
            if duration <= 0:
                continue
            shifts.append(
                {
                    "start_clock_sec": round(start, 3),
                    "end_clock_sec": round(end, 3),
                    "start_clock": format_time(start),
                    "end_clock": format_time(end),
                    "duration_sec": duration,
                    "sightings": len(cluster),
                    "sight_hi": sight_hi,
                    "sight_lo": sight_lo,
                }
            )
        shifts = bridge_shifts(shifts, held, mates)
        play_to_buzzer(shifts, period, seen[-1])
        for shift in shifts:
            shift.pop("sight_lo", None)
            shift.pop("sight_hi", None)
        spans = [(item["start_clock_sec"], item["end_clock_sec"]) for item in shifts]
        total = round(union_length(spans), 3)
        game_total += total
        period_rows.append({"period": period, "total_sec": total, "shifts": shifts})
    return {
        "current_period": current_period,
        "periods": period_rows,
        "total_sec": round(game_total, 3),
    }


def parse_easyocr(results, min_conf: float) -> tuple[str, float] | None:
    """EasyOCR 결과에서 정확히 두 자리 숫자만 꺼낸다. 부분 일치는 받지 않는다."""
    glyphs: list[dict] = []
    for box, text, conf in results:
        conf = float(conf)
        digits = "".join(ch for ch in str(text) if ch.isdigit())
        if not digits or conf < min_conf:
            continue
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        glyphs.append(
            {
                "x": min(xs),
                "y": sum(ys) / len(ys),
                "w": max(xs) - min(xs),
                "h": max(max(ys) - min(ys), 1.0),
                "text": digits,
                "conf": conf,
            }
        )
    if len(glyphs) == 1 and len(glyphs[0]["text"]) == 2:
        return glyphs[0]["text"], glyphs[0]["conf"]
    singles = [glyph for glyph in glyphs if len(glyph["text"]) == 1]
    if len(glyphs) == 2 and len(singles) == 2:
        singles.sort(key=lambda glyph: glyph["x"])
        left, right = singles
        y_close = abs(left["y"] - right["y"]) <= 0.6 * max(left["h"], right["h"])
        gap = right["x"] - (left["x"] + left["w"])
        limit = 1.2 * max(left["w"], right["w"])
        x_close = -0.3 * left["w"] <= gap <= limit
        if y_close and x_close:
            return left["text"] + right["text"], min(left["conf"], right["conf"])
    return None


def accept_digits(text: str, conf: float, min_conf: float) -> str | None:
    text = text.strip()
    if conf < min_conf or len(text) != 2 or not text.isdigit():
        return None
    return text


def back_visible(xy: np.ndarray | None, conf: np.ndarray | None) -> bool:
    """등이 카메라를 향하는지 본다.

    사람 기준 왼쪽 어깨(인덱스 5)가 화면 왼쪽에 있으면 등을 보인 것이다.
    헬멧 때문에 얼굴 점이 약할 때는, 어깨가 충분히 넓고 얼굴 점이 거의 없을 때만
    등판으로 인정한다.
    """
    if xy is None or conf is None:
        return False
    if float(conf[L_SHOULDER]) < 0.40 or float(conf[R_SHOULDER]) < 0.40:
        return False
    shoulder_w = abs(float(xy[L_SHOULDER, 0] - xy[R_SHOULDER, 0]))
    if float(conf[L_HIP]) >= 0.25 and float(conf[R_HIP]) >= 0.25:
        torso_h = abs(
            float((xy[L_SHOULDER, 1] + xy[R_SHOULDER, 1]) * 0.5)
            - float((xy[L_HIP, 1] + xy[R_HIP, 1]) * 0.5)
        )
    else:
        torso_h = shoulder_w / 0.55
    if shoulder_w < 10 or torso_h < 10:
        return False
    if shoulder_w / torso_h < 0.35:
        return False
    if float(xy[L_SHOULDER, 0]) < float(xy[R_SHOULDER, 0]):
        return True
    face_conf = float(np.mean([conf[index] for index in FACE_KEYPOINTS]))
    shoulder_conf = float(conf[L_SHOULDER] + conf[R_SHOULDER]) * 0.5
    return face_conf < 0.15 and shoulder_conf >= 0.50 and shoulder_w / torso_h >= 0.45


def oblique_back(xy: np.ndarray | None, conf: np.ndarray | None) -> bool:
    """정면이 아니면 비스듬한 등도 등판 글자 후보로 둔다."""
    if xy is None or conf is None:
        return False
    if float(conf[L_SHOULDER]) < 0.30 or float(conf[R_SHOULDER]) < 0.30:
        return False
    shoulder_w = abs(float(xy[L_SHOULDER, 0] - xy[R_SHOULDER, 0]))
    if shoulder_w < 8.0:
        return False
    face_conf = float(np.mean([conf[index] for index in FACE_KEYPOINTS]))
    return face_conf < 0.35


def clip_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return x1, y1, x2, y2


def torso_box(
    person: tuple[float, float, float, float],
    xy: np.ndarray | None,
    conf: np.ndarray | None,
    frame_shape: tuple[int, ...],
) -> tuple[float, float, float, float] | None:
    """등번호가 있는 몸통만 남긴다. 헬멧과 다리, 스틱은 빼는 쪽이다."""
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = person
    if xy is not None and conf is not None and float(conf[L_SHOULDER]) > 0.30 and float(conf[R_SHOULDER]) > 0.30:
        xs = [float(xy[L_SHOULDER, 0]), float(xy[R_SHOULDER, 0])]
        ys = [float(xy[L_SHOULDER, 1]), float(xy[R_SHOULDER, 1])]
        if float(conf[L_HIP]) > 0.25 and float(conf[R_HIP]) > 0.25:
            xs += [float(xy[L_HIP, 0]), float(xy[R_HIP, 0])]
            ys += [float(xy[L_HIP, 1]), float(xy[R_HIP, 1])]
            top = min(ys)
            bottom = max(ys)
        else:
            top = min(ys)
            bottom = top + 0.55 * (y2 - y1)
        left, right = min(xs), max(xs)
        box_w = max(8.0, right - left)
        box_h = max(8.0, bottom - top)
        left -= box_w * 0.35
        right += box_w * 0.35
        top -= box_h * 0.12
        bottom += box_h * 0.08
        return clip_box(left, top, right, bottom, frame_w, frame_h)

    person_h = y2 - y1
    person_w = x2 - x1
    return clip_box(
        x1 + 0.12 * person_w,
        y1 + 0.16 * person_h,
        x2 - 0.12 * person_w,
        y1 + 0.70 * person_h,
        frame_w,
        frame_h,
    )


def arm_strip(frame: np.ndarray, xy: np.ndarray, conf: np.ndarray, start: int, end: int) -> np.ndarray | None:
    """어깨-팔꿈치 또는 팔꿈치-손목 축을 수평으로 돌려 팔뚝 숫자 띠를 만든다."""
    if float(conf[start]) < 0.35 or float(conf[end]) < 0.35:
        return None
    p1 = xy[start].astype(np.float32)
    p2 = xy[end].astype(np.float32)
    delta = p2 - p1
    length = float(np.hypot(float(delta[0]), float(delta[1])))
    if length < 14.0:
        return None
    frame_h, frame_w = frame.shape[:2]
    pad = length * 0.65
    x1 = int(max(0, min(float(p1[0]), float(p2[0])) - pad))
    y1 = int(max(0, min(float(p1[1]), float(p2[1])) - pad))
    x2 = int(min(frame_w, max(float(p1[0]), float(p2[0])) + pad))
    y2 = int(min(frame_h, max(float(p1[1]), float(p2[1])) + pad))
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    roi = frame[y1:y2, x1:x2]
    center = (p1 + p2) * 0.5 - np.array([x1, y1], dtype=np.float32)
    angle = float(np.degrees(np.arctan2(float(delta[1]), float(delta[0]))))
    matrix = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, 1.0)
    rotated = cv2.warpAffine(
        roi,
        matrix,
        (roi.shape[1], roi.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    strip_w = int(max(24, length * 1.20))
    strip_h = int(max(16, length * 0.62))
    left = int(round(float(center[0]) - strip_w / 2))
    top = int(round(float(center[1]) - strip_h / 2))
    return slice_box(rotated, (left, top, left + strip_w, top + strip_h))


def sleeve_geometry_ok(xy: np.ndarray | None, conf: np.ndarray | None) -> bool:
    """팔 축이 숫자 띠를 만들 수 있는 길이인지 본다. 이미지를 돌리지는 않는다."""
    if xy is None or conf is None:
        return False
    for shoulder, elbow, wrist in (
        (L_SHOULDER, L_ELBOW, L_WRIST),
        (R_SHOULDER, R_ELBOW, R_WRIST),
    ):
        end = wrist if float(conf[wrist]) >= 0.30 else elbow
        for joint in (end, elbow):
            if float(conf[shoulder]) < 0.35 or float(conf[joint]) < 0.35:
                continue
            delta = xy[joint].astype(np.float32) - xy[shoulder].astype(np.float32)
            if float(np.hypot(float(delta[0]), float(delta[1]))) >= 14.0:
                return True
    return False


def sleeve_strips(frame: np.ndarray, xy: np.ndarray | None, conf: np.ndarray | None) -> list[np.ndarray]:
    """양팔에서 숫자가 있을 법한 띠를 길이가 긴 순으로 최대 두 장 고른다."""
    if xy is None or conf is None:
        return []
    ranked: list[tuple[float, np.ndarray]] = []
    for shoulder, elbow, wrist in (
        (L_SHOULDER, L_ELBOW, L_WRIST),
        (R_SHOULDER, R_ELBOW, R_WRIST),
    ):
        end = wrist if float(conf[wrist]) >= 0.30 else elbow
        strip = arm_strip(frame, xy, conf, shoulder, end)
        if strip is None and end != elbow:
            strip = arm_strip(frame, xy, conf, shoulder, elbow)
        if strip is None:
            continue
        ranked.append((float(strip.shape[1]), strip))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [strip for _length, strip in ranked[:2]]


def slice_box(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return frame[y1:y2, x1:x2].copy()


def in_banned_region(
    box: tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
) -> bool:
    x1, y1, x2, y2 = box
    center_y = (y1 + y2) * 0.5
    box_w = x2 - x1
    box_h = y2 - y1
    if center_y < frame_h * SCOREBUG_TOP_RATIO and box_h < frame_h * 0.28:
        return True
    # 모서리 스코어버그는 낮고 좁다. 키가 큰 선수 박스는 남긴다.
    corner = x1 < frame_w * 0.32 or x2 > frame_w * 0.68
    if y1 < frame_h * 0.16 and box_h < frame_h * 0.18 and box_w < frame_w * 0.35 and corner:
        return True
    if y1 > frame_h * 0.90 and box_h < frame_h * 0.12 and box_w > frame_w * 0.45:
        return True
    return False


def is_sharp(bgr: np.ndarray, min_variance: float) -> bool:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) >= min_variance


def preprocess_for_ocr(bgr: np.ndarray) -> np.ndarray | None:
    """작은 등번호만 확대하고 CLAHE로 대비를 올린다. 마지막 블러는 숫자 획을 지워서 넣지 않는다."""
    height, width = bgr.shape[:2]
    if height < 20 or width < 12:
        return None
    scale = 3.0 if height < 140 else max(1.0, 160.0 / height)
    if scale > 1.05:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def upscale_color(bgr: np.ndarray) -> np.ndarray:
    height = bgr.shape[0]
    if height >= 140:
        return bgr
    return cv2.resize(bgr, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)


def color_for_ocr(bgr: np.ndarray) -> np.ndarray:
    """등판이 작을수록 컬러 상태로 더 키운다. 56px 등번호는 12배에서 읽힌다."""
    height = bgr.shape[0]
    if height < 70:
        scale = 12.0
    elif height < 110:
        scale = 8.0
    elif height < 220:
        scale = 3.0
    else:
        return bgr
    return cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


@dataclass
class ColorSignature:
    hist: np.ndarray
    mean_v: float
    mean_bgr: np.ndarray


def _jersey_masks(bgr: np.ndarray) -> tuple[float, float, float] | None:
    """어깨의 빨강 비율, 몸통의 흰색 비율, 몸통 밝기를 반환한다."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    height, width = hsv.shape[:2]
    if height < 12 or width < 8:
        return None
    shoulders = hsv[: max(4, int(height * 0.50)), :]
    torso = hsv[int(height * 0.25) : int(height * 0.80), int(width * 0.15) : int(width * 0.85)]
    if torso.size == 0 or shoulders.size == 0:
        return None
    red = ((shoulders[:, :, 0] <= 12) | (shoulders[:, :, 0] >= 168)) & (shoulders[:, :, 1] > 80) & (
        shoulders[:, :, 2] > 50
    )
    white = (torso[:, :, 1] < 80) & (torso[:, :, 2] > 130)
    return float(red.mean()), float(white.mean()), float(torso[:, :, 2].mean())


def is_red_white_jersey(bgr: np.ndarray | None) -> bool:
    """루비덕스 계열. 흰 몸통에 빨간 어깨다. 검은 유니폼 26번은 여기서 탈락한다."""
    if bgr is None or bgr.size == 0:
        return False
    measured = _jersey_masks(bgr)
    if measured is None:
        return False
    red_ratio, white_ratio, mean_v = measured
    # 원거리 등판은 짙은 빨간 숫자가 흰 몸통 밝기를 95 근처까지 끌어내린다.
    return red_ratio >= 0.08 and white_ratio >= 0.15 and mean_v >= 92.0


def is_dark_jersey(bgr: np.ndarray | None) -> bool:
    if bgr is None or bgr.size == 0:
        return False
    measured = _jersey_masks(bgr)
    if measured is None:
        return False
    red_ratio, _white_ratio, mean_v = measured
    return mean_v < 90.0 and red_ratio < 0.06


def jersey_signature(bgr: np.ndarray) -> ColorSignature | None:
    height, width = bgr.shape[:2]
    if height < 8 or width < 8:
        return None
    roi = bgr[int(height * 0.25) : int(height * 0.80), int(width * 0.20) : int(width * 0.80)]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return ColorSignature(
        hist=hist,
        mean_v=float(hsv[:, :, 2].mean()),
        mean_bgr=roi.reshape(-1, 3).mean(axis=0).astype(np.float32),
    )


def color_distance(a: ColorSignature, b: ColorSignature) -> float:
    hist_d = float(cv2.compareHist(a.hist, b.hist, cv2.HISTCMP_BHATTACHARYYA))
    value_d = abs(a.mean_v - b.mean_v) / 255.0
    return 0.75 * hist_d + 0.25 * value_d


def blend_signature(base: ColorSignature, new: ColorSignature, alpha: float = 0.15) -> ColorSignature:
    hist = (1.0 - alpha) * base.hist + alpha * new.hist
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return ColorSignature(
        hist=hist,
        mean_v=(1.0 - alpha) * base.mean_v + alpha * new.mean_v,
        mean_bgr=(1.0 - alpha) * base.mean_bgr + alpha * new.mean_bgr,
    )


@dataclass
class LogicalTrack:
    logical_id: int
    samples: list[float] = field(default_factory=list)
    votes: dict[str, int] = field(default_factory=dict)
    target_vote_times: list[float] = field(default_factory=list)
    confirmed: bool = False
    rejected: bool = False
    last_ocr_time: float = -1.0e9
    last_seen: float = 0.0
    last_embed_time: float = -1.0e9
    center: np.ndarray | None = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    embed: np.ndarray | None = None
    signature: ColorSignature | None = None
    last_track_id: int | None = None
    color_misses: int = 0
    name_hits: int = 0
    red_white_hits: int = 0
    text_confirmed: bool = False
    jersey_embed: np.ndarray | None = None
    last_jersey_time: float = -1.0e9
    pending_embed: np.ndarray | None = None
    pending_embed_time: float = -1.0e9
    trail: list[tuple[float, float, float]] = field(default_factory=list)


@dataclass
class Stats:
    frames: int = 0
    detections: int = 0
    trackable: int = 0
    ocr_attempts: int = 0
    blur_rejected: int = 0
    back_rejected: int = 0
    small_rejected: int = 0
    region_rejected: int = 0
    confirmed: int = 0
    rebound: int = 0
    other_team: int = 0
    sleeve_hits: int = 0
    sleeve_attempts: int = 0
    prototype_matches: int = 0
    reads: dict[str, int] = field(default_factory=dict)


class TeamLock:
    def __init__(self, hint_bgr: tuple[int, int, int] | None) -> None:
        self.hint = None if hint_bgr is None else np.array(hint_bgr, dtype=np.float32)
        self.signature: ColorSignature | None = None

    def matches(self, signature: ColorSignature | None) -> bool:
        if signature is None:
            return False
        if self.hint is not None and float(np.linalg.norm(signature.mean_bgr - self.hint)) > 90.0:
            return False
        if self.signature is None:
            return True
        return color_distance(self.signature, signature) <= TEAM_COLOR_MAX_DIST

    def adopt(self, signature: ColorSignature) -> None:
        if self.signature is None:
            self.signature = signature
            return
        self.signature = blend_signature(self.signature, signature)

    def describe(self) -> str:
        if self.signature is None:
            return "아직 없음"
        bgr = self.signature.mean_bgr
        return f"BGR({bgr[0]:.0f}, {bgr[1]:.0f}, {bgr[2]:.0f}), 밝기 {self.signature.mean_v:.0f}"


class AppearanceEncoder:
    """트랙 ID가 바뀐 직후 같은 선수인지 볼 때 쓰는 외형 벡터.

    BoT-SORT 안의 ReID가 가림 구간을 담당하고, 이 벡터는 ID가 끊긴 뒤
    1초 안팎의 재연결에만 쓴다.
    """

    def __init__(self, device: str) -> None:
        self.device = device
        self.net = None
        self.transform = None
        self._load()

    def _load(self) -> None:
        try:
            import torch
            from PIL import Image
            from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

            weights = MobileNet_V3_Small_Weights.DEFAULT
            net = mobilenet_v3_small(weights=weights)
            net.classifier = torch.nn.Identity()
            self.net = net.to(self.device).eval()
            self.transform = weights.transforms()
            self._torch = torch
            self._image = Image
            print("  외형 벡터: MobileNetV3-Small")
        except Exception as exc:
            print(f"  외형 벡터를 불러오지 못했습니다 ({exc}). 색과 위치만으로 재연결합니다.")
            self.net = None

    def embed(self, bgr: np.ndarray) -> np.ndarray | None:
        if self.net is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        batch = self.transform(self._image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with self._torch.inference_mode():
            feat = self.net(batch)
            feat = self._torch.nn.functional.normalize(feat.float(), dim=1)
        return feat.squeeze(0).detach().cpu().numpy()


class ParseqReader:
    def __init__(self, device: str) -> None:
        import torch
        from PIL import Image

        print("  PARSeq 가중치를 받는 중...")
        self.torch = torch
        self.image = Image
        self.device = device
        self.model = torch.hub.load("baudm/parseq", "parseq", pretrained=True, trust_repo=True).eval()
        self.model.to(device)
        self.transform = self._build_transform()

    def _build_transform(self):
        try:
            from strhub.data.module import SceneTextDataModule

            img_size = getattr(getattr(self.model, "hparams", None), "img_size", (32, 128))
            return SceneTextDataModule.get_transform(img_size)
        except Exception:
            from torchvision.transforms import InterpolationMode
            from torchvision.transforms import functional as F

            class _ResizeNorm:
                def __call__(self, image):
                    image = F.resize(image, [32, 128], interpolation=InterpolationMode.BICUBIC)
                    tensor = F.to_tensor(image)
                    return F.normalize(tensor, [0.5], [0.5])

            return _ResizeNorm()

    def read(self, bgr: np.ndarray) -> tuple[str, float]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(self.image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            logits = self.model(tensor)
            pred = logits.softmax(-1)
            labels, confs = self.model.tokenizer.decode(pred)
        text = str(labels[0]).strip()
        conf = confs[0]
        if self.torch.is_tensor(conf):
            values = conf.detach().float().cpu().reshape(-1)
            score = float(values.min()) if values.numel() else 0.0
        else:
            score = float(conf)
        return text, score


class NumberReader:
    def __init__(self, use_cuda: bool, enable_parseq: bool) -> None:
        self.easy = None
        self.parseq = None
        self.disagreements = 0
        self._init_easy(use_cuda)
        if enable_parseq:
            self._init_parseq("cuda" if use_cuda else "cpu")
        if self.easy is None and self.parseq is None:
            raise RuntimeError("등번호 인식기를 불러오지 못했습니다. easyocr 설치를 확인해 주세요.")

    def _init_easy(self, use_cuda: bool) -> None:
        import easyocr

        try:
            self.easy = easyocr.Reader(["en"], gpu=use_cuda, verbose=False)
            print(f"  EasyOCR: {'GPU' if use_cuda else 'CPU'}, 숫자만")
        except Exception as exc:
            if not use_cuda:
                raise
            print(f"  EasyOCR GPU 초기화 실패 ({exc}). CPU로 다시 시도합니다.")
            self.easy = easyocr.Reader(["en"], gpu=False, verbose=False)
            print("  EasyOCR: CPU, 숫자만")

    def _init_parseq(self, device: str) -> None:
        try:
            self.parseq = ParseqReader(device)
            print("  PARSeq: 사용")
        except Exception as exc:
            print(f"  PARSeq를 불러오지 못했습니다 ({exc}). EasyOCR만 사용합니다.")
            self.parseq = None

    def read(self, color_bgr: np.ndarray, gray: np.ndarray | None, color_only: bool = False) -> tuple[str, float] | None:
        accepted: list[tuple[str, float]] = []
        if self.parseq is not None:
            try:
                text, conf = self.parseq.read(upscale_color(color_bgr))
                digits = accept_digits(text, conf, MIN_OCR_CONF)
                if digits:
                    accepted.append((digits, conf))
            except Exception:
                self.parseq = None
        if self.easy is not None:
            # 작은 등판의 짙은 빨간 숫자는 흑백 CLAHE에서 사라지고, 컬러를 키우면 읽힌다.
            color = color_for_ocr(color_bgr)
            raw = self.easy.readtext(color, allowlist="0123456789", detail=1, paragraph=False, min_size=6)
            parsed = parse_easyocr(raw, 0.35)
            if parsed is None and gray is not None and not color_only:
                raw = self.easy.readtext(gray, allowlist="0123456789", detail=1, paragraph=False, min_size=6)
                parsed = parse_easyocr(raw, 0.35)
            if parsed and parsed[0] != TARGET_NUMBER and parsed[1] < MIN_OCR_CONF:
                parsed = None
            if parsed:
                accepted.append(parsed)
        if not accepted:
            return None
        if len(accepted) == 1:
            return accepted[0]
        if accepted[0][0] == accepted[1][0]:
            return accepted[0][0], min(accepted[0][1], accepted[1][1])
        self.disagreements += 1
        return None

    def read_name(self, color_bgr: np.ndarray | None, gray: np.ndarray | None, name: str) -> bool:
        """등번호 위쪽 이름 띠에서 목표 이름을 찾는다. 컬러를 먼저 보고, 없으면 흑백을 본다."""
        if self.easy is None or not name:
            return False
        bands: list[np.ndarray] = []
        if color_bgr is not None and color_bgr.size > 0:
            color = color_for_ocr(color_bgr)
            bands.append(color[: max(8, int(color.shape[0] * 0.42))])
        if gray is not None and gray.size > 0:
            bands.append(gray[: max(8, int(gray.shape[0] * 0.42))])
        target = name.upper()
        for band in bands:
            raw = self.easy.readtext(
                band,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                detail=1,
                paragraph=False,
                min_size=6,
            )
            for _box, text, conf in raw:
                if float(conf) < 0.55:
                    continue
                letters = "".join(ch for ch in str(text).upper() if ch.isalpha())
                if letters == target:
                    return True
        return False

    def read_scorebug(self, bgr: np.ndarray) -> list[str]:
        """피리어드와 경기 시계 알약만 읽는다. 숫자와 st/nd/rd, 콜론만 허용한다."""
        if self.easy is None or bgr.size == 0:
            return []
        enlarged = cv2.resize(bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        try:
            raw = self.easy.readtext(
                enlarged,
                allowlist="0123456789stndrOT|:",
                detail=1,
                paragraph=False,
                min_size=6,
            )
        except Exception:
            return []
        texts: list[str] = []
        for _box, text, conf in raw:
            if float(conf) < 0.30:
                continue
            token = str(text).strip()
            if token:
                texts.append(token)
        return texts


class SamRefiner:
    def __init__(self, device: str) -> None:
        self.predictor = None
        self.frame_token = -1
        self._load(device)

    @classmethod
    def disabled(cls) -> SamRefiner:
        refiner = cls.__new__(cls)
        refiner.predictor = None
        refiner.frame_token = -1
        return refiner

    def _load(self, device: str) -> None:
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            print("  SAM2 미설치. 포즈로 자른 등판만 사용합니다. (pip install sam2)")
            return
        try:
            if hasattr(SAM2ImagePredictor, "from_pretrained"):
                self.predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-small")
            else:
                from sam2.build_sam import build_sam2_hf

                model = build_sam2_hf("facebook/sam2.1-hiera-small", device=device)
                self.predictor = SAM2ImagePredictor(model)
            model_ref = getattr(self.predictor, "model", None)
            if model_ref is not None and hasattr(model_ref, "to"):
                model_ref.to(device)
            print("  SAM2: 등판 마스크 정제 사용")
        except Exception as exc:
            print(f"  SAM2를 불러오지 못했습니다 ({exc}). 포즈 crop만 사용합니다.")
            self.predictor = None

    @property
    def enabled(self) -> bool:
        return self.predictor is not None

    def refine(
        self,
        frame_bgr: np.ndarray,
        box: tuple[float, float, float, float],
        frame_token: int,
    ) -> np.ndarray | None:
        if self.predictor is None:
            return None
        try:
            if self.frame_token != frame_token:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                self.predictor.set_image(rgb)
                self.frame_token = frame_token
            masks, scores, _ = self.predictor.predict(
                box=np.array(box, dtype=np.float32),
                multimask_output=False,
            )
        except Exception as exc:
            name = type(exc).__name__
            if "OutOfMemory" in name:
                self.predictor = None
                print("  GPU 메모리가 부족해 SAM2를 껐습니다.")
                return None
            return None
        if float(scores[0]) < 0.50:
            return None
        mask = np.asarray(masks[0]) > 0.5
        ys, xs = np.where(mask)
        if xs.size < 30:
            return None
        crop = frame_bgr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].copy()
        sub = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        if crop.size == 0 or sub.size == 0:
            return None
        median = np.median(crop[sub], axis=0)
        crop[~sub] = median
        return crop


def write_tracker_yaml(path: Path, sample_fps: float, with_reid: bool) -> None:
    buffer_frames = max(10, int(round(3.0 * sample_fps)))
    reid = "True" if with_reid else "False"
    path.write_text(
        "\n".join(
            [
                "tracker_type: botsort",
                "track_high_thresh: 0.25",
                "track_low_thresh: 0.1",
                "new_track_thresh: 0.30",
                f"track_buffer: {buffer_frames}",
                "match_thresh: 0.8",
                "fuse_score: True",
                "gmc_method: sparseOptFlow",
                "proximity_thresh: 0.5",
                "appearance_thresh: 0.75",
                f"with_reid: {reid}",
                "model: auto",
                "",
            ]
        ),
        encoding="utf-8",
    )


def load_pose_model():
    from ultralytics import YOLO

    errors: list[str] = []
    for name in POSE_WEIGHTS:
        try:
            model = YOLO(name)
            return model, name
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError("포즈 모델을 불러오지 못했습니다.\n" + "\n".join(errors))


def seek_video(cap: cv2.VideoCapture, fps: float, start_sec: float) -> None:
    target = int(round(start_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    landed = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if abs(landed - target) > int(fps * 2):
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
        landed = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    landed_sec = landed / fps if fps else 0.0
    if start_sec > 0 and abs(landed_sec - start_sec) > 3.0:
        print(
            f"경고: 요청 시각은 {format_time(start_sec)}인데 디코더가 "
            f"{format_time(landed_sec)} 근처로 이동했습니다. 키프레임 영향일 수 있습니다."
        )


def iterate_window(cap: cv2.VideoCapture, fps: float, start_sec: float, end_sec: float, sample_fps: float):
    """시작 시각을 첫 프레임의 타임스탬프로 맞추고, 샘플 간격마다만 디코딩한다."""
    ok, first = cap.read()
    if not ok:
        return
    msec = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
    reported_next = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    index_time = max(0, reported_next - 1) / fps
    if 0.05 <= msec <= 8 * 3600:
        origin = msec
    else:
        origin = index_time
    if start_sec > 1.0 and origin + 2.0 < start_sec:
        print(
            f"경고: 시크 이후 첫 프레임이 {format_time(origin)}입니다. "
            f"{format_time(start_sec)}까지 프레임을 건너뛰며 진행합니다."
        )
    period = 1.0 / sample_fps
    next_t = start_sec
    index = 0

    def consider(current: float, frame: np.ndarray | None):
        nonlocal next_t
        if frame is None or current < start_sec or current > end_sec:
            return None
        if current + 1e-4 < next_t:
            return None
        next_t = current + period
        return current, frame

    yielded = consider(origin, first)
    if yielded is not None:
        yield yielded
    index = 1
    while True:
        current = origin + index / fps
        if current > end_sec:
            break
        want = current + 1e-4 >= max(next_t, start_sec)
        if want:
            ok, frame = cap.read()
            if not ok:
                break
            yielded = consider(current, frame)
            if yielded is not None:
                yield yielded
        else:
            if not cap.grab():
                break
        index += 1


def unpack_keypoints(result, index: int, count: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None or keypoints.xy is None or len(keypoints.xy) != count:
        return None, None
    xy = keypoints.xy[index].detach().cpu().numpy()
    conf_tensor = keypoints.conf
    if conf_tensor is None or len(conf_tensor) != count:
        conf = np.ones((xy.shape[0],), dtype=np.float32)
    else:
        conf = conf_tensor[index].detach().cpu().numpy()
    return xy, conf


class PlayerTimelineTracker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.video_path = Path(args.video)
        self.target = str(args.number)
        self.target_name = str(args.name)
        self.start_sec = float(args.start)
        self.end_sec = None if args.end is None else float(args.end)
        self.sample_fps = float(args.sample_fps)
        self.imgsz = int(args.imgsz)
        self.save_crops = bool(args.save_crops)
        self.sample_dt = 1.0 / self.sample_fps
        self.stats = Stats()
        self.logicals: dict[int, LogicalTrack] = {}
        self.track_to_logical: dict[int, int] = {}
        self.next_logical_id = 1
        self.team = TeamLock(args.team_bgr)
        self.allow_oblique = ALLOW_OBLIQUE and not bool(getattr(args, "no_oblique", False))
        self.prototype_embed: np.ndarray | None = None
        self.prototype_signature: ColorSignature | None = None
        self._text_hits: list[tuple[float, str, np.ndarray, ColorSignature | None]] = []
        self.wide_refs: list[tuple[float, np.ndarray]] = []
        self._last_report = self.start_sec
        self._crop_seq = 0
        self._sleeve_left = 0
        self.clock_readings: list[ClockReading] = []
        self.play_sightings: list[float] = []
        self.teammate_sightings: list[float] = []
        self._clock_anchors: list[tuple[float, int, float]] = []
        self._last_clock_ocr = -1.0e9
        self._debug_dir = Path("debug_crops")

        import torch

        self.torch = torch
        self.use_cuda = torch.cuda.is_available()
        self.device = "cuda" if self.use_cuda else "cpu"
        torch.set_grad_enabled(False)
        if self.use_cuda:
            torch.backends.cudnn.benchmark = True

        print("AI 모델과 등번호 인식기를 초기화합니다...")
        self.model, self.model_name = load_pose_model()
        print(f"  검출/자세: {self.model_name} ({self.device}, imgsz={self.imgsz})")
        self.tracker_yaml = Path(__file__).with_name("botsort_hockey.yaml")
        self.with_reid = ENABLE_REID and not args.no_reid
        write_tracker_yaml(self.tracker_yaml, self.sample_fps, self.with_reid)
        print(
            "  추적: BoT-SORT, 카메라 움직임 보정"
            + (", YOLO 특징 ReID" if self.with_reid else "")
        )
        self.reader = NumberReader(self.use_cuda, ENABLE_PARSEQ and not args.no_parseq)
        self.encoder = AppearanceEncoder(self.device)
        if ENABLE_SAM2 and not args.no_sam2:
            self.sam = SamRefiner(self.device)
        else:
            self.sam = SamRefiner.disabled()
            print("  SAM2: 끔")

    def run(self) -> list[tuple[float, float]]:
        if not self.video_path.exists():
            raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {self.video_path}")
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"영상을 열 수 없습니다: {self.video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 1.0:
            fps = 30.0
            print("FPS를 읽지 못해 30으로 가정합니다.")
        self.fps = fps
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if frame_count > 0 else 0.0
        if self.end_sec is None:
            self.end_sec = duration
        elif duration > 0:
            self.end_sec = min(self.end_sec, duration)
        seek_video(cap, fps, self.start_sec)
        print(
            f"\n분석 구간 {format_time(self.start_sec)} ~ {format_time(self.end_sec)} "
            f"(추적 {self.sample_fps:.1f}fps, OCR {OCR_INTERVAL_SEC:.2f}초 간격, "
            f"목표 {self.target_name} 또는 {self.target}번, 빨강+흰색"
            + (", 측면 등판 포함" if self.allow_oblique else "")
            + ")"
        )
        try:
            for current, frame in iterate_window(cap, fps, self.start_sec, self.end_sec, self.sample_fps):
                self._process_frame(frame, current)
                self._report(current)
        finally:
            cap.release()
        self._apply_prototype()
        self._stitch_confirmed()
        return self._intervals()

    def _report(self, current: float) -> None:
        if current - self._last_report < 30.0:
            return
        self._last_report = current
        summary = self.playtime_summary()
        period = summary["current_period"]
        if period is None:
            period_text = "P- 00:00"
        else:
            seconds = next(
                (row["total_sec"] for row in summary["periods"] if row["period"] == period),
                0.0,
            )
            period_text = f"P{period} {format_time(seconds)}"
        print(
            f"  진행 {format_time(current)} / {format_time(self.end_sec)}  "
            f"{period_text}  합계 {format_time(summary['total_sec'])}  "
            f"처리 {self.stats.frames}프레임  확정 {self.stats.confirmed}",
            flush=True,
        )

    def _track(self, frame: np.ndarray):
        kwargs = dict(
            persist=True,
            tracker=str(self.tracker_yaml),
            conf=DET_CONF,
            iou=0.5,
            classes=[0],
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            max_det=40,
        )
        try:
            return self.model.track(frame, **kwargs)
        except Exception:
            if not self.with_reid:
                raise
            print("  ReID 추적 초기화에 실패해 모션 추적만 사용합니다.")
            self.with_reid = False
            write_tracker_yaml(self.tracker_yaml, self.sample_fps, False)
            self.model.predictor = None
            return self.model.track(frame, **kwargs)

    def _read_scorebug(self, frame: np.ndarray, current: float) -> None:
        if current - self._last_clock_ocr < OCR_INTERVAL_SEC:
            return
        self._last_clock_ocr = current
        crop = scorebug_clock_crop(frame)
        if crop is None:
            return
        reading = choose_scorebug(
            self.reader.read_scorebug(crop),
            self.clock_readings,
            current,
            self._clock_anchors,
        )
        if reading is not None:
            self.clock_readings.append(reading)

    def _note_play_sighting(self, current: float) -> None:
        if self.play_sightings and current - self.play_sightings[-1] < 0.15:
            return
        self.play_sightings.append(current)

    def _note_teammate(self, current: float) -> None:
        if self.teammate_sightings and current - self.teammate_sightings[-1] < 0.15:
            return
        self.teammate_sightings.append(current)

    def _detected_times(self) -> list[float]:
        times: list[float] = []
        for state in self.logicals.values():
            if state.confirmed:
                times.extend(state.samples)
        return times or list(self.play_sightings)

    def playtime_summary(self) -> dict:
        return build_playtime(self.clock_readings, self._detected_times(), self.teammate_sightings)

    def _process_frame(self, frame: np.ndarray, current: float) -> None:
        self.stats.frames += 1
        self._read_scorebug(frame, current)
        results = self._track(frame)
        if not results:
            return
        result = results[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            return
        frame_h, frame_w = frame.shape[:2]
        min_track_h = max(36.0, frame_h * MIN_TRACK_HEIGHT_RATIO)
        min_ocr_h = max(48.0, frame_h * MIN_OCR_HEIGHT_RATIO)
        ids = boxes.id.detach().cpu().numpy().astype(int)
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        count = len(ids)
        active_ids = {int(track_id) for track_id in ids}
        self.stats.detections += count

        candidates: list[dict] = []
        for index in range(count):
            track_id = int(ids[index])
            person = tuple(float(v) for v in xyxy[index])
            det_conf = float(confs[index])
            box_h = person[3] - person[1]
            box_w = person[2] - person[0]
            aspect = box_h / max(box_w, 1.0)
            xy, kp_conf = unpack_keypoints(result, index, count)
            region_bad = in_banned_region(person, frame_w, frame_h)
            if region_bad:
                self.stats.region_rejected += 1
            too_small = box_h < min_track_h
            if too_small:
                self.stats.small_rejected += 1
            trackable = (
                det_conf >= TRACK_DET_CONF
                and not region_bad
                and not too_small
                and 0.70 <= aspect <= 6.0
            )
            if not trackable:
                state = self.logicals.get(self.track_to_logical.get(track_id, -1))
                if state is not None and not region_bad:
                    state.last_seen = current
                continue

            center = np.array([(person[0] + person[2]) * 0.5, (person[1] + person[3]) * 0.5], dtype=np.float32)
            torso = torso_box(person, xy, kp_conf, frame.shape)
            crop = slice_box(frame, torso) if torso is not None else None
            state = self._touch_track(track_id, current, center, box_h, crop, active_ids)
            if state.rejected:
                continue
            if state.confirmed and is_dark_jersey(crop):
                continue
            self.stats.trackable += 1
            state.samples.append(current)
            state.trail.append((current, float(center[0]), float(center[1])))
            showing_back = back_visible(xy, kp_conf)
            oblique = self.allow_oblique and not showing_back and oblique_back(xy, kp_conf)
            torso_text = showing_back or oblique
            red_white = crop is not None and is_red_white_jersey(crop)
            if red_white:
                state.red_white_hits += 1
                self._remember_jersey(state, crop, current)
                if not state.confirmed:
                    self._match_prototype(state, current)
            if det_conf >= OCR_DET_CONF and box_h >= min_ocr_h and not torso_text:
                self.stats.back_rejected += 1
            readable = (
                not state.confirmed
                and red_white
                and det_conf >= OCR_DET_CONF
                and box_h >= min_ocr_h
                and crop is not None
                and torso is not None
                and current - state.last_ocr_time >= OCR_INTERVAL_SEC
                and (torso_text or xy is not None)
            )
            if readable:
                candidates.append(
                    {
                        "state": state,
                        "torso": torso,
                        "crop": crop,
                        "area": box_w * box_h,
                        "torso_text": torso_text,
                        "xy": xy,
                        "kp_conf": kp_conf,
                    }
                )

        candidates.sort(key=lambda item: item["area"], reverse=True)
        self._sleeve_left = 2
        for item in candidates[:MAX_OCR_PER_FRAME]:
            self._read_candidate(frame, current, item)

    def _touch_track(
        self,
        track_id: int,
        current: float,
        center: np.ndarray,
        box_h: float,
        crop: np.ndarray | None,
        active_ids: set[int],
    ) -> LogicalTrack:
        known_id = self.track_to_logical.get(track_id)
        if known_id is not None:
            state = self.logicals[known_id]
        else:
            signature = jersey_signature(crop) if crop is not None else None
            embed = self.encoder.embed(crop) if crop is not None else None
            rebound = self._rebind(current, center, box_h, signature, embed, active_ids)
            if rebound is None:
                state = LogicalTrack(logical_id=self.next_logical_id)
                self.logicals[self.next_logical_id] = state
                self.next_logical_id += 1
            else:
                state = rebound
                self.stats.rebound += 1
                previous_id = state.last_track_id
                if previous_id is not None and previous_id != track_id:
                    if self.track_to_logical.get(previous_id) == state.logical_id:
                        self.track_to_logical.pop(previous_id, None)
            self.track_to_logical[track_id] = state.logical_id
            state.last_track_id = track_id
            if signature is not None:
                state.signature = (
                    signature if state.signature is None else blend_signature(state.signature, signature, 0.25)
                )
            if embed is not None:
                state.pending_embed = embed
                state.pending_embed_time = current
                if state.embed is None:
                    state.embed = embed
                else:
                    mixed = 0.8 * state.embed + 0.2 * embed
                    norm = float(np.linalg.norm(mixed))
                    state.embed = mixed / max(norm, 1e-6)
                state.last_embed_time = current

        if state.center is not None:
            dt = current - state.last_seen
            if 0.0 < dt <= 1.0:
                instant = (center - state.center) / dt
                state.velocity = 0.6 * state.velocity + 0.4 * instant
        state.center = center
        state.last_seen = current
        if state.rejected:
            return state
        if crop is not None and current - state.last_embed_time >= 0.40:
            signature = jersey_signature(crop)
            embed = self.encoder.embed(crop)
            if signature is not None:
                state.signature = (
                    signature if state.signature is None else blend_signature(state.signature, signature, 0.25)
                )
            if embed is not None:
                state.pending_embed = embed
                state.pending_embed_time = current
                if state.embed is None:
                    state.embed = embed
                else:
                    mixed = 0.8 * state.embed + 0.2 * embed
                    norm = float(np.linalg.norm(mixed))
                    state.embed = mixed / max(norm, 1e-6)
                state.last_embed_time = current
        return state

    def _rebind(
        self,
        current: float,
        center: np.ndarray,
        box_h: float,
        signature: ColorSignature | None,
        embed: np.ndarray | None,
        active_ids: set[int],
    ) -> LogicalTrack | None:
        # 기준 외형이 생기기 전에는 끊긴 ID를 붙이지 않는다.
        if self.prototype_embed is None or embed is None:
            return None
        best: LogicalTrack | None = None
        best_score = -1.0e9
        for state in self.logicals.values():
            if not state.confirmed or not state.text_confirmed or state.center is None:
                continue
            if state.last_track_id in active_ids:
                continue
            gap = current - state.last_seen
            if gap < self.sample_dt * 0.5 or gap > RECONNECT_SEC:
                continue
            predicted = state.center + state.velocity * gap
            dist = float(np.linalg.norm(predicted - center))
            limit = max(120.0, 2.5 * box_h)
            if dist > limit:
                continue
            if signature is not None and state.signature is not None:
                if color_distance(signature, state.signature) > 0.42:
                    continue
            if state.jersey_embed is not None and float(np.dot(embed, state.jersey_embed)) < STITCH_MATCH:
                continue
            elif state.jersey_embed is None:
                continue
            score = float(np.dot(embed, state.jersey_embed)) - dist / limit
            if score > best_score:
                best_score = score
                best = state
        return best

    def _read_candidate(self, frame: np.ndarray, current: float, item: dict) -> None:
        state: LogicalTrack = item["state"]
        crop = item["crop"]
        sharp = is_sharp(crop, BLUR_MIN_VARIANCE)
        if not sharp:
            self.stats.blur_rejected += 1
        torso_crop = crop
        if sharp and self.sam.enabled:
            refined = self.sam.refine(frame, item["torso"], self.stats.frames)
            if refined is not None and refined.shape[0] >= 20 and refined.shape[1] >= 12:
                torso_crop = refined
        gray = preprocess_for_ocr(torso_crop) if sharp else None
        named = False
        read: tuple[str, float] | None = None
        did_read = False
        if sharp and item["torso_text"] and gray is not None:
            did_read = True
            if torso_crop.shape[0] >= 80:
                named = bool(self.reader.read_name(torso_crop, gray, self.target_name))
            read = self.reader.read(torso_crop, gray)
        if named:
            state.name_hits += 1
            self._note_text_hit("name", torso_crop, current)
        number = None
        conf = 0.0
        if read is not None:
            number, conf = read
            if REQUIRE_RED_WHITE and number == self.target and not is_red_white_jersey(torso_crop):
                self.stats.other_team += 1
                number = None
            else:
                self.stats.reads[number] = self.stats.reads.get(number, 0) + 1
                signature = jersey_signature(torso_crop)
                if signature is not None:
                    state.signature = (
                        signature if state.signature is None else blend_signature(state.signature, signature, 0.2)
                    )
                self._vote(state, number, current, conf)
                if number == self.target and conf >= 0.50:
                    self._note_text_hit("number", torso_crop, current)
                elif number != self.target and conf >= MIN_OCR_CONF and is_red_white_jersey(torso_crop):
                    self._note_teammate(current)

        sleeve_number = None
        back_missed = read is None or number != self.target or not item["torso_text"]
        if not state.confirmed and back_missed:
            sleeve_number, sleeve_conf, sleeve_ran = self._read_sleeves(frame, item)
            if sleeve_ran:
                did_read = True
            if sleeve_number == self.target:
                self.stats.sleeve_hits += 1
                self.stats.reads[sleeve_number] = self.stats.reads.get(sleeve_number, 0) + 1
                self._vote(state, sleeve_number, current, sleeve_conf)
                self._note_text_hit("number", item["crop"], current)
        if did_read:
            state.last_ocr_time = current
            self.stats.ocr_attempts += 1
            if self.save_crops and sharp:
                self._save_crop(torso_crop, current, state.logical_id)

        if named and not state.confirmed:
            self._confirm(state, current, f"이름 {self.target_name}", text=True)

        saw_number = (number == self.target and conf >= TARGET_MIN_CONF) or sleeve_number == self.target
        if named and saw_number:
            self._adopt_prototype(torso_crop, current)
        if named or saw_number:
            self._note_play_sighting(current)

    def _read_sleeves(self, frame: np.ndarray, item: dict) -> tuple[str | None, float, bool]:
        crop = item["crop"]
        if crop.shape[0] < 70 or self._sleeve_left <= 0:
            return None, 0.0, False
        if not ENABLE_SLEEVE_OCR:
            if not sleeve_geometry_ok(item.get("xy"), item.get("kp_conf")):
                return None, 0.0, False
            self._sleeve_left -= 1
            return None, 0.0, True
        strips = sleeve_strips(frame, item.get("xy"), item.get("kp_conf"))
        if not strips:
            return None, 0.0, False
        for strip in strips[:1]:
            self._sleeve_left -= 1
            self.stats.sleeve_attempts += 1
            parsed = self.reader.read(strip, None, color_only=True)
            if parsed is None and self._sleeve_left > 0:
                self._sleeve_left -= 1
                self.stats.sleeve_attempts += 1
                flipped = cv2.rotate(strip, cv2.ROTATE_180)
                parsed = self.reader.read(flipped, None, color_only=True)
            if parsed is None:
                return None, 0.0, True
            number, conf = parsed
            if number == self.target and conf >= TARGET_MIN_CONF:
                return number, conf, True
        return None, 0.0, True

    def _remember_jersey(self, state: LogicalTrack, crop: np.ndarray, current: float) -> None:
        if state.jersey_embed is not None and current - state.last_jersey_time < 0.40:
            return
        embed = state.pending_embed if state.pending_embed_time == current else self.encoder.embed(crop)
        signature = jersey_signature(crop)
        state.last_jersey_time = current
        if signature is not None:
            state.signature = signature if state.signature is None else blend_signature(state.signature, signature, 0.25)
        if embed is None:
            return
        if state.jersey_embed is None:
            state.jersey_embed = embed
            return
        mixed = 0.8 * state.jersey_embed + 0.2 * embed
        norm = float(np.linalg.norm(mixed))
        state.jersey_embed = mixed / max(norm, 1e-6)

    def _prototype_cosine(self, state: LogicalTrack) -> float | None:
        if self.prototype_embed is None or state.jersey_embed is None:
            return None
        cosine = float(np.dot(state.jersey_embed, self.prototype_embed))
        if cosine < PROTO_MATCH:
            return None
        if self.prototype_signature is not None and state.signature is not None:
            if color_distance(state.signature, self.prototype_signature) > 0.55:
                return None
        return cosine

    def _number_conflict(self, state: LogicalTrack) -> bool:
        target_votes = state.votes.get(self.target, 0)
        other_votes = sum(count for num, count in state.votes.items() if num != self.target)
        return other_votes >= 2 and other_votes > target_votes

    def _match_prototype(self, state: LogicalTrack, current: float) -> None:
        if state.confirmed or state.red_white_hits < 1 or self._number_conflict(state):
            return
        cosine = self._prototype_cosine(state)
        if cosine is None:
            return
        self.stats.prototype_matches += 1
        self._confirm(state, current, f"외형 {cosine:.2f}")

    def _endpoint_gap(
        self,
        fragment: LogicalTrack,
        anchor: LogicalTrack,
    ) -> tuple[float, float] | None:
        if not fragment.trail or not anchor.trail:
            return None
        early_f = min(fragment.trail, key=lambda point: point[0])
        late_f = max(fragment.trail, key=lambda point: point[0])
        early_a = min(anchor.trail, key=lambda point: point[0])
        late_a = max(anchor.trail, key=lambda point: point[0])
        if late_f[0] < early_a[0]:
            gap = early_a[0] - late_f[0]
            dist = float(np.hypot(late_f[1] - early_a[1], late_f[2] - early_a[2]))
            return gap, dist
        if late_a[0] < early_f[0]:
            gap = early_f[0] - late_a[0]
            dist = float(np.hypot(early_f[1] - late_a[1], early_f[2] - late_a[2]))
            return gap, dist
        best = 1.0e9
        for time_f, x_f, y_f in fragment.trail:
            for time_a, x_a, y_a in anchor.trail:
                if abs(time_f - time_a) > 0.45:
                    continue
                best = min(best, float(np.hypot(x_f - x_a, y_f - y_a)))
        if best > 1.0e8:
            best = float(np.hypot(late_f[1] - late_a[1], late_f[2] - late_a[2]))
        return 0.0, best

    def _wide_hit(self, embed: np.ndarray) -> tuple[float, float] | None:
        best_cos = -1.0
        best_time = 0.0
        for seen, ref in self.wide_refs:
            cosine = float(np.dot(embed, ref))
            if cosine > best_cos:
                best_cos = cosine
                best_time = seen
        if best_cos < 0.0:
            return None
        return best_cos, best_time

    def _stitch_confirmed(self) -> None:
        """번호가 읽힌 원거리 외형, 그리고 이어지는 위치와 외형으로 번호 없는 구간을 채운다."""
        anchors = [state for state in self.logicals.values() if state.text_confirmed and state.trail and state.jersey_embed is not None]
        if not anchors and not self.wide_refs:
            return
        if self.wide_refs:
            times = ", ".join(format_time(seen) for seen, _ref in self.wide_refs[:6])
            print(f"  원거리 외형 {len(self.wide_refs)}장 @ {times}", flush=True)
        linked = 0
        wide_linked = 0
        for _pass in range(FOLLOW_PASSES):
            gained = 0
            for state in self.logicals.values():
                if state.confirmed or state.rejected or state.text_confirmed:
                    continue
                if state.red_white_hits < 2 or state.jersey_embed is None or not state.trail:
                    continue
                if self._number_conflict(state):
                    continue
                mid = (min(point[0] for point in state.trail) + max(point[0] for point in state.trail)) * 0.5
                wide = self._wide_hit(state.jersey_embed)
                if wide is not None and wide[0] >= WIDE_MATCH and abs(mid - wide[1]) <= 30.0:
                    state.confirmed = True
                    self.stats.prototype_matches += 1
                    wide_linked += 1
                    gained += 1
                    print(
                        f"  원거리 외형 확정: logical {state.logical_id} @ {format_time(mid)} ({wide[0]:.2f})",
                        flush=True,
                    )
                    continue
                best: LogicalTrack | None = None
                best_gap = 1.0e9
                for anchor in anchors:
                    measured = self._endpoint_gap(state, anchor)
                    if measured is None:
                        continue
                    gap, dist = measured
                    if gap > FOLLOW_HOP_SEC:
                        continue
                    limit = 160.0 if gap <= 0.2 else 140.0 + 80.0 * gap
                    if dist > limit:
                        continue
                    if float(np.dot(state.jersey_embed, anchor.jersey_embed)) < STITCH_MATCH:
                        continue
                    if gap < best_gap:
                        best_gap = gap
                        best = anchor
                if best is None:
                    continue
                best.samples.extend(state.samples)
                best.trail.extend(state.trail)
                state.rejected = True
                linked += 1
                gained += 1
                self.stats.rebound += 1
            if gained == 0:
                break
        if linked or wide_linked:
            print(f"  번호 없이 이어 붙임: 위치 {linked}개, 원거리 외형 {wide_linked}개", flush=True)

    def _note_text_hit(self, kind: str, crop: np.ndarray, current: float) -> None:
        """이름과 번호가 2초 안에 같이 보이면, 번호가 읽힌 등판을 기준 외형으로 둔다."""
        embed = self.encoder.embed(crop)
        if embed is None:
            return
        signature = jersey_signature(crop)
        if kind == "number" and crop.shape[0] < 120:
            self.wide_refs.append((current, embed.copy()))
        self._text_hits.append((current, kind, embed, signature))
        self._text_hits = [hit for hit in self._text_hits if current - hit[0] <= 2.0]
        if self.prototype_embed is not None:
            return
        numbers = [hit for hit in self._text_hits if hit[1] == "number"]
        names = [hit for hit in self._text_hits if hit[1] == "name"]
        if not numbers or not names:
            return
        _seen, _kind, chosen, chosen_sig = numbers[-1]
        self.prototype_embed = chosen
        self.prototype_signature = chosen_sig
        print(f"  기준 외형 저장 @ {format_time(current)}: {self.target_name}과 {self.target}이 같이 보임", flush=True)
        if chosen_sig is not None and chosen_sig.mean_v >= 100:
            locked_before = self.team.signature is not None or self.team.hint is not None
            self.team.adopt(chosen_sig)
            if not locked_before and self.team.hint is None:
                print(f"  팀 색 고정: {self.team.describe()}", flush=True)

    def _apply_prototype(self) -> None:
        """영상 앞부분에서 번호가 안 읽힌 빨강+흰색 트랙을, 나중에 생긴 기준 외형으로 확정한다."""
        if self.prototype_embed is None:
            print("  기준 외형 없음: 이름과 번호가 같은 장면에서 같이 보이지 않았습니다.", flush=True)
            return
        ranked: list[tuple[float, LogicalTrack]] = []
        for state in self.logicals.values():
            if state.jersey_embed is None or state.red_white_hits < 3:
                continue
            ranked.append((float(np.dot(state.jersey_embed, self.prototype_embed)), state))
        if ranked:
            ranked.sort(key=lambda item: item[0], reverse=True)
            above = {gate: sum(1 for cos, _state in ranked if cos >= gate) for gate in (0.75, 0.84, 0.90)}
            top = ", ".join(f"{cos:.2f}" for cos, _state in ranked[:8])
            print(f"  외형 유사 트랙 0.75/0.84/0.90: {above[0.75]}/{above[0.84]}/{above[0.90]}  상위 {top}", flush=True)
        for state in self.logicals.values():
            if state.confirmed or state.rejected or state.red_white_hits < 3 or self._number_conflict(state):
                continue
            cosine = self._prototype_cosine(state)
            if cosine is None:
                continue
            self.stats.prototype_matches += 1
            self._confirm(state, state.last_seen, f"외형 {cosine:.2f}")

    def _adopt_prototype(self, crop: np.ndarray, current: float) -> None:
        embed = self.encoder.embed(crop)
        if embed is None:
            return
        if self.prototype_embed is not None:
            return
        signature = jersey_signature(crop)
        self.prototype_embed = embed
        self.prototype_signature = signature
        print(f"  기준 외형 저장 @ {format_time(current)}: {self.target_name}과 {self.target}이 같이 보임", flush=True)
        if signature is not None and signature.mean_v >= 100:
            locked_before = self.team.signature is not None or self.team.hint is not None
            self.team.adopt(signature)
            if not locked_before and self.team.hint is None:
                print(f"  팀 색 고정: {self.team.describe()}", flush=True)

    def _confirm(self, state: LogicalTrack, current: float, reason: str, text: bool = False) -> None:
        if state.confirmed or state.rejected:
            return
        state.confirmed = True
        if text:
            state.text_confirmed = True
        self.stats.confirmed += 1
        target_votes = state.votes.get(self.target, 0)
        other_votes = sum(count for num, count in state.votes.items() if num != self.target)
        print(
            f"  확정: logical {state.logical_id} @ {format_time(current)} ({reason}) "
            f"이름 {state.name_hits}회, {self.target} 득표 {target_votes} (다른 번호 {other_votes})",
            flush=True,
        )

    def _vote(self, state: LogicalTrack, number: str, current: float, conf: float = 0.0) -> None:
        if state.confirmed:
            return
        state.votes[number] = state.votes.get(number, 0) + 1
        if number != self.target or conf < TARGET_MIN_CONF:
            return
        state.target_vote_times.append(current)
        target_votes = state.votes.get(self.target, 0)
        other_votes = sum(count for num, count in state.votes.items() if num != self.target)
        # 빨강+흰색에서 26 한 번이면 확정한다. 다른 번호가 더 많으면 보류한다.
        if target_votes > other_votes or other_votes == 0:
            self._confirm(state, current, f"번호 {number} {conf:.2f}", text=True)

    def _save_crop(self, crop: np.ndarray, current: float, logical_id: int) -> None:
        if self._crop_seq >= 40:
            return
        self._debug_dir.mkdir(exist_ok=True)
        name = f"{self._crop_seq:03d}_{format_time(current).replace(':', '')}_L{logical_id}.jpg"
        cv2.imwrite(str(self._debug_dir / name), crop)
        self._crop_seq += 1

    def _intervals(self) -> list[tuple[float, float]]:
        times: list[float] = []
        for state in self.logicals.values():
            if state.confirmed:
                times.extend(state.samples)
        return merge_intervals(times, self.sample_dt, MERGE_GAP_SEC, self.end_sec)

    def save(self, intervals: list[tuple[float, float]]) -> tuple[Path, Path]:
        stem = f"{self.video_path.stem}_player_{self.target}_timeline"
        out_dir = self.video_path.parent if self.video_path.parent.as_posix() not in ("", ".") else Path.cwd()
        json_path = out_dir / f"{stem}.json"
        csv_path = out_dir / f"{stem}.csv"
        payload = {
            "target_number": self.target,
            "target_name": self.target_name,
            "video": str(self.video_path),
            "model": self.model_name,
            "window": {"start_sec": self.start_sec, "end_sec": self.end_sec},
            "sample_fps": self.sample_fps,
            "team_color": self.team.describe(),
            "intervals": [
                {
                    "start_sec": round(start, 3),
                    "end_sec": round(end, 3),
                    "start": format_time(start),
                    "end": format_time(end),
                    "duration_sec": round(end - start, 3),
                }
                for start, end in intervals
            ],
            "total_duration_sec": round(sum(end - start for start, end in intervals), 3),
            "playtime": self._playtime_payload(),
            "diagnostics": {
                "frames": self.stats.frames,
                "detections": self.stats.detections,
                "trackable": self.stats.trackable,
                "ocr_attempts": self.stats.ocr_attempts,
                "blur_rejected": self.stats.blur_rejected,
                "back_rejected": self.stats.back_rejected,
                "small_rejected": self.stats.small_rejected,
                "region_rejected": self.stats.region_rejected,
                "reader_disagreements": self.reader.disagreements,
                "confirmed_tracks": self.stats.confirmed,
                "rebinds": self.stats.rebound,
                "other_team_rejected": self.stats.other_team,
                "sleeve_hits": self.stats.sleeve_hits,
                "sleeve_attempts": self.stats.sleeve_attempts,
                "prototype_matches": self.stats.prototype_matches,
                "digit_reads": dict(sorted(self.stats.reads.items(), key=lambda item: item[1], reverse=True)),
            },
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["start_sec", "end_sec", "start_clock", "end_clock", "duration_sec"])
            for start, end in intervals:
                writer.writerow([f"{start:.3f}", f"{end:.3f}", format_time(start), format_time(end), f"{end - start:.3f}"])
        return json_path, csv_path

    def _playtime_payload(self) -> dict:
        summary = self.playtime_summary()
        return {
            "total_sec": summary["total_sec"],
            "periods": summary["periods"],
        }


def print_summary(tracker: PlayerTimelineTracker, intervals: list[tuple[float, float]], json_path: Path, csv_path: Path) -> None:
    print(f"\n{tracker.target_name} {tracker.target}")
    playtime = tracker.playtime_summary()
    shown = [row for row in playtime["periods"] if row["total_sec"] > 0 or row["shifts"]]
    print()
    if not shown:
        print("플레이타임을 계산하지 못했습니다.")
    for index, row in enumerate(shown):
        if index:
            print()
        print(f"{format_period_label(row['period'])} - {format_play_length(row['total_sec'])}")
        for shift in row["shifts"]:
            print(
                f"{shift['start_clock']}~{shift['end_clock']} - {format_play_length(shift['duration_sec'])}"
            )

    print(f"\n등번호 {tracker.target} 출연 구간")
    if not intervals:
        print("  확정된 구간이 없습니다.")
    total = 0.0
    for start, end in intervals:
        duration = end - start
        total += duration
        print(f"  {format_time(start)} - {format_time(end)}  ({duration:.1f}초)")
    print(f"구간 {len(intervals)}개, 합계 {format_time(total)} ({total:.1f}초)")
    stats = tracker.stats
    print(
        "진단: "
        f"프레임 {stats.frames}, 검출 {stats.detections}, OCR {stats.ocr_attempts}, "
        f"등 아님 {stats.back_rejected}, 흐림 {stats.blur_rejected}, "
        f"스코어보드 {stats.region_rejected}, 작음 {stats.small_rejected}, "
        f"재연결 {stats.rebound}, 팔뚝 {stats.sleeve_hits}/{stats.sleeve_attempts}, 외형 {stats.prototype_matches}, 상대팀 제외 {stats.other_team}"
    )
    if stats.reads:
        top = ", ".join(f"{num}:{count}" for num, count in list(stats.reads.items())[:8])
        print(f"읽힌 숫자: {top}")
    if not intervals and stats.ocr_attempts == 0 and stats.back_rejected > 20:
        print("등판으로 판정된 선수가 거의 없습니다. 포즈가 정면을 등으로 뒤집으면 등판 조건을 완화해야 합니다.")
    if not intervals and stats.blur_rejected > stats.ocr_attempts and stats.blur_rejected > 20:
        print("흐림 탈락이 많습니다. BLUR_MIN_VARIANCE를 15 정도로 낮춰 보세요.")
    print(f"저장: {json_path}")
    print(f"저장: {csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="아이스하키 영상에서 특정 등번호의 출연 구간을 뽑습니다.")
    parser.add_argument("--video", default=VIDEO_PATH, help="분석할 영상 경로")
    parser.add_argument("--number", default=TARGET_NUMBER, help="찾을 등번호")
    parser.add_argument("--name", default=TARGET_NAME, help="유니폼에 적힌 이름. 예: EAN")
    parser.add_argument("--start", type=float, default=START_TIME_SEC, help="시작 시각(초). 기본은 영상 처음")
    parser.add_argument("--end", type=float, default=END_TIME_SEC, help="끝 시각(초). 기본은 영상 끝")
    parser.add_argument("--sample-fps", type=float, default=SAMPLE_FPS, help="검출/추적 샘플 주기")
    parser.add_argument("--imgsz", type=int, default=IMGSZ, help="YOLO 입력 크기")
    parser.add_argument("--team-bgr", type=parse_bgr, default=TEAM_HINT_BGR, help="목표 팀 유니폼색 B,G,R. 예: 230,230,230")
    parser.add_argument("--no-parseq", action="store_true", help="PARSeq를 쓰지 않고 EasyOCR만 사용")
    parser.add_argument("--no-sam2", action="store_true", help="SAM2 마스크 정제를 끄기")
    parser.add_argument("--no-reid", action="store_true", help="BoT-SORT ReID를 끄기")
    parser.add_argument("--no-oblique", action="store_true", help="비스듬한 등은 등판 OCR에서 제외")
    parser.add_argument("--save-crops", action="store_true", default=SAVE_DEBUG_CROPS, help="OCR에 넣은 등판 crop을 저장")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.end is not None and args.end <= args.start:
        raise SystemExit("끝 시각은 시작 시각보다 커야 합니다.")
    if not str(args.number).isdigit() or len(str(args.number)) != 2:
        raise SystemExit("등번호는 두 자리 숫자여야 합니다. 예: 26")
    args.name = str(args.name).strip().upper()
    if not args.name.isalpha():
        raise SystemExit("이름은 영문이어야 합니다. 예: EAN")
    tracker = PlayerTimelineTracker(args)
    intervals: list[tuple[float, float]] = []
    try:
        intervals = tracker.run()
    except KeyboardInterrupt:
        print("\n중단되었습니다. 지금까지 확정된 구간을 저장합니다.")
        tracker._apply_prototype()
        tracker._stitch_confirmed()
        intervals = tracker._intervals()
    json_path, csv_path = tracker.save(intervals)
    print_summary(tracker, intervals, json_path, csv_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
