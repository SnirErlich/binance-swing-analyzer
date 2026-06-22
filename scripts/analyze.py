#!/usr/bin/env python3
"""
Binance 4H Swing Trade Analyzer v2
8-section output: backtest, leaderboard, current state, volume check,
winner selection, top pick box, backup box, warnings.
"""

import sys
import os
import json
import time
import tempfile
import requests
import numpy as np
from datetime import datetime, timezone

# reconfigure stdout to UTF-8 so box-drawing / emoji chars print on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Config ───────────────────────────────────────────────────────────────────
INTERVAL   = "4h"
LIMIT      = 500
BASE_URL   = "https://api.binance.com/api/v3/klines"
MAX_FWD    = 60   # candles to resolve a trade before timeout
MIN_TRADES = 3    # min trades for a valid leaderboard row


# ── Fetch ────────────────────────────────────────────────────────────────────
def fetch_candles(symbol: str) -> list:
    r = requests.get(BASE_URL,
                     params={"symbol": symbol.upper(), "interval": INTERVAL, "limit": LIMIT},
                     timeout=15)
    r.raise_for_status()
    return [{"ts":     int(c[0]),
             "open":   float(c[1]),
             "high":   float(c[2]),
             "low":    float(c[3]),
             "close":  float(c[4]),
             "volume": float(c[5])} for c in r.json()]


# ── Indicators ───────────────────────────────────────────────────────────────
def _ema(v: np.ndarray, p: int) -> np.ndarray:
    out = np.full_like(v, np.nan)
    k = 2.0 / (p + 1)
    out[p - 1] = np.mean(v[:p])
    for i in range(p, len(v)):
        out[i] = v[i] * k + out[i - 1] * (1 - k)
    return out


def _rsi(closes: np.ndarray, p: int = 14) -> np.ndarray:
    d = np.diff(closes)
    g, l = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    ag = np.full(len(closes), np.nan)
    al = np.full(len(closes), np.nan)
    ag[p] = np.mean(g[:p])
    al[p] = np.mean(l[:p])
    for i in range(p + 1, len(closes)):
        ag[i] = (ag[i - 1] * (p - 1) + g[i - 1]) / p
        al[i] = (al[i - 1] * (p - 1) + l[i - 1]) / p
    rs = np.where(al == 0, np.inf, ag / al)
    return 100.0 - 100.0 / (1 + rs)


def _atr(hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, p: int = 14) -> np.ndarray:
    tr = np.maximum(hi[1:] - lo[1:],
         np.maximum(np.abs(hi[1:] - cl[:-1]),
                    np.abs(lo[1:] - cl[:-1])))
    out = np.full(len(cl), np.nan)
    out[p] = np.mean(tr[:p])
    for i in range(p + 1, len(cl)):
        out[i] = (out[i - 1] * (p - 1) + tr[i - 1]) / p
    return out


def _vol_avg20(vol: np.ndarray) -> np.ndarray:
    out = np.full(len(vol), np.nan)
    for i in range(20, len(vol)):
        out[i] = np.mean(vol[i - 20:i])
    return out


def compute_indicators(candles: list) -> dict:
    cl = np.array([c["close"]  for c in candles])
    hi = np.array([c["high"]   for c in candles])
    lo = np.array([c["low"]    for c in candles])
    vo = np.array([c["volume"] for c in candles])
    return {
        "closes":   cl,
        "highs":    hi,
        "lows":     lo,
        "volumes":  vo,
        "ema20":    _ema(cl, 20),
        "ema50":    _ema(cl, 50),
        "ema200":   _ema(cl, 200),
        "rsi14":    _rsi(cl, 14),
        "atr14":    _atr(hi, lo, cl, 14),
        "vol_avg":  _vol_avg20(vo),
    }


# ── Signal detection ─────────────────────────────────────────────────────────
def check_signals(ind: dict, i: int) -> list:
    """Return list of signal names firing at bar i (needs i >= 1)."""
    if i < 1:
        return []
    cl, lo  = ind["closes"],  ind["lows"]
    r, rp   = ind["rsi14"][i], ind["rsi14"][i - 1]
    e20, e50, e200 = ind["ema20"][i], ind["ema50"][i], ind["ema200"][i]
    e20p, e50p     = ind["ema20"][i - 1], ind["ema50"][i - 1]
    c = cl[i]

    if any(np.isnan(v) for v in [r, rp, e20, e50, e200]):
        return []

    out = []

    # RSI Bounce: RSI crosses up through 35, price above EMA200
    if rp < 35 <= r and c > e200:
        out.append("RSI Bounce")

    # RSI 40 Cross: RSI crosses up through 40, price above EMA200 and EMA20
    if rp < 40 <= r and c > e200 and c > e20:
        out.append("RSI 40 Cross")

    # EMA50 Bounce: price reclaims EMA50 from below, RSI > 40
    if cl[i - 1] < e50p and c >= e50 and r > 40:
        out.append("EMA50 Bounce")

    # EMA Pullback: bull stack + RSI 40-55 + prev low touched EMA20 then recovered
    if c > e20 > e50 > e200 and 40 <= r <= 55 and lo[i - 1] <= e20p:
        out.append("EMA Pullback")

    # RSI Bounce Below EMA200: RSI crosses up through 35, below EMA200 but above EMA50
    if rp < 35 <= r and c < e200 and c > e50:
        out.append("RSI Bounce Below EMA200")

    return out


# ── Backtest engine ───────────────────────────────────────────────────────────
def backtest(ind: dict, candles: list) -> dict:
    cl, hi, lo = ind["closes"], ind["highs"], ind["lows"]
    atr14 = ind["atr14"]
    n = len(cl)
    trades_list = []
    last_i = -1

    for i in range(210, n - 1):
        if i <= last_i + 5:
            continue
        sigs = check_signals(ind, i)
        if not sigs:
            continue

        entry     = float(cl[i])
        swing_low = float(np.min(lo[max(0, i - 5):i + 1]))
        sl        = swing_low - 0.3 * float(atr14[i])
        if np.isnan(sl) or sl >= entry:
            continue
        tp = entry + 2.0 * (entry - sl)

        # resolve forward
        result = "timeout"
        for j in range(i + 1, min(i + MAX_FWD + 1, n)):
            if float(lo[j]) <= sl:
                result = "loss"
                break
            if float(hi[j]) >= tp:
                result = "win"
                break
        if result == "timeout":
            continue

        win_pct  = (tp - entry) / entry * 100.0 if result == "win"  else None
        loss_pct = (entry - sl) / entry * 100.0  if result == "loss" else None

        trades_list.append({
            "i":           i,
            "signal":      sigs[0],
            "all_signals": sigs,
            "entry":       round(entry, 8),
            "sl":          round(sl, 8),
            "tp":          round(tp, 8),
            "result":      result,
            "win_pct":     round(win_pct,  4) if win_pct  is not None else None,
            "loss_pct":    round(loss_pct, 4) if loss_pct is not None else None,
            "dt":          datetime.utcfromtimestamp(candles[i]["ts"] / 1000).strftime("%Y-%m-%d %H:%M"),
        })
        last_i = i

    wins_t   = [t for t in trades_list if t["result"] == "win"]
    losses_t = [t for t in trades_list if t["result"] == "loss"]
    total    = len(trades_list)

    if total == 0:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
                "ev": 0.0, "score": 0.0, "trades_list": []}

    win_rate     = len(wins_t) / total
    avg_win_pct  = float(np.mean([t["win_pct"]  for t in wins_t]))   if wins_t   else 0.0
    avg_loss_pct = float(np.mean([t["loss_pct"] for t in losses_t])) if losses_t else 0.0
    ev           = (len(wins_t) * 2.0 - len(losses_t)) / total

    if len(wins_t) >= 3:
        wp_arr   = [t["win_pct"] for t in wins_t]
        std_win  = float(np.std(wp_arr))
        sharpe   = avg_win_pct / std_win if std_win > 0 else 1.0
    else:
        sharpe = 1.0

    score = ev * win_rate * sharpe

    return {
        "trades":       total,
        "wins":         len(wins_t),
        "losses":       len(losses_t),
        "win_rate":     round(win_rate * 100, 1),
        "avg_win_pct":  round(avg_win_pct, 2),
        "avg_loss_pct": round(avg_loss_pct, 2),
        "ev":           round(ev, 3),
        "score":        round(score, 4),
        "trades_list":  trades_list,
    }


# ── Current state ─────────────────────────────────────────────────────────────
def current_state(symbol: str, candles: list, ind: dict) -> dict:
    cl, hi, lo = ind["closes"], ind["highs"], ind["lows"]
    rsi14, atr14 = ind["rsi14"], ind["atr14"]
    e20, e50, e200 = ind["ema20"], ind["ema50"], ind["ema200"]
    vols, va = ind["volumes"], ind["vol_avg"]

    price    = float(cl[-1])
    rsi_val  = float(rsi14[-1])
    atr_val  = float(atr14[-1])
    curr_vol = float(vols[-1])
    avg_vol  = float(va[-1])
    vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 0.0
    vol_label = ("elevated" if vol_ratio >= 1.2 else
                 "low"       if vol_ratio < 0.5  else "average")

    e20v  = float(e20[-1])
    e50v  = float(e50[-1])
    e200v = float(e200[-1])

    ema_pos = []
    ema_pos.append("above EMA20"  if price > e20v  else "below EMA20")
    ema_pos.append("above EMA50"  if price > e50v  else "below EMA50")
    ema_pos.append("above EMA200" if price > e200v else "below EMA200")

    rsi_trend = [round(float(rsi14[-5 + k]), 1) for k in range(5)]

    swing_low = float(np.min(lo[-5:]))
    sl = swing_low - 0.3 * atr_val
    tp = price + 2.0 * (price - sl)
    sl_pct = (price - sl) / price * 100.0
    tp_pct = (tp - price)  / price * 100.0

    last_ts = candles[-1]["ts"]
    last_dt = datetime.utcfromtimestamp(last_ts / 1000).strftime("%Y-%m-%d %H:%M UTC")

    curr_signals = check_signals(ind, len(cl) - 1)

    return {
        "symbol":       symbol,
        "price":        price,
        "rsi":          round(rsi_val, 1),
        "atr":          round(atr_val, 8),
        "vol_ratio":    round(vol_ratio, 2),
        "vol_label":    vol_label,
        "ema20":        round(e20v,  8),
        "ema50":        round(e50v,  8),
        "ema200":       round(e200v, 8),
        "ema_pos":      ema_pos,
        "rsi_trend":    rsi_trend,
        "sl":           round(sl, 8),
        "tp":           round(tp, 8),
        "sl_pct":       round(sl_pct, 2),
        "tp_pct":       round(tp_pct, 2),
        "last_dt":      last_dt,
        "last_ts_ms":   last_ts,
        "curr_signals": curr_signals,
    }


# ── Formatting helpers ────────────────────────────────────────────────────────
def fmt(p: float) -> str:
    if p >= 1000: return f"{p:.2f}"
    if p >= 10:   return f"{p:.3f}"
    if p >= 1:    return f"{p:.4f}"
    return f"{p:.6f}"

def sep(char="=", w=62):
    return char * w

def hdr(title: str, w=62):
    print(f"\n{sep('=', w)}")
    print(f"  {title}")
    print(sep("=", w))


# ── Section printers ──────────────────────────────────────────────────────────
def print_s1(results: list):
    hdr("SECTION 1 -- BACKTEST ENGINE  (5 signals, 4H, 500 candles)")
    signal_names = [
        "RSI Bounce", "RSI 40 Cross", "EMA50 Bounce",
        "EMA Pullback", "RSI Bounce Below EMA200",
    ]
    for r in results:
        bt  = r["bt"]
        tl  = bt["trades_list"]
        sig_counts = {s: 0 for s in signal_names}
        for t in tl:
            if t["signal"] in sig_counts:
                sig_counts[t["signal"]] += 1
        parts = ", ".join(f"{s}: {sig_counts[s]}" for s in signal_names if sig_counts[s] > 0)
        print(f"  {r['symbol']:<12}  {bt['trades']:>3} trades  "
              f"({bt['wins']}W/{bt['losses']}L)  score={bt['score']:.4f}"
              + (f"  [{parts}]" if parts else "  [no trades]"))


def print_s2(results: list):
    hdr("SECTION 2 -- LEADERBOARD  (sorted by composite score)")
    cols = f"{'Symbol':<12} {'Trades':>6} {'Wins':>5} {'Losses':>7} " \
           f"{'WinRate':>8} {'AvgWin%':>8} {'AvgLoss%':>9} {'EV/Trade':>9} {'Score':>8}"
    print(f"  {cols}")
    print("  " + "-" * (len(cols)))
    for r in results:
        bt = r["bt"]
        print(f"  {r['symbol']:<12} {bt['trades']:>6} {bt['wins']:>5} {bt['losses']:>7} "
              f"{bt['win_rate']:>7.1f}% {bt['avg_win_pct']:>7.2f}% "
              f"{bt['avg_loss_pct']:>8.2f}% {bt['ev']:>+9.3f} {bt['score']:>8.4f}")


def print_s3(results: list):
    hdr("SECTION 3 -- CURRENT STATE  (score order)")
    for r in results:
        cs  = r["cs"]
        rsi_str = " -> ".join(str(v) for v in cs["rsi_trend"])
        ema_str = " | ".join(cs["ema_pos"])
        sigs    = ", ".join(cs["curr_signals"]) if cs["curr_signals"] else "none"
        print(f"\n  {r['symbol']}")
        print(f"    Price       : {fmt(cs['price'])}   RSI: {cs['rsi']}   "
              f"Vol: {cs['vol_ratio']:.2f}x avg -- {cs['vol_label']}")
        print(f"    EMAs        : {ema_str}")
        print(f"    RSI 5-bar   : {rsi_str}")
        print(f"    SL (current): {fmt(cs['sl'])}  (-{cs['sl_pct']:.2f}%)")
        print(f"    TP (current): {fmt(cs['tp'])}  (+{cs['tp_pct']:.2f}%)")
        print(f"    Signal now  : {sigs}")
        print(f"    Last candle : {cs['last_dt']}")


def print_s4(results: list):
    hdr("SECTION 4 -- VOLUME / CANDLE VALIDITY CHECK")
    warned = False
    for r in results:
        cs = r["cs"]
        if cs["vol_ratio"] < 0.3:
            print(f"  WARNING  {r['symbol']} last candle volume is very low "
                  f"({cs['vol_ratio']:.2f}x avg) -- candle may still be forming. "
                  f"Wait for close before entering.")
            warned = True
    if not warned:
        print("  All candles have adequate volume (>= 0.3x avg).")


def setup_quality(cs: dict, bt: dict) -> float:
    """Numeric quality for current setup — used to pick winner vs backup."""
    q = bt["score"]
    if cs["rsi"] > 70:
        q = -abs(q)
    elif cs["rsi"] > 65:
        q *= 0.5
    if cs["vol_ratio"] < 0.3:
        q *= 0.5
    if cs["curr_signals"]:
        q *= 1.25
    return q


def pick_winners(results: list):
    ranked = sorted(range(len(results)),
                    key=lambda i: setup_quality(results[i]["cs"], results[i]["bt"]),
                    reverse=True)
    top_i    = ranked[0] if len(ranked) >= 1 else None
    backup_i = ranked[1] if len(ranked) >= 2 else None
    return top_i, backup_i


def print_s5(top: dict, backup, results: list):
    hdr("SECTION 5 -- WINNER SELECTION")
    cs, bt = top["cs"], top["bt"]

    reason = []
    reason.append(
        f"  {top['symbol']} leads with score {bt['score']:.4f} "
        f"({bt['win_rate']:.1f}% win rate over {bt['trades']} trades, EV {bt['ev']:.3f}R)."
    )

    if cs["curr_signals"]:
        reason.append(
            f"  An active signal is firing right now ({cs['curr_signals'][0]}), "
            f"confirming current entry timing."
        )
    else:
        reason.append(
            f"  No signal is active at the last close -- consider waiting for a fresh trigger "
            f"rather than entering at market."
        )

    if cs["rsi"] > 65:
        reason.append(
            f"  RSI is elevated at {cs['rsi']}, suggesting momentum may be stretched; "
            f"reduce position size accordingly."
        )
    elif cs["rsi"] < 35:
        reason.append(
            f"  RSI at {cs['rsi']} is oversold, increasing the probability of a bounce setup."
        )

    if backup:
        cs2, bt2 = backup["cs"], backup["bt"]
        delta = bt["score"] - bt2["score"]
        reason.append(
            f"  {backup['symbol']} is the backup; it scores {bt2['score']:.4f} "
            f"({delta:+.4f} vs {top['symbol']}) with "
            f"{'weaker backtest results' if bt2['ev'] < bt['ev'] else 'a weaker current setup quality'}."
        )

    for line in reason:
        print(line)


def active_signal(r: dict) -> str:
    sigs = r["cs"]["curr_signals"]
    if sigs:
        return sigs[0]
    tl = r["bt"]["trades_list"]
    if tl:
        return tl[-1]["signal"] + " (last historical)"
    return "No active signal"


def trade_box(r: dict, label: str) -> str:
    cs, bt = r["cs"], r["bt"]
    price   = cs["price"]
    sl, tp  = cs["sl"], cs["tp"]
    sl_pct, tp_pct = cs["sl_pct"], cs["tp_pct"]
    risk = price - sl
    rr   = (tp - price) / risk if risk > 0 else 2.0
    sig  = active_signal(r)
    W = 44
    bar = "═" * W
    return (
        f"\n{bar}\n"
        f"  {label}: {r['symbol']}\n"
        f"{bar}\n"
        f"  ENTRY       : ${fmt(price)}\n"
        f"  STOP LOSS   : ${fmt(sl)}  (-{sl_pct:.2f}%)\n"
        f"  TAKE PROFIT : ${fmt(tp)}  (+{tp_pct:.2f}%)\n"
        f"  R:R         : 1:{rr:.1f}\n"
        f"  Signal      : {sig}\n"
        f"  Backtest    : {bt['win_rate']:.1f}% win rate | {bt['trades']} trades | EV {bt['ev']:.3f}R\n"
        f"{bar}"
    )


def print_s6(top: dict):
    hdr("SECTION 6 -- TOP PICK TRADE SETUP")
    print(trade_box(top, "TOP PICK"))


def print_s7(backup: dict, top: dict):
    hdr("SECTION 7 -- BACKUP TRADE")
    print(trade_box(backup, "BACKUP"))
    delta = top["bt"]["score"] - backup["bt"]["score"]
    reason = ("lower backtest score" if backup["bt"]["score"] < top["bt"]["score"]
              else "weaker current setup quality")
    print(f"\n  {backup['symbol']} ranks below {top['symbol']} due to {reason} "
          f"(score {backup['bt']['score']:.4f} vs {top['bt']['score']:.4f}, "
          f"delta {delta:+.4f}).")


def print_s8(top: dict):
    hdr("SECTION 8 -- WARNINGS")
    cs    = top["cs"]
    found = False

    # candle close within 30 minutes
    next_close_ms = cs["last_ts_ms"] + 4 * 3600 * 1000
    mins_left     = (next_close_ms - time.time() * 1000) / 60_000
    if 0 < mins_left <= 30:
        close_str = datetime.utcfromtimestamp(next_close_ms / 1000).strftime("%H:%M")
        print(f"  Wait for candle close at {close_str} UTC")
        found = True

    # counter-trend
    if "below EMA200" in cs["ema_pos"]:
        print("  Counter-trend trade -- consider 50% position size")
        found = True

    # RSI overbought
    if cs["rsi"] > 70:
        print("  RSI overbought -- disqualified despite high score")
        found = True

    # price extended above EMA20
    if cs["price"] > cs["ema20"] + 2 * cs["atr"]:
        print("  Price extended -- entry is far from EMA20")
        found = True

    if not found:
        print("  No active warnings.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze.py BTCUSDT ETHUSDT ...")
        sys.exit(1)

    pairs   = [s.upper() for s in sys.argv[1:]]
    results = []

    for symbol in pairs:
        print(f"Fetching {symbol} ...")
        try:
            candles = fetch_candles(symbol)
            ind     = compute_indicators(candles)
            bt      = backtest(ind, candles)
            cs      = current_state(symbol, candles, ind)
            results.append({"symbol": symbol, "bt": bt, "cs": cs})
        except Exception as e:
            print(f"  {symbol} failed: {e}")
        time.sleep(0.3)

    if not results:
        print("No results.")
        sys.exit(1)

    results.sort(key=lambda r: r["bt"]["score"], reverse=True)

    print_s1(results)
    print_s2(results)
    print_s3(results)
    print_s4(results)

    top_i, backup_i = pick_winners(results)
    top    = results[top_i]
    backup = results[backup_i] if backup_i is not None else None

    print_s5(top, backup, results)
    print_s6(top)
    if backup:
        print_s7(backup, top)
    print_s8(top)

    # JSON — strip trades_list from output to keep file small
    out = []
    for r in results:
        bt_clean = {k: v for k, v in r["bt"].items() if k != "trades_list"}
        out.append({"symbol": r["symbol"], "backtest": bt_clean, "current_state": r["cs"]})

    out_path = os.path.join(tempfile.gettempdir(), "binance_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
