"""
NIFTY 50 Backtest Dashboard
Uso: streamlit run app.py
"""
import warnings, math
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.stats import norm
import io

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NIFTY 50 Backtest Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ── responsive metrics ── */
[data-testid="stMetric"] {
    background: #1e1e2e;
    border-radius: 10px;
    padding: 12px 16px;
    border: 1px solid #333;
}
[data-testid="stMetricValue"] { font-size: 1.4rem !important; }
[data-testid="stMetricLabel"] { font-size: 0.8rem !important; }
/* mobile: stack sidebar */
@media (max-width: 768px) {
    [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
    section[data-testid="stSidebar"] { min-width: 100% !important; }
}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
DATA_FILE = "NIFTY_50_Historical_28062019_to_28062026.xlsx"
NIFTY_LOT_SIZE = 50

# ── DATA & INDICATORS ─────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)   # refresh every 1 hour
def load_data():
    """
    Primary  : Yahoo Finance (^NSEI) — real data updated daily
    Fallback : local Excel file (if Yahoo is unavailable)
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker("^NSEI")
        yf_df  = ticker.history(period="10y", interval="1d", auto_adjust=True)
        if yf_df is None or len(yf_df) < 100:
            raise ValueError("Not enough data from Yahoo Finance")
        yf_df = yf_df.reset_index()
        yf_df.columns = [c.upper() for c in yf_df.columns]
        # Yahoo returns 'DATE' as timezone-aware; strip timezone
        if hasattr(yf_df["DATE"].dtype, "tz") or str(yf_df["DATE"].dtype) == "datetime64[ns, America/New_York]":
            yf_df["DATE"] = yf_df["DATE"].dt.tz_localize(None)
        yf_df["DATE"] = pd.to_datetime(yf_df["DATE"])
        # Keep only needed columns
        for col in ["DIVIDENDS", "STOCK SPLITS", "CAPITAL GAINS"]:
            if col in yf_df.columns:
                yf_df = yf_df.drop(columns=[col])
        if "VOLUME" not in yf_df.columns:
            yf_df["VOLUME"] = 0
        df = yf_df[["DATE","OPEN","HIGH","LOW","CLOSE","VOLUME"]].copy()
        df = df.sort_values("DATE").reset_index(drop=True)
        df["RETURNS"] = df["CLOSE"].pct_change()
        data_source = "📡 Live data — Yahoo Finance (^NSEI)"
    except Exception as e:
        # Fallback to local Excel
        df = pd.read_excel(DATA_FILE)
        df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]
        df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True)
        df = df.sort_values("DATE").reset_index(drop=True)
        for col in list(df.columns):
            if "SHARES"   in col: df.rename(columns={col: "VOLUME"},   inplace=True)
            if "TURNOVER" in col: df.rename(columns={col: "TURNOVER"}, inplace=True)
        df["RETURNS"] = df["CLOSE"].pct_change()
        data_source = "📁 Local file (Yahoo Finance unavailable)"
    df.attrs["source"] = data_source
    return df

def add_indicators(df, sma_fast, sma_slow, rsi_period, bb_period, bb_std,
                   macd_fast, macd_slow, macd_sig):
    c = df["CLOSE"]
    df = df.copy()
    df["SMA_FAST"] = c.rolling(sma_fast).mean()
    df["SMA_SLOW"] = c.rolling(sma_slow).mean()
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(rsi_period).mean()
    loss  = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    bb_m = c.rolling(bb_period).mean()
    bb_s = c.rolling(bb_period).std()
    df["BB_MID"] = bb_m
    df["BB_UP"]  = bb_m + bb_std * bb_s
    df["BB_LOW"] = bb_m - bb_std * bb_s
    ema_f = c.ewm(span=macd_fast, adjust=False).mean()
    ema_s = c.ewm(span=macd_slow, adjust=False).mean()
    df["MACD"]     = ema_f - ema_s
    df["MACD_SIG"] = df["MACD"].ewm(span=macd_sig, adjust=False).mean()
    hl  = df["HIGH"] - df["LOW"]
    hpc = (df["HIGH"] - df["CLOSE"].shift()).abs()
    lpc = (df["LOW"]  - df["CLOSE"].shift()).abs()
    df["ATR"] = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()
    return df

# ── SIGNALS ───────────────────────────────────────────────────────────────────
def sig_sma(df):
    return (df["SMA_FAST"] > df["SMA_SLOW"]).astype(int).diff().fillna(0).astype(int)

def sig_rsi(df, oversold, overbought):
    s = pd.Series(0, index=df.index)
    s[(df["RSI"] < oversold)   & (df["RSI"].shift() >= oversold)]   =  1
    s[(df["RSI"] > overbought) & (df["RSI"].shift() <= overbought)]  = -1
    return s

def sig_bb(df):
    s = pd.Series(0, index=df.index)
    s[(df["CLOSE"] > df["BB_LOW"]) & (df["CLOSE"].shift() <= df["BB_LOW"])] =  1
    s[(df["CLOSE"] < df["BB_UP"])  & (df["CLOSE"].shift() >= df["BB_UP"])]  = -1
    return s

def sig_macd(df):
    return (df["MACD"] > df["MACD_SIG"]).astype(int).diff().fillna(0).astype(int)

# ── BLACK-SCHOLES ─────────────────────────────────────────────────────────────
def bs_price(S, K, T, r, sigma, opt="call"):
    if T <= 0:
        return max(0.0, (S-K) if opt=="call" else (K-S))
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    if opt == "call":
        return S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
    return K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def atm_strike(spot):
    return round(spot / 50) * 50

# ── BACKTEST ENGINE ───────────────────────────────────────────────────────────
def run_backtest(df, signals, mode, initial_capital, futures_margin,
                 option_iv, option_expiry_days, risk_free_rate,
                 brokerage_pct, slippage_ticks, stop_loss_pct):

    def ep(px, d=1): return px + d * slippage_ticks * 0.05
    def br(n):       return n * brokerage_pct

    cash = float(initial_capital)
    pos = 0; entry_px = 0.0; qty = 0; opt_data = None
    equity_curve = []
    trades = []

    for i in range(len(df)):
        sig   = signals.iloc[i]
        close = df["CLOSE"].iloc[i]
        date  = df["DATE"].iloc[i]

        # ── stop-loss check ──────────────────────────────────────────────────
        if stop_loss_pct > 0 and pos != 0 and mode == "SPOT":
            if pos == 1 and close < entry_px * (1 - stop_loss_pct/100):
                ep2 = ep(close, -1)
                pnl = (ep2 - entry_px)*qty - br(ep2*qty)
                cash += qty*ep2 - br(ep2*qty)
                trades.append({"date":date,"type":"STOP_LOSS","price":ep2,"qty":qty,"pnl":round(pnl,2)})
                qty = 0; pos = 0
            elif pos == -1 and close > entry_px * (1 + stop_loss_pct/100):
                ep2 = ep(close, 1)
                pnl = (entry_px - ep2)*qty - br(ep2*qty)
                cash += entry_px*qty + pnl
                trades.append({"date":date,"type":"STOP_LOSS","price":ep2,"qty":qty,"pnl":round(pnl,2)})
                qty = 0; pos = 0

        # ── SPOT ─────────────────────────────────────────────────────────────
        if mode == "SPOT":
            opnl = (close-entry_px)*qty if pos==1 else (entry_px-close)*qty if pos==-1 else 0
            equity_curve.append(cash + max(0, opnl))
            if sig == 1 and pos != 1:
                if pos == -1:
                    ep2 = ep(close,-1); pnl = (entry_px-ep2)*qty - br(ep2*qty)
                    cash += entry_px*qty + pnl
                    trades.append({"date":date,"type":"S_EXIT","price":ep2,"qty":qty,"pnl":round(pnl,2)}); qty=0
                ep2 = ep(close,1); qty = int(cash//ep2)
                if qty > 0:
                    cash -= qty*ep2 + br(qty*ep2); pos=1; entry_px=ep2
                    trades.append({"date":date,"type":"BUY","price":ep2,"qty":qty,"pnl":0})
            elif sig == -1 and pos != -1:
                if pos == 1:
                    ep2 = ep(close,-1); pnl = (ep2-entry_px)*qty - br(ep2*qty)
                    cash += qty*ep2 - br(ep2*qty)
                    trades.append({"date":date,"type":"SELL","price":ep2,"qty":qty,"pnl":round(pnl,2)}); qty=0
                ep2 = ep(close,-1); qty = int(cash//ep2)
                if qty > 0:
                    cash -= qty*ep2; pos=-1; entry_px=ep2
                    trades.append({"date":date,"type":"S_ENTRY","price":ep2,"qty":qty,"pnl":0})

        # ── FUTURE ───────────────────────────────────────────────────────────
        elif mode == "FUTURE":
            m1 = close * NIFTY_LOT_SIZE * futures_margin
            ml = int(cash // m1) if m1 > 0 else 0
            opnl = (close-entry_px)*qty*NIFTY_LOT_SIZE if pos==1 else \
                   (entry_px-close)*qty*NIFTY_LOT_SIZE if pos==-1 else 0
            equity_curve.append(max(0, cash + opnl))
            if sig == 1 and pos != 1:
                if pos == -1:
                    ep2 = ep(close,-1)
                    pnl = (entry_px-ep2)*qty*NIFTY_LOT_SIZE - br(ep2*qty*NIFTY_LOT_SIZE)
                    cash += entry_px*NIFTY_LOT_SIZE*qty*futures_margin + pnl
                    trades.append({"date":date,"type":"FUT_SE","price":ep2,"qty":qty,"pnl":round(pnl,2)}); qty=0
                if ml > 0:
                    ep2=ep(close,1); qty=ml
                    cash -= qty*ep2*NIFTY_LOT_SIZE*futures_margin
                    pos=1; entry_px=ep2
                    trades.append({"date":date,"type":"FUT_BUY","price":ep2,"qty":qty,"pnl":0})
            elif sig == -1 and pos != -1:
                if pos == 1:
                    ep2 = ep(close,-1)
                    pnl = (ep2-entry_px)*qty*NIFTY_LOT_SIZE - br(ep2*qty*NIFTY_LOT_SIZE)
                    cash += entry_px*NIFTY_LOT_SIZE*qty*futures_margin + pnl
                    trades.append({"date":date,"type":"FUT_SELL","price":ep2,"qty":qty,"pnl":round(pnl,2)}); qty=0
                if ml > 0:
                    ep2=ep(close,-1); qty=ml
                    cash -= qty*ep2*NIFTY_LOT_SIZE*futures_margin
                    pos=-1; entry_px=ep2
                    trades.append({"date":date,"type":"FUT_SS","price":ep2,"qty":qty,"pnl":0})

        # ── OPTION ───────────────────────────────────────────────────────────
        elif mode == "OPTION":
            opt_val = 0.0
            if opt_data:
                dh = (date - opt_data["ed"]).days
                Tr = max(0,(option_expiry_days-dh)/365)
                opt_val = bs_price(close,opt_data["K"],Tr,risk_free_rate,option_iv,opt_data["ot"]) \
                          * opt_data["lots"] * NIFTY_LOT_SIZE
            equity_curve.append(cash + opt_val)
            if opt_data:
                dh = (date - opt_data["ed"]).days
                Tr = max(0,(option_expiry_days-dh)/365)
                cf = (sig==-1 and opt_data["ot"]=="call") or \
                     (sig== 1 and opt_data["ot"]=="put")  or Tr<=(2/365)
                if cf:
                    xp = bs_price(close,opt_data["K"],Tr,risk_free_rate,option_iv,opt_data["ot"])
                    pnl = (xp-opt_data["ep"])*opt_data["lots"]*NIFTY_LOT_SIZE - br(xp*opt_data["lots"]*NIFTY_LOT_SIZE)
                    cash += opt_val
                    trades.append({"date":date,"type":"OPT_EXIT","price":xp,"qty":opt_data["lots"],"pnl":round(pnl,2)})
                    opt_data=None; pos=0
            if opt_data is None and sig in (1,-1):
                ot = "call" if sig==1 else "put"
                K  = atm_strike(close)
                T0 = option_expiry_days/365
                ep2 = bs_price(close,K,T0,risk_free_rate,option_iv,ot)
                lots = max(1, int(cash*0.2/(ep2*NIFTY_LOT_SIZE))) if ep2>0 else 0
                if lots > 0:
                    cost = lots*ep2*NIFTY_LOT_SIZE + br(lots*ep2*NIFTY_LOT_SIZE)
                    if cash >= cost:
                        cash -= cost
                        opt_data = {"ed":date,"K":K,"ot":ot,"ep":ep2,"lots":lots}
                        pos = 1 if sig==1 else -1
                        trades.append({"date":date,"type":"OPT_BUY","price":ep2,"qty":lots,"pnl":0})

    # force-close last bar
    if len(df) > 0:
        last_close = df["CLOSE"].iloc[-1]
        last_date  = df["DATE"].iloc[-1]
        if mode in ("SPOT","FUTURE") and pos != 0:
            ep2 = ep(last_close, -1 if pos==1 else 1)
            if mode == "SPOT":
                pnl = (ep2-entry_px)*qty if pos==1 else (entry_px-ep2)*qty
                pnl -= br(ep2*qty)
                cash += qty*ep2 if pos==1 else entry_px*qty + pnl
            else:
                pnl = (ep2-entry_px)*qty*NIFTY_LOT_SIZE if pos==1 else (entry_px-ep2)*qty*NIFTY_LOT_SIZE
                pnl -= br(ep2*qty*NIFTY_LOT_SIZE)
                cash += entry_px*NIFTY_LOT_SIZE*qty*futures_margin + pnl
            trades.append({"date":last_date,"type":"CLOSE_ALL","price":ep2,"qty":qty,"pnl":round(pnl,2)})
            if equity_curve: equity_curve[-1] = max(0, cash)
        elif mode == "OPTION" and opt_data:
            dh = (last_date - opt_data["ed"]).days
            Tr = max(0,(option_expiry_days-dh)/365)
            xp = bs_price(last_close,opt_data["K"],Tr,risk_free_rate,option_iv,opt_data["ot"])
            pnl = (xp-opt_data["ep"])*opt_data["lots"]*NIFTY_LOT_SIZE - br(xp*opt_data["lots"]*NIFTY_LOT_SIZE)
            cash += xp*opt_data["lots"]*NIFTY_LOT_SIZE
            trades.append({"date":last_date,"type":"OPT_CLOSE","price":xp,"qty":opt_data["lots"],"pnl":round(pnl,2)})
            if equity_curve: equity_curve[-1] = max(0, cash)

    return equity_curve, trades

# ── METRICS ───────────────────────────────────────────────────────────────────
def calc_metrics(equity_curve, trades_list, initial_capital, risk_free_rate):
    eq  = np.array(equity_curve, dtype=float)
    if len(eq) < 2:
        return {}
    ret   = np.diff(eq) / np.where(eq[:-1]==0, 1, eq[:-1])
    years = len(eq) / 252
    total_ret = (eq[-1] - initial_capital) / initial_capital * 100
    cagr      = ((eq[-1]/initial_capital)**(1/years) - 1)*100 if years>0 and eq[-1]>0 else -100
    dd        = eq / np.maximum.accumulate(np.where(eq==0,1e-9,eq)) - 1
    max_dd    = dd.min() * 100
    ann_ret   = ret.mean() * 252
    ann_std   = ret.std()  * np.sqrt(252)
    sharpe    = (ann_ret - risk_free_rate) / ann_std if ann_std > 0 else 0
    down      = ret[ret < 0]
    sortino   = (ann_ret - risk_free_rate) / (down.std()*np.sqrt(252)) if len(down)>0 and down.std()>0 else 0
    td        = pd.DataFrame(trades_list)
    pnls      = td[td["pnl"] != 0]["pnl"] if len(td)>0 else pd.Series(dtype=float)
    nt        = len(pnls)
    win_rate  = float((pnls > 0).sum()) / nt * 100 if nt > 0 else 0
    gp = float(pnls[pnls>0].sum()); gl = abs(float(pnls[pnls<0].sum()))
    pf = gp/gl if gl > 0 else float("inf")
    avg_win  = float(pnls[pnls>0].mean()) if (pnls>0).any() else 0
    avg_loss = float(pnls[pnls<0].mean()) if (pnls<0).any() else 0
    return {
        "final_equity": round(eq[-1],0), "total_ret": round(total_ret,2),
        "cagr": round(cagr,2), "max_dd": round(max_dd,2),
        "sharpe": round(sharpe,2), "sortino": round(sortino,2),
        "num_trades": nt, "win_rate": round(win_rate,2),
        "profit_factor": round(pf,2), "avg_win": round(avg_win,2),
        "avg_loss": round(avg_loss,2),
    }

# ════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ════════════════════════════════════════════════════════════════════════════
st.title("📈 NIFTY 50 — Trading Signal Dashboard")
st.caption("Based on 7 years of real NIFTY 50 data (2019–2026)")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("💰 Your Capital")
    initial_capital = st.number_input("Starting Capital (Rs)", 100_000, 10_000_000,
                                       1_000_000, step=100_000,
                                       help="How much money you want to trade with")
    stop_loss_pct = st.slider("Stop-Loss (%)", 0.0, 20.0, 5.0, 0.5,
                               help="Max loss before exiting a trade. Recommended: 3-7%")

    st.subheader("📊 Strategy")
    strategy = st.selectbox("Strategy", ["SMA Crossover", "RSI Mean Reversion",
                                          "Bollinger Bands", "MACD Signal"])
    mode = st.selectbox("Instrument", ["SPOT", "FUTURE", "OPTION"],
                         help="SPOT = buy/sell index directly | FUTURE = leveraged | OPTION = buy calls/puts")

    st.subheader("📐 Indicator Settings")
    if strategy == "SMA Crossover":
        sma_fast = st.slider("Fast MA (days)", 5, 50, 20)
        sma_slow = st.slider("Slow MA (days)", 20, 200, 50)
        rsi_period=14; rsi_os=35; rsi_ob=65; bb_p=20; bb_s=2.0; mf=12; ms=26; mg=9
    elif strategy == "RSI Mean Reversion":
        rsi_period = st.slider("RSI Period", 5, 30, 14)
        rsi_os     = st.slider("Oversold level (buy below)", 10, 45, 35)
        rsi_ob     = st.slider("Overbought level (sell above)", 55, 90, 65)
        sma_fast=20; sma_slow=50; bb_p=20; bb_s=2.0; mf=12; ms=26; mg=9
    elif strategy == "Bollinger Bands":
        bb_p = st.slider("BB Period", 10, 50, 20)
        bb_s = st.slider("Band width", 1.0, 3.0, 2.0, 0.1)
        sma_fast=20; sma_slow=50; rsi_period=14; rsi_os=35; rsi_ob=65; mf=12; ms=26; mg=9
    else:
        mf = st.slider("MACD Fast EMA", 5, 20, 12)
        ms = st.slider("MACD Slow EMA", 15, 40, 26)
        mg = st.slider("MACD Signal",    5, 15,  9)
        sma_fast=20; sma_slow=50; rsi_period=14; rsi_os=35; rsi_ob=65; bb_p=20; bb_s=2.0

    with st.expander("⚙️ Advanced Settings"):
        futures_margin     = st.slider("Futures Margin (%)", 5, 30, 12) / 100
        option_iv          = st.slider("Options IV (%)", 5, 50, 18) / 100
        option_expiry_days = st.slider("Days to Expiry (Options)", 7, 60, 25)
        risk_free_rate     = st.slider("Risk-Free Rate (%)", 3.0, 10.0, 6.5) / 100
        brokerage_pct      = st.slider("Brokerage (%)", 0.01, 0.5, 0.03) / 100
        slippage_ticks     = st.slider("Slippage (ticks)", 0, 10, 2)

    st.subheader("📅 Date Range")
    df_raw = load_data()
    min_date = df_raw["DATE"].min().date()
    max_date = df_raw["DATE"].max().date()
    date_range = st.date_input("Select period",
                                value=(min_date, max_date),
                                min_value=min_date, max_value=max_date)
# ── MAIN PANEL ────────────────────────────────────────────────────────────────
if len(date_range) == 2:
    start_d, end_d = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_d, end_d = df_raw["DATE"].min(), df_raw["DATE"].max()

df_f = df_raw[(df_raw["DATE"] >= start_d) & (df_raw["DATE"] <= end_d)].reset_index(drop=True)
df_f = add_indicators(df_f, sma_fast, sma_slow, rsi_period, bb_p, bb_s, mf, ms, mg)

sig_map = {"SMA Crossover": sig_sma(df_f),
           "RSI Mean Reversion": sig_rsi(df_f, rsi_os, rsi_ob),
           "Bollinger Bands": sig_bb(df_f),
           "MACD Signal": sig_macd(df_f)}
signals = sig_map[strategy]

with st.spinner("Calculando..."):
    eq_curve, trades_list = run_backtest(
        df_f, signals, mode, initial_capital, futures_margin,
        option_iv, option_expiry_days, risk_free_rate,
        brokerage_pct, slippage_ticks, stop_loss_pct)
    metrics = calc_metrics(eq_curve, trades_list, initial_capital, risk_free_rate)
    bh_eq = [initial_capital * (df_f["CLOSE"].iloc[i] / df_f["CLOSE"].iloc[0]) for i in range(len(df_f))]
    bh_m  = calc_metrics(bh_eq, [], initial_capital, risk_free_rate)

# ── SEÑAL ACTUAL ──────────────────────────────────────────────────────────────
st.markdown("---")
last = df_f.iloc[-1]
prev = df_f.iloc[-2] if len(df_f) > 1 else df_f.iloc[-1]
last_date_str = last["DATE"].strftime("%d %b %Y")
price_chg = ((last["CLOSE"] - prev["CLOSE"]) / prev["CLOSE"]) * 100
atr_val   = last["ATR"] if not np.isnan(last["ATR"]) else last["CLOSE"] * 0.01

# Compute current signal for each strategy
def current_signal(df, strat_name, ros=35, rob=65):
    if strat_name == "SMA Crossover":
        if df["SMA_FAST"].iloc[-1] > df["SMA_SLOW"].iloc[-1] and \
           df["SMA_FAST"].iloc[-2] <= df["SMA_SLOW"].iloc[-2]:
            return 1, "Fresh BUY crossover today"
        elif df["SMA_FAST"].iloc[-1] > df["SMA_SLOW"].iloc[-1]:
            return 1, "Uptrend active — fast MA above slow MA"
        elif df["SMA_FAST"].iloc[-1] < df["SMA_SLOW"].iloc[-1]:
            return -1, "Downtrend active — fast MA below slow MA"
        return 0, "No clear trend"
    elif strat_name == "RSI Mean Reversion":
        rsi = df["RSI"].iloc[-1]
        if rsi < ros:   return 1,  "RSI {:.0f} — Oversold → good time to BUY".format(rsi)
        if rsi > rob:   return -1, "RSI {:.0f} — Overbought → consider SELLING".format(rsi)
        return 0, "RSI {:.0f} — Neutral zone, wait".format(rsi)
    elif strat_name == "Bollinger Bands":
        c = df["CLOSE"].iloc[-1]
        if c <= df["BB_LOW"].iloc[-1]:  return 1,  "Price hit lower band → likely to bounce up"
        if c >= df["BB_UP"].iloc[-1]:   return -1, "Price hit upper band → likely to pull back"
        pct = (c - df["BB_LOW"].iloc[-1]) / (df["BB_UP"].iloc[-1] - df["BB_LOW"].iloc[-1]) * 100
        return 0, "Price is {:.0f}% inside the bands — wait".format(pct)
    elif strat_name == "MACD Signal":
        if df["MACD"].iloc[-1] > df["MACD_SIG"].iloc[-1] and \
           df["MACD"].iloc[-2] <= df["MACD_SIG"].iloc[-2]:
            return 1,  "Fresh bullish MACD crossover today"
        elif df["MACD"].iloc[-1] > df["MACD_SIG"].iloc[-1]:
            return 1,  "MACD above signal line — bullish"
        elif df["MACD"].iloc[-1] < df["MACD_SIG"].iloc[-1]:
            return -1, "MACD below signal line — bearish"
        return 0, "MACD neutral"
    return 0, "Neutral"

strategies_all = ["SMA Crossover", "RSI Mean Reversion", "Bollinger Bands", "MACD Signal"]
all_signals = []
for s in strategies_all:
    ros_v = rsi_os if s == "RSI Mean Reversion" else 35
    rob_v = rsi_ob if s == "RSI Mean Reversion" else 65
    sv, desc = current_signal(df_f, s, ros_v, rob_v)
    all_signals.append((s, sv, desc))

buy_count  = sum(1 for _, sv, _ in all_signals if sv == 1)
sell_count = sum(1 for _, sv, _ in all_signals if sv == -1)
consensus  = "🟢 BUY"  if buy_count >= 3 else \
             "🔴 SELL" if sell_count >= 3 else \
             "🟡 WAIT"

# SL/TP based on ATR
sl_price = round(last["CLOSE"] - 2 * atr_val, 2)
tp_price = round(last["CLOSE"] + 3 * atr_val, 2)
sl_pct   = round(2 * atr_val / last["CLOSE"] * 100, 2)
tp_pct   = round(3 * atr_val / last["CLOSE"] * 100, 2)

# Support/Resistance (20-day high/low)
recent = df_f.tail(20)
support    = round(recent["LOW"].min(), 2)
resistance = round(recent["HIGH"].max(), 2)

st.subheader("📡 TODAY'S SIGNAL — {}".format(last_date_str))

# Price ticker
pc1, pc2, pc3, pc4 = st.columns(4)
pc1.metric("💹 NIFTY 50 Price", "{:,.2f}".format(last["CLOSE"]),
            delta="{:+.2f}% today".format(price_chg))
pc2.metric("📊 Momentum (RSI)", "{:.0f} / 100".format(last["RSI"]) if not np.isnan(last["RSI"]) else "—",
            delta="Oversold <35  |  Overbought >65", delta_color="off")
pc3.metric("🔺 Resistance (20-day high)", "{:,.2f}".format(resistance),
            delta="Price ceiling — sell zone", delta_color="off")
pc4.metric("🔻 Support (20-day low)", "{:,.2f}".format(support),
            delta="Price floor — buy zone", delta_color="off")

st.markdown("<br>", unsafe_allow_html=True)

# Signal cards for all 4 strategies
sig_cols = st.columns(4)
icons  = {1: "🟢", -1: "🔴", 0: "⚪"}
labels = {1: "BUY",   -1: "SELL",  0: "WAIT"}
for col, (sname, sv, desc) in zip(sig_cols, all_signals):
    col.metric(
        label="{} {}".format(icons[sv], sname),
        value=labels[sv],
        delta=desc,
        delta_color="normal" if sv == 1 else ("inverse" if sv == -1 else "off")
    )

st.markdown("<br>", unsafe_allow_html=True)

# Big consensus box + SL/TP
con1, con2, con3, con4 = st.columns(4)
agree = max(buy_count, sell_count)
con1.metric("🎯 OVERALL SIGNAL  ({}/4 agree)".format(agree), consensus,
             delta="Based on all 4 strategies", delta_color="off")
con2.metric("🛑 Stop-Loss  (exit if price drops here)",
             "Rs {:,.0f}".format(sl_price),
             delta="{}% below current price".format(sl_pct), delta_color="inverse")
con3.metric("✅ Take-Profit  (exit when price reaches here)",
             "Rs {:,.0f}".format(tp_price),
             delta="{}% above current price".format(tp_pct))
con4.metric("📐 Daily Volatility",
             "±Rs {:.0f}".format(atr_val),
             delta="Average daily price move", delta_color="off")

st.markdown("---")

# ── BACKTEST RESULTS ──────────────────────────────────────────────────────────
st.subheader("📊 Strategy Performance — {} [{}]".format(strategy, mode))

r1c1, r1c2, r1c3 = st.columns(3)
fin_eq  = metrics.get("final_equity", 0)
tot_ret = metrics.get("total_ret", 0)
cagr_v  = metrics.get("cagr", 0)
r1c1.metric("💰 Final Capital",
             "Rs {:,.0f}".format(fin_eq),
             delta="Profit: Rs {:,.0f}".format(fin_eq - initial_capital))
r1c2.metric("📈 Total Return",
             "{}%".format(tot_ret),
             delta="{:+.1f}% vs Buy&Hold".format(tot_ret - bh_m.get("total_ret",0)))
r1c3.metric("🚀 Annual Return (CAGR)",
             "{}% / year".format(cagr_v),
             delta="{:+.1f}% vs Buy&Hold".format(cagr_v - bh_m.get("cagr",0)))

r2c1, r2c2, r2c3 = st.columns(3)
r2c1.metric("📉 Worst Drawdown",  "{}%".format(metrics.get("max_dd",0)),
             delta="Max loss from peak", delta_color="off")
r2c2.metric("🎯 Win Rate",        "{}%".format(metrics.get("win_rate",0)),
             delta="% of trades that made money", delta_color="off")
r2c3.metric("💹 Profit Factor",   str(metrics.get("profit_factor",0)),
             delta="Gains / Losses ratio (>1 is good)", delta_color="off")

r3c1, r3c2, r3c3 = st.columns(3)
r3c1.metric("🔢 Total Trades",    str(metrics.get("num_trades",0)),
             delta="Number of buy/sell operations", delta_color="off")
r3c2.metric("📊 Avg Winning Trade","Rs {:,.0f}".format(metrics.get("avg_win",0)),
             delta="Average profit per winning trade", delta_color="off")
r3c3.metric("📊 Avg Losing Trade", "Rs {:,.0f}".format(abs(metrics.get("avg_loss",0))),
             delta="Average loss per losing trade", delta_color="off")

st.markdown("---")

# ── CHARTS & TRADE LOG ────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Equity & Drawdown", "🕯️ Price Chart & Indicators", "📋 Trade History"])

with tab1:
    eq_arr = np.array(eq_curve, dtype=float)
    dates  = df_f["DATE"].iloc[:len(eq_arr)]
    dd_arr = eq_arr / np.maximum.accumulate(np.where(eq_arr==0,1e-9,eq_arr)) - 1

    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True, row_heights=[0.65, 0.35],
                        subplot_titles=["Equity (Rs)", "Drawdown (%)"])

    fig.add_trace(go.Scatter(x=dates, y=eq_arr/1e5, name="Estrategia",
                             line=dict(color="#00d26a", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_f["DATE"].iloc[:len(bh_eq)], y=np.array(bh_eq)/1e5,
                             name="Buy & Hold", line=dict(color="#4da6ff", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=dd_arr*100, name="Drawdown",
                             fill="tozeroy", fillcolor="rgba(255,75,75,0.2)",
                             line=dict(color="rgba(255,75,75,0.8)", width=1)), row=2, col=1)

    # trade markers
    td = pd.DataFrame(trades_list)
    if len(td) > 0:
        buys  = td[td["type"].str.contains("BUY|ENTRY", na=False)]
        sells = td[td["type"].str.contains("SELL|EXIT|STOP", na=False)]
        for _, row in buys.iterrows():
            idx = df_f[df_f["DATE"]==row["date"]].index
            if len(idx)>0 and idx[0]<len(eq_arr):
                fig.add_vline(x=row["date"], line_color="lime", line_width=0.8,
                              line_dash="dot", opacity=0.6, row=1, col=1)
        for _, row in sells.iterrows():
            idx = df_f[df_f["DATE"]==row["date"]].index
            if len(idx)>0 and idx[0]<len(eq_arr):
                fig.add_vline(x=row["date"], line_color="red", line_width=0.8,
                              line_dash="dot", opacity=0.6, row=1, col=1)

    fig.update_layout(height=550, template="plotly_dark", showlegend=True,
                      margin=dict(l=40,r=40,t=40,b=20),
                      legend=dict(orientation="h", y=1.05))
    fig.update_yaxes(title_text="Rs Lakh", row=1, col=1)
    fig.update_yaxes(title_text="DD %",    row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**vs Buy & Hold (just holding NIFTY 50):**")
    comp_cols = st.columns(4)
    comp_cols[0].metric("Strategy Return",  "{}%".format(metrics.get("total_ret",0)),
                         delta="{:+.1f}% vs B&H".format(metrics.get("total_ret",0)-bh_m.get("total_ret",0)))
    comp_cols[1].metric("Strategy CAGR",    "{}%/yr".format(metrics.get("cagr",0)),
                         delta="{:+.1f}% vs B&H".format(metrics.get("cagr",0)-bh_m.get("cagr",0)))
    comp_cols[2].metric("Buy & Hold Return","{}%".format(bh_m.get("total_ret",0)),
                         delta="Baseline — no trading at all", delta_color="off")
    comp_cols[3].metric("Strategy Max Loss","{}%".format(metrics.get("max_dd",0)),
                         delta="{:+.1f}% vs B&H".format(metrics.get("max_dd",0)-bh_m.get("max_dd",0)),
                         delta_color="inverse")

with tab2:
    fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55,0.25,0.20],
                         subplot_titles=["NIFTY 50 Price + Indicators", "RSI (Momentum)", "MACD (Trend)"])
    # Candlestick
    fig2.add_trace(go.Candlestick(x=df_f["DATE"], open=df_f["OPEN"], high=df_f["HIGH"],
                                   low=df_f["LOW"], close=df_f["CLOSE"], name="NIFTY 50",
                                   increasing_line_color="#00d26a", decreasing_line_color="#ff4b4b"),
                   row=1, col=1)
    fig2.add_trace(go.Scatter(x=df_f["DATE"], y=df_f["SMA_FAST"], name="SMA Fast",
                               line=dict(color="orange", width=1.2)), row=1, col=1)
    fig2.add_trace(go.Scatter(x=df_f["DATE"], y=df_f["SMA_SLOW"], name="SMA Slow",
                               line=dict(color="red", width=1.2)), row=1, col=1)
    fig2.add_trace(go.Scatter(x=df_f["DATE"], y=df_f["BB_UP"], name="BB Up",
                               line=dict(color="rgba(150,100,255,0.5)", width=1, dash="dot")), row=1, col=1)
    fig2.add_trace(go.Scatter(x=df_f["DATE"], y=df_f["BB_LOW"], name="BB Low",
                               line=dict(color="rgba(150,100,255,0.5)", width=1, dash="dot"),
                               fill="tonexty", fillcolor="rgba(150,100,255,0.05)"), row=1, col=1)
    # RSI
    fig2.add_trace(go.Scatter(x=df_f["DATE"], y=df_f["RSI"], name="RSI",
                               line=dict(color="#ffd700", width=1.2)), row=2, col=1)
    fig2.add_hline(y=70, line_color="red",  line_dash="dash", line_width=0.8, row=2, col=1)
    fig2.add_hline(y=30, line_color="lime", line_dash="dash", line_width=0.8, row=2, col=1)
    # MACD
    macd_color = ["#00d26a" if v>=0 else "#ff4b4b" for v in df_f["MACD"]-df_f["MACD_SIG"]]
    fig2.add_trace(go.Bar(x=df_f["DATE"], y=df_f["MACD"]-df_f["MACD_SIG"],
                           name="MACD Hist", marker_color=macd_color, opacity=0.7), row=3, col=1)
    fig2.add_trace(go.Scatter(x=df_f["DATE"], y=df_f["MACD"], name="MACD",
                               line=dict(color="#4da6ff", width=1)), row=3, col=1)
    fig2.add_trace(go.Scatter(x=df_f["DATE"], y=df_f["MACD_SIG"], name="Signal",
                               line=dict(color="orange", width=1)), row=3, col=1)

    fig2.update_layout(height=650, template="plotly_dark", showlegend=True,
                       xaxis_rangeslider_visible=False, margin=dict(l=40,r=40,t=40,b=20))
    fig2.update_yaxes(range=[0,100], row=2, col=1)
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    if trades_list:
        td_df = pd.DataFrame(trades_list)
        td_df["date"] = td_df["date"].dt.strftime("%Y-%m-%d")
        td_df["result"] = td_df["pnl"].apply(lambda x: "WIN 🟢" if x>0 else ("LOSS 🔴" if x<0 else "—"))
        st.dataframe(td_df[["date","type","price","qty","pnl","result"]].rename(
            columns={"date":"Date","type":"Action","price":"Price (Rs)",
                     "qty":"Quantity","pnl":"Profit/Loss (Rs)","result":"Result"}),
            use_container_width=True, height=400)
        total_pnl = td_df["pnl"].sum()
        color = "green" if total_pnl >= 0 else "red"
        st.markdown("**Total Profit/Loss from closed trades: Rs {:,.0f}**".format(total_pnl))
        csv_buf = io.StringIO()
        td_df.drop("result", axis=1).to_csv(csv_buf, index=False)
        st.download_button("⬇️ Download Trade History (CSV)", csv_buf.getvalue(),
                            file_name="trades_{}_{}.csv".format(strategy.replace(" ","_"), mode),
                            mime="text/csv")
    else:
        st.info("No trades were generated with this configuration.")

st.markdown("---")
st.caption("Disclaimer: This dashboard is for educational purposes only. Past performance does not guarantee future results.")
