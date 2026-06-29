"""
NIFTY 50 - Strategy Backtester with F&O Simulation
Data: 28-Jun-2019 to 28-Jun-2026 (daily OHLCV)
Strategies: SMA Crossover, RSI, Bollinger Bands, MACD
Modes: SPOT / FUTURE (lot=50, margin 12%) / OPTION (ATM Call/Put, Black-Scholes)
Run:  python nifty50_backtest.py
Out:  backtest_results.png   trade_log.csv
"""
import warnings, math
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATA_FILE          = "NIFTY_50_Historical_28062019_to_28062026.xlsx"
INITIAL_CAPITAL    = 1_000_000
NIFTY_LOT_SIZE     = 50
FUTURES_MARGIN     = 0.12
OPTION_IV          = 0.18
OPTION_EXPIRY_DAYS = 25
RISK_FREE_RATE     = 0.065
BROKERAGE_PCT      = 0.0003
SLIPPAGE_TICKS     = 2
SMA_FAST, SMA_SLOW             = 20, 50
RSI_PERIOD                     = 14
RSI_OVERSOLD, RSI_OVERBOUGHT   = 35, 65
BB_PERIOD, BB_STD              = 20, 2.0
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9

# ── DATA ─────────────────────────────────────────────────────────────────────
def load_data(path):
    df = pd.read_excel(path)
    df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True)
    df = df.sort_values("DATE").reset_index(drop=True)
    for col in list(df.columns):
        if "SHARES"   in col: df.rename(columns={col: "VOLUME"},   inplace=True)
        if "TURNOVER" in col: df.rename(columns={col: "TURNOVER"}, inplace=True)
    df["RETURNS"] = df["CLOSE"].pct_change()
    return df

# ── INDICATORS ────────────────────────────────────────────────────────────────
def add_indicators(df):
    c = df["CLOSE"]
    df["SMA_FAST"] = c.rolling(SMA_FAST).mean()
    df["SMA_SLOW"] = c.rolling(SMA_SLOW).mean()
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss  = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    bb_m = c.rolling(BB_PERIOD).mean()
    bb_s = c.rolling(BB_PERIOD).std()
    df["BB_MID"] = bb_m
    df["BB_UP"]  = bb_m + BB_STD * bb_s
    df["BB_LOW"] = bb_m - BB_STD * bb_s
    ema_f = c.ewm(span=MACD_FAST, adjust=False).mean()
    ema_s = c.ewm(span=MACD_SLOW, adjust=False).mean()
    df["MACD"]      = ema_f - ema_s
    df["MACD_SIG"]  = df["MACD"].ewm(span=MACD_SIG, adjust=False).mean()
    hl  = df["HIGH"] - df["LOW"]
    hpc = (df["HIGH"] - df["CLOSE"].shift()).abs()
    lpc = (df["LOW"]  - df["CLOSE"].shift()).abs()
    df["ATR"] = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()
    return df

# ── SIGNALS ───────────────────────────────────────────────────────────────────
def sig_sma(df):
    return (df["SMA_FAST"] > df["SMA_SLOW"]).astype(int).diff().fillna(0).astype(int)

def sig_rsi(df):
    s = pd.Series(0, index=df.index)
    s[(df["RSI"] < RSI_OVERSOLD)   & (df["RSI"].shift() >= RSI_OVERSOLD)]   =  1
    s[(df["RSI"] > RSI_OVERBOUGHT) & (df["RSI"].shift() <= RSI_OVERBOUGHT)]  = -1
    return s

def sig_bb(df):
    s = pd.Series(0, index=df.index)
    s[(df["CLOSE"] > df["BB_LOW"]) & (df["CLOSE"].shift() <= df["BB_LOW"])]  =  1
    s[(df["CLOSE"] < df["BB_UP"])  & (df["CLOSE"].shift() >= df["BB_UP"])]   = -1
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

# ── ENGINE ────────────────────────────────────────────────────────────────────
class BacktestEngine:
    def __init__(self, df, signals, mode="SPOT", name="Strategy"):
        self.df           = df.copy().reset_index(drop=True)
        self.signals      = signals.reset_index(drop=True)
        self.mode         = mode.upper()
        self.name         = "{} [{}]".format(name, mode)
        self.trades       = []
        self.equity_curve = []
        self._run()

    def _ep(self, px, d=1):  return px + d * SLIPPAGE_TICKS * 0.05
    def _br(self, n):        return n * BROKERAGE_PCT
    def _rec(self, dt, tp, px, q, pnl):
        self.trades.append({"date": dt, "type": tp, "price": px, "qty": q, "pnl": round(pnl, 2)})

    def _run(self):
        df = self.df; eq = self.equity_curve
        cash = float(INITIAL_CAPITAL)
        pos = 0; entry_px = 0.0; qty = 0; opt_data = None

        for i in range(len(df)):
            sig = self.signals.iloc[i]
            close = df["CLOSE"].iloc[i]
            date  = df["DATE"].iloc[i]

            # ─ SPOT ──────────────────────────────────────────────────────────
            if self.mode == "SPOT":
                opnl = (close-entry_px)*qty if pos==1 else (entry_px-close)*qty if pos==-1 else 0
                eq.append(cash + max(0, opnl))
                if sig == 1 and pos != 1:
                    if pos == -1:
                        ep = self._ep(close,-1); pnl = (entry_px-ep)*qty - self._br(ep*qty)
                        cash += entry_px*qty + pnl; self._rec(date,"S_EXIT",ep,qty,pnl); qty=0
                    ep = self._ep(close,1); qty = int(cash//ep)
                    if qty > 0:
                        cash -= qty*ep + self._br(qty*ep); pos=1; entry_px=ep
                        self._rec(date,"BUY",ep,qty,0)
                elif sig == -1 and pos != -1:
                    if pos == 1:
                        ep = self._ep(close,-1); pnl = (ep-entry_px)*qty - self._br(ep*qty)
                        cash += qty*ep - self._br(ep*qty); self._rec(date,"SELL",ep,qty,pnl); qty=0
                    ep = self._ep(close,-1); qty = int(cash//ep)
                    if qty > 0:
                        cash -= qty*ep; pos=-1; entry_px=ep; self._rec(date,"S_ENTRY",ep,qty,0)

            # ─ FUTURE ────────────────────────────────────────────────────────
            elif self.mode == "FUTURE":
                m1 = close * NIFTY_LOT_SIZE * FUTURES_MARGIN
                ml = int(cash // m1) if m1 > 0 else 0
                opnl = (close-entry_px)*qty*NIFTY_LOT_SIZE if pos==1 else \
                       (entry_px-close)*qty*NIFTY_LOT_SIZE if pos==-1 else 0
                eq.append(cash + opnl)
                if sig == 1 and pos != 1:
                    if pos == -1:
                        ep = self._ep(close,-1)
                        pnl = (entry_px-ep)*qty*NIFTY_LOT_SIZE - self._br(ep*qty*NIFTY_LOT_SIZE)
                        cash += entry_px*NIFTY_LOT_SIZE*qty*FUTURES_MARGIN + pnl
                        self._rec(date,"FUT_SE",ep,qty,pnl); qty=0
                    if ml > 0:
                        ep=self._ep(close,1); qty=ml
                        cash -= qty*ep*NIFTY_LOT_SIZE*FUTURES_MARGIN
                        pos=1; entry_px=ep; self._rec(date,"FUT_BUY",ep,qty,0)
                elif sig == -1 and pos != -1:
                    if pos == 1:
                        ep = self._ep(close,-1)
                        pnl = (ep-entry_px)*qty*NIFTY_LOT_SIZE - self._br(ep*qty*NIFTY_LOT_SIZE)
                        cash += entry_px*NIFTY_LOT_SIZE*qty*FUTURES_MARGIN + pnl
                        self._rec(date,"FUT_SELL",ep,qty,pnl); qty=0
                    if ml > 0:
                        ep=self._ep(close,-1); qty=ml
                        cash -= qty*ep*NIFTY_LOT_SIZE*FUTURES_MARGIN
                        pos=-1; entry_px=ep; self._rec(date,"FUT_SS",ep,qty,0)

            # ─ OPTION ────────────────────────────────────────────────────────
            elif self.mode == "OPTION":
                opt_val = 0.0
                if opt_data:
                    dh = (date - opt_data["ed"]).days
                    Tr = max(0,(OPTION_EXPIRY_DAYS-dh)/365)
                    opt_val = bs_price(close,opt_data["K"],Tr,RISK_FREE_RATE,OPTION_IV,opt_data["ot"]) \
                              * opt_data["lots"] * NIFTY_LOT_SIZE
                eq.append(cash + opt_val)
                if opt_data:
                    dh = (date - opt_data["ed"]).days
                    Tr = max(0,(OPTION_EXPIRY_DAYS-dh)/365)
                    cf = (sig==-1 and opt_data["ot"]=="call") or \
                         (sig== 1 and opt_data["ot"]=="put")  or Tr<=(2/365)
                    if cf:
                        xp = bs_price(close,opt_data["K"],Tr,RISK_FREE_RATE,OPTION_IV,opt_data["ot"])
                        pnl = (xp-opt_data["ep"])*opt_data["lots"]*NIFTY_LOT_SIZE \
                              - self._br(xp*opt_data["lots"]*NIFTY_LOT_SIZE)
                        cash += opt_val
                        self._rec(date,"OPT_X_"+opt_data["ot"].upper(),xp,opt_data["lots"],pnl)
                        opt_data=None; pos=0
                if opt_data is None and sig in (1,-1):
                    ot = "call" if sig==1 else "put"
                    K  = atm_strike(close)
                    T0 = OPTION_EXPIRY_DAYS/365
                    ep = bs_price(close,K,T0,RISK_FREE_RATE,OPTION_IV,ot)
                    lots = max(1, int(cash*0.2 / (ep*NIFTY_LOT_SIZE))) if ep>0 else 0
                    if lots > 0:
                        cost = lots*ep*NIFTY_LOT_SIZE + self._br(lots*ep*NIFTY_LOT_SIZE)
                        if cash >= cost:
                            cash -= cost
                            opt_data = {"ed":date,"K":K,"ot":ot,"ep":ep,"lots":lots}
                            pos = 1 if sig==1 else -1
                            self._rec(date,"OPT_BUY_"+ot.upper(),ep,lots,0)

        # force-close any open position on last bar
        last_close = df["CLOSE"].iloc[-1]
        last_date  = df["DATE"].iloc[-1]
        if self.mode in ("SPOT","FUTURE") and pos != 0:
            ep = self._ep(last_close, -1 if pos==1 else 1)
            if self.mode == "SPOT":
                pnl = (ep-entry_px)*qty if pos==1 else (entry_px-ep)*qty
                pnl -= self._br(ep*qty)
                cash += qty*ep if pos==1 else entry_px*qty + pnl
            else:
                pnl = (ep-entry_px)*qty*NIFTY_LOT_SIZE if pos==1 else (entry_px-ep)*qty*NIFTY_LOT_SIZE
                pnl -= self._br(ep*qty*NIFTY_LOT_SIZE)
                cash += entry_px*NIFTY_LOT_SIZE*qty*FUTURES_MARGIN + pnl
            self._rec(last_date,"CLOSE_ALL",ep,qty,pnl)
            eq[-1] = cash
        elif self.mode == "OPTION" and opt_data:
            dh = (last_date - opt_data["ed"]).days
            Tr = max(0,(OPTION_EXPIRY_DAYS-dh)/365)
            xp = bs_price(last_close,opt_data["K"],Tr,RISK_FREE_RATE,OPTION_IV,opt_data["ot"])
            pnl = (xp-opt_data["ep"])*opt_data["lots"]*NIFTY_LOT_SIZE
            pnl -= self._br(xp*opt_data["lots"]*NIFTY_LOT_SIZE)
            cash += xp*opt_data["lots"]*NIFTY_LOT_SIZE
            self._rec(last_date,"OPT_CLOSE",xp,opt_data["lots"],pnl)
            eq[-1] = cash

        self.final_equity = eq[-1] if eq else INITIAL_CAPITAL

    # ── metrics ───────────────────────────────────────────────────────────────
    def metrics(self):
        eq  = np.array(self.equity_curve, dtype=float)
        ret = np.diff(eq) / eq[:-1]
        n   = len(self.df)
        years = n / 252

        total_ret = (eq[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        cagr      = ((eq[-1]/INITIAL_CAPITAL)**(1/years) - 1)*100 if years>0 else 0

        dd     = eq / np.maximum.accumulate(eq) - 1
        max_dd = dd.min() * 100

        ann_ret = ret.mean() * 252
        ann_std = ret.std()  * np.sqrt(252)
        sharpe  = (ann_ret - RISK_FREE_RATE) / ann_std if ann_std > 0 else 0

        down    = ret[ret < 0]
        sortino = (ann_ret - RISK_FREE_RATE) / (down.std()*np.sqrt(252)) if len(down)>0 else 0

        trades_df = pd.DataFrame(self.trades)
        pnl_trades = trades_df[trades_df["pnl"] != 0]["pnl"] if len(trades_df)>0 else pd.Series(dtype=float)
        num_trades = len(pnl_trades)
        win_rate   = (pnl_trades > 0).sum() / num_trades * 100 if num_trades > 0 else 0
        gross_p    = pnl_trades[pnl_trades>0].sum()
        gross_l    = abs(pnl_trades[pnl_trades<0].sum())
        pf         = gross_p / gross_l if gross_l > 0 else float("inf")

        return {
            "Strategy"      : self.name,
            "Final Equity"  : round(eq[-1], 0),
            "Total Ret %"   : round(total_ret, 2),
            "CAGR %"        : round(cagr, 2),
            "Max DD %"      : round(max_dd, 2),
            "Sharpe"        : round(sharpe, 2),
            "Sortino"       : round(sortino, 2),
            "# Trades"      : num_trades,
            "Win Rate %"    : round(win_rate, 2),
            "Profit Factor" : round(pf, 2),
        }


# ── PERFORMANCE TABLE ─────────────────────────────────────────────────────────
def print_table(results):
    keys = list(results[0].keys())
    col_w = {k: max(len(k), max(len(str(r[k])) for r in results))+2 for k in keys}
    sep = "+" + "+".join("-"*col_w[k] for k in keys) + "+"
    hdr = "|" + "|".join(k.center(col_w[k]) for k in keys) + "|"
    print("\n" + sep)
    print(hdr)
    print(sep)
    for r in results:
        row = "|" + "|".join(str(r[k]).center(col_w[k]) for k in keys) + "|"
        print(row)
    print(sep + "\n")


# ── CHARTS ────────────────────────────────────────────────────────────────────
def plot_results(df, backtests):
    n = len(backtests)
    fig, axes = plt.subplots(n+1, 1, figsize=(16, 4*(n+1)), sharex=False)
    fig.suptitle("NIFTY 50 – Backtest Results (2019-2026)", fontsize=14, fontweight="bold")

    # Price + indicators on top panel
    ax0 = axes[0]
    ax0.plot(df["DATE"], df["CLOSE"], color="steelblue", lw=1.2, label="NIFTY 50 Close")
    ax0.plot(df["DATE"], df["SMA_FAST"], color="orange",  lw=0.9, label="SMA20", alpha=0.8)
    ax0.plot(df["DATE"], df["SMA_SLOW"], color="red",     lw=0.9, label="SMA50", alpha=0.8)
    ax0.fill_between(df["DATE"], df["BB_UP"], df["BB_LOW"], alpha=0.07, color="purple", label="BB")
    ax0.set_ylabel("Price (INR)")
    ax0.legend(loc="upper left", fontsize=7)
    ax0.set_title("NIFTY 50 Price + Indicators")
    ax0.grid(True, alpha=0.3)

    colors = plt.cm.tab10.colors
    for idx, bt in enumerate(backtests):
        ax = axes[idx+1]
        eq = np.array(bt.equity_curve, dtype=float)
        dates = df["DATE"].iloc[:len(eq)]
        ax.plot(dates, eq/1e5, color=colors[idx % len(colors)], lw=1.3, label=bt.name)

        # Drawdown shading
        dd = eq / np.maximum.accumulate(eq) - 1
        ax2 = ax.twinx()
        ax2.fill_between(dates, dd*100, 0, alpha=0.2, color="red")
        ax2.set_ylabel("DD %", fontsize=7, color="red")
        ax2.tick_params(axis="y", labelcolor="red", labelsize=7)
        ax2.set_ylim(-100, 5)

        # Trade markers
        td = pd.DataFrame(bt.trades)
        if len(td) > 0:
            buys  = td[td["type"].str.contains("BUY|ENTRY",  na=False)]
            sells = td[td["type"].str.contains("SELL|EXIT",  na=False)]
            for _, row in buys.iterrows():
                idx_d = df[df["DATE"]==row["date"]].index
                if len(idx_d)>0 and idx_d[0]<len(eq):
                    ax.axvline(row["date"], color="green", lw=0.4, alpha=0.5)
            for _, row in sells.iterrows():
                idx_d = df[df["DATE"]==row["date"]].index
                if len(idx_d)>0 and idx_d[0]<len(eq):
                    ax.axvline(row["date"], color="red", lw=0.4, alpha=0.5)

        ax.set_ylabel("Equity (Rs Lakh)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        m = bt.metrics()
        info = "CAGR:{cagr}%  MaxDD:{mdd}%  Sharpe:{sh}  Trades:{tr}  WinRate:{wr}%".format(
            cagr=m["CAGR %"], mdd=m["Max DD %"], sh=m["Sharpe"],
            tr=m["# Trades"], wr=m["Win Rate %"])
        ax.set_title(bt.name + "   |   " + info, fontsize=9)

    plt.tight_layout()
    plt.savefig("backtest_results.png", dpi=150, bbox_inches="tight")
    print("Chart saved: backtest_results.png")
    plt.show()


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    df = load_data(DATA_FILE)
    df = add_indicators(df)
    print("Rows: {}  |  {:%d-%b-%Y} to {:%d-%b-%Y}".format(
        len(df), df["DATE"].iloc[0], df["DATE"].iloc[-1]))

    # Define strategy + mode combos to run
    combos = [
        ("SMA Crossover", sig_sma,  "SPOT"),
        ("SMA Crossover", sig_sma,  "FUTURE"),
        ("RSI RevMean",   sig_rsi,  "SPOT"),
        ("RSI RevMean",   sig_rsi,  "FUTURE"),
        ("Bollinger",     sig_bb,   "SPOT"),
        ("Bollinger",     sig_bb,   "FUTURE"),
        ("MACD",          sig_macd, "SPOT"),
        ("MACD",          sig_macd, "FUTURE"),
        ("SMA Crossover", sig_sma,  "OPTION"),
        ("MACD",          sig_macd, "OPTION"),
    ]

    backtests = []
    results   = []
    print("\nRunning backtests...")
    for name, sig_fn, mode in combos:
        signals = sig_fn(df)
        bt = BacktestEngine(df, signals, mode=mode, name=name)
        backtests.append(bt)
        m = bt.metrics()
        results.append(m)
        print("  Done: {}".format(bt.name))

    # Print performance table
    print_table(results)

    # Save trade log
    all_trades = []
    for bt in backtests:
        for t in bt.trades:
            t["strategy"] = bt.name
            all_trades.append(t)
    if all_trades:
        pd.DataFrame(all_trades).to_csv("trade_log.csv", index=False)
        print("Trade log saved: trade_log.csv")

    # Plot (only first 6 to keep chart readable)
    print("\nGenerating chart...")
    plot_results(df, backtests[:6])


if __name__ == "__main__":
    main()
