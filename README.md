# 📈 NIFTY 50 Backtest Dashboard

Dashboard interactivo para probar estrategias de trading en el índice NIFTY 50 (India) con datos históricos reales 2019-2026, incluyendo simulación de Futuros y Opciones (F&O).

## 🚀 Demo en vivo
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## 📊 Estrategias incluidas
| Estrategia | Descripción |
|---|---|
| **SMA Crossover** | Cruce de medias móviles (tendencia) |
| **RSI Mean Reversion** | Reversión a la media con RSI |
| **Bollinger Bands** | Ruptura de bandas de Bollinger |
| **MACD Signal** | Cruce de línea MACD y señal |

## 🛠️ Instrumentos
- **SPOT** — Exposición directa al índice
- **FUTURE** — Futuros Nifty (lot=50, margen 12% SPAN)
- **OPTION** — Compra de Call/Put ATM (Black-Scholes)

## 📐 Métricas de performance
- CAGR (Retorno anual compuesto)
- Sharpe Ratio & Sortino Ratio
- Máximo Drawdown
- Win Rate & Profit Factor
- Ganancia/Pérdida promedio por trade

## ⚙️ Configuración desde la interfaz
- Capital inicial (Rs)
- Stop-Loss automático (%)
- Períodos de indicadores (SMA, RSI, BB, MACD)
- Rango de fechas personalizado
- Parámetros F&O (margen, IV, días a vencimiento)

## 📁 Archivos
```
app.py                          → Dashboard web (Streamlit)
nifty50_backtest.py             → Script backtest completo
nifty50_backtest_COLAB.py       → Versión Google Colab
requirements.txt                → Dependencias Python
NIFTY_50_Historical_*.xlsx      → Datos históricos NIFTY 50
```

## 🖥️ Correr localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy en Streamlit Cloud
1. Fork este repositorio
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta el repositorio
4. Main file: `app.py`
5. Click Deploy

## 📈 Resultados del backtest (2019-2026)
| Estrategia | Retorno Total | CAGR | Sharpe |
|---|---|---|---|
| SMA Crossover [SPOT] | +133% | 13.09% | 0.75 |
| SMA Crossover [OPTION] | +120% | 12.19% | 0.33 |
| Buy & Hold NIFTY 50 | +112% | 11.3% | — |

> ⚠️ **Disclaimer:** Este dashboard es solo para fines educativos y de análisis. No constituye asesoramiento financiero. Los resultados pasados no garantizan rendimientos futuros.

---
Built with Python · Streamlit · Plotly · Pandas
