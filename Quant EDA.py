# === Imports ===
import yfinance as yf
import pandas as pd
import numpy as np
from pykalman import KalmanFilter
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import warnings
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

ticker = 'GOOGL'
df = yf.download(ticker, start='2010-01-01', end='2025-01-01')

# Clean MultiIndex
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Squeeze price columns
for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
    if df[col].ndim > 1:
        df[col] = df[col].squeeze()

# Create 'Close' Series variable
Close = df['Close'].squeeze()
High = df['High'].squeeze()
Low = df['Low'].squeeze()
Open = df['Open'].squeeze()
Volume = df['Volume'].squeeze()

df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

# === 2. Kalman Filter Smoothing ===
kf = KalmanFilter(initial_state_mean=df['Close'].values[0], n_dim_obs=1)
state_means, _ = kf.em(df['Close'].values).smooth(df['Close'].values)
df['kf_smooth'] = state_means

# === 3. Kalman Filter Slope (Bull/Bear Regime) ===
df['kf_slope'] = pd.Series(df['kf_smooth']).diff()
df['regime'] = np.where(df['kf_slope'] > 0, 1, -1)

# === 4. Bollinger Bands ===

bb = BollingerBands(close=Close, window=20, window_dev=2)
bb_upper = bb.bollinger_hband()
bb_lower = bb.bollinger_lband()

# Align output with the main DataFrame
df['bb_upper'] = bb_upper.reindex(df.index)
df['bb_lower'] = bb_lower.reindex(df.index)
df['bb_width'] = df['bb_upper'] - df['bb_lower']

# === 5. Z-Score ===
rolling_mean = df['Close'].rolling(window=20).mean()
rolling_std = df['Close'].rolling(window=20).std()
df['z_score'] = (df['Close'] - rolling_mean) / rolling_std

# === 6. RSI ===
df['rsi'] = RSIIndicator(close=df['Close'], window=14).rsi()

# === 7. ATR ===
df['atr'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()

# === 8. ACF / PACF ===
acf_vals = acf(df['log_return'], nlags=20)
pacf_vals = pacf(df['log_return'], nlags=20)

# === 9. 1st/2nd Derivatives ===
df['1st_deriv'] = df['Close'].diff()
df['2nd_deriv'] = df['1st_deriv'].diff()

# === 10. Price-Volume Interactions ===
df['pv'] = df['Close'] * df['Volume']
df['vol_pressure'] = df['log_return'] * df['Volume']
df['rel_volume'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
df['vol_spike'] = df['rel_volume'] > 2
df['rel_vol_pressure'] = df['log_return'] * df['rel_volume']

# === 11. Relative Volatility Features ===
df['volatility_20'] = df['log_return'].rolling(window=20).std()
df['rel_atr'] = df['atr'] / df['Close']
df['vol_ratio'] = df['log_return'].rolling(window=5).std() / df['log_return'].rolling(window=20).std()
df['bb_z_width'] = df['bb_width'] / df['Close']

# === 12. Subtract ARIMA(0,1,0) ===
model = ARIMA(df['Close'], order=(0, 1, 0)).fit()
df['arima_resid'] = model.resid

# === 13. Recalculate Indicators on Residuals ===
resid = df['arima_resid'].dropna()

df['resid_z'] = (resid - resid.rolling(20).mean()) / resid.rolling(20).std()
df['resid_rsi'] = RSIIndicator(close=resid, window=14).rsi()
df['resid_1st_deriv'] = resid.diff()
df['resid_2nd_deriv'] = df['resid_1st_deriv'].diff()

df['resid_vol_pressure'] = df['resid_1st_deriv'] * df['Volume']
df['resid_rel_vol_pressure'] = df['resid_1st_deriv'] * df['rel_volume']

df['resid_volatility'] = df['arima_resid'].rolling(window=20).std()
df['resid_vol_ratio'] = df['arima_resid'].rolling(window=5).std() / df['arima_resid'].rolling(window=20).std()
df['resid_rel_volatility'] = df['resid_volatility'] / df['Close']

## ------------------------------------------------- ##
## ---- Close with b-bands and bull/bear regime ---- ##
## ------------------------------------------------- ##

plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Close'], label='Close Price', color='black', linewidth=1)

plt.plot(df.index, df['bb_upper'], label='Bollinger Upper', color='blue', linewidth=1)
plt.plot(df.index, df['bb_lower'], label='Bollinger Lower', color='blue', linewidth=1)
plt.fill_between(df.index, df['bb_lower'], df['bb_upper'], color='blue', alpha=0.15, label='Bollinger Range')
plt.fill_between(df.index, df['Close'].min(), df['Close'].max(),
                 where=(df['regime'] == 1), facecolor='green', alpha=0.2, label='Bull Regime')
plt.fill_between(df.index, df['Close'].min(), df['Close'].max(),
                 where=(df['regime'] == -1), facecolor='red', alpha=0.25, label='Bear Regime')

plt.title(f"{ticker} Bull/Bear Regimes with Bollinger Bands", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

## ------------------------------------------------- ##
## ---------------- derivatives  ------------------- ##
## ------------------------------------------------- ##

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# === 1st Derivative (Velocity) ===
axes[0].plot(df.index, df['1st_deriv'], color='blue', label='1st Derivative (Velocity)')
axes[0].axhline(0, color='gray', linestyle='--', linewidth=1)
axes[0].set_title(f"{ticker} - 1st Derivative with Bull/Bear Regimes")
axes[0].set_ylabel("ΔPrice")
axes[0].legend(loc='upper left')
axes[0].grid(True, linestyle='--', alpha=0.3)
axes[0].fill_between(df.index, axes[0].get_ylim()[0], axes[0].get_ylim()[1],
                     where=(df['regime'] == 1), facecolor='green', alpha=0.08)
axes[0].fill_between(df.index, axes[0].get_ylim()[0], axes[0].get_ylim()[1],
                     where=(df['regime'] == -1), facecolor='red', alpha=0.08)

# === 2nd Derivative (Acceleration) ===
axes[1].plot(df.index, df['2nd_deriv'], color='purple', label='2nd Derivative (Acceleration)')
axes[1].axhline(0, color='gray', linestyle='--', linewidth=1)
axes[1].set_title(f"{ticker} - 2nd Derivative with Bull/Bear Regimes")
axes[1].set_ylabel("Δ(ΔPrice)")
axes[1].set_xlabel("Date")
axes[1].legend(loc='upper left')
axes[1].grid(True, linestyle='--', alpha=0.3)
axes[1].fill_between(df.index, axes[1].get_ylim()[0], axes[1].get_ylim()[1],
                     where=(df['regime'] == 1), facecolor='green', alpha=0.08)
axes[1].fill_between(df.index, axes[1].get_ylim()[0], axes[1].get_ylim()[1],
                     where=(df['regime'] == -1), facecolor='red', alpha=0.08)

plt.tight_layout()
plt.show()

## ------------------------------------------------- ##
## ---------------- Z-Score ------------------------ ##
## ------------------------------------------------- ##

plt.figure(figsize=(14, 5))

# Plot Z-score
plt.plot(df.index, df['z_score'], label='Z-Score', color='purple', linewidth=1.5)

# Horizontal Thresholds
plt.axhline(2, color='red', linestyle='--', linewidth=1, label='Overbought (+2)')
plt.axhline(-2, color='blue', linestyle='--', linewidth=1, label='Oversold (−2)')
plt.axhline(0, color='gray', linestyle='-', linewidth=1)

# Shade Bull/Bear Regimes
plt.fill_between(df.index, df['z_score'].min(), df['z_score'].max(),
                 where=(df['regime'] == 1), facecolor='green', alpha=0.1)
plt.fill_between(df.index, df['z_score'].min(), df['z_score'].max(),
                 where=(df['regime'] == -1), facecolor='red', alpha=0.1)

# Labels and Layout
plt.title(f"{ticker} Z-Score with Bull/Bear Regime Shading", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Z-Score")
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

## ------------------------------------------------- ##
## ---------------- RSI & ATR ---------------------- ##
## ------------------------------------------------- ##

# RSI: price momentum
# RSI > 70 → Overbought → Price may be due for a pullback
# RSI < 30 → Oversold → Price may be due for a bounce
# RSI rising → Upward momentum building
# RSI falling → Downward momentum building
#
# ATR: Market volatility
# High ATR → Increased volatility (e.g., earnings, news, panic, breakout)
# Low ATR → Compression, range-bound behavior

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(df.index, df['rsi'], label='RSI (14)', color='teal', linewidth=1.5)
axes[0].axhline(70, color='red', linestyle='--', linewidth=1, label='Overbought')
axes[0].axhline(30, color='blue', linestyle='--', linewidth=1, label='Oversold')
axes[0].set_ylabel("RSI")
axes[0].set_title(f"{ticker} RSI & ATR with Bull/Bear Regimes")
axes[0].legend(loc='upper left')
axes[0].grid(True, linestyle='--', alpha=0.3)

axes[0].fill_between(df.index, axes[0].get_ylim()[0], axes[0].get_ylim()[1],
                     where=(df['regime'] == 1), facecolor='green', alpha=0.08)
axes[0].fill_between(df.index, axes[0].get_ylim()[0], axes[0].get_ylim()[1],
                     where=(df['regime'] == -1), facecolor='red', alpha=0.08)

axes[1].plot(df.index, df['atr'], label='ATR (14)', color='darkorange', linewidth=1.5)
axes[1].set_ylabel("ATR")
axes[1].set_xlabel("Date")
axes[1].legend(loc='upper left')
axes[1].grid(True, linestyle='--', alpha=0.3)

axes[1].fill_between(df.index, axes[1].get_ylim()[0], axes[1].get_ylim()[1],
                     where=(df['regime'] == 1), facecolor='green', alpha=0.08)
axes[1].fill_between(df.index, axes[1].get_ylim()[0], axes[1].get_ylim()[1],
                     where=(df['regime'] == -1), facecolor='red', alpha=0.08)

plt.tight_layout()
plt.show()

## ------------------------------------------------- ##
## -------- Price - Volume interactions ------------ ##
## ------------------------------------------------- ##

date_format = mdates.DateFormatter('%Y')
fig1, axs1 = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig1.suptitle(f"{ticker} - Price × Volume Features with Bull/Bear Regime", fontsize=16)

def shade(ax):  # Regime shading helper
    ax.fill_between(df.index, ax.get_ylim()[0], ax.get_ylim()[1],
                    where=(df['regime'] == 1), facecolor='green', alpha=0.06)
    ax.fill_between(df.index, ax.get_ylim()[0], ax.get_ylim()[1],
                    where=(df['regime'] == -1), facecolor='red', alpha=0.06)

# PV
axs1[0].plot(df.index, df['pv'], color='teal', label='Price × Volume (PV)')
axs1[0].set_ylabel('PV')
axs1[0].legend(loc='upper left')
axs1[0].grid(True)
shade(axs1[0])

# Volume Pressure
axs1[1].plot(df.index, df['vol_pressure'], color='purple', label='Log Return × Volume')
axs1[1].set_ylabel('Vol Pressure')
axs1[1].legend(loc='upper left')
axs1[1].grid(True)
shade(axs1[1])

# Relative Volume
axs1[2].plot(df.index, df['rel_volume'], color='darkorange', label='Relative Volume')
axs1[2].axhline(2, linestyle='--', color='red', label='Volume Spike Threshold')
axs1[2].set_ylabel('Rel Volume')
axs1[2].legend(loc='upper left')
axs1[2].grid(True)
shade(axs1[2])

# Relative Volume Pressure
axs1[3].plot(df.index, df['rel_vol_pressure'], color='green', label='Rel Vol Pressure')
axs1[3].set_ylabel('Rel Vol Pressure')
axs1[3].legend(loc='upper left')
axs1[3].grid(True)
shade(axs1[3])

axs1[-1].xaxis.set_major_formatter(date_format)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

## ------------------------------------------------- ##
## -------------- Relative Volatitlity ------------- ##
## ------------------------------------------------- ##

# === Plot 2: Relative Volatility Features ===
fig2, axs2 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig2.suptitle(f"{ticker} - Relative Volatility Features with Bull/Bear Regime", fontsize=16)

# Volatility Ratio
axs2[0].plot(df.index, df['vol_ratio'], color='navy', label='Volatility Ratio (5d / 20d)')
axs2[0].axhline(1, linestyle='--', color='gray')
axs2[0].set_ylabel('Vol Ratio')
axs2[0].legend(loc='upper left')
axs2[0].grid(True)
shade(axs2[0])

# Relative ATR
axs2[1].plot(df.index, df['rel_atr'], label='Relative ATR', color='darkred')
axs2[1].set_ylabel('Rel ATR')
axs2[1].legend(loc='upper left')
axs2[1].grid(True)
shade(axs2[1])

# BB Z-Width
axs2[2].plot(df.index, df['bb_z_width'], label='BB Z-Width', color='darkblue')
axs2[2].set_ylabel('BB Z-Width')
axs2[2].legend(loc='upper left')
axs2[2].grid(True)
shade(axs2[2])

axs2[-1].xaxis.set_major_formatter(date_format)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


## ------------------------------------------------- ##
## ---------------- ACF & PACF --------------------- ##
## ------------------------------------------------- ##

# Ensure Kalman smoothed is a Series
kalman_series = pd.Series(df['kf_smooth'].dropna(), index=df.index)

# === Plot ACF and PACF ===
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

plot_acf(kalman_series, lags=40, ax=axes[0], title='ACF of Kalman Smoothed Close')
plot_pacf(kalman_series, lags=40, ax=axes[1], title='PACF of Kalman Smoothed Close', method='ywm')

plt.tight_layout()
plt.show()

## ------------------------------------------------- ##
## ----------- ARIMA(0, 1, 0) residuals ------------ ##
## ------------------------------------------------- ##

plt.figure(figsize=(12, 4))
plt.plot(df['arima_resid'], label='ARIMA(0,1,0) Residuals')
plt.axhline(0, color='gray', linestyle='--')
plt.title(f"{ticker} - ARIMA(0,1,0) Residuals (Deviation from Random Walk)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

# Drop NaNs
resid = df['arima_resid'].dropna()

# Run ADF test
adf_result = adfuller(resid)

# Output results
print(" ADF Statistic < Critical Value --> reject null, series stationary")
print("ADF Test Statistic:", adf_result[0])
print("p-value < 0.05 strong evidence of stationarity")
print("p-value:", adf_result[1])
print("Critical Values:\n")
for key, value in adf_result[4].items():
    print(f"   {key}: {value}")

## ------------------------------------------------- ##
## --------- Post-ARIMA(0, 1, 0) ACF/PACF ---------- ##
## ------------------------------------------------- ##

# Create side-by-side ACF & PACF plots
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

plot_acf(resid, lags=40, ax=axes[0], title='ACF of ARIMA(0,1,0) Residuals')
plot_pacf(resid, lags=40, ax=axes[1], title='PACF of ARIMA(0,1,0) Residuals', method='ywm')

plt.tight_layout()
plt.show()

## ------------------------------------------------- ##
## ---------- Post-ARIMA(0, 1, 0) Stats ------------ ##
## ------------------------------------------------- ##

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

def shade_regimes(ax):
    ax.fill_between(df.index, ax.get_ylim()[0], ax.get_ylim()[1],
                    where=(df['regime'] == 1), facecolor='green', alpha=0.08)
    ax.fill_between(df.index, ax.get_ylim()[0], ax.get_ylim()[1],
                    where=(df['regime'] == -1), facecolor='red', alpha=0.08)

# === 1. Z-Score Plot ===
axes[0].plot(df.index, df['resid_z'], label='Z-Score of Residuals', color='navy')
axes[0].axhline(2, color='red', linestyle='--')
axes[0].axhline(-2, color='green', linestyle='--')
axes[0].axhline(0, color='gray', linestyle='-')
axes[0].set_ylabel("Z-Score")
axes[0].set_title("Z-Score of ARIMA(0,1,0) Residuals")
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.3)
shade_regimes(axes[0])
axes[0].fill_between(df.index, axes[1].get_ylim()[0], axes[1].get_ylim()[1],
                     where=(df['regime'] == 1), facecolor='green', alpha=0.08)
axes[0].fill_between(df.index, axes[1].get_ylim()[0], axes[1].get_ylim()[1],
                     where=(df['regime'] == -1), facecolor='red', alpha=0.08)

# === 2. RSI Plot ===
axes[1].plot(df.index, df['resid_rsi'], label='RSI (Residual)', color='purple')
axes[1].axhline(70, color='red', linestyle='--')
axes[1].axhline(30, color='green', linestyle='--')
axes[1].set_ylabel("RSI")
axes[1].set_title("RSI of ARIMA(0,1,0) Residuals")
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.3)
shade_regimes(axes[1])

axes[1].fill_between(df.index, axes[1].get_ylim()[0], axes[1].get_ylim()[1],
                     where=(df['regime'] == 1), facecolor='green', alpha=0.08)
axes[1].fill_between(df.index, axes[1].get_ylim()[0], axes[1].get_ylim()[1],
                     where=(df['regime'] == -1), facecolor='red', alpha=0.08)

# === 3. 1st Derivative ===
axes[2].plot(df.index, df['resid_1st_deriv'], label='1st Derivative', color='blue')
axes[2].axhline(0, color='gray', linestyle='--')
axes[2].set_ylabel("Δ Residual")
axes[2].set_title("1st Derivative of Residuals")
axes[2].legend()
axes[2].grid(True, linestyle='--', alpha=0.3)
shade_regimes(axes[2])

axes[2].fill_between(df.index, axes[1].get_ylim()[0], axes[1].get_ylim()[1],
                     where=(df['regime'] == 1), facecolor='green', alpha=0.08)
axes[2].fill_between(df.index, axes[1].get_ylim()[0], axes[1].get_ylim()[1],
                     where=(df['regime'] == -1), facecolor='red', alpha=0.08)

# === 4. 2nd Derivative ===
axes[3].plot(df.index, df['resid_2nd_deriv'], label='2nd Derivative', color='darkorange')
axes[3].axhline(0, color='gray', linestyle='--')
axes[3].set_ylabel("Δ(Δ Residual)")
axes[3].set_title("2nd Derivative of Residuals")
axes[3].legend()
axes[3].grid(True, linestyle='--', alpha=0.3)
shade_regimes(axes[3])

axes[3].fill_between(df.index, axes[1].get_ylim()[0], axes[1].get_ylim()[1],
                     where=(df['regime'] == 1), facecolor='green', alpha=0.08)
axes[3].fill_between(df.index, axes[1].get_ylim()[0], axes[1].get_ylim()[1],
                     where=(df['regime'] == -1), facecolor='red', alpha=0.08)

plt.xlabel("Date")
plt.tight_layout()
plt.show()

## ------------------------------------------------- ##
## ----- P-V interactions after ARIMA(0, 1, 0) ----- ##
## ------------------------------------------------- ##

fig1, axs1 = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig1.suptitle(f"{ticker} - Residual Price × Volume Features with Bull/Bear Regime", fontsize=16)

# Residual Vol Pressure
axs1[0].plot(df.index, df['resid_vol_pressure'], color='purple', label='Residual Vol Pressure')
axs1[0].set_ylabel('Vol Pressure')
axs1[0].legend(loc='upper left')
axs1[0].grid(True)
shade(axs1[0])  # uses the same shading helper from before

axes[0].fill_between(df.index, axes[1].get_ylim()[0], axes[1].get_ylim()[1],
                     where=(df['regime'] == 1), facecolor='green', alpha=0.08)
axes[0].fill_between(df.index, axes[1].get_ylim()[0], axes[1].get_ylim()[1],
                     where=(df['regime'] == -1), facecolor='red', alpha=0.08)

# Residual Relative Vol Pressure
axs1[1].plot(df.index, df['resid_rel_vol_pressure'], color='green', label='Residual Rel Vol Pressure')
axs1[1].set_ylabel('Rel Vol Pressure')
axs1[1].legend(loc='upper left')
axs1[1].grid(True)
shade(axs1[1])

axes[1].fill_between(df.index, axes[1].get_ylim()[0], axes[1].get_ylim()[1],
                     where=(df['regime'] == 1), facecolor='green', alpha=0.08)
axes[1].fill_between(df.index, axes[1].get_ylim()[0], axes[1].get_ylim()[1],
                     where=(df['regime'] == -1), facecolor='red', alpha=0.08)

axs1[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()



## ------------------------------------------------- ##
## --- Relative Volatitlity after ARIMA(0, 1, 0) --- ##
## ------------------------------------------------- ##

fig2, axs2 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig2.suptitle(f"{ticker} - Residual Volatility Features with Bull/Bear Regime", fontsize=16)

# Residual Vol Ratio
axs2[0].plot(df.index, df['resid_vol_ratio'], color='navy', label='Residual Volatility Ratio (5d / 20d)')
axs2[0].axhline(1, linestyle='--', color='gray')
axs2[0].set_ylabel('Vol Ratio')
axs2[0].legend(loc='upper left')
axs2[0].grid(True)
shade(axs2[0])

# Residual Relative Volatility
axs2[1].plot(df.index, df['resid_rel_volatility'], color='darkred', label='Residual Relative Volatility')
axs2[1].set_ylabel('Rel Volatility')
axs2[1].legend(loc='upper left')
axs2[1].grid(True)
shade(axs2[1])

# Residual Absolute Volatility
axs2[2].plot(df.index, df['resid_volatility'], label='Residual Volatility (20d STD)', color='blue')
axs2[2].set_ylabel('Volatility')
axs2[2].legend(loc='upper left')
axs2[2].grid(True)
shade(axs2[2])

axs2[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
