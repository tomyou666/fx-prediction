"""
PSO-LSTM 為替予測の共通モジュール（設計書準拠）。
データ読込・前処理（MODWTローリング・テクニカル・周期エンコーディング）・
シーケンス作成・EWMAスケーリング・LSTM構築・PSO・評価を提供する。
目的変数: TARGET_HORIZON 本先の対数収益 y_t = ln(P_{t+h}/P_t)。h は共通パラメータで指定。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pywt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import ta
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pyswarms as ps
from typing import Any

# 定数（設計書付録）
L2_LAMBDA = 1e-4
DROPOUT_RATE = 0.2
BATCH_SIZE = 1024
NEURON_BOUNDS = (50, 300)
EPOCH_BOUNDS = (50, 300)
LAYER_BOUNDS = (1, 3)
PSO_W = 0.8
PSO_C1 = 1.5
PSO_C2 = 1.5
PSO_PARTICLES = 20
PSO_ITERS = 5
WT_LEVEL = 3
WT_WAVELET = "haar"
CLIP_SIGMA = 3
CORR_THRESHOLD = 0.95
PERIOD_COLS = ("dow_sin", "dow_cos", "hour_sin", "hour_cos")
# 何本先の対数収益で学習・推論するか（設計書 TARGET_HORIZON）。6 のときは 6 本先。
TARGET_HORIZON = 6
# 対数収益のスケーリング係数（学習時に掛ける。評価・可視化前に 1/TARGET_SCALE を掛けて戻す）
TARGET_SCALE = 1e4

# CSVで使用する列（US02Y_close は使用しない）
CSV_PRICE_COLS = ["open", "high", "low", "close", "volume", "vwap"]
CSV_MACRO_COLS = ["EURUSD_close", "EURJPY_close"]


def load_csv(
    path: str,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """指定パスのCSVを読み込み、timestampをdatetimeにし、列名・型を統一して返す。
    US02Y_close は使用しない（読み込まない）。
    """
    df = pd.read_csv(path)
    if timestamp_column not in df.columns:
        raise ValueError(f"列 '{timestamp_column}' がCSVに存在しません")
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    df = df.set_index(timestamp_column)
    use_cols = [c for c in CSV_PRICE_COLS + CSV_MACRO_COLS if c in df.columns]
    df = df[use_cols].copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def resample_ohlcv(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """OHLCVを指定分足にリサンプル。5分足の場合はそのままコピーを返す。"""
    if minutes == 5:
        return df.copy()
    rule = f"{minutes}min"
    ohlc = df["open"].resample(rule).first()
    high = df["high"].resample(rule).max()
    low = df["low"].resample(rule).min()
    close = df["close"].resample(rule).last()
    volume = df["volume"].resample(rule).sum()
    out = pd.concat([ohlc, high, low, close, volume], axis=1)
    out.columns = ["open", "high", "low", "close", "volume"]
    for c in CSV_MACRO_COLS:
        if c in df.columns:
            out[c] = df[c].resample(rule).last()
    return out.dropna(how="all").ffill()


def _modwt_rolling_one(
    arr: np.ndarray,
    end_idx: int,
    window_len: int,
    wavelet: str,
    level: int,
) -> float:
    """時点 end_idx において、過去のみの窓 [end_idx - window_len + 1, end_idx] で
    SWT（pywt）で分解・再構成し、再構成系列の末尾1点を返す。未来データを使わない。
    """
    start = max(0, end_idx - window_len + 1)
    window = arr[start : end_idx + 1].astype(np.float64)
    n = len(window)
    # SWTは長さが2^levelで割り切れる必要がある
    min_len = 2**level
    if n < min_len:
        return np.nan
    # 末尾を2^levelの倍数に切り詰める
    trim = n % (2**level)
    if trim > 0:
        window = window[trim:]
    if len(window) < min_len:
        return np.nan
    try:
        coeffs = pywt.swt(window, wavelet, level=level)
        # coeffs = [(cA_n, cD_n), (cA_n-1, cD_n-1), ...]
        detail_coeffs = [c[1] for c in coeffs]
        sigma = np.median(np.abs(detail_coeffs[-1])) / 0.6745 if detail_coeffs else 0.0
        uthresh = sigma * np.sqrt(2 * np.log(len(window))) if sigma > 0 else 0.0
        new_coeffs = []
        for cA, cD in coeffs:
            cD_th = pywt.threshold(cD, uthresh, mode="soft")
            new_coeffs.append((cA, cD_th))
        rec = pywt.iswt(new_coeffs, wavelet)
        val = float(rec[-1])
        # sigma=0 のときは閾値処理なし＝再構成は入力に近い。iswt が NaN を返すことがあるので窓末尾で補う
        if np.isnan(val) and not np.any(np.isnan(window)) and np.isfinite(window[-1]):
            val = float(window[-1])
        return val
    except Exception:
        return np.nan


def modwt_rolling_denoise(
    series: pd.Series,
    wavelet: str = WT_WAVELET,
    level: int = WT_LEVEL,
    window_len: int | None = None,
) -> pd.Series:
    """MODWT（Haar, level 3）に相当するノイズ低減をローリング窓で適用する。
    各時点 t では t より過去のデータのみを使用し、未来データを使わない。
    """
    if window_len is None:
        window_len = max(64, 2**level * 8)
    arr = np.asarray(series, dtype=np.float64).ravel()
    # inf は pywt で NaN を生むため、事前に NaN に置換
    arr = np.where(np.isinf(arr), np.nan, arr)
    out = np.full(len(arr), np.nan, dtype=np.float64)
    for t in range(len(arr)):
        out[t] = _modwt_rolling_one(arr, t, window_len, wavelet, level)
    return pd.Series(out, index=series.index, name=series.name)


def add_log_ratio_and_wt(
    df: pd.DataFrame,
    price_cols: list[str],
    wavelet: str = WT_WAVELET,
    level: int = WT_LEVEL,
    volume_safe: bool = True,
) -> pd.DataFrame:
    """価格・出来高を対数比率 ln(x_t/x_{t-1}) に変換し、WTでノイズ低減した列を追加する。
    volume_safe が True のときは volume 列に 1 を足して log(0) を防ぐ。
    """
    out = df.copy()
    for col in price_cols:
        if col not in out.columns:
            continue
        s = out[col].astype(float)
        if volume_safe and col == "volume":
            s = s + 1.0
        log_ratio = np.log(s / s.shift(1))
        log_ratio = log_ratio.replace([np.inf, -np.inf], np.nan)
        wt_s = modwt_rolling_denoise(log_ratio, wavelet=wavelet, level=level)
        out[f"ln_{col}"] = wt_s
    return out


def add_usd_eur_log_diff(
    df: pd.DataFrame,
    wavelet: str = WT_WAVELET,
    level: int = WT_LEVEL,
) -> pd.DataFrame:
    """USD/JPY と EUR/JPY の対数剥離を計算し、WT適用後に追加する。"""
    if "close" not in df.columns or "EURJPY_close" not in df.columns:
        return df
    # 対数剥離: ln(USDJPY) - ln(EURJPY) のスプレッド的な量
    log_diff = np.log(df["close"].astype(float)) - np.log(
        df["EURJPY_close"].astype(float)
    )
    log_diff = pd.Series(log_diff.values, index=df.index)
    wt_s = modwt_rolling_denoise(log_diff, wavelet=wavelet, level=level)
    df = df.copy()
    df["usd_eur_log_diff"] = wt_s
    return df


def add_technical_indicators_log_diff(df: pd.DataFrame) -> pd.DataFrame:
    """テクニカル指標を追加し、終値との対数乖離 ln(I_t/P_t) で統一する。"""
    out = df.copy()
    close = out["close"].astype(float)

    macd = ta.trend.MACD(close=close)
    macd_hist = macd.macd_diff()
    out["ln_macd_diff"] = np.log(1.0 + np.abs(macd_hist) / (close + 1e-10))

    cci = ta.trend.cci(
        out["high"].astype(float),
        out["low"].astype(float),
        close,
    )
    # CCI は -300～+300 程度。300 を足して 0 以上にしつつ水準を保つ（abs より情報量が多い）
    out["ln_cci_diff"] = np.log(1.0 + np.maximum(0, cci + 300) / (close + 1e-10))

    atr = ta.volatility.average_true_range(
        out["high"].astype(float),
        out["low"].astype(float),
        close,
    )
    out["ln_atr_diff"] = np.log(atr / (close + 1e-10) + 1e-10)

    eps = 1e-10
    boll = ta.volatility.BollingerBands(close=close)
    out["ln_boll_diff"] = np.log(boll.bollinger_mavg() / (close + eps) + eps)

    ema20 = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    out["ln_ema20_diff"] = np.log(ema20 / (close + eps) + eps)

    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    out["ln_ma5_diff"] = np.log(ma5 / (close + eps) + eps)
    out["ln_ma10_diff"] = np.log(ma10 / (close + eps) + eps)

    mtm6 = close - close.shift(6)
    mtm12 = close - close.shift(12)
    out["ln_mtm6_diff"] = np.log(1.0 + mtm6 / (close + 1e-10))
    out["ln_mtm12_diff"] = np.log(1.0 + mtm12 / (close + 1e-10))

    roc = ta.momentum.ROCIndicator(close, window=10).roc()
    out["ln_roc_diff"] = np.log(1.0 + roc / 100.0)

    hl = (out["high"].astype(float) + out["low"].astype(float)) / 2
    diff = close - hl
    hl_range = out["high"].astype(float) - out["low"].astype(float)
    ema1 = diff.ewm(span=14, adjust=False).mean()
    ema2 = ema1.ewm(span=14, adjust=False).mean()
    range_ema1 = hl_range.ewm(span=14, adjust=False).mean()
    range_ema2 = range_ema1.ewm(span=14, adjust=False).mean()
    smi = 100 * (ema2 / (0.5 * range_ema2.replace(0, np.nan)))
    out["ln_smi_diff"] = np.log(1.0 + (smi + 100) / 200.0)

    denom = (out["high"] - out["low"]).replace(0, np.nan)
    wvad = ((out["close"] - out["open"]) / denom) * out["volume"]
    out["ln_wvad_diff"] = np.log(1.0 + np.abs(wvad) / (close + 1e-10))

    return out


def add_period_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """曜日・時間を sin/cos 周期エンコーディングで追加する。正規化は行わない。"""
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        return out
    dow = out.index.dayofweek
    hour = out.index.hour
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out


def preprocess_5m_pipeline(
    df: pd.DataFrame,
    wavelet: str = WT_WAVELET,
    level: int = WT_LEVEL,
    steps_ahead: int | None = None,
) -> pd.DataFrame:
    """設計書に沿った5分足前処理パイプライン。
    リサンプル済みのDataFrameを受け、対数比率+WT・テクニカル・周期・目的変数まで一括適用する。
    steps_ahead: 何本先の対数収益を目的変数にするか。None のときは TARGET_HORIZON を使用。
    """
    if steps_ahead is None:
        steps_ahead = TARGET_HORIZON
    # マクロは前方補完（CSV結合済みの場合はffillのみ）
    out = df.ffill()
    price_cols = [
        c for c in ["open", "high", "low", "close", "volume"] if c in out.columns
    ]
    out = add_log_ratio_and_wt(out, price_cols, wavelet=wavelet, level=level)
    if "EURJPY_close" in out.columns:
        eur_jpy = out["EURJPY_close"].astype(float)
        log_ratio = np.log(eur_jpy / eur_jpy.shift(1))
        log_ratio = log_ratio.replace([np.inf, -np.inf], np.nan)
        out["eur_jpy_ln_close"] = modwt_rolling_denoise(
            log_ratio, wavelet=wavelet, level=level
        )
    out = add_usd_eur_log_diff(out, wavelet=wavelet, level=level)
    out = add_technical_indicators_log_diff(out)
    out = add_period_encoding(out)
    out["target_log_return"] = build_target_log_return(
        out, close_col="close", steps_ahead=steps_ahead
    )
    return out


def build_target_log_return(
    df: pd.DataFrame,
    close_col: str = "close",
    steps_ahead: int | None = None,
) -> pd.Series:
    """steps_ahead 本先の対数収益 y_t = ln(P_{t+h}/P_t) を計算して返す。
    steps_ahead が None のときは TARGET_HORIZON を使用。末尾 h 行は NaN。"""
    if steps_ahead is None:
        steps_ahead = TARGET_HORIZON
    close = pd.Series(np.asarray(df[close_col]).ravel(), index=df.index)
    next_close = close.shift(-steps_ahead)
    log_ret = np.log(next_close / close)
    log_ret.name = "target_log_return"
    return log_ret


def _flatten_column_index(df: pd.DataFrame) -> pd.DataFrame:
    """MultiIndex列を1段階の文字列列名に変換する。"""
    if df.columns.nlevels <= 1:
        return df.copy()
    new_cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            part = [str(x).strip() for x in c if x]
            new_cols.append("_".join(part) if part else str(c[0]))
        else:
            new_cols.append(str(c))
    out = df.copy()
    out.columns = new_cols
    return out


def remove_high_corr_features(
    df: pd.DataFrame,
    target_col: str,
    threshold: float = CORR_THRESHOLD,
    protect_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """目的変数との絶対相関が threshold を超える特徴量を削除し、削除した列名のリストも返す。
    target_col および protect_cols は削除しない。
    """
    df = _flatten_column_index(df)
    protect = set(protect_cols or [])
    corr = df.corr(numeric_only=True)
    if target_col not in corr.columns:
        raise ValueError(f"target_col '{target_col}' not found in the DataFrame.")
    corr_series = corr[target_col].abs()
    if isinstance(corr_series, pd.DataFrame):
        corr_series = corr_series.squeeze()
    names = list(corr_series.index)
    vals = np.asarray(corr_series).ravel()
    is_high = vals > threshold
    not_self = np.array([n != target_col for n in names], dtype=bool)
    drop_cols = [
        names[i]
        for i in range(len(names))
        if is_high[i] and not_self[i] and names[i] not in protect
    ]
    return df.drop(columns=drop_cols), drop_cols


def create_sequences(
    features: np.ndarray,
    target: np.ndarray,
    lookback: int,
    close_col_idx: int | None = None,
    close_series: pd.Series | None = None,
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """ルックバック長 L の窓で入力 X と目的変数 y を生成する。
    窓の末尾時点 t に対応する目的変数は y_t（1本先の対数収益）。
    close_col_idx を指定した場合、各サンプルの最終ステップの終値インデックスから last_close を返す。
    close_series を指定した場合、各サンプルに対応する時点の終値（価格スケール復元用）を返す。
    """
    xs, ys = [], []
    for i in range(lookback, len(features)):
        xs.append(features[i - lookback : i])
        ys.append(target[i])
    X = np.array(xs)
    y = np.array(ys)

    if close_series is not None:
        # サンプル i は時点 lookback+i に対応。復元用 last_close は close[lookback+i]
        last_close = np.asarray(
            close_series.iloc[lookback : len(features)], dtype=np.float64
        )
        return X, y, last_close
    if close_col_idx is not None:
        last_close = features[np.arange(lookback - 1, len(features) - 1), close_col_idx]
        return X, y, last_close
    return X, y


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    last_close: np.ndarray | None = None,
) -> Any:
    """時系列順を保ち、先頭から train_ratio を訓練ブロック、そのうち val_ratio を検証、残りをテストに分割する。
    train_ratio=0.8, val_ratio=0.2 のとき、訓練64% / 検証16% / テスト20%。
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(train_end * (1 - val_ratio))
    X_train, X_val, X_test = X[:val_end], X[val_end:train_end], X[train_end:]
    y_train, y_val, y_test = y[:val_end], y[val_end:train_end], y[train_end:]
    if last_close is not None:
        lc_train = last_close[:val_end]
        lc_val = last_close[val_end:train_end]
        lc_test = last_close[train_end:]
        return (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            lc_train,
            lc_val,
            lc_test,
        )
    return X_train, y_train, X_val, y_val, X_test, y_test


class EWMAScaler:
    """EWMAに基づくローリング標準化。±clip_sigma*σでクリップしてから標準化。
    訓練データのみでフィットし、検証・テストは訓練の状態を引き継いで逐次適用する（リーク防止）。
    周期エンコーディング列はスケールしない。
    """

    def __init__(
        self,
        alpha: float = 0.06,
        clip_sigma: float = CLIP_SIGMA,
        exclude_cols: tuple[str, ...] = PERIOD_COLS,
        eps: float = 1e-8,
    ):
        self.alpha = alpha
        self.clip_sigma = clip_sigma
        self.exclude_cols = exclude_cols
        self.eps = eps
        self.ewma_mean_: np.ndarray | None = None
        self.ewma_var_: np.ndarray | None = None
        self.scale_cols_: np.ndarray | None = None  # True = スケールする

    def fit(
        self,
        X_3d: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "EWMAScaler":
        """訓練データ (samples, lookback, features) でEWMAをフィットする。時系列順に逐次更新。"""
        n_samples, lookback, n_features = X_3d.shape
        X_flat = X_3d.reshape(-1, n_features)
        if feature_names is None:
            scale_cols = np.ones(n_features, dtype=bool)
        else:
            scale_cols = np.array(
                [c not in self.exclude_cols for c in feature_names],
                dtype=bool,
            )
        self.scale_cols_ = scale_cols
        ewma_mean = np.zeros(n_features)
        ewma_mean[scale_cols] = np.nan_to_num(
            X_flat[0, scale_cols], nan=0.0, posinf=0.0, neginf=0.0
        )
        ewma_var = np.ones(n_features) * self.eps
        for i in range(len(X_flat)):
            x = X_flat[i]
            clip_val = self.clip_sigma * np.sqrt(ewma_var + self.eps)
            x_clipped = x.copy()
            x_clipped[scale_cols] = np.clip(
                x[scale_cols],
                ewma_mean[scale_cols] - clip_val[scale_cols],
                ewma_mean[scale_cols] + clip_val[scale_cols],
            )
            old_mean = ewma_mean[scale_cols].copy()
            ewma_mean[scale_cols] = (
                self.alpha * x_clipped[scale_cols]
                + (1 - self.alpha) * ewma_mean[scale_cols]
            )
            ewma_var[scale_cols] = (
                self.alpha * (x_clipped[scale_cols] - old_mean) ** 2
                + (1 - self.alpha) * ewma_var[scale_cols]
            )
        self.ewma_mean_ = ewma_mean
        self.ewma_var_ = ewma_var
        return self

    def transform(
        self,
        X_3d: np.ndarray,
        state_mean: np.ndarray | None = None,
        state_var: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """X_3d をスケールし、更新された (mean, var) 状態を返す。時系列順に逐次更新。"""
        if self.ewma_mean_ is None:
            raise ValueError("fit が呼ばれていません")
        n_samples, lookback, n_features = X_3d.shape
        scale_cols = self.scale_cols_
        mean = state_mean.copy() if state_mean is not None else self.ewma_mean_.copy()
        var = state_var.copy() if state_var is not None else self.ewma_var_.copy()
        # 時系列順 (sample0_t0..tL, sample1_t0..tL, ...) でフラットに処理
        X_flat = X_3d.reshape(-1, n_features)
        out_flat = X_flat.copy()
        std = np.sqrt(var + self.eps)
        clip_val = self.clip_sigma * std
        for i in range(len(X_flat)):
            x = X_flat[i]
            x_clipped = x.copy()
            x_clipped[scale_cols] = np.clip(
                x[scale_cols],
                mean[scale_cols] - clip_val[scale_cols],
                mean[scale_cols] + clip_val[scale_cols],
            )
            x_scaled = x.copy()
            x_scaled[scale_cols] = (x_clipped[scale_cols] - mean[scale_cols]) / (
                std[scale_cols] + self.eps
            )
            out_flat[i] = x_scaled
            old_mean = mean[scale_cols].copy()
            mean[scale_cols] = (
                self.alpha * x_clipped[scale_cols] + (1 - self.alpha) * mean[scale_cols]
            )
            var[scale_cols] = (
                self.alpha * (x_clipped[scale_cols] - old_mean) ** 2
                + (1 - self.alpha) * var[scale_cols]
            )
            std = np.sqrt(var + self.eps)
            clip_val = self.clip_sigma * std
        out = out_flat.reshape(X_3d.shape)
        return out, mean, var

    def fit_transform(
        self,
        X_3d: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> np.ndarray:
        """訓練データでフィットし、訓練データを変換して返す。"""
        self.fit(X_3d, feature_names)
        out, _, _ = self.transform(X_3d)
        return out


def scale_ewma_train_val_test(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str] | None = None,
    alpha: float = 0.06,
    clip_sigma: float = CLIP_SIGMA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, EWMAScaler]:
    """訓練データのみでEWMAスケーラーをフィットし、訓練・検証・テストに適用する。
    検証は訓練の最終状態から、テストは検証の最終状態から逐次適用する。
    """
    scaler = EWMAScaler(alpha=alpha, clip_sigma=clip_sigma)
    X_train_s = scaler.fit_transform(X_train, feature_names)
    X_val_s, mean_v, var_v = scaler.transform(X_val)
    X_test_s, _, _ = scaler.transform(X_test, state_mean=mean_v, state_var=var_v)
    return X_train_s, X_val_s, X_test_s, scaler


def build_lstm_model(
    input_shape: tuple[int, ...],
    num_layers: int,
    num_units: int,
    l2_lambda: float | None = None,
    dropout_rate: float | None = None,
    dense_units: int | None = None,
) -> keras.Model:
    """指定した層数・ユニット数でスタックLSTMを構築。出力はスカラー（1時点先の対数収益）。"""
    if l2_lambda is None:
        l2_lambda = L2_LAMBDA
    if dropout_rate is None:
        dropout_rate = DROPOUT_RATE
    if dense_units is None:
        dense_units = num_units
    l2_reg = keras.regularizers.l2(l2_lambda)
    model = keras.Sequential()
    for i in range(num_layers):
        return_sequences = i < num_layers - 1
        lstm_kw = dict(
            kernel_regularizer=l2_reg,
            recurrent_regularizer=l2_reg,
            bias_regularizer=l2_reg,
        )
        if i == 0:
            model.add(
                layers.LSTM(
                    num_units,
                    return_sequences=return_sequences,
                    input_shape=input_shape,
                    **lstm_kw,
                )
            )
        else:
            model.add(
                layers.LSTM(
                    num_units,
                    return_sequences=return_sequences,
                    **lstm_kw,
                )
            )
        model.add(layers.Dropout(dropout_rate))
    model.add(
        layers.Dense(
            1024,
            activation="relu",
            kernel_regularizer=l2_reg,
            bias_regularizer=l2_reg,
        )
    )
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, kernel_regularizer=l2_reg, bias_regularizer=l2_reg))
    model.compile(optimizer="adam", loss="mae")
    return model


def pso_optimize(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_shape: tuple[int, ...],
    csv_log_path: str = "pso_training.csv",
    *,
    batch_size: int | None = None,
    neuron_bounds: tuple[int, int] | None = None,
    epoch_bounds: tuple[int, int] | None = None,
    layer_bounds: tuple[int, int] | None = None,
    pso_w: float | None = None,
    pso_c1: float | None = None,
    pso_c2: float | None = None,
    pso_particles: int | None = None,
    pso_iters: int | None = None,
    strategy: tf.distribute.Strategy | None = None,
) -> tuple[np.ndarray, float]:
    """PSOでLSTMのハイパーパラメータを探索し、検証RMSEを最小化する。
    戻り値: (best_pos, best_cost)。best_pos は [ユニット数, エポック数, 層数]。
    """
    batch_size = batch_size or BATCH_SIZE
    neuron_bounds = neuron_bounds or NEURON_BOUNDS
    epoch_bounds = epoch_bounds or EPOCH_BOUNDS
    layer_bounds = layer_bounds or LAYER_BOUNDS
    pso_w = pso_w if pso_w is not None else PSO_W
    pso_c1 = pso_c1 if pso_c1 is not None else PSO_C1
    pso_c2 = pso_c2 if pso_c2 is not None else PSO_C2
    pso_particles = pso_particles or PSO_PARTICLES
    pso_iters = pso_iters or PSO_ITERS

    def _objective(particles: np.ndarray) -> np.ndarray:
        costs = []
        for particle in particles:
            units = int(np.clip(round(particle[0]), *neuron_bounds))
            epochs = int(np.clip(round(particle[1]), *epoch_bounds))
            n_layers = int(np.clip(round(particle[2]), *layer_bounds))
            tf.keras.backend.clear_session()
            if strategy is not None:
                with strategy.scope():
                    model = build_lstm_model(input_shape, n_layers, units)
            else:
                model = build_lstm_model(input_shape, n_layers, units)
            es = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            )
            csv_log = keras.callbacks.CSVLogger(csv_log_path)
            model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=[es, csv_log],
            )
            preds = model.predict(X_val, verbose=0)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            costs.append(rmse)
        return np.array(costs)

    options = {"c1": pso_c1, "c2": pso_c2, "w": pso_w}
    lower = np.array([neuron_bounds[0], epoch_bounds[0], layer_bounds[0]])
    upper = np.array([neuron_bounds[1], epoch_bounds[1], layer_bounds[1]])
    optimizer = ps.single.GlobalBestPSO(
        n_particles=pso_particles,
        dimensions=3,
        options=options,
        bounds=(lower, upper),
    )
    best_cost, best_pos = optimizer.optimize(_objective, iters=pso_iters, verbose=False)
    return best_pos, float(best_cost)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-8,
) -> tuple[float, float, float, float]:
    """RMSE, MAE, MAPE(%), R2 を対数収益スケールで計算する。MAPEは分母に y_true+eps を使用。"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.asarray(y_true) + eps))) * 100.0
    r2 = r2_score(y_true, y_pred)
    return float(rmse), float(mae), float(mape), float(r2)
