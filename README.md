# FX Prediction: LSTM–PSO 為替予測

USD/JPY の **1本先対数収益** を、LSTM と粒子群最適化（PSO）を組み合わせたハイブリッドモデルで予測するプロジェクトです。5分足の為替データを対象とし、論文の枠組みに沿って実装しています。

## 概要

- **目的変数**: USD/JPY の **対数収益**（1本先）$y_t = \ln(P_{t+1}/P_t)$
- **手法**: LSTM による時系列予測 + PSO によるハイパーパラメータ（ユニット数・層数・エポック数）探索
- **データ**: 5分足為替（開発時は CSV、本番はリアルタイム取得を想定）
- **評価**: 対数収益スケールで RMSE 等を算出。可視化時は必要に応じて終値スケールに復元

### 参考論文との対応

| 論文の要素 | 本実装での対応 |
|------------|----------------|
| LSTM による時系列予測 | 説明変数シーケンスから 1 時点先の対数収益を予測 |
| PSO によるハイパーパラメータ探索 | ニューロン数・隠れ層数・エポック数を PSO で探索 |
| 目的関数 | 検証データでの RMSE を最小化 |
| 株価指数 | USD/JPY 為替（5分足）に適用 |

## 技術的な方針

金融時系列の性質（非定常性・ノイズ・非線形性・ボラティリティクラスタリング・外れ値・カレンダー効果）に対し、次の方針を採用しています。

- **定常性**: 対数収益・対数乖離に変換し、定常に近いデータのみを入力
- **ノイズ低減**: MODWT（Haar, level 3）によるウェーブレット変換
- **非線形・長期依存**: LSTM で学習
- **ハイパーパラメータ**: PSO でユニット数・層数・エポック数を探索
- **スケール・外れ値**: EWMA ローリング標準化、±3σ_ewma でクリップ
- **周期**: 曜日・時間を sin/cos 周期エンコーディングで考慮

## リポジトリ構成

```
.
├── docs/
│   ├── 01_basic_design.md              # 基本設計
│   ├── 02_detailed_design_theory.md    # 詳細設計（理論）
│   └── 03_detailed_design_implementation.md  # 詳細設計（実装）
├── data/
│   └── merged-usdjpy-base-*.csv        # 5分足・マクロ結合済み生データ（開発用）
├── pso_lstm_common.py                  # 共通モジュール（読込・前処理・PSO・LSTM・評価）
├── pso_lstm_5m.ipynb                   # 5分足：学習・PSO・評価のメインノートブック
├── pso_lstm_5m_visualize.ipynb         # 可視化・結果確認用ノートブック
├── pyproject.toml / uv.lock            # 依存関係・実行環境
├── .python-version                     # Python バージョン
└── .devcontainer/                      # Dev Container 定義（任意）
```

| 成果物 | 役割 |
|--------|------|
| **pso_lstm_common.py** | データ読込、前処理（WT・テクニカル・正規化）、シーケンス作成・分割、LSTM 構築、PSO、評価指標の共通関数 |
| **pso_lstm_5m.ipynb** | 5分足パイプラインのエントリポイント。CSV 読込 → 前処理 → PSO＋LSTM 学習 → 評価 |
| **pso_lstm_5m_visualize.ipynb** | 学習結果の可視化（実測 vs 予測、学習曲線など） |

## 実行環境

- **Python**: 3.11+
- **パッケージ管理**: [uv](https://github.com/astral-sh/uv)（`uv sync` で依存関係インストール）
- **主な依存**: TensorFlow/Keras, PyWavelets, pyswarms, pandas, scikit-learn, ta など（`pyproject.toml` 参照）

### セットアップ

```bash
# 依存関係のインストール
uv sync

# Jupyter の起動（ノートブックから実行）
uv run jupyter lab
```

Dev Container 利用時はコンテナ内で同様に `uv run jupyter lab` を実行します。

## パイプライン概要

1. **CSV 読込** → タイムスタンプ・列の統一
2. **前処理** → リサンプル・マクロ結合・対数変換・MODWT・テクニカル指標・目的変数作成
3. **シーケンス作成・分割** → ルックバック窓で (X, y) を生成、時系列順に Train/Val/Test 分割
4. **スケーリング** → 訓練データのみで EWMA パラメータを算出し、±3σ クリップ後に標準化
5. **PSO + LSTM 学習** → 検証 RMSE を最小化するハイパーパラメータを探索し、最良モデルで評価
6. **評価・可視化** → RMSE / MAE / MAPE / R2 を算出、実測 vs 予測のプロットなど

## データ

- **入力**: 5分足 CSV（timestamp, open, high, low, close, volume, vwap, EURUSD_close, EURJPY_close など）。開発時は 2024-03-01 ～ 2025-10-31 を想定。
- **説明変数**: 対数比率/対数乖離 + MODWT、テクニカル指標（対数乖離）、曜日・時間の sin/cos エンコーディング。EWMA ローリング標準化と ±3σ クリップを適用。
- **目的変数**: 1本先の対数収益 $y_t = \ln(P_{t+1}/P_t)$。

詳細な列定義・前処理仕様は `docs/01_basic_design.md` の「データの概要」を参照してください。

## ハイパーパラメータ（代表値）

| 分類 | パラメータ | 代表値・範囲 |
|------|------------|----------------------|
| シーケンス | lookback | 100 |
| 分割 | train_ratio / val_ratio | 0.8 / 0.2（訓練 64% / 検証 16% / テスト 20%） |
| 前処理 | wt_level, clip_sigma | 3, ±3σ_ewma |
| PSO 探索 | ニューロン数 / エポック数 / 層数 | (50, 300) / (50, 300) / (1, 3) |
| PSO | 粒子数, 反復回数, w, c1, c2 | 20, 5, 0.8, 1.5, 1.5 |

## ドキュメント

- [01_basic_design.md](docs/01_basic_design.md) — 基本設計（目的・構成・データ・パイプライン概要）
- [02_detailed_design_theory.md](docs/02_detailed_design_theory.md) — 理論（問題定式化・LSTM/PSO/前処理の根拠）
- [03_detailed_design_implementation.md](docs/03_detailed_design_implementation.md) — 実装（読込・前処理・学習・評価の仕様、ハイパーパラメータ一覧）

## 注意事項

- 本実装は研究・検証用です。実運用での取引判断には利用しないでください。
- MODWT は「未来データを使わない」ローリング窓で適用する設計です（データリーク防止）。
