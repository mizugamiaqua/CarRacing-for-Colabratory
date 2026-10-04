# CarRacing-for-Colabratory

Google Colab で Gymnasium の `CarRacing-v3` を PPO (Stable-Baselines3) で学習し、**コースを1周完走**できるエージェントを作るプロジェクトです。

## 使い方 (All Run)

1. [`CarRacing_PPO_Colab.ipynb`](CarRacing_PPO_Colab.ipynb) を Colab で開く
   (GitHub 上のファイルを `https://colab.research.google.com/github/mizugamiaqua/CarRacing-for-Colabratory/blob/main/CarRacing_PPO_Colab.ipynb` で開けます)
2. ランタイム → ランタイムのタイプを変更 → **GPU (T4)**
3. ランタイム → **すべてのセルを実行**
   - Drive へのアクセス許可を求められます (チェックポイント保存用)。
   - 学習 → TensorBoard → 評価 (平均スコア・周回完走率) → 動画表示 まで自動で行います。

学習が切れても (Colab の切断など)、同じノートブックを再度 All Run すれば Drive 上の最新チェックポイントから再開します。

ローカル実行: `pip install -r requirements.txt && python -m carracing.train --out runs/x`

## 構成

| ファイル | 内容 |
|---|---|
| `carracing/config.py` | ハイパーパラメータ |
| `carracing/train.py` | 学習 (チェックポイント・自動再開・評価付き) |
| `carracing/evaluate.py` | 評価・周回完走判定・動画出力 |
| `CarRacing_PPO_Colab.ipynb` | Colab 用ノートブック |

## アルゴリズム

PPO + CnnPolicy。96×96 RGB 画像を2フレームスタック、8並列環境、gSDE (状態依存探索)、学習率線形減衰、報酬正規化。
設定は rl-baselines3-zoo の CarRacing 用 PPO 設定に準拠しており、この設定は約400万ステップで1周完走・スコア 800〜900 台が報告されている構成です。

## 学習時間・推奨スペック

> ⚠️ 下記は一般的な Colab 環境での**目安**です (この環境では実機計測していません)。スループットは割り当てられる CPU/GPU により大きく変わります。

CarRacing は物理シミュレーション＋画像レンダリングが CPU 律速で、GPU は更新時にのみ効きます。

| 環境 | 想定スループット | 400万ステップ |
|---|---|---|
| Colab 無料 T4 GPU + 2 vCPU | 約 400〜700 step/s | **約 2〜3 時間** |
| Colab Pro (L4/A100 + 多めの vCPU) | 約 700〜1200 step/s | 約 1〜1.5 時間 |
| CPU のみ (GPUなし) | 約 150〜300 step/s | 5 時間以上 (非推奨) |

- 推奨: **GPU ランタイム (T4 以上)**、RAM 12GB 以上 (標準で足りる)、Drive 空き容量 1GB 程度。
- 目安の到達度: 約100万 step で道なり走行 (スコア 300〜600)、約200万 step で完走し始め、約400万 step で安定 (平均 800〜900 前後)。乱数シードでばらつくため、完走率が低ければ `STEPS` を 600万〜800万に増やして再実行 (再開されます)。
- 動作確認だけなら `STEPS = 200_000` で約5〜10分。
- 無料版 Colab は 12 時間程度で切断されることがあり、またGPU枠が制限されることがあります。Drive にチェックポイントを保存しているので再実行で再開できます。
- 「1周完走」は全タイル通過 (`tile_visited_count == len(track)`) で判定し、`evaluate` が完走率を表示します。
