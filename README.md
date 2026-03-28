# GeNeLy C++

GeNeLy C++ は、オセロ用のハイブリッド探索エンジンです。  
αβ探索、MCTS、ニューラルネットワーク評価、C++ 拡張、ONNX Runtime を組み合わせて動作します。

## 主な構成

- `genely.py`
  起動用エントリーポイントです。
- `gui/`
  GUI 本体、設定ダイアログ、探索制御を置いています。
- `core/`
  盤面処理、NN 推論、MCTS 連携などの Python 側ロジックです。
- `engine/`
  `pybind11` で公開している C++ 探索エンジンです。
- `data/`
  重み、ONNX モデル、定石、各種キャッシュを置きます。
- `scripts/`
  補助スクリプトです。

## 現在の特徴

- C++ αβ探索
- MCTS 併用
- ONNX Runtime による NN 推論
- GPU 利用時は `CUDAExecutionProvider` を優先
- CPU 環境では MCTS の時間を自動で抑えて αβ に配分
- Egaroucid 形式 `.egbk3` 定石の読み込み
- JSON 定石との切り替え
- pondering 対応
- 終盤 exact 探索の早期開始と読み切り最適化

## 必要環境

- Windows
- Python 3.13 系推奨
- Visual Studio Build Tools または Visual Studio
- CUDA を使う場合は、ONNX Runtime GPU が要求する CUDA / cuDNN 環境

## セットアップ

仮想環境を使う前提です。

```powershell
.\.venv\Scripts\Activate.ps1
```

必要パッケージを入れたあと、C++ 拡張をビルドします。

```powershell
python setup.py build_ext --inplace
```

## 起動

```powershell
python genely.py
```

## モデルと定石

- ONNX モデル: `data/model_best.onnx`
- PyTorch 元モデル: `data/model_best.pth`
- Egaroucid 定石: `data/book.egbk3`
- JSON 定石: `data/opening_book.json`

GUI では起動時に定石ソースを選べます。  
Egaroucid 定石を選んだ場合は定石使用率は 100% 固定です。

## ビルド生成物

主な生成物は以下です。

- `engine/othello_engine.cp313-win_amd64.pyd`
- `build/`
- `data/opening_book_egbk3_cache.npz`

## 補足

- ONNX GPU が使えない環境では自動で CPU 実行にフォールバックします。
- exact 探索の開始タイミングは思考時間設定に応じて変わります。
- 終盤では勝敗優先のまま、石差も考慮して手を選びます。
