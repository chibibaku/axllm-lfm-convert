# axllm-lfm-convert

LFM2.5-2.6Bを取得・校正・変換し、AX650/AX8850用のモデルパッケージを作成します。

モデルは別の [Hugging Faceリポジトリ](https://huggingface.co/chibibaku/axllm-lfm) で管理します。

## 使い方

Ubuntu 24.04 x86_64を推奨します。Docker、Python 3.10以降、32 GiB以上のRAMを用意してください。
開始時には空きRAM 18 GiB、空きディスク20 GiBを確認します。工程によってさらに容量が必要になる場合があります。

runtime・convert・modelを同じ親ディレクトリに配置して実行します。

```bash
cd axllm-lfm-convert
./lfm setup --device cpu
./lfm build --profile ../axllm-lfm-model/profiles/FP32 --device cpu
```

BF16 KVを使う場合は `--profile ../axllm-lfm-model/profiles/BF16` を指定します。
CUDA校正を使う場合は `setup` と `build` の両方に `--device cuda` を指定してください。

`setup` は専用venvとPulsar2 7.0-patch1を準備します。Pulsar2は初回に自動取得し、Dockerへ読み込みます。
Ubuntuパッケージの導入にはrootまたはsudoが必要です。
`build` はモデル取得、FP32 teacherによる校正、ONNX export、Pulsar2 compile、パッケージ作成を順に実行します。

成果物は各プロファイルの `dist/` に生成されます。AX650/AX8850へコピーして実行します。

```bash
axllm run /path/to/LFM2.5-2.6B-FP32
axllm run /path/to/LFM2.5-2.6B-BF16 --kv-storage disk
```

## 再開と補助操作

```bash
./lfm verify --profile ../axllm-lfm-model/profiles/FP32
./lfm calibrate --profile ../axllm-lfm-model/profiles/FP32 --device cpu
./lfm package --profile ../axllm-lfm-model/profiles/FP32 --assets /path/to/compiled-payload --source /path/to/model-snapshot
./lfm clean --yes
```

失敗後は同じ `build` を再実行できます。設定と完了記録、必要ファイルの存在を基に途中結果を再利用します。
ファイル内容のハッシュ照合は行いません。変換コード・元モデル・校正条件を変更した場合や、キャッシュの破損が疑われる場合は
`./lfm clean --all --yes` で生成物とキャッシュを削除してから作り直してください。
FP32／BF16は共通のNPU成果物を利用します。
`--source` には対応するLFM2.5-2.6Bの取得済みモデルを指定できます。

`verify` は必要な48グラフ、tokenizer、設定、embeddingサイズを確認します。追加のメモなどは許容します。
失敗時は `logs/failure.log` に詳細を残します。`--keep-failure-logs` で複数の失敗ログを保持できます。
`clean` は管理用マーカーがある生成物のみを削除します。

## 量子化と検証範囲

FP32／BF16はKVの保存形式です。NPUグラフの入出力はFP32、行列処理には16bit設定を使い、
正規化・残差・SoftmaxなどにはFP32を指定します。コンパイラ内部の全演算・重みの実行精度は未検証です。
校正時には元のTransformers実装との数値比較を行います。

[動作確認と制限](docs/validation.md)を参照してください。
GitHub Actionsでは軽量な単体テストを実行します。全モデル変換と実機確認は手元の環境で行ってください。

## ライセンス

元モデルは `LiquidAI/LFM2.5-2.6B` です。取得revisionはコード内に指定しています。
配布パッケージにはモデルのLICENSE、公式tokenizer、chat template、実行設定を同梱します。
モデルの利用・再配布条件は同梱ライセンスを確認してください。
