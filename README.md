### 開発環境
- Macbook Air Early 2014(macOS Mojave)
- Python 3.6.5

### ライブラリ
| 名前 | バージョン |
| - | - |
| TensorFlow | 1.14.0 |
| numpy | 1.16.4 |
| Pillow | 6.1.0 |
| matplotlib | 3.1.1 |

### 環境構築（Python編）
1. Homebrew インストール
2. pyenv インストール
3. pyenv-virtualenv インストール
4. ~/.bash_profile を以下を追加または編集(使用環境によって多少差異あり)
```
# １行目は `brew doctor` の WARNING 対策
alias brew="PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin brew"
export PYENV_ROOT="${HOME}/.pyenv"
export PATH="${PYENV_ROOT}/shims:$PATH"
eval "$(pyenv init -)"
```  
5. `source ~/.bash_profile` で設定を反映
6. `pyenv install [python version]` で python をインストール
7. `pyenv global [4.で指定した python version]` でグローバル環境で使用する python のバージョンを設定
8. `pyenv virstualenv [python version] [任意の名前]` で仮想環境作成
9. `pyenv global [8.で指定した任意の名前]` で環境を切り替え

### 参考
- [Python未経験エンジニアがMacでTensorFlowの実行環境+快適なコーディング環境を構築するまで](https://qiita.com/KazaKago/items/587ac1224afc2c9350f1)
- [pyenvでPythonのバージョンを切り替えられない場合の対処法＋](https://qiita.com/TheHiro/items/88d885ef6a4d25ec3020)
- [pyenv-virtualenvでディレクトリ単位のpython環境構築](https://qiita.com/niwak2/items/5490607be32202ce1314)

### SSDLite_MobileNet_v2 のtfliteファイル作成
1. [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) COCOデータセットを用いて学習済みの COCO-trained models から [ssdlite_mobilenet_v2_coco](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz) をダウンロード
2. Python の Tensorflow をインストールし、`tflite_convert` コマンドが使用できるようにしておく
3. 2.でダウンロードしたものを解凍し、内部にある `frozen_inference_graph.pb` を以下コマンドの `graph_def_file` に指定して以下を実行し、tfliteファイルへ変換
```
tflite_convert --graph_def_file=frozen_inference_graph.pb
               --output_file=ssdlite_mobilenet_v2_coco.tflite
               --inference_type=FLOAT
               --input_shape=1,300,300,3
               --input_array=Preprocessor/sub
               --output_arrays=concat,concat_1
```

### 参考
- [object_detection_ssd_coco.md](https://github.com/freedomtan/tensorflow/blob/object_detection_tflite_object_dtection_python/tensorflow/contrib/lite/examples/python/object_detection_ssd_coco.md)
- [How to Convert Tensorflow Model to TF Lite Model](https://github.com/nnsuite/nnstreamer/wiki/%5BTF-Lite%5D-How-to-Convert-Tensorflow-Model-to-TF-Lite-Model)