# Poetic Cybermetrics — 詩のサイバメトリクス

**Poetic Waveform Analytics**

このリポジトリは、詩を「数値化」して波形として可視化する実験的プロジェクトです。  
詩行ごとの **Divergence（詩的跳躍度）** を埋め込みモデルを用いて計算し、  
Streamlit アプリでインタラクティブに分析・描画できます。



## 公開中の Streamlit アプリ

現在、このリポジトリから以下の 2 種のアプリを公開しています：

1. **単一モデル版**  
   [poetic-waveform-analytics-divergence.streamlit.app](https://poetic-waveform-analytics-divergence.streamlit.app/)  
   - SBERT 系モデル（英語寄り / 多言語対応）  
   - 入力テキストの Divergence 波形を描画  
   - 参照は直前 k 行（k 任意）

2. **４モデル比較版**  
   [poetic-waveform-analytics-divergence_4models.streamlit.app](https://poetic-waveform-analytics-divergence_4models.streamlit.app/)  
   - SBERT（英語寄り / 多言語対応）  
   - OpenAI Embeddings（small / large）  
   - Ruri-v3-30m  
   - ４モデルの波形を比較・重ね描画  
   - CSV（wide形式）出力に対応  
   - 参照は直前3行（k=3固定）



## 背景
このプロジェクトは「詩の数値化／Poetic Cybermetrics」をテーマにしています。  
従来の詩学や批評を補完する新しい方法として、  
詩を埋め込みベクトル空間に写像し、その文脈からの逸脱度を波形として可視化します。  

- Divergence（逸脱度／詩的跳躍度）  
- Resonance（共鳴度／余韻）※現在は実装を試行中です。  

これらの指標を通じて、詩の実作における**主観的な感覚（内観）をより鋭敏に捉え直すこと**を目指しています。  
数値化は外部に伝えるためだけでなく、創作者自身が自らの言葉の揺らぎを見つめ直すための装置でもあります。
戯画的に言えば、詩人の感覚（勘）に測定装置（計器盤）を接続する方法の開発を目指しています。



## 詩的跳躍度（Divergence）の定義

各行ベクトル $x_i$ と参照ベクトル $ref_i$ のコサイン類似度をもとに定義します。

$$
D_i = 1 - \cos(x_i, ref_i)
$$

- コサイン類似度 $\cos(x_i, ref_i)$ の値域は **[-1, +1]** です。
- したがって Divergence の値域は **[0, 2]** となります。  
  - $\cos = 1$ のとき $D = 0$ （完全に同一方向）  
  - $\cos = 0$ のとき $D = 1$ （直交＝無関連）  
  - $\cos = -1$ のとき $D = 2$ （正反対方向）  

> 注：言語埋め込みベクトルにおいては負の相関（$\cos < 0$）はほとんど見られません。実際の多くの Divergence 値は **0〜1** の範囲に収まります。



 ### コサイン類似度とは？

ベクトル同士がつくる「角度」を測る指標です。  
二つのベクトル $A, B$ に対して、

$$
\cos(A, B) = \frac{A \cdot B}{\|A\| \, \|B\|}
$$

と定義され、値域は **[-1, +1]** です。  

- **+1** に近いほど「同じ方向を向いている」＝意味が強く重なり合う。  
- **0** 付近では「直交する」＝つながりが感じられない。  
- **-1** に近いほど「真逆の方向」＝意味が反転して響き合う。  

言いかえれば、**コサイン類似度は“語と語のあいだに生まれる角度”を測る物差し**です。  
その角度が狭ければ「近さ」、広ければ「隔たり」、そして隔たりの中に「余白」が生まれ「詩性」が宿るのではないでしょうか。詩的跳躍度は、この「角度の差異」をゆらぎとして可視化する試みです。
なお、自然言語処理では「コサイン類似度」は、文章や語の埋め込みベクトル同士の意味的な近さを測る基本的な手法として広く使われています。 



## セッションとデータ保持

- **アップロードされた詩行データはセッション内でのみ利用され、終了時に保持されません。**  
- 利用者の入力は永続的に保存されることはありません。  



## APIキーについて

- **OpenAI Embeddings 使用時は API Key が必須** です。  
  - secrets → 環境変数 → 入力 UI の順に探索します。  
  - セッション中のみ保持され、終了時に揮発します。  
- Hugging Face Hub Token は任意ですが、Ruri 系モデル利用時に推奨されます。



## セットアップ

依存ライブラリは `requirements.txt` に記載しています。  
以下で環境を準備できます。

```bash
git clone https://github.com/pmypoetry-lab/poetic-waveform-analytics.git
cd poetic-waveform-analytics
pip install -r requirements.txt
```



## 使い方

ローカルで起動する場合:

```streamlit run poetic_divergence_min.py```



## ライセンス
MIT License.  
このリポジトリは自由に利用・改変・配布できます。  
詳細は [LICENSE](./LICENSE) をご覧ください。


## 謝辞
本プロジェクトは、詩的知性（Poetic Intelligence）の探究の一環として進められています。  
詩と数理の交差点に関心を寄せるすべての人に感謝します。
