# Poetic Waveform Analytics

**Poetic Cybermetrics — 詩のサイバメトリクス**

このリポジトリは、詩を「数値化」して波形として可視化する実験的プロジェクトです。  
詩行ごとの **Divergence（詩的跳躍度）** を埋め込みモデルを用いて計算し、  
Streamlit アプリでインタラクティブに分析・描画できます。

[公開アプリはこちら](https://poetic-waveform-analytics-divergence.streamlit.app/)



## 背景
このプロジェクトは「詩の数値化／Poetic Cybermetrics」をテーマにしています。  
従来の詩学や批評を補完する新しい方法として、  
詩を埋め込みベクトル空間に写像し、その文脈からの逸脱度を波形として可視化します。  

- Divergence（逸脱度／詩的跳躍度）  
- Resonance（共鳴度／余韻）※現在は実装を試行中です。  

これらの指標を通じて、詩の実作における**主観的な感覚（内観）をより鋭敏に捉え直すこと**を目指しています。  
数値化は外部に伝えるためだけでなく、創作者自身が自らの言葉の揺らぎを見つめ直すための装置でもあります。
戯画的に言えば、詩人の感覚（勘）に測定装置（計器盤）を接続する方法の開発を目指しています。



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
