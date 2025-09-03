# -*- coding: utf-8 -*-
"""
Poetic Divergence — minimal Streamlit app (model selectable)
- 入力: テキスト貼り付け or DOCX アップロード
- 出力: Divergence 波形（前3行との相対逸脱）、CSV（raw/normalized）
- 依存: streamlit, numpy, pandas, matplotlib
       （任意）sentence-transformers, python-docx, scikit-learn
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.font_manager import FontProperties

def use_japanese_font() -> tuple[FontProperties | None, str]:
    """
    リポジトリ同梱フォントを優先して登録し、使用中の family 名を返す。
    戻り値: (FontProperties or None, family_name or "")
    """
    # 1) リポジトリ同梱候補（置いた実ファイル名に合わせて並べる）
    candidates = [
        "fonts/NotoSansJP-Bold.ttf",
        "fonts/NotoSansJP-Regular.ttf",
        "fonts/NotoSerifJP-Bold.ttf",
        "fonts/NotoSerifJP-Regular.ttf"
    ]

    # DejaVuは日本語NGなので無効化
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    # --- 同梱TTFを優先して addfont し、family 名を取得 ---
    for path in candidates:
        if os.path.exists(path):
            try:
                fm.fontManager.addfont(path)
                prop = FontProperties(fname=path)
                family = prop.get_name()          # ← 内部family名（例: "Noto Sans JP"）
                matplotlib.rcParams["font.family"] = family
                matplotlib.rcParams["font.sans-serif"] = [family]
                return prop, family
            except Exception:
                pass

              return prop, fam

    return None, ""  # どうしても見つからない場合

# 一度だけ実行
_JP_PROP, _JP_FAMILY = use_japanese_font()
try:
    import streamlit as st
    st.caption(f"matplotlibフォント: {_JP_FAMILY or '未設定（フォールバック）'}")
except Exception:
    pass


# 日本語フォントを指定
jp_font_path = "fonts/NotoSansJP-Regular.ttf"  # リポジトリ内のパス
jp_font = fm.FontProperties(fname=jp_font_path)

plt.rcParams["font.family"] = jp_font.get_name()


# ========= オプショナル依存の確認 =========
_HAS_SBERT = False
_HAS_SKLEARN = False
_HAS_DOCX = False
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBERT = True
except Exception:
    _HAS_SBERT = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

try:
    import docx
    _HAS_DOCX = True
except Exception:
    _HAS_DOCX = False


# ========= ユーティリティ =========
def split_poem_lines(text: str) -> list[str]:
    """空行を落とし、前後空白を整えて行配列に。"""
    if not text:
        return []
    lines = [ln.strip() for ln in text.replace("\r\n", "\n").split("\n")]
    return [ln for ln in lines if ln != ""]


def safe_minmax_scale(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mn, mx = float(np.min(x)), float(np.max(x))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cos(a,b) （ベクトルが0なら0扱い）"""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    sim = float(np.dot(a, b) / denom)
    return max(0.0, 1.0 - sim)


# ========= 埋め込み（SBERT → TF-IDF → BoW） =========
@st.cache_resource(show_spinner=False)
def load_sbert(model_name: str):
    # 引数（モデル名）でキャッシュキーが分かれる
    return SentenceTransformer(model_name)

def embed_lines(lines: list[str], model_name: str) -> np.ndarray:
    if len(lines) == 0:
        return np.zeros((0, 1), dtype=float)

    # 1) SBERT
    if _HAS_SBERT:
        try:
            model = load_sbert(model_name)
            vec = model.encode(lines, convert_to_numpy=True, normalize_embeddings=False)
            return np.asarray(vec, dtype=float)
        except Exception:
            pass

    # 2) TF-IDF（フォールバック）
    if _HAS_SKLEARN:
        try:
            tfidf = TfidfVectorizer()
            m = tfidf.fit_transform(lines)
            return m.toarray().astype(float)
        except Exception:
            pass

    # 3) 単純 BoW（文字ベース・最終フォールバック）
    vocab = {}
    rows = []
    for ln in lines:
        for ch in ln:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    V = len(vocab)
    for ln in lines:
        v = np.zeros(V, dtype=float)
        for ch in ln:
            idx = vocab.get(ch)
            if idx is not None:
                v[idx] += 1.0
        rows.append(v)
    return np.vstack(rows) if rows else np.zeros((0, 1), dtype=float)


# ========= Divergence（前 window 行の平均ベクトルとの 1-cos 距離） =========
def compute_divergence(lines: list[str], window: int = 3, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> tuple[np.ndarray, np.ndarray]:
    """
    returns: (raw, normalized)
    raw: 各行の 1−cos 距離（前 window 行の平均 vs 当詩行）
    normalized: min-max で 0..1
    """
    if not lines:
        z = np.zeros((0,), dtype=float)
        return z, z

    emb = embed_lines(lines, model_name=model_name)  # (n, d)
    n = emb.shape[0]
    raw = np.zeros(n, dtype=float)

    for t in range(n):
        if t < window:           # window 揃うまで 0（旧版互換仕様）
            raw[t] = 0.0
            continue
        ctx = emb[t-window:t].mean(axis=0)
        denom = (np.linalg.norm(emb[t]) * np.linalg.norm(ctx)) or 1e-9  # 微小値保護
        sim = float(np.dot(emb[t], ctx) / denom)
        raw[t] = max(0.0, 1.0 - sim)

    normed = safe_minmax_scale(raw)
    return raw, normed


# ========= Streamlit UI =========
st.set_page_config(page_title="Poetic Divergence (minimal)", layout="centered")

st.title("Divergence（詩的跳躍度）波形描画アプリ")
st.caption("入力テキスト（またはDOCX）から、行ごとの Divergence を計算して波形表示し、CSV を出力します。")

with st.sidebar:
    st.subheader("設定")

    # --- 埋め込みモデル選択（ラベル→IDのマッピング） ---
    MODEL_OPTIONS = {
        "all-MiniLM-L6-v2（英語寄り）": "all-MiniLM-L6-v2",
        "paraphrase-multilingual-MiniLM-L12-v2（多言語推奨）": "paraphrase-multilingual-MiniLM-L12-v2",
    }

    label = st.selectbox(
        "埋め込みモデル",
        options=list(MODEL_OPTIONS.keys()),
        index=0,
        help="簡易分析には all-MiniLM-L6-v2 を選択。日本語詩には多言語モデル（下）を推奨。"
    )
    model_choice = MODEL_OPTIONS[label]  # ← 実際に使うID

    if not _HAS_SBERT:
        st.warning("sentence-transformers が未インストールのため、TF-IDF/BoW で代替します。requirements.txt に追加してください。")

    window = st.number_input("文脈ウィンドウ（直前の行数）", min_value=1, max_value=10, value=3, step=1)
    use_normalized = st.toggle("グラフを 0..1 正規化で表示", value=True)
    csv_mode = st.radio(
        "CSVに含める列",
        ["rawのみ", "normalizedのみ", "rawとnormalized（両方）"],
        index=2,
        horizontal=False
    )
    st.caption("※ 表示とCSVは独立設定です。")

tab1, tab2 = st.tabs(["テキスト入力 / DOCX", "結果"])

with tab1:
    st.markdown("**1) 入力方法を選択**")
    up = st.file_uploader("DOCX（任意）をアップロード", type=["docx"])
    text = st.text_area("またはテキストを直接貼り付け", height=200, placeholder="ここに詩行を貼り付け / 改行で区切り")

    lines: list[str] = []
    if up is not None and _HAS_DOCX:
        try:
            doc = docx.Document(up)
            raw_txt = "\n".join(p.text for p in doc.paragraphs)
            lines = split_poem_lines(raw_txt)
            st.success(f"DOCX から {len(lines)} 行を読み取りました。")
        except Exception as e:
            st.error(f"DOCX 読み取りに失敗: {e}")
    elif up is not None and not _HAS_DOCX:
        st.warning("python-docx が未インストールのため、DOCX を読み取れません。テキスト貼り付けをご利用ください。")

    if not lines and text.strip():
        lines = split_poem_lines(text)

    if lines:
        st.info(f"解析対象の行数: {len(lines)}")
    else:
        st.warning("詩行がありません。DOCX をアップロードするか、テキストを貼り付けてください。")


with tab2:
    if not lines:
        st.stop()

    with st.spinner("Divergence 計算中…"):
        raw, normed = compute_divergence(lines, window=window, model_name=model_choice)

    # ---- グラフ表示 ----
    y = normed if use_normalized else raw
    fig = plt.figure(figsize=(10, 3.5), dpi=150)
    plt.plot(np.arange(1, len(lines) + 1), y, linewidth=2)
    plt.xlabel("行番号")   # 日本語でもOK
    plt.ylabel("Divergence " + ("(0..1 正規化)" if use_normalized else "(raw 1−cos)"))
    plt.title(f"Divergence 波形 [{label}]")
    plt.grid(alpha=0.3)
    st.pyplot(fig, clear_figure=True)

    # ---- CSV 出力 ----
    df = pd.DataFrame({
        "line_index": np.arange(1, len(lines) + 1, dtype=int),
        "line_text": lines,
        "divergence_raw": raw,
        "divergence_norm": normed,
        "embedding_model": [model_choice] * len(lines),
        "window": [int(window)] * len(lines),
    })
    if csv_mode == "rawのみ":
        out_df = df[["line_index", "line_text", "divergence_raw", "embedding_model", "window"]]
        fname = "divergence_raw.csv"
    elif csv_mode == "normalizedのみ":
        out_df = df[["line_index", "line_text", "divergence_norm", "embedding_model", "window"]]
        fname = "divergence_normalized.csv"
    else:
        out_df = df[["line_index", "line_text", "divergence_raw", "divergence_norm", "embedding_model", "window"]]
        fname = "divergence_raw_and_normalized.csv"

    csv_buf = io.StringIO()
    out_df.to_csv(csv_buf, index=False)
    st.download_button("CSVをダウンロード", csv_buf.getvalue().encode("utf-8-sig"),
                       file_name=fname, mime="text/csv")

    # 概要統計
    st.caption("— 概要統計（参考） —")
    st.write(pd.DataFrame({
        "lines": [len(lines)],
        "raw_mean": [float(np.mean(raw)) if len(raw) else 0.0],
        "raw_range": [float(np.max(raw) - np.min(raw)) if len(raw) else 0.0],
        "norm_mean": [float(np.mean(normed)) if len(normed) else 0.0],
        "norm_range": [float(np.max(normed) - np.min(normed)) if len(normed) else 0.0],
        "model": [model_choice],
        "window": [int(window)],
    }))
