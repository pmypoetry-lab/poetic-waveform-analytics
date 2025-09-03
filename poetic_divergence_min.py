# -*- coding: utf-8 -*-
"""
Poetic Divergence — minimal Streamlit app
- 入力: テキスト貼り付け or DOCX アップロード
- 出力: Divergence 波形（前3行との相対逸脱）、CSV（raw/normalized）
- 依存: streamlit, numpy, pandas, matplotlib
       （任意）sentence-transformers, python-docx, scikit-learn
"""

import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ========= オプショナル依存の準備（あるものだけ使う） =========
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


# ========= 埋め込み（可能なら SBERT、次点 TF-IDF、最後に単純Bag-of-words） =========
@st.cache_resource(show_spinner=False)
def load_sbert():
    # 日本語も安定する多言語モデル
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    return SentenceTransformer(model_name)

def embed_lines(lines: list[str]) -> np.ndarray:
    if len(lines) == 0:
        return np.zeros((0, 1), dtype=float)

    # 1) SBERT
    if _HAS_SBERT:
        try:
            model = load_sbert()
            vec = model.encode(lines, convert_to_numpy=True, normalize_embeddings=False)
            return np.asarray(vec, dtype=float)
        except Exception:
            pass

    # 2) TF-IDF
    if _HAS_SKLEARN:
        try:
            tfidf = TfidfVectorizer()
            m = tfidf.fit_transform(lines)
            return m.toarray().astype(float)
        except Exception:
            pass

    # 3) 単純 BoW（手作り）
    vocab = {}
    rows = []
    for ln in lines:
        toks = list(ln)
        for t in toks:
            if t not in vocab:
                vocab[t] = len(vocab)
    V = len(vocab)
    for ln in lines:
        v = np.zeros(V, dtype=float)
        for t in ln:
            idx = vocab.get(t)
            if idx is not None:
                v[idx] += 1.0
        rows.append(v)
    return np.vstack(rows) if rows else np.zeros((0, 1), dtype=float)


# ========= Divergence（前 window 行の平均ベクトルとの 1-cos 距離） =========
def compute_divergence(lines: list[str], window: int = 3, normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    returns: (raw, normalized)
    raw: 各行の 1−cos 距離（前 window 行の平均 vs 当詩行）
    normalized: min-max で 0..1
    """
    if not lines:
        z = np.zeros((0,), dtype=float)
        return z, z

    emb = embed_lines(lines)  # (n, d)
    n = emb.shape[0]
    raw = np.zeros(n, dtype=float)

    for t in range(n):
        start = max(0, t - window)
        if start >= t:
            raw[t] = 0.0  # 文脈がない先頭付近は 0
            continue
        ctx = emb[start:t].mean(axis=0)
        raw[t] = cosine_distance(emb[t], ctx)

    normed = safe_minmax_scale(raw) if normalize else raw.copy()
    return raw, normed


# ========= Streamlit UI =========
st.set_page_config(page_title="Poetic Divergence (minimal)", layout="centered")

st.title("Divergence（詩的跳躍度）最小アプリ")
st.caption("入力テキスト（またはDOCX）から、行ごとの Divergence を計算して波形表示し、CSV を出力します。")

with st.sidebar:
    st.subheader("設定")
    window = st.number_input("文脈ウィンドウ（直前の行数）", min_value=1, max_value=10, value=3, step=1)
    use_normalized = st.toggle("0..1 に正規化して表示/出力", value=True)
    st.caption("※ 正規化は詩内の相対比較に便利です。raw は生の 1−cos 値。")

tab1, tab2 = st.tabs(["テキスト入力 / DOCX", "結果"])

with tab1:
    st.markdown("**1) 入力方法を選択**")
    up = st.file_uploader("DOCX（任意）をアップロード", type=["docx"])
    text = st.text_area("またはテキストを直接貼り付け", height=200, placeholder="ここに詩行を貼り付け / 改行で区切り")

    lines: list[str] = []
    if up is not None and _HAS_DOCX:
        try:
            doc = docx.Document(up)
            raw = "\n".join([p.text for p in doc.paragraphs])
            lines = split_poem_lines(raw)
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
        raw, normed = compute_divergence(lines, window=window, normalize=True)

    y = normed if use_normalized else raw
    df = pd.DataFrame({
        "line_index": np.arange(1, len(lines) + 1, dtype=int),
        "line_text": lines,
        "divergence_raw": raw,
        "divergence_norm": normed
    })

    # ---- 波形描画 ----
    fig = plt.figure(figsize=(10, 3.5), dpi=150)
    plt.plot(np.arange(1, len(lines) + 1), y, linewidth=2)
    plt.xlabel("Line")
    plt.ylabel("Divergence" + (" (0..1)" if use_normalized else " (raw 1−cos)"))
    plt.title("Divergence Waveform")
    plt.grid(alpha=0.3)
    st.pyplot(fig, clear_figure=True)

    # ---- CSV ダウンロード ----
    csv_buf = io.StringIO()
    # 出力カラムは最小限（波形と行テキストに限定）
    out_cols = ["line_index", "line_text", "divergence_raw", "divergence_norm"]
    df[out_cols].to_csv(csv_buf, index=False)
    st.download_button(
        label="CSVをダウンロード",
        data=csv_buf.getvalue().encode("utf-8-sig"),
        file_name="divergence_scores.csv",
        mime="text/csv"
    )

    # 簡単な統計
    st.caption("— 概要統計（参考） —")
    st.write(pd.DataFrame({
        "lines": [len(lines)],
        "raw_mean": [float(np.mean(raw)) if len(raw) else 0.0],
        "raw_range": [float(np.max(raw) - np.min(raw)) if len(raw) else 0.0]
    }))
