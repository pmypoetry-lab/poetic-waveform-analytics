# -*- coding: utf-8 -*-
"""
Poetic Divergence — Four Models Unified (minimum)
- 逐次ローカル基準のみ／ref = 直前3行の平均（k=3 固定）
- 行ベクトルおよび参照ベクトルは明示的に L2 正規化
- 入力: テキスト貼り付け or DOCX アップロード
- 出力: Divergence 波形（前3行との相対逸脱）、CSV（raw/normalized）
- モデル選択: SBERT（英語寄り / 多言語）, OpenAI Embeddings, Ruri-v3-30m
- API/Token: OPENAI_API_KEY（env/secrets/入力）, HUGGINGFACE_HUB_TOKEN（env/入力）
"""


import io
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm



# --- 日本語フォント設定（公開/非公開 共通対策） ---
import os, matplotlib
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# 探索候補（src/fonts と repo 直下 fonts の両方を見る）
candidates = [
    BASE_DIR / "fonts" / "NotoSansJP-Regular.ttf",
    BASE_DIR / "fonts" / "NotoSansJP-Bold.ttf",
    Path.cwd() / "fonts" / "NotoSansJP-Regular.ttf",
    Path.cwd() / "fonts" / "NotoSansJP-Bold.ttf",
]

family = None
for p in candidates:
    if p.exists():
        try:
            fm.fontManager.addfont(str(p))         # ← 実体を登録
            prop = FontProperties(fname=str(p))
            family = prop.get_name()               # 例: "Noto Sans JP"
            break
        except Exception:
            pass

if family:
    matplotlib.rcParams["font.family"] = family
    matplotlib.rcParams["font.sans-serif"] = [family]
else:
    # 最終フォールバック（Linux想定）
    matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAGothic", "DejaVu Sans"]
    matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "IPAGothic", "DejaVu Sans"]

matplotlib.rcParams["axes.unicode_minus"] = False
# --- end font setup ---



WINDOW_K = 3  # rolling window size


# ===== Optional deps detection =====
_HAS_DOCX = False
try:
    import docx
    _HAS_DOCX = True
except Exception:
    _HAS_DOCX = False

_HAS_SBERT = False
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_SBERT = True
except Exception:
    _HAS_SBERT = False

_HAS_SKLEARN = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

_HAS_OPENAI = False
try:
    from openai import OpenAI  # type: ignore
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False
    OpenAI = None  # type: ignore

_HAS_TF = False
try:
    from transformers import AutoTokenizer, AutoModel, AutoConfig  # type: ignore
    import torch  # type: ignore
    _HAS_TF = True
except Exception:
    _HAS_TF = False

# ===== Utilities =====
def split_poem_lines(text: str) -> List[str]:
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

def l2_normalize_rows(M: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    if M.size == 0:
        return M
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return M / norms

def cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < eps:
        return 0.0
    sim = float(np.dot(a, b) / denom)
    return max(0.0, 1.0 - sim)

# ===== Embedding backends =====
# SBERT (or TF-IDF fallback)
@st.cache_resource(show_spinner=False)
def _load_sbert(model_name: str):
    return SentenceTransformer(model_name)

def embed_sbert(lines: List[str], model_name: str) -> np.ndarray:
    if len(lines) == 0:
        return np.zeros((0, 1), dtype=float)
    if _HAS_SBERT:
        try:
            model = _load_sbert(model_name)
            vec = model.encode(lines, convert_to_numpy=True, normalize_embeddings=False)
            return np.asarray(vec, dtype=float)
        except Exception as e:
            st.warning(f"SBERT 失敗（{e}）。TF-IDF にフォールバックします。")
    if _HAS_SKLEARN:
        tfidf = TfidfVectorizer()
        m = tfidf.fit_transform(lines)
        return m.toarray().astype(float)
    # 最終: 文字BoW
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


# OpenAI
def _get_openai_client_session():
    """セッション専有の OpenAI クライアントを返す（secrets/env/session_state を参照）。UIはここで描画しない。"""
    if not _HAS_OPENAI:
        return None

    # 1) secrets
    api_key = None
    try:
        api_key = st.secrets["openai"]["api_key"]
    except Exception:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            api_key = None

    # 2) env
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "") or None

    # 3) session_state（サイドバーの「認証」で入力された値を使う）
    if not api_key:
        api_key = st.session_state.get("manual_openai_api_key") or None

    if not api_key:
        return None

    # セッション内でキーが変わったら作り直す
    if st.session_state.get("_openai_client") is None or st.session_state.get("_openai_client_key") != api_key:
        try:
            st.session_state["_openai_client"] = OpenAI(api_key=api_key)
            st.session_state["_openai_client_key"] = api_key
        except Exception as e:
            st.error(f"OpenAI クライアント初期化に失敗: {e}")
            return None

    return st.session_state["_openai_client"]


# 互換ラッパー（既存呼び出し維持用）
def _get_openai_client():
    return _get_openai_client_session()


def embed_openai(lines: List[str], model_name: str) -> np.ndarray:
    if len(lines) == 0:
        return np.zeros((0, 1), dtype=float)
    client = _get_openai_client_session()
    if client is None:
        return np.zeros((len(lines), 1), dtype=float)
    try:
        resp = client.embeddings.create(model=model_name, input=lines)
        vecs = [np.array(d.embedding, dtype=float) for d in resp.data]
        return np.vstack(vecs) if vecs else np.zeros((len(lines), 1), dtype=float)
    except Exception as e:
        st.error(f"OpenAI Embeddings 取得に失敗: {e}")
        return np.zeros((len(lines), 1), dtype=float)





# ---- Helper: Hugging Face token retrieval (secrets -> env -> session_state) ----
def _get_hf_token() -> str | None:
    # 1) secrets
    token = None
    try:
        token = st.secrets["huggingface"]["token"]
    except Exception:
        try:
            token = st.secrets["HF_TOKEN"]
        except Exception:
            token = None

    # 2) env
    if not token:
        token = os.getenv("HF_TOKEN", "") or None

    # 3) session_state（UIは描画しない）
    if not token:
        token = st.session_state.get("manual_hf_token") or None

    return token






# Ruri (Transformers)
@st.cache_resource(show_spinner=False)
def _load_ruri(model_id: str, device: str = "cpu"):
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(model_id, config=cfg, trust_remote_code=True)
    mdl.to(device).eval()
    return tok, mdl

def embed_ruri(lines: List[str], model_id: str, max_seq_len: int = 128, batch_size: int = 8, device: str = "cpu") -> np.ndarray:
    if not _HAS_TF:
        st.error("transformers/torch が見つかりません。requirements を確認してください。")
        return np.zeros((len(lines), 1), dtype=float)
    if not lines:
        return np.zeros((0, 1), dtype=float)
    # HF token (optional)
    token = _get_hf_token()
    if token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    tok, mdl = _load_ruri(model_id=model_id, device=device)
    vecs = []
    with torch.no_grad():
        for i in range(0, len(lines), batch_size):
            batch = lines[i:i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=max_seq_len, return_tensors="pt").to(device)
            out = mdl(**enc)
            hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            mean = (summed / counts).cpu().numpy().astype(np.float32)
            vecs.append(mean)
    return np.vstack(vecs) if vecs else np.zeros((len(lines), 1), dtype=np.float32)

# ===== Divergence (rolling k=3) =====
def compute_divergence_local3(emb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if emb.size == 0:
        z = np.zeros((0,), dtype=float)
        return z, z
    emb = l2_normalize_rows(emb)
    n = emb.shape[0]
    raw = np.zeros(n, dtype=float)
    for t in range(n):
        if t < WINDOW_K:
            raw[t] = 0.0
            continue
        ctx = emb[t-WINDOW_K:t].mean(axis=0)
        ctx_norm = np.linalg.norm(ctx) or 1e-9
        ctx = ctx / ctx_norm
        raw[t] = cosine_distance(emb[t], ctx)
    normed = safe_minmax_scale(raw)
    return raw, normed

# ===== UI =====
# ===== Overlay (4 models) =====
def _ensure_openai_ready() -> bool:
    if not _HAS_OPENAI:
        st.error("openai パッケージが見つかりません。")
        return False
    client = _get_openai_client_session()
    return client is not None

def _ensure_ruri_ready() -> bool:
    if not _HAS_TF:
        st.error("transformers/torch が見つかりません。requirements を確認してください。")
        return False
    return True

def compute_divergence_for_backend(lines: List[str], backend: str, model_choice: str,
                                   max_seq_len: int = 128, batch_size: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    if backend == "SBERT":
        emb = embed_sbert(lines, model_choice)
    elif backend == "OpenAI":
        emb = embed_openai(lines, model_choice)
    else:  # Ruri
        emb = embed_ruri(lines, model_choice, max_seq_len=int(max_seq_len), batch_size=int(batch_size), device="cpu")
    return compute_divergence_local3(emb)

def overlay_four_models(lines: List[str], max_seq_len: int, batch_size: int):
    """Compute and plot four-model overlay (same axis). Return dict of results and combined DataFrame."""
    results = {}

    # 1) SBERT-en
    sbert_en = "all-MiniLM-L6-v2"
    raw_en, norm_en = compute_divergence_for_backend(lines, "SBERT", sbert_en, max_seq_len, batch_size)
    results[("SBERT-en", sbert_en)] = (raw_en, norm_en)

    # 2) SBERT-multi
    sbert_multi = "paraphrase-multilingual-MiniLM-L12-v2"
    raw_multi, norm_multi = compute_divergence_for_backend(lines, "SBERT", sbert_multi, max_seq_len, batch_size)
    results[("SBERT-multi", sbert_multi)] = (raw_multi, norm_multi)

    # 3) OpenAI small（利用可なら）
    if _ensure_openai_ready():
        openai_model = "text-embedding-3-small"
        raw_oa, norm_oa = compute_divergence_for_backend(lines, "OpenAI", openai_model, max_seq_len, batch_size)
        results[("OpenAI", openai_model)] = (raw_oa, norm_oa)
    else:
        st.warning("OpenAI Embeddings をスキップしました（キー未設定またはパッケージ未導入）。")

    # 4) Ruri（利用可なら）
    if _ensure_ruri_ready():
        ruri_model = "cl-nagoya/ruri-v3-30m"
        raw_ru, norm_ru = compute_divergence_for_backend(lines, "Ruri-v3-30m", ruri_model, max_seq_len, batch_size)
        results[("Ruri", ruri_model)] = (raw_ru, norm_ru)
    else:
        st.warning("Ruri（transformers/torch）をスキップしました。")

    # ---- Plot overlay (same axis) ----
    fig = plt.figure(figsize=(11.5, 3.8), dpi=150)
    x = np.arange(1, len(lines) + 1)
    for (backend_name, model_name), (raw, normed) in results.items():
        plt.plot(x, normed, linewidth=1.25, label=f"{backend_name} ({model_name})")
    plt.xlabel("行番号")
    plt.ylabel("Divergence (0..1 正規化)")
    plt.title("4モデル重ね描画（逐次ローカル k=3）")
    plt.grid(alpha=0.3)
    # leave room for legend on the right
    plt.subplots_adjust(right=0.78)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0., frameon=False)
    st.pyplot(fig, clear_figure=True)

    # ---- Combined CSV (long format) ----
    recs = []
    for (backend_name, model_name), (raw, normed) in results.items():
        for i, (rv, nv) in enumerate(zip(raw, normed), start=1):
            recs.append({
                "line_index": i,
                "line_text": lines[i-1] if 1 <= i <= len(lines) else "",
                "divergence_raw": float(rv),
                "divergence_norm": float(nv),
                "backend": backend_name,
                "embedding_model": model_name,
                "model_key": f"{backend_name}:{model_name}",
                "window": int(WINDOW_K),
                "reference": "rolling_k3",
                "l2_normalized": True,
            })
    df_overlay = pd.DataFrame(recs)
    return results, df_overlay

st.set_page_config(page_title="Poetic Divergence — 4 Models (k=3)", layout="centered")
st.title("Divergence（詩的跳躍度）— 逐次ローカル基準（k=3）/ 4モデル統合")
st.caption("入力テキスト（またはDOCX）から、行ごとの Divergence（直前3行基準）を計算して波形表示／CSV出力します。")


with st.sidebar:
    if st.button("🔒 入力したキー/トークンを忘れる", help="このセッション内で保持している OpenAI/HF の情報を破棄します。"):
        for k in [
            "manual_openai_api_key", "manual_hf_token",
            "_openai_client", "_openai_client_key",
            "openai_client", "openai_api_key"
        ]:
            st.session_state.pop(k, None)
        st.success("このセッションのキー/トークンを破棄しました。")


with st.sidebar:
    st.markdown("### 認証")
    # --- OpenAI key ---
    _openai_key = None
    try:
        _openai_key = st.secrets["openai"]["api_key"]
    except Exception:
        try:
            _openai_key = st.secrets.get("OPENAI_API_KEY", None)
        except Exception:
            _openai_key = None
    if not _openai_key:
        _openai_key = os.getenv("OPENAI_API_KEY", "") or None
    if not _openai_key:
        val = st.session_state.get("manual_openai_api_key", "")
        val = st.text_input("🔑 OpenAI API Key", value=val, type="password",
                            key="openai_api_key_input",
                            help="一度入力すると、このセッション中は保持されます。")
        if val:
            st.session_state["manual_openai_api_key"] = val

    # --- HF token ---
    _hf_tok = None
    try:
        _hf_tok = st.secrets["huggingface"]["token"]
    except Exception:
        try:
            _hf_tok = st.secrets.get("HF_TOKEN", None)
        except Exception:
            _hf_tok = None
    if not _hf_tok:
        _hf_tok = os.getenv("HF_TOKEN", "") or None
    if not _hf_tok:
        val = st.session_state.get("manual_hf_token", "")
        val = st.text_input("🪪 Hugging Face Hub Token", value=val, type="password",
                            key="hf_token_input",
                            help="一度入力すると、このセッション中は保持されます。未入力でも公開モデルは利用可能な場合があります。")
        if val:
            st.session_state["manual_hf_token"] = val


with st.sidebar:

    max_seq_len = 128
    batch_size  = 8
    
    st.subheader("モデル選択")
    backend = st.selectbox(
        "Embedding backend",
        ["SBERT", "OpenAI", "Ruri-v3-30m"],
        index=0
    )

    if backend == "SBERT":
        MODEL_OPTIONS = {
            "all-MiniLM-L6-v2（英語寄り）": "all-MiniLM-L6-v2",
            "paraphrase-multilingual-MiniLM-L12-v2（多言語推奨）": "paraphrase-multilingual-MiniLM-L12-v2",
        }
        label = st.selectbox("SBERTモデル", options=list(MODEL_OPTIONS.keys()), index=1)
        model_choice = MODEL_OPTIONS[label]
        extra_note = "（SBERT→TF-IDF→BoW の順でフォールバック）"

    elif backend == "OpenAI":
        MODEL_OPTIONS = {
            "text-embedding-3-small（コスト軽・高速）": "text-embedding-3-small",
            "text-embedding-3-large（高精度）": "text-embedding-3-large",
        }
        label = st.selectbox("OpenAI Embeddings", options=list(MODEL_OPTIONS.keys()), index=0)
        model_choice = MODEL_OPTIONS[label]
        extra_note = "（secrets/env/入力で API Key 取得）"

    elif backend == "Ruri-v3-30m":
        model_choice = st.text_input("Ruri モデルID", "cl-nagoya/ruri-v3-30m（日本語特化）")
        max_seq_len = st.number_input("max_seq_length", 32, 512, 128, step=16)
        batch_size = st.number_input("batch_size", 1, 64, 8, step=1)

        # ★ ここでは text_input を出さない。状態のみ表示。
        _tok_state = "設定済み" if _get_hf_token() else "未設定"
        st.caption(f"HF Hub Token: {_tok_state}（サイドバー上部の『認証』で設定）")

        extra_note = "（HF Hub Token は任意）"






    st.caption(f"参照系: 逐次ローカル基準（k={WINDOW_K} 固定） {extra_note}")
    do_overlay = st.checkbox("4モデル重ね描画（同軸オーバーレイ）を表示", value=False, key="do_overlay_chk",
                                 help="SBERT英語/多言語 + OpenAI + Ruri を同軸に重ねます（利用可能なバックエンドのみ）。")
    use_normalized = st.toggle("グラフを 0..1 正規化で表示", value=True)
    csv_mode = st.radio("CSVに含める列", ["rawのみ", "normalizedのみ", "rawとnormalized（両方）"], index=2)

tab1, tab2 = st.tabs(["テキスト入力 / DOCX", "結果"])

with tab1:
    st.markdown("**1) 入力方法を選択**")
    up = st.file_uploader("DOCX（任意）をアップロード", type=["docx"])
    text = st.text_area("またはテキストを直接貼り付け", height=200, placeholder="ここに詩行を貼り付け / 改行で区切り")

    lines: List[str] = []
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
    # Model-specific prerequisites
    if backend == "OpenAI" and _HAS_OPENAI:
        client = _get_openai_client_session()
        if client is None:
            st.stop()
    if backend == "Ruri-v3-30m" and not _HAS_TF:
        st.error("transformers/torch が見つかりません。requirements を確認してください。")
        st.stop()

    if not lines:
        st.stop()

    with st.spinner("Divergence 計算中…"):
        if backend == "SBERT":
            emb = embed_sbert(lines, model_choice)
        elif backend == "OpenAI":
            emb = embed_openai(lines, model_choice)
        else:  # Ruri
            emb = embed_ruri(lines, model_choice, max_seq_len=int(max_seq_len), batch_size=int(batch_size), device="cpu")

        raw, normed = compute_divergence_local3(emb)
    # ---- Plot ----
    y = normed if use_normalized else raw
    fig = plt.figure(figsize=(10, 3.5), dpi=150)
    plt.plot(np.arange(1, len(lines) + 1), y, linewidth=2)
    plt.xlabel("行番号")
    plt.ylabel("Divergence " + ("(0..1 正規化)" if use_normalized else "(raw 1−cos)"))
    plt.title(f"Divergence（逐次ローカル k=3）[{backend}: {model_choice}]")
    plt.grid(alpha=0.3)
    st.pyplot(fig, clear_figure=True)

    # ---- CSV ----
    df = pd.DataFrame({
        "line_index": np.arange(1, len(lines) + 1, dtype=int),
        "line_text": lines,
        "divergence_raw": raw,
        "divergence_norm": normed,
        "backend": [backend] * len(lines),
        "embedding_model": [model_choice] * len(lines),
        "window": [int(WINDOW_K)] * len(lines),
        "reference": ["rolling_k3"] * len(lines),
        "l2_normalized": [True] * len(lines),
    })
    if csv_mode == "rawのみ":
        out_df = df[["line_index", "line_text", "divergence_raw", "backend", "embedding_model", "window", "reference", "l2_normalized"]]
        fname = "divergence_unified_raw_k3.csv"
    elif csv_mode == "normalizedのみ":
        out_df = df[["line_index", "line_text", "divergence_norm", "backend", "embedding_model", "window", "reference", "l2_normalized"]]
        fname = "divergence_unified_normalized_k3.csv"
    else:
        out_df = df[["line_index", "line_text", "divergence_raw", "divergence_norm", "backend", "embedding_model", "window", "reference", "l2_normalized"]]
        fname = "divergence_unified_raw_and_normalized_k3.csv"

    # ---- CSV ----
    csv_buf = io.StringIO()
    out_df.to_csv(csv_buf, index=False)
    st.download_button(
        "CSVをダウンロード",
        csv_buf.getvalue().encode("utf-8-sig"),
        file_name=fname,
        mime="text/csv",
    )


    # ---- Stats (skip first WINDOW_K lines) ----
    valid_raw = raw[WINDOW_K:] if len(raw) > WINDOW_K else np.array([])
    valid_normed = normed[WINDOW_K:] if len(normed) > WINDOW_K else np.array([])

    st.caption("— 概要統計（参考） —")
    st.write(pd.DataFrame({
        "lines": [len(lines)],
        "raw_mean": [float(np.mean(valid_raw)) if len(valid_raw) else 0.0],
        "raw_range": [float(np.max(valid_raw) - np.min(valid_raw)) if len(valid_raw) else 0.0],
        "norm_mean": [float(np.mean(valid_normed)) if len(valid_normed) else 0.0],
        "norm_range": [float(np.max(valid_normed) - np.min(valid_normed)) if len(valid_normed) else 0.0],
        "backend": [backend],
        "model": [model_choice],
        "window": [int(WINDOW_K)],
        "reference": ["rolling_k3"]
    }))


# ---- 4-model overlay (below single-model results) ----
if do_overlay:
    st.markdown("---")
    st.subheader("4モデル重ね描画")
    _max_seq_len = int(max_seq_len) if "max_seq_len" in locals() else 128
    _batch_size  = int(batch_size)  if "batch_size"  in locals() else 8
    results_overlay, df_overlay = overlay_four_models(lines, _max_seq_len, _batch_size)

    # ---- convert to wide format ----
    df_wide = (
        df_overlay
        .pivot(index=["line_index", "line_text"], columns="model_key", values="divergence_norm")
        .reset_index()
    )
    base_cols = ["line_index", "line_text"]
    model_cols = sorted([c for c in df_wide.columns if c not in base_cols])
    df_wide = df_wide[base_cols + model_cols]
    df_wide = df_wide.fillna("")

    # ---- download button ----
    st.subheader("重ね描画データCSV（wide形式）")
    csv_buf2 = io.StringIO()
    df_wide.to_csv(csv_buf2, index=False)
    st.download_button(
        "重ね描画データをダウンロード",
        csv_buf2.getvalue().encode("utf-8-sig"),
        file_name="divergence_overlay_four_models_k3_wide.csv",
        mime="text/csv",
    )

    # ---- convert to wide format (raw) ----
    df_wide_raw = (
        df_overlay
        .pivot(index=["line_index", "line_text"], columns="model_key", values="divergence_raw")
        .reset_index()
    )
    base_cols = ["line_index", "line_text"]
    model_cols = sorted([c for c in df_wide_raw.columns if c not in base_cols])
    df_wide_raw = df_wide_raw[base_cols + model_cols]
    df_wide_raw = df_wide_raw.fillna("")

    # ---- download button (raw wide) ----
    st.download_button(
        "重ね描画データをダウンロード（raw wide形式）",
        df_wide_raw.to_csv(index=False).encode("utf-8-sig"),
        file_name="divergence_overlay_four_models_k3_raw_wide.csv",
        mime="text/csv",
    )

