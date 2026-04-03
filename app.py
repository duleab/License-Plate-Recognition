"""
License Plate Recognition System — Streamlit Deployment
Run:  streamlit run app.py
Requires: ultralytics, easyocr, opencv-python-headless, streamlit
"""

import re
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="LPR System · License Plate Recognition",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# STYLE — Premium dark UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:        #080B10;
    --surface:   #0E1117;
    --surface2:  #141920;
    --border:    #1C2333;
    --border-hi: #2A3650;
    --accent:    #4F8EF7;
    --accent2:   #7B61FF;
    --green:     #22D3A5;
    --text:      #E2E8F0;
    --muted:     #4A5568;
    --muted2:    #718096;
    --danger:    #FC5A5A;
    --warn:      #F6AD55;
    --radius:    10px;
    --radius-sm: 6px;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stApp"] {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

/* scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 4px; }

/* ── sidebar ─────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stNumberInput label { color: var(--muted2) !important; font-size: 0.75rem !important; }

/* ── hide default chrome ─────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
.stDeployButton { display: none; }

/* ── headings ────────────────────────────────────────── */
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; }

/* ── file uploader ───────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--surface2);
    border: 1.5px dashed var(--border-hi);
    border-radius: var(--radius);
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(79,142,247,0.08);
}

/* ── primary button ──────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #fff;
    border: none;
    border-radius: var(--radius-sm);
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.82rem;
    letter-spacing: 0.04em;
    padding: 0.55rem 1.3rem;
    cursor: pointer;
    transition: opacity 0.15s, transform 0.1s, box-shadow 0.15s;
    box-shadow: 0 2px 12px rgba(79,142,247,0.25);
}
.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(79,142,247,0.4);
}
.stButton > button:active { transform: translateY(0); }

/* ── sliders ─────────────────────────────────────────── */
[data-testid="stSlider"] > div > div > div > div { background: var(--accent) !important; }

/* ── metrics ─────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    position: relative;
    overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
[data-testid="stMetricLabel"] { color: var(--muted2) !important; font-size: 0.72rem !important; letter-spacing: 0.06em; text-transform: uppercase; }
[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* ── tabs ────────────────────────────────────────────── */
[data-testid="stTabs"] { border-bottom: 1px solid var(--border); }
[data-testid="stTabs"] button {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    color: var(--muted2);
    padding: 0.6rem 1.2rem;
    border-bottom: 2px solid transparent;
    transition: color 0.2s, border-color 0.2s;
}
[data-testid="stTabs"] button:hover { color: var(--text); }
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent);
    border-bottom: 2px solid var(--accent);
}

/* ── dataframe ───────────────────────────────────────── */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: var(--radius); }

/* ── info/warning/error ──────────────────────────────── */
[data-testid="stAlert"] { border-radius: var(--radius-sm); }

/* ── progress bar ────────────────────────────────────── */
[data-testid="stProgress"] > div > div { background: linear-gradient(90deg, var(--accent), var(--accent2)) !important; }

/* ── custom components ───────────────────────────────── */
.lpr-logo {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-bottom: 0.25rem;
}
.lpr-logo .dot { color: var(--accent); }
.lpr-tagline { font-size: 0.68rem; color: var(--muted); letter-spacing: 0.08em; text-transform: uppercase; }

.section-label {
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.35rem;
    margin: 1.2rem 0 0.8rem;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.3rem 0.75rem;
    border-radius: 100px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.status-pill.ready  { background: rgba(34,211,165,0.12); color: var(--green); border: 1px solid rgba(34,211,165,0.25); }
.status-pill.error  { background: rgba(252, 90, 90,0.12); color: var(--danger); border: 1px solid rgba(252,90,90,0.25); }
.status-pill.idle   { background: rgba(74,85,104,0.2); color: var(--muted2); border: 1px solid var(--border); }
.status-pill .dot-pulse { width:6px; height:6px; border-radius:50%; animation: pulse 1.8s ease-in-out infinite; }
.status-pill.ready .dot-pulse { background: var(--green); }
.status-pill.error .dot-pulse { background: var(--danger); }
.status-pill.idle  .dot-pulse { background: var(--muted2); animation: none; }

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.8); }
}

.plate-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.plate-card:hover { border-color: var(--border-hi); box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
.plate-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, var(--accent), var(--accent2));
}
.plate-card.warn::before { background: var(--warn); }
.plate-card.danger::before { background: var(--danger); }

.plate-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 500;
    color: var(--text);
    letter-spacing: 0.1em;
}
.plate-num.warn   { color: var(--warn); }
.plate-num.danger { color: var(--danger); }
.plate-idx  { font-size: 0.6rem; color: var(--muted); letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.25rem; }
.plate-meta { font-size: 0.7rem; color: var(--muted2); margin-top: 0.4rem; font-family: 'JetBrains Mono', monospace; }

.conf-bar-wrap { margin-top: 0.5rem; }
.conf-bar-label { display: flex; justify-content: space-between; font-size: 0.62rem; color: var(--muted); margin-bottom: 0.2rem; }
.conf-bar { height: 3px; background: var(--border); border-radius: 2px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 2px; background: linear-gradient(90deg, var(--accent), var(--accent2)); transition: width 0.4s ease; }

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    line-height: 1.1;
    background: linear-gradient(135deg, var(--text) 40%, var(--muted2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub { color: var(--muted2); font-size: 0.85rem; margin-top: 0.3rem; line-height: 1.6; }
.hero-divider { border: none; border-top: 1px solid var(--border); margin: 1.25rem 0 1.5rem; }

.upload-hint {
    text-align: center;
    padding: 2.5rem 1rem;
    color: var(--muted2);
    font-size: 0.82rem;
    line-height: 2;
}
.upload-hint .icon { font-size: 2rem; margin-bottom: 0.5rem; }
.upload-hint strong { color: var(--text); display: block; font-family: 'Space Grotesk', sans-serif; font-size: 0.95rem; margin-bottom: 0.3rem; }

.stats-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin: 0.8rem 0;
}
.stat-chip {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.3rem 0.7rem;
    font-size: 0.7rem;
    color: var(--muted2);
    font-family: 'JetBrains Mono', monospace;
}
.stat-chip span { color: var(--accent); font-weight: 600; }

.model-path-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.2rem 0.5rem;
    color: var(--green);
    display: inline-block;
    margin-top: 0.3rem;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════
if "detector" not in st.session_state:
    st.session_state.detector = None
if "reader" not in st.session_state:
    st.session_state.reader = None
if "model_status" not in st.session_state:
    st.session_state.model_status = "idle"   # idle | ready | error
if "model_error" not in st.session_state:
    st.session_state.model_error = ""
if "loaded_model_path" not in st.session_state:
    st.session_state.loaded_model_path = ""


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def load_models_now(model_path: str, use_gpu: bool):
    """Actually load YOLO + EasyOCR — call only on button press."""
    from ultralytics import YOLO
    import easyocr
    detector = YOLO(model_path)
    reader   = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
    return detector, reader


def preprocess_plate(crop_bgr: np.ndarray) -> np.ndarray:
    gray     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    h, w     = denoised.shape
    target_w = max(1, int(w * 80 / h))
    return cv2.resize(denoised, (target_w, 80))


def clean_text(raw: str) -> str:
    cleaned = raw.upper().strip()
    cleaned = re.sub(r"[^A-Z0-9\- ]", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def detect_plates(image_bgr, detector, reader, conf, ocr_min, pad):
    results     = detector(image_bgr, conf=conf, verbose=False)
    h_img, w_img = image_bgr.shape[:2]
    plates      = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        det_conf        = float(box.conf[0].cpu())
        px = int((x2 - x1) * pad); py = int((y2 - y1) * pad)
        x1c = max(0, x1 - px); y1c = max(0, y1 - py)
        x2c = min(w_img, x2 + px); y2c = min(h_img, y2 + py)
        crop = image_bgr[y1c:y2c, x1c:x2c]
        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            continue
        processed   = preprocess_plate(crop)
        ocr_results = reader.readtext(processed, detail=1)
        kept        = [(t, c) for _, t, c in ocr_results if c >= ocr_min]
        raw_text    = " ".join([t for t, _ in kept])
        ocr_conf    = float(np.mean([c for _, c in kept])) if kept else 0.0
        plates.append({
            "bbox"    : [x1c, y1c, x2c, y2c],
            "det_conf": round(det_conf, 3),
            "text"    : clean_text(raw_text),
            "raw_text": raw_text,
            "ocr_conf": round(ocr_conf, 3),
            "crop"    : crop,
        })
    return plates


def annotate(image_bgr, plates,
             box_color=(79, 142, 247), text_color=(255, 255, 255)):
    out = image_bgr.copy()
    for p in plates:
        x1, y1, x2, y2 = p["bbox"]
        label = p["text"] if p["text"] else f"plate ({p['det_conf']:.2f})"
        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        cv2.rectangle(out, (x1, y1 - th - bl - 8), (x1 + tw + 8, y1), box_color, -1)
        cv2.putText(out, label, (x1 + 4, y1 - bl - 3),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, text_color, 1, cv2.LINE_AA)
        sub = f"det {p['det_conf']:.2f}  ocr {p['ocr_conf']:.2f}"
        cv2.putText(out, sub, (x1 + 2, y2 + 14),
                    cv2.FONT_HERSHEY_PLAIN, 0.8, box_color, 1, cv2.LINE_AA)
    return out


def plate_card_html(plate, index):
    text     = plate["text"] or "(unreadable)"
    det, ocr = plate["det_conf"], plate["ocr_conf"]
    css      = "plate-card"
    num_css  = "plate-num"
    if not plate["text"] or ocr < 0.3:
        css += " warn"; num_css += " warn"
    det_pct = int(det * 100)
    ocr_pct = int(ocr * 100)
    return f"""
    <div class="{css}">
        <div class="plate-idx">Plate {index}</div>
        <div class="{num_css}">{text}</div>
        <div class="conf-bar-wrap">
            <div class="conf-bar-label"><span>Detection</span><span>{det:.3f}</span></div>
            <div class="conf-bar"><div class="conf-bar-fill" style="width:{det_pct}%"></div></div>
        </div>
        <div class="conf-bar-wrap">
            <div class="conf-bar-label"><span>OCR</span><span>{ocr:.3f}</span></div>
            <div class="conf-bar"><div class="conf-bar-fill" style="width:{ocr_pct}%;background:linear-gradient(90deg,#22D3A5,#4F8EF7)"></div></div>
        </div>
        <div class="plate-meta">raw: {plate['raw_text'] or '—'}</div>
    </div>
    """


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo
    st.markdown("""
    <div class="lpr-logo">🔍 <span>LPR<span class="dot">·</span>Studio</span></div>
    <div class="lpr-tagline">License Plate Recognition</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Model</div>', unsafe_allow_html=True)
    model_path = st.text_input(
        "Weights path",
        value="plate_detector_best.pt",
        label_visibility="collapsed",
        placeholder="plate_detector_best.pt",
    )
    use_gpu = st.toggle("Use GPU (CUDA)", value=False)

    # Load button + status
    col_btn, col_stat = st.columns([2, 3])
    with col_btn:
        load_clicked = st.button("⚡ Load", use_container_width=True)
    with col_stat:
        status = st.session_state.model_status
        if status == "ready":
            st.markdown('<div class="status-pill ready"><div class="dot-pulse"></div>Ready</div>', unsafe_allow_html=True)
        elif status == "error":
            st.markdown('<div class="status-pill error"><div class="dot-pulse"></div>Failed</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-pill idle"><div class="dot-pulse"></div>Not loaded</div>', unsafe_allow_html=True)

    if load_clicked:
        if not Path(model_path).exists():
            st.session_state.model_status = "error"
            st.session_state.model_error  = f"File not found: `{model_path}`"
            st.session_state.detector = None
            st.session_state.reader   = None
        else:
            with st.spinner("Loading YOLO + EasyOCR… (first run may download OCR models, ~30s)"):
                try:
                    d, r = load_models_now(model_path, use_gpu)
                    st.session_state.detector        = d
                    st.session_state.reader          = r
                    st.session_state.model_status    = "ready"
                    st.session_state.loaded_model_path = model_path
                    st.session_state.model_error     = ""
                    st.rerun()
                except Exception as e:
                    st.session_state.model_status = "error"
                    st.session_state.model_error  = str(e)
                    st.session_state.detector = None
                    st.session_state.reader   = None

    if st.session_state.model_status == "error":
        st.error(st.session_state.model_error, icon="🚫")

    if st.session_state.model_status == "ready":
        st.markdown(
            f'<div class="model-path-badge">{Path(st.session_state.loaded_model_path).name}</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-label">Detection</div>', unsafe_allow_html=True)
    conf_threshold = st.slider("Confidence threshold", 0.10, 0.95, 0.50, 0.05)
    ocr_min_conf   = st.slider("OCR min confidence",   0.05, 0.50, 0.10, 0.05)
    pad_ratio      = st.slider("Crop padding",         0.00, 0.15, 0.05, 0.01)

    st.markdown('<div class="section-label">Video</div>', unsafe_allow_html=True)
    skip_frames = st.slider("Process every N frames", 1, 6, 2, 1)
    max_frames  = st.number_input("Max frames (0 = all)", min_value=0, value=0, step=50)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="stats-row">
        <div class="stat-chip">YOLO<span>26s</span></div>
        <div class="stat-chip">EasyOCR</div>
        <div class="stat-chip">CLAHE</div>
        <div class="stat-chip"><span>27,900</span> imgs</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
detector = st.session_state.detector
reader   = st.session_state.reader

# Hero header
st.markdown("""
<div class="hero-title">License Plate Recognition</div>
<div class="hero-sub">YOLO26s detection &nbsp;·&nbsp; EasyOCR reading &nbsp;·&nbsp; CLAHE bilateral preprocessing</div>
<hr class="hero-divider">
""", unsafe_allow_html=True)

# Model not loaded banner
if detector is None:
    st.markdown("""
    <div style="background:rgba(79,142,247,0.06);border:1px solid rgba(79,142,247,0.2);
                border-radius:10px;padding:1.4rem 1.6rem;margin-bottom:1.5rem;display:flex;
                align-items:center;gap:1rem;">
        <div style="font-size:1.8rem;">⚡</div>
        <div>
            <div style="font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:0.95rem;margin-bottom:0.2rem;">
                Models not loaded yet
            </div>
            <div style="font-size:0.78rem;color:#718096;line-height:1.5;">
                Click <strong style="color:#4F8EF7">⚡ Load</strong> in the sidebar to initialise YOLO26s + EasyOCR.<br>
                First run downloads OCR weights (~30 seconds). Subsequent loads are instant.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

tab_image, tab_video = st.tabs(["📷  IMAGE", "🎬  VIDEO"])


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_image:
    uploaded_img = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="img_upload",
        label_visibility="collapsed",
    )

    if uploaded_img is None:
        st.markdown("""
        <div class="upload-hint">
            <div class="icon">🖼️</div>
            <strong>Drop an image here</strong>
            Supports JPG · PNG · BMP · WEBP<br>
            Load models first, then upload to detect plates
        </div>
        """, unsafe_allow_html=True)

    elif detector is None:
        st.warning("⚡ Load the models first using the sidebar button.", icon="⚠️")

    else:
        file_bytes = np.frombuffer(uploaded_img.read(), np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Could not decode image — please try another file.", icon="🚫")
        else:
            with st.spinner("Running detection + OCR..."):
                t0         = time.time()
                plates     = detect_plates(img_bgr, detector, reader,
                                           conf_threshold, ocr_min_conf, pad_ratio)
                elapsed_ms = (time.time() - t0) * 1000
                annotated  = annotate(img_bgr, plates)

            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Plates detected", len(plates))
            m2.metric("Inference time",  f"{elapsed_ms:.0f} ms")
            m3.metric("Mean det conf",
                      f"{np.mean([p['det_conf'] for p in plates]):.3f}" if plates else "—")
            m4.metric("Mean OCR conf",
                      f"{np.mean([p['ocr_conf'] for p in plates if p['text']]):.3f}"
                      if any(p["text"] for p in plates) else "—")

            st.markdown("<br>", unsafe_allow_html=True)

            # Side-by-side images
            col_orig, col_det = st.columns(2, gap="medium")
            with col_orig:
                st.markdown('<div class="section-label">Original</div>', unsafe_allow_html=True)
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col_det:
                st.markdown('<div class="section-label">Detection result</div>', unsafe_allow_html=True)
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

            # Plate cards
            if plates:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">Plate readings</div>', unsafe_allow_html=True)
                crop_cols = st.columns(min(len(plates), 4))
                for i, (col, p) in enumerate(zip(crop_cols, plates), 1):
                    with col:
                        st.image(cv2.cvtColor(p["crop"], cv2.COLOR_BGR2RGB),
                                 use_container_width=True)
                        st.markdown(plate_card_html(p, i), unsafe_allow_html=True)

                # Download
                _, enc = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
                st.download_button(
                    label="⬇ Download annotated image",
                    data=enc.tobytes(),
                    file_name=f"lpr_{uploaded_img.name}",
                    mime="image/jpeg",
                )
            else:
                st.info("No plates detected at the current threshold. Try lowering the confidence slider.", icon="🔍")


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_video:
    uploaded_vid = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov", "mkv"],
        key="vid_upload",
        label_visibility="collapsed",
    )

    if uploaded_vid is None:
        st.markdown("""
        <div class="upload-hint">
            <div class="icon">🎬</div>
            <strong>Drop a video here</strong>
            Supports MP4 · AVI · MOV · MKV<br>
            Frame-by-frame detection with live preview
        </div>
        """, unsafe_allow_html=True)

    elif detector is None:
        st.warning("⚡ Load the models first using the sidebar button.", icon="⚠️")

    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(uploaded_vid.read())
            input_path = tmp_in.name

        cap          = cv2.VideoCapture(input_path)
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        limit      = int(max_frames) if max_frames > 0 else total_frames
        to_process = min(limit, total_frames)

        vi1, vi2, vi3, vi4 = st.columns(4)
        vi1.metric("Resolution",   f"{width}×{height}")
        vi2.metric("Frame rate",   f"{fps:.1f} fps")
        vi3.metric("Total frames", total_frames)
        vi4.metric("Will process", to_process)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("▶ Run detection on video"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
                output_path = tmp_out.name

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps / skip_frames, (width, height))

            cap           = cv2.VideoCapture(input_path)
            progress_bar  = st.progress(0)
            status_text   = st.empty()
            preview_slot  = st.empty()

            frame_results = []
            all_texts     = []
            last_plates   = []
            frame_idx     = 0
            processed     = 0
            t_start       = time.time()

            while True:
                ret, frame = cap.read()
                if not ret or frame_idx >= to_process:
                    break

                if frame_idx % skip_frames == 0:
                    plates      = detect_plates(frame, detector, reader,
                                               conf_threshold, ocr_min_conf, pad_ratio)
                    annotated_f = annotate(frame, plates)
                    last_plates = plates
                    processed  += 1
                    texts       = [p["text"] for p in plates if p["text"]]
                    all_texts.extend(texts)
                    frame_results.append({"frame": frame_idx, "n_plates": len(plates), "texts": texts})

                    if processed % 15 == 1:
                        preview_slot.image(
                            cv2.cvtColor(annotated_f, cv2.COLOR_BGR2RGB),
                            caption=f"Frame {frame_idx} — {len(plates)} plate(s) detected",
                            use_container_width=True,
                        )
                else:
                    annotated_f = annotate(frame, last_plates)

                writer.write(annotated_f)
                frame_idx += 1

                pct     = frame_idx / to_process
                elapsed = time.time() - t_start
                fps_cur = processed / max(elapsed, 0.01)
                progress_bar.progress(min(pct, 1.0))
                status_text.markdown(
                    f'<div class="stat-chip" style="display:inline-block">'
                    f'Frame <span>{frame_idx}</span>/{to_process} &nbsp;·&nbsp; '
                    f'<span>{fps_cur:.1f}</span> fps &nbsp;·&nbsp; {elapsed:.1f}s</div>',
                    unsafe_allow_html=True,
                )

            cap.release()
            writer.release()
            preview_slot.empty()
            status_text.empty()
            progress_bar.empty()

            elapsed_total = time.time() - t_start
            det_frames    = sum(1 for r in frame_results if r["n_plates"] > 0)
            unique_txt    = list(dict.fromkeys(all_texts))

            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Frames processed",  processed)
            sm2.metric("Frames with plates", det_frames)
            sm3.metric("Unique plates read", len(unique_txt))
            sm4.metric("Processing speed",  f"{processed / elapsed_total:.1f} fps")

            if unique_txt:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">Unique plates detected</div>', unsafe_allow_html=True)
                cols = st.columns(min(len(unique_txt), 4))
                for col, txt in zip(cols, unique_txt[:4]):
                    col.markdown(
                        f'<div class="plate-card"><div class="plate-num">{txt}</div></div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("<br>", unsafe_allow_html=True)
            with open(output_path, "rb") as vf:
                st.download_button(
                    label="⬇ Download processed video",
                    data=vf.read(),
                    file_name=f"lpr_{uploaded_vid.name}",
                    mime="video/mp4",
                )

            if frame_results:
                with st.expander("📋 Frame-by-frame results"):
                    import pandas as pd
                    df = pd.DataFrame([
                        {"Frame": r["frame"], "Plates": r["n_plates"],
                         "Texts": ", ".join(r["texts"]) or "—"}
                        for r in frame_results if r["n_plates"] > 0
                    ])
                    st.dataframe(df, use_container_width=True, hide_index=True)
