"""
License Plate Recognition System — Streamlit Deployment
Run:  streamlit run app.py
Requires: ultralytics, easyocr, opencv-python, streamlit
"""

import re
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import easyocr

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="License Plate Recognition",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# STYLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg:        #0A0C0F;
    --surface:   #111418;
    --border:    #1E2328;
    --border-hi: #2D3540;
    --accent:    #C8F135;
    --accent-dk: #9BBF1A;
    --text:      #E8EDF2;
    --muted:     #5A6370;
    --danger:    #FF4C4C;
    --warn:      #F5A623;
    --success:   #3DD68C;
    --radius:    6px;
}

html, body, [data-testid="stApp"] {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Mono', monospace;
}

/* ── sidebar ─────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { font-family: 'DM Mono', monospace; }

/* ── headings ────────────────────────────────────────── */
h1, h2, h3, h4 {
    font-family: 'Syne', sans-serif;
    letter-spacing: -0.02em;
}

/* ── hide default Streamlit chrome ───────────────────── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── file uploader ───────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--surface);
    border: 1px dashed var(--border-hi);
    border-radius: var(--radius);
    padding: 1.5rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent);
}

/* ── buttons ─────────────────────────────────────────── */
.stButton > button {
    background: var(--accent);
    color: #0A0C0F;
    border: none;
    border-radius: var(--radius);
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    padding: 0.55rem 1.4rem;
    cursor: pointer;
    transition: background 0.15s, transform 0.1s;
    text-transform: uppercase;
}
.stButton > button:hover {
    background: var(--accent-dk);
    transform: translateY(-1px);
}
.stButton > button:active { transform: translateY(0); }

/* ── sliders ─────────────────────────────────────────── */
[data-testid="stSlider"] > div > div > div > div {
    background: var(--accent) !important;
}

/* ── metrics ─────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
}
[data-testid="stMetricLabel"] { color: var(--muted); font-size: 0.75rem; }
[data-testid="stMetricValue"] {
    color: var(--accent);
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
}

/* ── result cards ────────────────────────────────────── */
.plate-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    margin-bottom: 0.6rem;
    font-family: 'DM Mono', monospace;
}
.plate-card .plate-text {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    color: var(--accent);
}
.plate-card .plate-meta {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 0.3rem;
}
.plate-card.warn { border-left-color: var(--warn); }
.plate-card.warn .plate-text { color: var(--warn); }
.plate-card.danger { border-left-color: var(--danger); }
.plate-card.danger .plate-text { color: var(--danger); }

/* ── section header ──────────────────────────────────── */
.section-label {
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}

/* ── wordmark ────────────────────────────────────────── */
.wordmark {
    font-family: 'Syne', sans-serif;
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--text);
    line-height: 1;
}
.wordmark span { color: var(--accent); }

/* ── status badge ────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 2px;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.badge-ok  { background: rgba(61,214,140,0.12); color: var(--success); }
.badge-warn{ background: rgba(245,166,35,0.12);  color: var(--warn);    }
.badge-err { background: rgba(255,76,76,0.12);   color: var(--danger);  }

/* ── progress bar override ───────────────────────────── */
[data-testid="stProgress"] > div > div {
    background: var(--accent) !important;
}

/* ── tab style ───────────────────────────────────────── */
[data-testid="stTabs"] button {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    letter-spacing: 0.06em;
    font-weight: 600;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    border-bottom: 2px solid var(--accent);
    color: var(--accent);
}

/* ── table ───────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: var(--radius);
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING  (cached — loads once per session)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_models(model_path: str, use_gpu: bool):
    detector = YOLO(model_path)
    reader   = easyocr.Reader(["en"], gpu=use_gpu)
    return detector, reader


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
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


def detect_plates(image_bgr: np.ndarray, detector, reader,
                  conf: float, ocr_min: float, pad: float):
    results  = detector(image_bgr, conf=conf, verbose=False)
    h_img, w_img = image_bgr.shape[:2]
    plates   = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        det_conf        = float(box.conf[0].cpu())

        px = int((x2 - x1) * pad);  py = int((y2 - y1) * pad)
        x1c = max(0, x1 - px);       y1c = max(0, y1 - py)
        x2c = min(w_img, x2 + px);   y2c = min(h_img, y2 + py)

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


def annotate(image_bgr: np.ndarray, plates: list,
             box_color=(0, 210, 60), text_color=(10, 12, 15)) -> np.ndarray:
    out = image_bgr.copy()
    for p in plates:
        x1, y1, x2, y2 = p["bbox"]
        label = p["text"] if p["text"] else f"plate ({p['det_conf']:.2f})"

        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)

        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
        cv2.rectangle(out, (x1, y1 - th - bl - 10), (x1 + tw + 8, y1), box_color, -1)
        cv2.putText(out, label, (x1 + 4, y1 - bl - 4),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, text_color, 2, cv2.LINE_AA)

        sub = f"det {p['det_conf']:.2f}  ocr {p['ocr_conf']:.2f}"
        cv2.putText(out, sub, (x1 + 2, y2 + 16),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, box_color, 1, cv2.LINE_AA)
    return out


def plate_card_html(plate: dict, index: int) -> str:
    text    = plate["text"] or "(no text read)"
    det     = plate["det_conf"]
    ocr     = plate["ocr_conf"]
    cls     = "plate-card"
    if not plate["text"]:
        cls += " warn"
    elif ocr < 0.3:
        cls += " warn"

    return f"""
    <div class="{cls}">
        <div style="font-size:0.65rem;letter-spacing:0.14em;
                    text-transform:uppercase;color:var(--muted);
                    margin-bottom:0.3rem;">Plate {index}</div>
        <div class="plate-text">{text}</div>
        <div class="plate-meta">
            Detection confidence: {det:.3f} &nbsp;|&nbsp;
            OCR confidence: {ocr:.3f} &nbsp;|&nbsp;
            Raw: {plate['raw_text'] or '—'}
        </div>
    </div>
    """


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="wordmark">LPR<span>.</span>System</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Model</div>', unsafe_allow_html=True)
    model_path = st.text_input(
        "Weights path",
        value="plate_detector_best.pt",
        help="Path to your YOLO26s .pt file",
        label_visibility="collapsed",
    )
    use_gpu = st.toggle("Use GPU", value=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Detection</div>', unsafe_allow_html=True)
    conf_threshold = st.slider("Confidence threshold", 0.10, 0.95, 0.50, 0.05)
    ocr_min_conf   = st.slider("OCR min confidence",   0.05, 0.50, 0.10, 0.05)
    pad_ratio      = st.slider("Crop padding",         0.00, 0.15, 0.05, 0.01)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Video</div>', unsafe_allow_html=True)
    skip_frames = st.slider("Process every N frames", 1, 6, 2, 1)
    max_frames  = st.number_input("Max frames (0 = all)", min_value=0, value=0, step=50)

    st.markdown("<br>", unsafe_allow_html=True)

    # Load models
    if st.button("Load / Reload Models"):
        st.cache_resource.clear()

    try:
        with st.spinner("Loading models..."):
            detector, reader = load_models(model_path, use_gpu)
        st.markdown('<span class="badge badge-ok">Models ready</span>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown('<span class="badge badge-err">Load failed</span>', unsafe_allow_html=True)
        st.caption(str(e))
        detector = reader = None

    st.markdown("<br><br>")
    st.markdown(
        '<div style="font-size:0.65rem;color:var(--muted);line-height:1.7;">'
        'YOLO26s + EasyOCR<br>'
        'CLAHE + bilateral preprocessing<br>'
        'Trained on 27,900 images'
        '</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<h1 style="font-size:2.2rem;font-weight:800;letter-spacing:-0.03em;'
    'margin-bottom:0.15rem;">License Plate Recognition</h1>'
    '<p style="color:var(--muted);font-size:0.8rem;margin-top:0;'
    'letter-spacing:0.04em;">YOLO26s detection — EasyOCR reading — '
    'CLAHE preprocessing</p>',
    unsafe_allow_html=True,
)
st.markdown("<hr style='border-color:var(--border);margin:1rem 0 1.5rem;'>",
            unsafe_allow_html=True)

tab_image, tab_video = st.tabs(["IMAGE", "VIDEO"])


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_image:
    uploaded_img = st.file_uploader(
        "Drop an image or click to browse",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="img_upload",
        label_visibility="collapsed",
    )

    if uploaded_img is not None and detector is not None:
        # decode
        file_bytes = np.frombuffer(uploaded_img.read(), np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Could not decode image.")
        else:
            with st.spinner("Running detection + OCR..."):
                t0     = time.time()
                plates = detect_plates(img_bgr, detector, reader,
                                       conf_threshold, ocr_min_conf, pad_ratio)
                elapsed_ms = (time.time() - t0) * 1000
                annotated  = annotate(img_bgr, plates)

            # ── metrics row ───────────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Plates detected", len(plates))
            m2.metric("Inference time",  f"{elapsed_ms:.0f} ms")
            m3.metric("Mean det conf",
                      f"{np.mean([p['det_conf'] for p in plates]):.3f}"
                      if plates else "—")
            m4.metric("Mean OCR conf",
                      f"{np.mean([p['ocr_conf'] for p in plates if p['text']]):.3f}"
                      if any(p['text'] for p in plates) else "—")

            st.markdown("<br>", unsafe_allow_html=True)

            # ── images ────────────────────────────────────────────────────────
            col_orig, col_det = st.columns(2, gap="medium")
            with col_orig:
                st.markdown('<div class="section-label">Original</div>',
                            unsafe_allow_html=True)
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                         use_container_width=True)
            with col_det:
                st.markdown('<div class="section-label">Detection result</div>',
                            unsafe_allow_html=True)
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                         use_container_width=True)

            # ── plate cards ───────────────────────────────────────────────────
            if plates:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">Plate readings</div>',
                            unsafe_allow_html=True)

                crop_cols = st.columns(min(len(plates), 4))
                for i, (col, p) in enumerate(zip(crop_cols, plates), 1):
                    with col:
                        st.image(cv2.cvtColor(p["crop"], cv2.COLOR_BGR2RGB),
                                 use_container_width=True)
                        st.markdown(plate_card_html(p, i), unsafe_allow_html=True)

                # download annotated
                _, annotated_enc = cv2.imencode(".jpg", annotated,
                                                [cv2.IMWRITE_JPEG_QUALITY, 92])
                st.download_button(
                    label="Download annotated image",
                    data=annotated_enc.tobytes(),
                    file_name=f"lpr_{uploaded_img.name}",
                    mime="image/jpeg",
                )
            else:
                st.info("No plates detected at the current confidence threshold. "
                        "Try lowering the threshold in the sidebar.")

    elif uploaded_img is not None and detector is None:
        st.warning("Models not loaded. Check the model path in the sidebar.")


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_video:
    uploaded_vid = st.file_uploader(
        "Drop a video or click to browse",
        type=["mp4", "avi", "mov", "mkv"],
        key="vid_upload",
        label_visibility="collapsed",
    )

    if uploaded_vid is not None and detector is not None:
        # write upload to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(uploaded_vid.read())
            input_path = tmp_in.name

        cap          = cv2.VideoCapture(input_path)
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        limit = int(max_frames) if max_frames > 0 else total_frames
        to_process = min(limit, total_frames)

        # ── video info ────────────────────────────────────────────────────────
        vi1, vi2, vi3, vi4 = st.columns(4)
        vi1.metric("Resolution", f"{width}x{height}")
        vi2.metric("Frame rate", f"{fps:.1f} fps")
        vi3.metric("Total frames", total_frames)
        vi4.metric("Will process", to_process)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Run detection on video"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
                output_path = tmp_out.name

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc,
                                     fps / skip_frames, (width, height))

            cap            = cv2.VideoCapture(input_path)
            progress_bar   = st.progress(0)
            status_text    = st.empty()
            preview_slot   = st.empty()

            frame_results  = []   # per-frame summary
            all_texts      = []
            last_plates    = []
            frame_idx      = 0
            processed      = 0
            t_start        = time.time()

            while True:
                ret, frame = cap.read()
                if not ret or frame_idx >= to_process:
                    break

                if frame_idx % skip_frames == 0:
                    plates      = detect_plates(frame, detector, reader,
                                                conf_threshold, ocr_min_conf,
                                                pad_ratio)
                    annotated_f = annotate(frame, plates)
                    last_plates = plates
                    processed  += 1

                    texts = [p["text"] for p in plates if p["text"]]
                    all_texts.extend(texts)
                    frame_results.append({
                        "frame"   : frame_idx,
                        "n_plates": len(plates),
                        "texts"   : texts,
                    })

                    # live preview every 15 processed frames
                    if processed % 15 == 1:
                        preview_slot.image(
                            cv2.cvtColor(annotated_f, cv2.COLOR_BGR2RGB),
                            caption=f"Frame {frame_idx} — {len(plates)} plate(s)",
                            use_container_width=True,
                        )
                else:
                    annotated_f = annotate(frame, last_plates)

                writer.write(annotated_f)
                frame_idx += 1

                pct = frame_idx / to_process
                progress_bar.progress(min(pct, 1.0))
                elapsed = time.time() - t_start
                fps_cur = processed / max(elapsed, 0.01)
                status_text.markdown(
                    f'<span style="font-size:0.75rem;color:var(--muted);">'
                    f'Frame {frame_idx} / {to_process} &nbsp;|&nbsp; '
                    f'{fps_cur:.1f} fps &nbsp;|&nbsp; '
                    f'{elapsed:.1f}s elapsed</span>',
                    unsafe_allow_html=True,
                )

            cap.release()
            writer.release()
            preview_slot.empty()
            status_text.empty()
            progress_bar.empty()

            elapsed_total = time.time() - t_start

            # ── summary metrics ───────────────────────────────────────────────
            det_frames = sum(1 for r in frame_results if r["n_plates"] > 0)
            unique_txt = list(dict.fromkeys(all_texts))  # preserve order, dedupe

            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Frames processed", processed)
            sm2.metric("Frames with plates", det_frames)
            sm3.metric("Unique plates read", len(unique_txt))
            sm4.metric("Processing speed", f"{processed / elapsed_total:.1f} fps")

            # ── unique plates list ────────────────────────────────────────────
            if unique_txt:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">Unique plates detected</div>',
                            unsafe_allow_html=True)
                cols = st.columns(min(len(unique_txt), 4))
                for col, txt in zip(cols, unique_txt[:4]):
                    col.markdown(
                        f'<div class="plate-card">'
                        f'<div class="plate-text">{txt}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # ── download processed video ──────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            with open(output_path, "rb") as vf:
                st.download_button(
                    label="Download processed video",
                    data=vf.read(),
                    file_name=f"lpr_{uploaded_vid.name}",
                    mime="video/mp4",
                )

            # ── frame-by-frame table ──────────────────────────────────────────
            if frame_results:
                with st.expander("Frame-by-frame results"):
                    import pandas as pd
                    df = pd.DataFrame([
                        {"Frame": r["frame"],
                         "Plates": r["n_plates"],
                         "Texts" : ", ".join(r["texts"]) or "—"}
                        for r in frame_results if r["n_plates"] > 0
                    ])
                    st.dataframe(df, use_container_width=True, hide_index=True)

    elif uploaded_vid is not None and detector is None:
        st.warning("Models not loaded. Check the model path in the sidebar.")
