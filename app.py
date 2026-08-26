from __future__ import annotations

import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

try:
    import torch
except Exception:  # pragma: no cover - torch should be installed with ultralytics
    torch = None

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - fallback when cv2 is unavailable
    cv2 = None

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

MODEL_FILES = {
    "YOLO26n": MODELS_DIR / "yolo26n.pt",
    "YOLO26s": MODELS_DIR / "yolo26s.pt",
    "YOLO26m": MODELS_DIR / "yolo26m.pt",
}

st.set_page_config(
    page_title="Maritime Vision | YOLO26",
    page_icon="⚓",
    layout="wide",
)


def apply_style() -> None:
    st.markdown("""
    <style>
    :root {
        --navy: #0A2342;
        --blue: #0D6EFD;
        --cyan: #18B6D4;
        --ink: #102A43;
        --muted: #486581;
        --line: #D9E2EC;
        --paper: #FFFFFF;
        --canvas: #F4F8FC;
    }

    .stApp { background: var(--canvas); color: var(--ink); }

    [data-testid="stSidebar"] {
        background: var(--navy);
        border-right: 0;
    }

    [data-testid="stSidebar"] * {
        color: #F7FBFF;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #16395E !important;
        border: 1px solid #4A779D !important;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] span {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] .stButton button {
        background: var(--cyan);
        border-color: var(--cyan);
        color: #06233D;
    }

    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] [data-testid="stWidgetLabel"] {
        color: #000000 !important;
    }

    [data-testid="stFileUploader"] button span {
        color: #FFFFFF !important;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 3.2rem;
        padding-bottom: 3rem;
    }

    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(120deg, var(--navy), #123F6B);
        border-radius: 22px;
        padding: 2.25rem 2.4rem;
        margin-bottom: 1.5rem;
    }

    .eyebrow {
        color: #8FE9F9 !important;
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .1em;
        text-transform: uppercase;
    }

    .hero h1 {
        color: #FFFFFF !important;
        font-size: clamp(2rem, 4vw, 3.35rem);
        line-height: 1.08;
        letter-spacing: -.04em;
        margin: .55rem 0;
    }

    .hero p {
        color: #D8F1F7 !important;
        font-size: 1.02rem;
        margin: 0;
    }

    .panel {
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1.2rem;
        min-height: 100%;
        box-shadow: 0 12px 28px rgba(10, 35, 66, .06);
    }

    .panel h3, h3 {
        color: var(--ink) !important;
        font-size: 1.15rem;
        margin: 0 0 .22rem;
    }

    .subtitle,
    [data-testid="stCaptionContainer"],
    .stCaption {
        color: var(--muted) !important;
    }

    .empty {
        background: #F8FBFE;
        border: 1px dashed #A9C7E2;
        border-radius: 12px;
        padding: 3.5rem 1rem;
        color: var(--muted) !important;
        text-align: center;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        background: var(--blue);
        color: #FFFFFF !important;
        border: 1px solid var(--blue);
        border-radius: 10px;
        font-weight: 700;
        min-height: 2.8rem;
        min-width: 12rem;
        width: auto !important;
        display: inline-block !important;
        padding: 0.7rem 1rem;
    }

    div.stButton > button {
        margin-right: 0.5rem;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        background: #0959CD;
        border-color: #0959CD;
    }
    div.stButton > button,
    div.stDownloadButton > button {
        white-space: nowrap;
    }
    [data-testid="stMetric"] {
        background: var(--paper);
        border: 1px solid var(--line);
        border-top: 3px solid var(--cyan);
        border-radius: 14px;
        padding: .9rem;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: var(--ink) !important;
    }

    /* status styling moved to inline HTML for precise control */
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return YOLO(path)


def get_model_info(model, path: Path) -> dict[str, Any]:
    """Kumpulkan metadata model untuk ditampilkan di sidebar."""
    try:
        params = sum(p.numel() for p in model.model.parameters())
    except Exception:
        params = 0
    size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0.0
    names = model.names or {}
    return {
        "params": params,
        "size_mb": size_mb,
        "classes": list(names.values()),
    }


def save_uploaded_video(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


def get_video_info(video_path: Path):
    if cv2 is None:
        return {
            "fps": 25.0,
            "frames": 0,
            "width": 0,
            "height": 0,
        }

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Video tidak dapat dibuka.")

    info = {
        "fps": capture.get(cv2.CAP_PROP_FPS) or 25.0,
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    return info


def process_video(
    video_path: Path,
    model,
    model_name: str,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
):
    OUTPUTS_DIR.mkdir(exist_ok=True)

    if cv2 is None:
        raise RuntimeError("OpenCV tidak tersedia di environment ini, sehingga proses video tidak bisa dijalankan.")

    info = get_video_info(video_path)

    capture = cv2.VideoCapture(str(video_path))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUTS_DIR / f"detections_{timestamp}.csv"

    # Prioritaskan codec yang bisa diputar browser: VP8 (WebM) → MP4V (MP4).
    # H.264 (avc1) di-skip karena OpenCV Windows butuh OpenH264 DLL eksternal
    # dan bisa "berhasil" palsu (file terbuat tapi stream kosong).
    writer = None
    output_path = None
    for suffix, fourcc in ((".webm", "VP80"), (".mp4", "mp4v")):
        candidate = OUTPUTS_DIR / f"detected_{timestamp}{suffix}"
        test_writer = cv2.VideoWriter(
            str(candidate),
            cv2.VideoWriter_fourcc(*fourcc),
            info["fps"],
            (info["width"], info["height"]),
        )
        if test_writer.isOpened():
            writer = test_writer
            output_path = candidate
            break
        test_writer.release()
        candidate.unlink(missing_ok=True)

    if writer is None or output_path is None:
        capture.release()
        raise RuntimeError("Output video tidak dapat dibuat. Periksa codec OpenCV.")

    progress = st.progress(0, text="Menyiapkan inferensi…")
    # replace st.status with a custom HTML details box so the header (kolom) can be styled
    status_placeholder = st.empty()
    initial_status_html = (
        "<details open style='margin-bottom:8px;'>"
        f"<div style='background:#FFFFFF; color:#000000; padding:0.6rem; border-radius:8px; margin-top:6px;'>Model aktif: <strong>{model_name}</strong> · conf {conf:.2f} · IoU {iou:.2f} · imgsz {imgsz}</div>"
        "</details>"
    )
    status_placeholder.markdown(initial_status_html, unsafe_allow_html=True)

    frame_index = 0
    object_count = 0
    inference_times = []
    detections: list[dict[str, Any]] = []
    frames_per_class: dict[str, set[int]] = {}
    class_names = model.names or {}
    process_started = time.perf_counter()

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            inference_started = time.perf_counter()
            result = model(frame, verbose=False, conf=conf, iou=iou, imgsz=imgsz)[0]
            inference_times.append(time.perf_counter() - inference_started)

            if result.boxes is not None and len(result.boxes):
                object_count += len(result.boxes)
                for cls_id, confidence, xyxy in zip(
                    result.boxes.cls.tolist(),
                    result.boxes.conf.tolist(),
                    result.boxes.xyxy.tolist(),
                ):
                    class_name = str(class_names.get(int(cls_id), int(cls_id)))
                    detections.append(
                        {
                            "frame": frame_index,
                            "class_id": int(cls_id),
                            "class_name": class_name,
                            "confidence": round(float(confidence), 4),
                            "x1": round(xyxy[0], 1),
                            "y1": round(xyxy[1], 1),
                            "x2": round(xyxy[2], 1),
                            "y2": round(xyxy[3], 1),
                        }
                    )
                    frames_per_class.setdefault(class_name, set()).add(frame_index)

            writer.write(result.plot())
            frame_index += 1

            percentage = frame_index / info["frames"] if info["frames"] else 0
            elapsed = time.perf_counter() - process_started
            eta_text = ""
            if frame_index > 5 and info["frames"]:
                remaining = elapsed / frame_index * (info["frames"] - frame_index)
                eta_text = f" · sisa ≈ {remaining:,.0f} dtk"
            progress.progress(
                min(percentage, 1.0),
                text=f"Memproses frame {frame_index:,} dari {info['frames']:,}{eta_text}",
            )
    finally:
        capture.release()
        writer.release()

    progress.progress(1.0, text="Deteksi selesai")

    detections_df = pd.DataFrame(
        detections,
        columns=["frame", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"],
    )
    detections_df.to_csv(csv_path, index=False)

    processing_time = time.perf_counter() - process_started
    summary = {
        "model": model_name,
        "frames": frame_index,
        "processing_time": processing_time,
        "average_ms": float(np.mean(inference_times) * 1000) if inference_times else 0,
        "fps": frame_index / processing_time if processing_time > 0 else 0,
        "resolution": f"{info['width']} × {info['height']}",
        "objects": object_count,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "csv_path": str(csv_path),
        "detections_csv": detections_df.to_csv(index=False).encode("utf-8"),
        "detections_df": detections_df,
        "per_class": (
            detections_df.groupby("class_name")
            .agg(
                total_deteksi=("confidence", "size"),
                rata2_confidence=("confidence", "mean"),
                jumlah_frame=("frame", "nunique"),
            )
            .round(3)
            .sort_values("total_deteksi", ascending=False)
            if not detections_df.empty
            else pd.DataFrame()
        ),
        "per_frame": (
            detections_df.groupby("frame").size()
            if not detections_df.empty
            else pd.Series(dtype=int)
        ),
    }
    return output_path, summary


def main():
    apply_style()
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    for key, value in {
        "model": None,
        "model_name": None,
        "source_path": None,
        "upload_key": None,
        "result_path": None,
        "summary": None,
        "prev_fps": None,
    }.items():
        st.session_state.setdefault(key, value)

    with st.sidebar:
        st.markdown("### ⚓ Maritime Vision")
        st.caption("Konfigurasi inferensi")

        selected_name = st.selectbox("Pilih model YOLO26", list(MODEL_FILES))
        selected_path = MODEL_FILES[selected_name]

        if st.button("Load Model"):
            if not selected_path.exists():
                st.error(f"Model belum ada: `models/{selected_path.name}`")
            else:
                with st.spinner(f"Memuat {selected_name}…"):
                    st.session_state.model = load_model(str(selected_path))
                st.session_state.model_name = selected_name

        st.divider()
        st.markdown("**Model aktif**")
        st.info(st.session_state.model_name or "Belum ada model yang dimuat.")

        if st.session_state.model is not None:
            model_info = get_model_info(st.session_state.model, MODEL_FILES[st.session_state.model_name])
            st.caption(
                f"Parameter: {model_info['params'] / 1e6:.1f} jt · Ukuran: {model_info['size_mb']:.1f} MB"
            )
            if model_info["classes"]:
                st.caption("Kelas: " + ", ".join(model_info["classes"]))

        st.divider()
        st.markdown("**Parameter inferensi**")
        conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
        iou = st.slider("IoU threshold (NMS)", 0.1, 0.9, 0.45, 0.05)
        imgsz = st.select_slider(
            "Ukuran input (imgsz)",
            options=[320, 416, 512, 640, 768, 960, 1280],
            value=640,
        )
        cuda_available = bool(torch is not None and torch.cuda.is_available())
        st.caption(f"Perangkat: {'GPU (CUDA)' if cuda_available else 'CPU'}")

    st.markdown("""
        <section class="hero">
        <div class="eyebrow">Sistem Pemantauan Maritim Otomatis</div>
        <h1>Deteksi dan klasifikasi kapal dari video UAV</h1>
        </section>
        """, unsafe_allow_html=True)

    upload = st.file_uploader(
        "Upload video UAV",
        type=["mp4", "avi", "mov", "mkv"],
    )

    source_path = None
    video_info = None

    if upload:
        upload_key = f"{upload.name}-{upload.size}"

        if st.session_state.upload_key != upload_key:
            st.session_state.source_path = str(save_uploaded_video(upload))
            st.session_state.upload_key = upload_key
            st.session_state.result_path = None
            st.session_state.summary = None

        source_path = Path(st.session_state.source_path)
        video_info = get_video_info(source_path)

    # Make the video panel span the full content width (no empty side columns)
    st.markdown(
        '<div class="panel"><h3>Video asli</h3>'
        '<p class="subtitle">Input dari kamera UAV</p>',
        unsafe_allow_html=True,
    )

    if source_path and source_path.exists():
        # Use raw bytes for st.video to avoid path handling issues.
        source_bytes = source_path.read_bytes()
        st.video(source_bytes)
        st.caption(
            f"{upload.name if upload else source_path.name} · {video_info['width']} × "
            f"{video_info['height']} · {video_info['frames']:,} frame"
        )
    else:
        st.markdown(
            '<div class="empty">Unggah video untuk melihat video asli.</div>',
            unsafe_allow_html=True,
        )

    # Tombol proses di dalam panel; tombol unduh ada di tab "Video hasil"
    process_clicked = st.button(
        "Proses deteksi",
        type="primary",
        disabled=not (source_path is not None and st.session_state.model is not None),
    )

    st.markdown("</div>", unsafe_allow_html=True)

    can_process = source_path is not None and st.session_state.model is not None

    if process_clicked:
        try:
            result_path, summary = process_video(
                source_path,
                st.session_state.model,
                st.session_state.model_name,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
            )
            st.session_state.prev_fps = (
                st.session_state.summary["fps"] if st.session_state.summary else None
            )
            st.session_state.result_path = str(result_path)
            st.session_state.summary = summary
            st.toast("Deteksi selesai!", icon="✅")
            st.rerun()
        except Exception as error:
            st.error(f"Proses deteksi gagal: {error}")

    if not can_process:
        st.caption("Unggah video dan muat model untuk mengaktifkan deteksi.")

    if st.session_state.summary:
        summary = st.session_state.summary
        st.markdown("### Ringkasan deteksi")

        fps_delta = None
        if st.session_state.prev_fps:
            fps_delta = f"{summary['fps'] - st.session_state.prev_fps:+.1f} FPS vs run sebelumnya"

        with st.container(horizontal=True):
            st.metric("Model", summary["model"], border=True)
            st.metric("Total frame", f"{summary['frames']:,}", border=True)
            st.metric("Resolusi", summary["resolution"], border=True)
            st.metric("Waktu proses", f"{summary['processing_time']:.2f} dtk", border=True)
            st.metric("Kecepatan", f"{summary['fps']:.1f} FPS", delta=fps_delta, border=True)
            st.metric(
                "Rata-rata inferensi",
                f"{summary['average_ms']:.2f} ms/frame",
                border=True,
            )
            st.metric("Objek terdeteksi", f"{summary['objects']:,}", border=True)

        st.caption(
            f"Parameter: conf {summary['conf']:.2f} · IoU {summary['iou']:.2f} · imgsz {summary['imgsz']}"
        )

        tab_video, tab_statistik, tab_data = st.tabs(["Video hasil", "Statistik", "Data deteksi"])

        with tab_video:
            result_path = Path(st.session_state.result_path)
            result_exists = result_path.exists()
            result_bytes = result_path.read_bytes() if result_exists else b""
            video_mime = "video/webm" if result_path.suffix == ".webm" else "video/mp4"
            if result_exists:
                st.video(result_bytes, format=video_mime)
            st.download_button(
                f"Unduh video hasil ({result_path.suffix})",
                data=result_bytes,
                file_name=result_path.name,
                mime=video_mime,
                disabled=not result_exists,
            )

        with tab_statistik:
            per_class = summary["per_class"]
            if per_class.empty:
                st.info("Tidak ada objek terdeteksi. Coba turunkan confidence threshold.")
            else:
                st.markdown("**Deteksi per kelas**")
                st.bar_chart(per_class["total_deteksi"])

                st.markdown("**Objek per frame**")
                per_frame = summary["per_frame"]
                if not per_frame.empty:
                    st.line_chart(per_frame)

                st.markdown("**Distribusi confidence**")
                st.area_chart(summary["detections_df"]["confidence"].value_counts().sort_index())

        with tab_data:
            per_class = summary["per_class"]
            if per_class.empty:
                st.info("Tidak ada data deteksi.")
            else:
                st.markdown("**Statistik per kelas**")
                st.dataframe(per_class, width="stretch")

                st.markdown("**Data mentah per deteksi**")
                st.dataframe(summary["detections_df"], width="stretch", hide_index=True)

                st.download_button(
                    "Unduh data deteksi (.csv)",
                    data=summary["detections_csv"],
                    file_name=Path(summary["csv_path"]).name,
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()