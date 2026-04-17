import io
import json
from typing import Any, Dict, Optional

import requests
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="Marine Image Recognition System Demo",
    page_icon=":ocean:",
    layout="wide",
)

DEFAULT_API_URL = "http://127.0.0.1:8000/predict"


def call_predict_api(api_url: str, image_bytes: bytes, filename: str) -> Dict[str, Any]:
    files = {"file": (filename, image_bytes, "image/jpeg")}
    response = requests.post(api_url, files=files, timeout=300)
    response.raise_for_status()
    return response.json()


def format_confidence(conf: Any) -> str:
    if conf is None:
        return "N/A"
    try:
        return f"{float(conf):.4f}"
    except Exception:
        return str(conf)


def normalize_image_type(image_type: Optional[str]) -> str:
    if not image_type:
        return "Unknown"
    x = str(image_type).lower()
    mapping = {
        "sonar": "Sonar",
        "biological": "Biological",
        "fish": "Fish",
        "coral": "Coral",
        "unknown": "Unknown",
    }
    return mapping.get(x, str(image_type))


def bool_to_english(v: bool) -> str:
    return "Yes" if v else "No"


def show_header():
    st.title("Marine Image Recognition System Demo")
    st.caption(
        "Multi-model marine image recognition powered by "
        "FastAPI + FAISS + YOLO + OceanCLIP"
    )


def show_summary_cards(result: Dict[str, Any]) -> None:
    core_result = result.get("result", {})
    final_result = core_result.get("final_result", {})
    modules = core_result.get("modules", {}) or {}

    retrieval = modules.get("retrieval", {}) or {}
    db_hit = retrieval.get("db_hit", False)

    image_type = final_result.get("image_type", "Unknown")
    target_label = final_result.get("primary_label", "None")
    confidence = final_result.get("confidence", None)
    stage = core_result.get("stage", "unknown")

    st.subheader("Recognition Result")

    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    with col1:
        st.metric("DB Hit", bool_to_english(bool(db_hit)))
    with col2:
        st.metric("Image Type", normalize_image_type(image_type))
    with col3:
        st.metric("Predicted Class", str(target_label), help=str(target_label))
    with col4:
        st.metric("Confidence", format_confidence(confidence))
    with col5:
        st.metric("Stage", str(stage))

    display_text = final_result.get("display_text")
    if display_text:
        st.success(display_text)


def step_status_text(has_data: bool, name: str) -> str:
    return f":white_check_mark: {name}" if has_data else f":o: {name}"


def show_pipeline(result: Dict[str, Any]) -> None:
    core_result = result.get("result", {})
    modules = core_result.get("modules", {}) or {}
    stage = core_result.get("stage", "")

    retrieval = modules.get("retrieval")
    sonar = modules.get("sonar")
    fish = modules.get("fish")
    coral = modules.get("coral")
    oceanclip = modules.get("oceanclip")
    fusion = modules.get("fusion")

    db_hit = False
    if isinstance(retrieval, dict):
        db_hit = bool(retrieval.get("db_hit", False))

    st.subheader("Pipeline Visualization")
    st.markdown("### Pipeline")

    if db_hit:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info(step_status_text(bool(retrieval), "FAISS Retrieval"))
        with c2:
            st.info(":white_check_mark: Database Hit")
        with c3:
            st.info(":pushpin: Final Result")
        st.markdown("---")
        st.markdown(
            "This sample **matched the FAISS retrieval database**, "
            "so the result was returned directly without further model inference."
        )
    else:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.info(step_status_text(bool(retrieval), "FAISS Retrieval"))
        with c2:
            st.info(step_status_text(bool(sonar), "Sonar Classifier"))
        with c3:
            st.info(step_status_text(bool(fish), "Fish Detector"))
        with c4:
            st.info(step_status_text(bool(coral), "Coral Detector"))
        with c5:
            st.info(step_status_text(bool(oceanclip), "OceanCLIP"))
        with c6:
            st.info(step_status_text(bool(fusion), "Fusion Output"))

        st.markdown("---")
        st.markdown(
            "This sample **did not match the FAISS database**. "
            "The system proceeds with multi-model inference: "
            "`Sonar Classifier + Fish Detector + Coral Detector + OceanCLIP`, "
            "then applies a fusion strategy for the final result."
        )


def show_module_details(result: Dict[str, Any]) -> None:
    core_result = result.get("result", {})
    final_result = core_result.get("final_result", {}) or {}
    modules = core_result.get("modules", {}) or {}

    st.subheader("Module Details")

    retrieval = modules.get("retrieval", {}) or {}
    sonar = modules.get("sonar", {}) or {}
    fish = modules.get("fish", {}) or {}
    coral = modules.get("coral", {}) or {}
    oceanclip = modules.get("oceanclip", {}) or {}
    fusion = modules.get("fusion", {}) or {}

    tabs = st.tabs([
        "FAISS Retrieval",
        "Sonar",
        "Fish Detector",
        "Coral Detector",
        "OceanCLIP",
        "Fusion",
        "Final Result",
    ])

    with tabs[0]:
        if retrieval:
            st.json(retrieval)
        else:
            st.info("No retrieval results.")

    with tabs[1]:
        if sonar:
            st.json(sonar)
        else:
            st.info("No Sonar results.")

    with tabs[2]:
        if fish:
            st.json(fish)
        else:
            st.info("No Fish detection results.")

    with tabs[3]:
        if coral:
            st.json(coral)
        else:
            st.info("No Coral detection results.")

    with tabs[4]:
        if oceanclip:
            st.json(oceanclip)
        else:
            st.info("No OceanCLIP results.")

    with tabs[5]:
        if fusion:
            st.json(fusion)
        else:
            st.info("No Fusion results.")

    with tabs[6]:
        st.json(final_result)


def show_result_panel(image: Image.Image, result: Dict[str, Any], show_raw_json: bool) -> None:
    show_summary_cards(result)
    st.divider()
    show_pipeline(result)
    st.divider()
    show_module_details(result)

    if show_raw_json:
        st.divider()
        st.subheader("Raw JSON Response")
        st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")


def main() -> None:
    show_header()

    with st.sidebar:
        st.header("Settings")
        api_url = st.text_input("FastAPI Endpoint", value=DEFAULT_API_URL)
        show_raw_json = st.checkbox("Show Raw JSON", value=True)

        st.markdown("---")
        st.markdown("### How It Works")
        st.write("1. Upload a marine image")
        st.write("2. Click \"Run Recognition\" to call the FastAPI endpoint")
        st.write("3. View retrieval match or multi-model fusion results")

    uploaded_file = st.file_uploader(
        "Upload an image for recognition",
        type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    if uploaded_file is None:
        st.info("Please upload an image to get started.")
        return

    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    left, right = st.columns([1, 1.25])

    with left:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with right:
        st.subheader("Run Recognition")
        run_button = st.button("Run Recognition", type="primary", use_container_width=True)

        if run_button:
            with st.spinner("Running model inference, please wait..."):
                try:
                    result = call_predict_api(
                        api_url=api_url,
                        image_bytes=image_bytes,
                        filename=uploaded_file.name
                    )
                    st.success("Recognition complete")
                    show_result_panel(image, result, show_raw_json)

                except requests.exceptions.RequestException as e:
                    st.error(f"API call failed: {e}")
                except Exception as e:
                    st.error(f"Processing failed: {e}")


if __name__ == "__main__":
    main()
