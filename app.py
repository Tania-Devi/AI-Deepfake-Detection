
import streamlit as st
from PIL import Image
from detector import analyze_image, analyze_video

st.set_page_config(
    page_title="Deepfake Detection AI",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101827, #172554);
    color: white;
}
.block-container {
    max-width: 1100px;
    padding-top: 2rem;
}
.hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(120deg, #1e3a8a, #4338ca);
    text-align: center;
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 38px;
}
.hero p {
    color: #dbeafe;
    font-size: 17px;
}
div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.06);
    padding: 15px;
    border-radius: 12px;
}
.stButton button {
    width: 100%;
    border-radius: 10px;
    background: #4f46e5;
    color: white;
    border: none;
    padding: 10px;
    font-weight: bold;
}
.stButton button:hover {
    background: #6366f1;
    color: white;
}
.result-card {
    padding: 20px;
    border-radius: 14px;
    background: #1e293b;
    border: 1px solid #475569;
    margin: 12px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🔍 AI Deepfake Detection</h1>
    <p>Analyze images and videos with a pretrained AI model</p>
</div>
""", unsafe_allow_html=True)


def show_result(scores):
    if not scores:
        st.error("No predictions available.")
        return

    best_label = max(scores, key=scores.get)
    confidence = scores[best_label]

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    if "fake" in best_label.lower():
        st.error(f"🚨 Prediction: FAKE ({confidence:.2%})")
    elif "real" in best_label.lower():
        st.success(f"✅ Prediction: REAL ({confidence:.2%})")
    else:
        st.info(f"Prediction: {best_label} ({confidence:.2%})")

    st.progress(float(confidence))
    st.caption("Prediction score from the model")

    st.markdown('</div>', unsafe_allow_html=True)
    st.warning(
        "AI predictions can be wrong and are not proof of authenticity."
    )


tab_image, tab_video = st.tabs(
    ["🖼️ Image Detection", "🎬 Video Detection"]
)

with tab_image:
    st.subheader("📷 Analyze an Image")
    st.write("Upload a JPG or PNG image to begin.")

    image_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if image_file:
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Analyze Image", type="primary"):
            try:
                with st.spinner("Analyzing image..."):
                    result = analyze_image(image)

                st.subheader("📊 Analysis Results")
                show_result({
                    item["label"]: item["score"]
                    for item in result["predictions"]
                })

            except Exception as error:
                st.error(f"Image analysis failed: {error}")


with tab_video:
    st.subheader("🎥 Analyze a Video")
    st.write("Upload a video to analyze selected frames.")

    video_file = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov"],
        key="video_upload"
    )

    if video_file:
        st.video(video_file)

        if st.button("🔍 Analyze Video", type="primary"):
            try:
                with st.spinner("Analyzing video frames..."):
                    result = analyze_video(video_file.getvalue())

                st.subheader("📊 Video Results")
                st.write(
                    f"Frames analyzed: {result['frames_analyzed']}"
                )
                show_result(result["average_scores"])

                with st.expander("View individual frame results"):
                    for frame in result["frame_results"]:
                        st.write(f"Frame {frame['frame']}")
                        for item in frame["predictions"]:
                            st.write(
                                f"- {item['label']}: {item['score']:.2%}"
                            )

            except Exception as error:
                st.error(f"Video analysis failed: {error}")
                st.info("Try a short MP4 video.")


st.divider()
st.caption(
    "AI Deepfake Detection Project | Educational use only"
)
