import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

st.set_page_config(page_title="Can AI See What You See?", page_icon="🧠", layout="wide")

@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

def prepare_image(image):
    image = image.convert("RGB").resize((224, 224))
    arr = np.asarray(image).astype("float32")
    return preprocess_input(np.expand_dims(arr, axis=0))

def predict(model, image):
    preds = model.predict(prepare_image(image), verbose=0)
    return [(label, float(score)) for _, label, score in decode_predictions(preds, top=5)[0]]

def label(x):
    return x.replace("_", " ").title()

st.title("🧠 Can AI See What You See?")
st.write("Upload an image and let a pretrained deep-learning model make its prediction.")
st.info("⚠️ This model cannot literally recognise everything. It can only predict categories represented in its training data.")

with st.sidebar:
    st.header("🎛️ Classroom Mode")
    mode = st.radio("Choose a mode", ["🔮 Predict", "🧑‍🎨 Human vs AI", "😈 Fool the AI", "🧠 What Did AI See?"])
    st.divider()
    st.caption("Model: MobileNetV2")
    st.caption("Pretrained on ImageNet")

model = load_model()

# uploaded = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png", "webp"])
st.subheader("📸 Give the AI an image")

input_method = st.radio(
    "Choose how you want to provide the image:",
    ["📁 Upload Image", "📷 Capture Image"],
    horizontal=True
)

uploaded = None

if input_method == "📁 Upload Image":
    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload an image from your device."
    )

else:
    uploaded = st.camera_input(
        "Take a picture"
    )
    
if not uploaded:
    st.subheader("🎯 How to use this in class")
    st.write("1. Upload an image.  2. Ask students to predict.  3. Reveal AI's answer.  4. Discuss whether it is correct.")
    st.stop()

try:
    image = Image.open(uploaded).convert("RGB")
except Exception:
    st.error("Could not read the image.")
    st.stop()

predictions = predict(model, image)
top_label, top_score = predictions[0]

left, right = st.columns([1, 1.4])
with left:
    st.subheader("👀 Your image")
    st.image(image, use_container_width=True)
    st.caption(f"{image.width} × {image.height} pixels")

with right:
    if mode == "🔮 Predict":
        st.subheader("🤖 AI prediction")
        st.success(f"### {label(top_label)}")
        st.metric("Confidence", f"{top_score*100:.1f}%")
        df = pd.DataFrame({"Prediction":[label(x[0]) for x in predictions],
                           "Confidence":[x[1]*100 for x in predictions]}).set_index("Prediction")
        st.bar_chart(df)
        st.caption("Confidence is not proof of correctness; it is the model's score for its learned classes.")

    elif mode == "🧑‍🎨 Human vs AI":
        st.subheader("🧑‍🎨 Predict first")
        guess = st.text_input("What do YOU think this is?", placeholder="Type your prediction...")
        if st.button("🔮 Reveal AI", type="primary", use_container_width=True):
            st.session_state.guess = guess
            st.session_state.reveal = True
        if st.session_state.get("reveal"):
            st.divider()
            st.success(f"🤖 AI: **{label(top_label)}** — {top_score*100:.1f}%")
            st.write(f"🧑 Human: **{st.session_state.get('guess','No guess')}**")
            st.info("Ask: Is the AI actually smarter, or did it recognise a pattern from training examples?")

    elif mode == "😈 Fool the AI":
        st.subheader("😈 Mission: Fool the AI")
        st.write("Try an unusual, ambiguous, artistic, poorly framed, or unexpected image.")
        st.success(f"AI's strongest guess: **{label(top_label)}** — {top_score*100:.1f}%")
        if top_score >= .70:
            st.warning("The model is confident. Can a confident AI still be wrong?")
        else:
            st.info("The model is uncertain. What visual ambiguity might be causing this?")
        st.markdown("**Discuss:** angle • lighting • background • ambiguity • unfamiliar object")

    else:
        st.subheader("🧠 What did AI see?")
        st.success(f"Top prediction: **{label(top_label)}**")
        for x, score in predictions:
            st.write(f"**{label(x)}** — {score*100:.1f}%")
            st.progress(min(score, 1.0))
        st.info("If it is wrong, investigate image quality, viewpoint, lighting, background, ambiguity, and whether the object belongs to a learned class.")

st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("Input", "Your image")
c2.metric("Model", "Deep Learning")
c3.metric("Output", "Top-5 predictions")
st.caption("Classroom demo: AI → ML → DL → Classification → Prediction → Confidence → Limitations")
