import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="AI Training Lab", page_icon="🧠", layout="wide")

# ---------- Session state ----------
defaults = {
    "score": 0, "correct": 0, "wrong": 0, "attempts": 0,
    "training_features": [], "training_labels": []
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------- Models ----------
@st.cache_resource(show_spinner="Loading the pretrained vision model...")
def feature_model():
    return MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

@st.cache_resource(show_spinner="Loading the ImageNet classifier...")
def imagenet_model():
    return MobileNetV2(weights="imagenet")

extractor = feature_model()

def image_feature(image):
    img = image.convert("RGB").resize((224, 224))
    arr = np.asarray(img).astype("float32")
    arr = preprocess_input(np.expand_dims(arr, 0))
    return extractor.predict(arr, verbose=0)[0]

def base_predict(image):
    model = imagenet_model()
    img = image.convert("RGB").resize((224, 224))
    arr = np.asarray(img).astype("float32")
    arr = preprocess_input(np.expand_dims(arr, 0))
    pred = model.predict(arr, verbose=0)
    return [(name.replace("_", " ").title(), float(score))
            for _, name, score in decode_predictions(pred, top=5)[0]]

def classroom_model():
    labels = st.session_state.training_labels
    if len(labels) < 2 or len(set(labels)) < 2:
        return None, None
    X = np.asarray(st.session_state.training_features)
    enc = LabelEncoder()
    y = enc.fit_transform(labels)
    clf = KNeighborsClassifier(
        n_neighbors=min(3, len(X)),
        weights="distance"
    )
    clf.fit(X, y)
    return clf, enc

def classroom_predict(image):
    clf, enc = classroom_model()
    if clf is None:
        return None
    feat = image_feature(image).reshape(1, -1)
    probs = clf.predict_proba(feat)[0]
    best = int(np.argmax(probs))
    return {
        "label": str(enc.inverse_transform([best])[0]),
        "confidence": float(probs[best]),
        "probabilities": {
            str(enc.inverse_transform([i])[0]): float(probs[i])
            for i in range(len(probs))
        }
    }

def reset():
    for k, v in defaults.items():
        st.session_state[k] = v
    st.rerun()

# ---------- Styling ----------
st.markdown("""
<style>
.title {font-size:3rem;font-weight:800}
.subtitle {font-size:1.2rem;opacity:.7}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧠 AI TRAINING LAB</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Teach it. Test it. Fool it.</div>',
    unsafe_allow_html=True
)

# ---------- Scoreboard ----------
a,b,c,d = st.columns(4)
a.metric("🏆 Class Score", st.session_state.score)
b.metric("🎯 AI Correct", st.session_state.correct)
c.metric("❌ AI Wrong", st.session_state.wrong)
d.metric("🧪 Attempts", st.session_state.attempts)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("🎮 Game")
    mode = st.radio("Choose a round", [
        "🟢 1. Meet the AI",
        "🟡 2. Beat the AI",
        "🔴 3. Teach the AI",
        "🔵 4. Final Exam",
        "😈 5. Fool the AI"
    ])
    st.divider()
    st.subheader("🧠 Classroom Memory")
    st.metric("Labelled examples", len(st.session_state.training_labels))
    st.metric("Classes learned", len(set(st.session_state.training_labels)))
    if st.session_state.training_labels:
        counts = pd.Series(st.session_state.training_labels).value_counts()
        st.dataframe(counts.rename("Examples").to_frame(), use_container_width=True)
    if st.button("♻️ Reset classroom AI", use_container_width=True):
        reset()

# ---------- Input ----------
st.subheader("📸 Give the AI an image")
method = st.radio(
    "Choose one:", ["📁 Upload Image", "📷 Capture Image"], horizontal=True
)
source = (
    st.file_uploader("Choose an image", type=["jpg","jpeg","png","webp"])
    if method == "📁 Upload Image"
    else st.camera_input("Take a picture")
)

if source is None:
    st.info("Upload or capture an image to begin.")
    st.stop()

image = Image.open(io.BytesIO(source.getvalue())).convert("RGB")
left, right = st.columns([1, 1.4])
with left:
    st.subheader("👀 Image")
    st.image(image, use_container_width=True)

# ---------- Round 1 ----------
if mode == "🟢 1. Meet the AI":
    with right:
        st.subheader("🤖 What does the pretrained AI think?")
        st.caption("This model was pretrained on ImageNet and has not learned from your classroom.")
        if st.button("🔮 Predict", type="primary", use_container_width=True):
            with st.spinner("AI is looking..."):
                results = base_predict(image)
            st.session_state.attempts += 1
            st.success(f"### {results[0][0]} — {results[0][1]*100:.1f}%")
            df = pd.DataFrame({
                "Prediction": [x[0] for x in results],
                "Confidence": [x[1]*100 for x in results]
            }).set_index("Prediction")
            st.bar_chart(df)
        st.info("Ask: Does high confidence guarantee that AI is correct?")

# ---------- Round 2 ----------
elif mode == "🟡 2. Beat the AI":
    with right:
        st.subheader("🧑‍🎨 Predict FIRST")
        guess = st.text_input("What do YOU think this is?")
        if st.button("🔮 Reveal AI", type="primary", use_container_width=True):
            results = base_predict(image)
            st.session_state.attempts += 1
            st.markdown(f"### 🧑 Human: **{guess or 'No answer'}**")
            st.markdown(f"### 🤖 AI: **{results[0][0]}** ({results[0][1]*100:.1f}%)")
            st.write("Was AI correct?")
            x,y = st.columns(2)
            with x:
                if st.button("✅ Correct", use_container_width=True):
                    st.session_state.correct += 1
                    st.session_state.score += 10
                    st.success("🎉 +10 points!")
            with y:
                if st.button("❌ Wrong", use_container_width=True):
                    st.session_state.wrong += 1
                    st.session_state.score += 10
                    st.success("🕵️ +10 detective points! You found an AI mistake.")

# ---------- Round 3 ----------
elif mode == "🔴 3. Teach the AI":
    with right:
        st.subheader("👩‍🏫 Teach the classroom AI")
        st.write("Give the image a correct label. The pretrained network extracts visual features; a small KNN classifier learns your classroom labels.")
        actual = st.text_input("What is this actually?", placeholder="water bottle")
        if st.button("🧠 Teach AI", type="primary", use_container_width=True):
            if not actual.strip():
                st.error("Enter a label first.")
            else:
                st.session_state.training_features.append(image_feature(image))
                st.session_state.training_labels.append(actual.strip().lower())
                st.session_state.score += 5
                st.success(f"Added **{actual.strip()}** to classroom memory. +5 points!")
                st.rerun()
        if st.session_state.training_labels:
            counts = pd.Series(st.session_state.training_labels).value_counts()
            st.dataframe(counts.rename("Examples").to_frame(), use_container_width=True)
        st.info("This is a simplified transfer-learning demonstration: the deep network is not retrained; a small classifier learns from its visual features.")

# ---------- Round 4 ----------
elif mode == "🔵 4. Final Exam":
    with right:
        st.subheader("🧪 Final Exam — unseen image")
        result = classroom_predict(image)
        if result is None:
            st.warning("Teach at least TWO different classes first.")
        else:
            st.success(f"### 🤖 {result['label']} — {result['confidence']*100:.1f}%")
            x,y = st.columns(2)
            with x:
                if st.button("✅ CORRECT", use_container_width=True):
                    st.session_state.correct += 1
                    st.session_state.attempts += 1
                    st.session_state.score += 10
                    st.balloons()
                    st.success("🎉 +10! AI generalised correctly.")
            with y:
                if st.button("❌ WRONG", use_container_width=True):
                    st.session_state.wrong += 1
                    st.session_state.attempts += 1
                    st.session_state.score += 10
                    st.warning("🕵️ +10 detective points! AI needs better examples.")
            probs = pd.DataFrame({
                "Class": list(result["probabilities"].keys()),
                "Score": [v*100 for v in result["probabilities"].values()]
            }).set_index("Class")
            st.bar_chart(probs)

# ---------- Round 5 ----------
else:
    with right:
        st.subheader("😈 FOOL THE AI")
        st.write("Find an unusual image that makes the classroom AI confidently wrong.")
        result = classroom_predict(image)
        if result is None:
            st.warning("Teach at least two classes first.")
        else:
            st.success(f"AI predicts **{result['label']}** — {result['confidence']*100:.1f}%")
            if result["confidence"] >= .8:
                st.warning("🔥 HIGH CONFIDENCE! Can you prove the AI is wrong?")
            else:
                st.info("The AI is uncertain. What makes this image difficult?")
            actual = st.text_input("If AI is wrong, what is the correct label?")
            if st.button("🕵️ Confirm AI was fooled", type="primary", use_container_width=True):
                if actual.strip():
                    st.session_state.score += 20
                    st.session_state.wrong += 1
                    st.session_state.attempts += 1
                    st.success(f"😈 +20! AI said **{result['label']}**, but you say **{actual.strip()}**.")
                else:
                    st.error("Enter the correct label first.")

# ---------- Learning map ----------
st.divider()
st.subheader("🎓 What just happened?")
mapping = [
    ("Images", "DATA"),
    ("Labels", "DATASET / LABELLING"),
    ("Pretrained network", "DEEP LEARNING"),
    ("KNN learning", "MACHINE LEARNING"),
    ("New images", "TEST DATA"),
    ("Correct / wrong", "EVALUATION"),
    ("Mistakes", "GENERALISATION"),
]
cols = st.columns(4)
for i, (action, concept) in enumerate(mapping):
    with cols[i % 4]:
        st.markdown(f"**{concept}**")
        st.caption(action)
