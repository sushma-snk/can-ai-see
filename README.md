# 🧠 Can AI See What You See?

A Streamlit classroom demo for B.Des Product Design and Interaction Design students.

It uses **MobileNetV2 pretrained on ImageNet**. Students upload images and explore prediction, confidence, human-vs-AI guessing, and attempts to fool the model.

## Run locally

Use Python 3.11 if possible.

```bash
py -3.11 -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the pretrained model weights.

## GitHub + Streamlit Cloud

Push `app.py`, `requirements.txt`, `README.md`, `.gitignore`, and `.streamlit/config.toml` to a GitHub repository. Then create a Streamlit Community Cloud app using `app.py` as the main file.

## Teaching point

The model cannot literally recognise anything. It can only predict classes represented in its training data. That limitation is useful for discussing training data, pattern recognition, confidence, generalisation, and model errors.
