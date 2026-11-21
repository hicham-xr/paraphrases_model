"""
Application Streamlit : détection de paraphrases (PAWS-X).
Lancer avec :  streamlit run app.py
Les modèles doivent se trouver dans le dossier ./models (produit par le notebook) :
  models/xlmr_fr, models/camembert_fr, models/xlmr_multi
"""
import os
import time

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "models"
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "XLM-R (français)": "xlmr_fr",
    "CamemBERT (français)": "camembert_fr",
    "XLM-R (multilingue)": "xlmr_multi",
}

EXAMPLES = {
    "Paraphrase (français)": (
        "Il a quitté l'entreprise en 2010 pour fonder sa propre société.",
        "En 2010, il a quitté l'entreprise afin de créer sa propre société.",
    ),
    "Inversion sujet/complément (non-paraphrase)": (
        "Marie a présenté Paul à Jean.",
        "Paul a présenté Marie à Jean.",
    ),
    "Paraphrase (reformulation)": (
        "La ville est située au nord de la rivière.",
        "La ville se trouve au nord de la rivière.",
    ),
    "Inversion (anglais, modèle multilingue)": (
        "The dog chased the cat across the garden.",
        "The cat chased the dog across the garden.",
    ),
}

st.set_page_config(page_title="Détecteur de paraphrases", page_icon="🔁", layout="centered")


@st.cache_resource(show_spinner="Chargement du modèle...")
def load_model(folder: str):
    path = os.path.join(MODEL_DIR, folder)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE)
    model.eval()
    return tok, model


@torch.no_grad()
def predict(tok, model, s1: str, s2: str):
    enc = tok(s1, s2, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
    probs = torch.softmax(model(**enc).logits, dim=-1)[0].cpu().numpy()
    return float(probs[0]), float(probs[1])  # (P(non-paraphrase), P(paraphrase))


# ---------------------------- Interface ----------------------------
st.title("🔁 Détecteur de paraphrases")
st.caption("Projet NLP/LLM · PAWS-X · Entrez deux phrases : le modèle dit si elles ont le même sens.")

available = {name: f for name, f in MODELS.items() if os.path.isdir(os.path.join(MODEL_DIR, f))}
if not available:
    st.error(
        f"Aucun modèle trouvé dans le dossier `{MODEL_DIR}/`. "
        "Exécute d'abord le notebook, puis place le dossier `models/` à côté de `app.py`."
    )
    st.stop()

with st.sidebar:
    st.header("Paramètres")
    model_name = st.selectbox("Modèle", list(available))
    threshold = st.slider("Seuil de décision (P(paraphrase))", 0.05, 0.95, 0.50, 0.05)
    compare = st.checkbox("Comparer avec tous les modèles disponibles", value=False)
    st.markdown(f"**Appareil :** `{DEVICE}`")

# Exemples prédéfinis (pratique pour la démo)
example = st.selectbox("Exemples prédéfinis", ["(saisie libre)"] + list(EXAMPLES))
default1, default2 = EXAMPLES.get(example, ("", ""))

s1 = st.text_area("Phrase 1", value=default1, height=80, key=f"s1_{example}")
s2 = st.text_area("Phrase 2", value=default2, height=80, key=f"s2_{example}")

if st.button("Analyser", type="primary"):
    if not s1.strip() or not s2.strip():
        st.warning("Merci de remplir les deux phrases.")
    else:
        targets = list(available) if compare else [model_name]
        for name in targets:
            tok, model = load_model(available[name])
            t0 = time.time()
            p_non, p_para = predict(tok, model, s1.strip(), s2.strip())
            elapsed = (time.time() - t0) * 1000
            is_para = p_para >= threshold

            st.subheader(name)
            if is_para:
                st.success("✅ **Paraphrase** : les deux phrases ont le même sens.")
            else:
                st.error("❌ **Non-paraphrase** : le sens est différent.")
            col1, col2 = st.columns(2)
            col1.metric("P(paraphrase)", f"{p_para * 100:.1f} %")
            col2.metric("P(non-paraphrase)", f"{p_non * 100:.1f} %")
            st.progress(min(max(p_para, 0.0), 1.0), text=f"Confiance « paraphrase » : {p_para * 100:.1f} %")
            st.caption(f"Temps d'inférence : {elapsed:.0f} ms")

st.divider()
st.caption(
    "Remarque : PAWS-X contient des paires aux mots très proches ; l'ordre des mots peut changer le sens. "
    "Les modèles français ne sont garantis que pour le français ; le modèle multilingue accepte aussi EN/ES/DE."
)
