# Interface de démo pour le projet PAWS-X (détection de paraphrases)
# lancement : streamlit run app.py   (ou run.bat)
# les modèles viennent du notebook -> dossier models/

import os
import time

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# nom affiché -> (dossier, description)
MODELS = {
    "XLM-R FR": ("xlmr_fr", "XLM-RoBERTa base, entraîné sur PAWS-X FR"),
    "CamemBERT FR": ("camembert_fr", "CamemBERT base, entraîné sur PAWS-X FR"),
    "XLM-R multilingue": ("xlmr_multi", "XLM-RoBERTa base, entraîné sur FR + EN + ES + DE"),
}

FR_ONLY = {"XLM-R FR", "CamemBERT FR"}  # entraînés seulement sur le français

# exemples par langue : nom -> (phrase 1, phrase 2)
EXAMPLES = {
    "Français": {
        "Reformulation simple": (
            "La ville est située au nord de la rivière.",
            "La ville se trouve au nord de la rivière.",
        ),
        "Même sens, phrase réordonnée": (
            "Il a quitté l'entreprise en 2010 pour fonder sa propre société.",
            "En 2010, il a quitté l'entreprise afin de créer sa propre société.",
        ),
        "Rôles inversés": (
            "Marie a présenté Paul à Jean.",
            "Paul a présenté Marie à Jean.",
        ),
    },
    "English": {
        "Reformulation simple": (
            "The city is located north of the river.",
            "The city lies to the north of the river.",
        ),
        "Même sens, phrase réordonnée": (
            "He left the company in 2010 to start his own business.",
            "In 2010, he left the company in order to start his own business.",
        ),
        "Rôles inversés": (
            "The dog chased the cat across the garden.",
            "The cat chased the dog across the garden.",
        ),
    },
    "Español": {
        "Reformulation simple": (
            "La ciudad está situada al norte del río.",
            "La ciudad se encuentra al norte del río.",
        ),
        "Même sens, phrase réordonnée": (
            "Dejó la empresa en 2010 para fundar su propia compañía.",
            "En 2010, dejó la empresa para crear su propia compañía.",
        ),
        "Rôles inversés": (
            "María presentó a Pablo a Juan.",
            "Pablo presentó a María a Juan.",
        ),
    },
    "Deutsch": {
        "Reformulation simple": (
            "Die Stadt liegt nördlich des Flusses.",
            "Die Stadt befindet sich nördlich des Flusses.",
        ),
        "Même sens, phrase réordonnée": (
            "Er verließ das Unternehmen 2010, um seine eigene Firma zu gründen.",
            "2010 verließ er das Unternehmen, um eine eigene Firma zu gründen.",
        ),
        "Rôles inversés": (
            "Der Hund jagte die Katze durch den Garten.",
            "Die Katze jagte den Hund durch den Garten.",
        ),
    },
    "Mixte (2 langues)": {
        "Même sens, FR / EN": (
            "La ville est située au nord de la rivière.",
            "The city is located north of the river.",
        ),
        "Même sens, ES / DE": (
            "La ciudad está situada al norte del río.",
            "Die Stadt liegt nördlich des Flusses.",
        ),
        "Rôles inversés, FR / ES": (
            "Marie a présenté Paul à Jean.",
            "Pablo presentó a María a Juan.",
        ),
        "Rôles inversés, EN / DE": (
            "The dog chased the cat across the garden.",
            "Die Katze jagte den Hund durch den Garten.",
        ),
    },
}

st.set_page_config(page_title="PAWS-X - paraphrases", layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 1000px; padding-top: 2.5rem;}
    h1 {font-size: 1.7rem !important; font-weight: 600 !important; margin-bottom: 0 !important;}
    .sous-titre {color: #6b7280; margin-bottom: 1.5rem; font-size: 0.95rem;}
    .res {border: 1px solid rgba(128,128,128,0.35); border-left-width: 4px; border-radius: 4px;
          padding: 0.8rem 1rem; margin-bottom: 0.8rem;}
    .res.oui {border-left-color: #2f855a;}
    .res.non {border-left-color: #c53030;}
    .res .nom {font-size: 0.8rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.04em;}
    .res .verdict {font-size: 1.15rem; font-weight: 600; margin: 0.15rem 0 0.5rem;}
    .barre {height: 6px; background: rgba(128,128,128,0.25); border-radius: 3px; overflow: hidden;}
    .barre div {height: 100%; background: currentColor; opacity: 0.6;}
    .res .details {font-size: 0.82rem; color: #6b7280; margin-top: 0.4rem;}
    .res .hors {text-transform: none; letter-spacing: 0; margin-left: 0.5rem; padding: 0 0.4rem;
                border: 1px solid #b7791f; border-radius: 3px; color: #b7791f;}
    </style>
    """,
    unsafe_allow_html=True,
)


def find_model(folder):
    for base in ("models", "."):
        path = os.path.join(base, folder)
        if os.path.isfile(os.path.join(path, "config.json")):
            return path
    return None


@st.cache_resource
def etat():
    # partagé entre les reruns : passe à "cpu" si le GPU plante
    return {"device": DEVICE}


@st.cache_resource(show_spinner="Chargement du modèle...")
def load_model(path, device):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
    model.eval()
    return tok, model


@torch.no_grad()
def predict(tok, model, s1, s2):
    t0 = time.time()
    dev = next(model.parameters()).device
    enc = tok(s1, s2, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(dev)
    probs = torch.softmax(model(**enc).logits, dim=-1)[0]
    p = probs[1].item()  # proba de la classe 1 = paraphrase
    return p, (time.time() - t0) * 1000


def run_model(path, s1, s2):
    # après une mise en veille, le contexte CUDA peut être perdu
    # (CUDA error: unknown error) : on bascule alors sur le CPU
    try:
        tok, model = load_model(path, etat()["device"])
        return predict(tok, model, s1, s2)
    except Exception as e:
        if etat()["device"] == "cpu" or "CUDA" not in str(e):
            raise
        etat()["device"] = "cpu"
        st.warning("Le GPU ne répond plus (souvent après une mise en veille) : "
                   "les modèles sont rechargés sur le CPU. Redémarre l'application pour réutiliser le GPU.")
        tok, model = load_model(path, "cpu")
        return predict(tok, model, s1, s2)


def show_result(name, p, seuil, ms, hors_langue=False):
    ok = p >= seuil
    verdict = "Paraphrase" if ok else "Pas une paraphrase"
    tag = ' <span class="hors">hors langue d\'entraînement</span>' if hors_langue else ""
    st.markdown(
        f"""
        <div class="res {'oui' if ok else 'non'}">
          <div class="nom">{name}{tag}</div>
          <div class="verdict">{verdict}</div>
          <div class="barre"><div style="width:{p * 100:.1f}%"></div></div>
          <div class="details">P(paraphrase) = {p:.3f} &nbsp;|&nbsp; seuil {seuil:.2f} &nbsp;|&nbsp; {ms:.0f} ms</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- chargement de la liste des modèles dispo ---
available = {}
for name, (folder, _) in MODELS.items():
    path = find_model(folder)
    if path:
        available[name] = path

st.title("Détection de paraphrases")
st.markdown(
    '<div class="sous-titre">Modèles fine-tunés sur PAWS-X. '
    "On donne deux phrases, le modèle prédit si elles veulent dire la même chose.</div>",
    unsafe_allow_html=True,
)

if not available:
    st.error("Pas de modèle trouvé dans models/. Il faut d'abord lancer le notebook pour les entraîner.")
    st.stop()

with st.sidebar:
    st.subheader("Réglages")
    langue = st.selectbox("Langue des phrases", list(EXAMPLES))
    francais = langue == "Français"
    # hors français, on propose le modèle multilingue par défaut
    noms = list(available)
    defaut = 0 if francais or "XLM-R multilingue" not in noms else noms.index("XLM-R multilingue")
    model_name = st.selectbox("Modèle", noms, index=defaut, key=f"modele_{langue}")
    st.caption(MODELS[model_name][1])
    seuil = st.slider("Seuil de décision", 0.05, 0.95, 0.50, 0.05)
    compare = st.checkbox("Comparer les modèles", value=False)
    st.caption(f"Exécution sur : {etat()['device']}")

if not francais and not compare and model_name in FR_ONLY:
    st.warning(f"{model_name} n'a été entraîné qu'en français : ses réponses en "
               f"« {langue} » ne sont pas fiables. Préfère XLM-R multilingue.")

examples = EXAMPLES[langue]
example = st.selectbox("Charger un exemple", ["-"] + list(examples), key=f"ex_{langue}")
d1, d2 = examples.get(example, ("", ""))

c1, c2 = st.columns(2)
s1 = c1.text_area("Phrase 1", value=d1, height=110, key=f"s1_{langue}_{example}")
s2 = c2.text_area("Phrase 2", value=d2, height=110, key=f"s2_{langue}_{example}")

if st.button("Analyser"):
    s1, s2 = s1.strip(), s2.strip()
    if not s1 or not s2:
        st.warning("Il faut remplir les deux phrases.")
    else:
        names = list(available) if compare else [model_name]
        rows = []
        for name in names:
            p, ms = run_model(available[name], s1, s2)
            hors = not francais and name in FR_ONLY
            show_result(name, p, seuil, ms, hors_langue=hors)
            rows.append({"Modèle": name, "Langue": langue, "P(paraphrase)": round(p, 3),
                         "Décision": "paraphrase" if p >= seuil else "non",
                         "Fiable": "non (modèle FR)" if hors else "oui"})

        if compare:
            st.markdown("**Récapitulatif**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.markdown("---")
st.caption(
    "Les paires de PAWS-X partagent presque tous leurs mots : changer l'ordre ou les rôles peut "
    "changer le sens. Les modèles FR ne sont fiables qu'en français, le modèle multilingue gère aussi EN/ES/DE."
)
