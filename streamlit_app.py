import warnings

import spacy
import streamlit as st
from spacy import displacy
import en_coref_md

warnings.filterwarnings("ignore", category=UserWarning)

# variables
DEFAULT_TEXT = "Netflix reported its first subscriber loss in more than a decade , dropping 200,00 subscribers in the first quarter  of 2022 , followed by a 35 percent stock price drop. It had also underperformed expectations falling short of analyst expectations by 5%"
ALL_NER_TYPES = [
    "CARDINAL",
    "DATE",
    "EVENT",
    "FAC",
    "GPE",
    "LANGUAGE",
    "LAW",
    "LOC",
    "MONEY",
    "NORP",
    "ORDINAL",
    "ORG",
    "PERCENT",
    "PERSON",
    "PRODUCT",
    "QUANTITY",
    "TIME",
    "WORK_OF_ART",
]
MODELS = ["en_core_web_md"]

input_text = st.text_area(
    "Input Sentence or Sentences seperated by fullstop", value=DEFAULT_TEXT, height=180
)
model = st.sidebar.selectbox("Model", MODELS)
nlp = spacy.load(model)
neuralcoref = en_coref_md.load()
with st.sidebar:
    quantifiable_types = st.multiselect(
        "NER Types",
        ALL_NER_TYPES,
        ["ORG"],
    )


def resolve_conference(input_text, neuralcoref=neuralcoref):
    test_sent = input_text

    doc = neuralcoref(test_sent)

    resolved_doc = doc._.coref_resolved
    return resolved_doc


def predict_with_awesome_ml_model(
    contents, ner_types=None, keywords=None, style="ent"
):
    displacy_options = {}
    displacy_options["ents"] = ner_types or []

    html = ""
    doc = nlp(contents)

    # loop sentences
    for sentence in doc.sents:

        # check for key entities
        tmp = nlp(str(sentence))
        html += displacy.render(tmp, style=style, options=displacy_options) + "<br>"

    return html


def show_ent(input_text):
    match_res = predict_with_awesome_ml_model(input_text, ner_types=quantifiable_types)
    return st.write(match_res, unsafe_allow_html=True)


resolved_input_text = resolve_conference(input_text, neuralcoref)

col1, col2 = st.columns(2)

with col1:
    st.header("With Conference Resolution")
    st.write(show_ent(resolved_input_text))

with col2:
    st.header("Without Conference Resolution")
    st.write(show_ent(input_text))
