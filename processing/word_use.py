import glob
from collections import Counter
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import scipy.sparse as spr
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import label_binarize
from spacy.tokens import Doc
from tqdm import tqdm
from umap.umap_ import UMAP

nlp = spacy.load("grc_odycy_joint_trf")


def load_works() -> list[dict]:
    works = glob.glob("data/spacy_objects/*")
    works = map(Path, works)
    works = [work for work in works if work.is_dir()]
    records = []
    for work in works:
        work_id = work.stem
        files = glob.glob(str(work.joinpath("*.spacy")))
        files = map(Path, files)
        for file in files:
            fable_name = file.stem
            doc = Doc(nlp.vocab).from_disk(file)
            records.append(dict(work_id=work_id, fable_name=fable_name, doc=doc))
    return records


def soft_ctf_idf(
    doc_topic_matrix: np.ndarray, doc_term_matrix: spr.csr_matrix
) -> np.ndarray:
    eps = np.finfo(float).eps
    term_importance = doc_topic_matrix.T @ doc_term_matrix
    overall_in_topic = np.abs(term_importance).sum(axis=1)
    n_docs = len(doc_topic_matrix)
    tf = (term_importance.T / (overall_in_topic + eps)).T
    idf = np.log(n_docs / (np.abs(term_importance).sum(axis=0) + eps))
    ctf_idf = tf * idf
    return ctf_idf


def get_highest(component, vocab, top_k=10) -> str:
    high = np.argpartition(-component, top_k)[:top_k]
    importance = component[high]
    high = high[np.argsort(-importance)]
    return " ".join(vocab[high])


def top_ctf_idf(labels, doc_term_matrix, vocab, top_k=10) -> dict[str, str]:
    unique_labels = np.unique(labels)
    binary_labels = label_binarize(labels, classes=unique_works)
    ctf_idf = soft_ctf_idf(binary_labels, doc_term_matrix)  # type: ignore
    res = {}
    for label, component in zip(unique_labels, ctf_idf):
        res[label] = get_highest(component, vocab, top_k)
    return res


def top_freq_group(labels, doc_term_matrix, vocab, top_k=10) -> dict[str, str]:
    unique_labels = np.unique(labels)
    res = {}
    for label in unique_labels:
        freq = np.squeeze(np.asarray(doc_term_matrix[labels == label].sum(axis=0)))
        res[label] = get_highest(freq, vocab, top_k)
    return res


def top_words(doc_term_matrix, vocab, top_k=5) -> list[str]:
    res = []
    for bow in doc_term_matrix:
        bow = np.squeeze(np.asarray(bow))
        res.append(get_highest(bow, vocab, top_k))
    return res


out_path = Path("results/stylistic_features.csv")
out_path.parent.mkdir(exist_ok=True)

print("Calculating vocabulary richness.")
data = pd.DataFrame(load_works())

print("Lemmatizing.")
lemmatized_text = data["doc"].map(lambda d: " ".join(tok.lemma_ for tok in d))

print("Counting frequencies.")
vectorizer = CountVectorizer()
dtm = vectorizer.fit_transform(lemmatized_text)
tf_idf = TfidfTransformer().fit_transform(dtm)
vocab = vectorizer.get_feature_names_out()

print("Calculating top words in works and fables.")
top_weighted_per_class = top_ctf_idf(data["work_id"], dtm, vocab)
top_freq_per_class = top_freq_group(data["work_id"], dtm, vocab)
data["top_frequency"] = top_words(dtm, vocab)
data["top_frequency_in_work"] = data["work_id"].map(top_freq_per_class)
data["top_tf-idf"] = top_words(tf_idf, vocab)
data["top_ctf-idf"] = data["work_id"].map(top_weighted_per_class)

print("Calculating positions with UMAP.")
data["x_umap"], data["y_umap"] = UMAP(n_components=2, metric="cosine").fit_transform(
    tf_idf
)

print("Saving results")
res = data.drop(columns=["doc"])
res.to_csv(out_path)

print("Done.")
