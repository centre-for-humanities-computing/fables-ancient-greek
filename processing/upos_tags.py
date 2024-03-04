import glob
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from spacy.tokens import Doc

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


def top_freq_group(
    labels, doc_term_matrix, vocab, top_k=10
) -> dict[str, tuple[str, float]]:
    unique_labels = np.unique(labels)
    res = {}
    for label in unique_labels:
        freq = np.squeeze(np.asarray(doc_term_matrix[labels == label].sum(axis=0)))
        freq = freq / freq.sum()
        high = np.argpartition(-freq, top_k)[:top_k]
        importance = freq[high]
        high = high[np.argsort(-importance)]
        res[label] = list(zip(vocab[high], freq[high]))
    return res


out_path = Path("results/upos_tags.csv")
out_path.parent.mkdir(exist_ok=True)

print("Loading data.")
data = pd.DataFrame(load_works())

print("Extracting UPOS tags.")
upos_docs = data["doc"].map(lambda d: " ".join(tok.pos_ for tok in d))

print("Counting UPOS frequencies.")
vectorizer = CountVectorizer()
upos_vecs = np.asarray(vectorizer.fit_transform(upos_docs).todense())
upos_vocab = vectorizer.get_feature_names_out()
rel_freqs = (upos_vecs.T / upos_vecs.sum(axis=1)).T
# rel_freqs = {}
# for work_id in data["work_id"].unique():
#     freq = upos_vecs[data["work_id"] == work_id].sum(axis=0)
#     freq = np.squeeze(np.asarray(freq))
#     rel_freq = freq / freq.sum()
#     rel_freqs[work_id] = dict(zip(upos_vocab, rel_freq))

print("Saving UPOS frequencies")
rel_freq_df = pd.DataFrame(rel_freqs, columns=upos_vocab)
rel_freq_df["fable_name"] = data["fable_name"]
rel_freq_df["work_id"] = data["work_id"]
rel_freq_df.to_csv(out_path)

print("Collecting UPOS n-grams")
n_gram_vectorizer = CountVectorizer(ngram_range=(2, 4))
upos_ngrams = n_gram_vectorizer.fit_transform(upos_docs)
ngram_vocab = n_gram_vectorizer.get_feature_names_out()

print("Calculating top n-gram patterns.")
top_freq_per_class = top_freq_group(data["work_id"], upos_ngrams, ngram_vocab)

print("Saving results")
out_path = Path("results/upos_patterns.csv")
res = pd.DataFrame(top_freq_per_class)
res.to_csv(out_path)

print("Done.")