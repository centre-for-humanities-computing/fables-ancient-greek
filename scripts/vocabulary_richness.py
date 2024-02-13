import glob
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import spacy
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


def moving_ttr(tokens: list[str], window_size: int = 50) -> list[float]:
    """Calculates moving type-token-ratios for each window in a text."""
    counter = Counter(tokens[:window_size])
    n_types = len(counter)
    ttrs = [n_types / window_size]
    for i in range(len(tokens) - window_size):
        old_word = tokens[i]
        new_word = tokens[i + window_size]
        counter[old_word] -= 1
        if not counter[old_word]:
            del counter[old_word]
        if new_word in counter:
            counter[new_word] += 1
        else:
            counter[new_word] = 1
        n_types = len(counter)
        ttrs.append(n_types / window_size)
    return ttrs


def mattr(tokens: list[str], window_size: int = 50) -> float:
    ttrs = moving_ttr(tokens, window_size)
    return np.mean(ttrs)


def ttr(tokens: list[str]) -> float:
    return len(set(tokens)) / len(tokens)


entries = load_works()
data = pd.DataFrame(entries)
data["tokens"] = data["doc"].map(lambda d: [token.lemma_ for token in d])
data["mattr_10"] = data["tokens"].map(partial(mattr, window_size=10))
data["mattr_50"] = data["tokens"].map(partial(mattr, window_size=50))
data["ttr"] = data["tokens"].map(ttr)

px.scatter(data, y="ttr", x="mattr_10", color="work_id", hover_data=["fable_name"])
