import glob
from pathlib import Path

import pandas as pd
import spacy
from spacy.tokens import Doc, Token
from tqdm import tqdm

nlp = spacy.load("grc_odycy_joint_trf")


def load_works(dat_path) -> list[dict]:
    works = list(dat_path.rglob("*.spacy"))
    records = []
    for work in tqdm(works):
        fable_name = work.stem
        doc = Doc(nlp.vocab).from_disk(work)
        work = str(work.parent).split("/")[-1]
        records.append(dict(fable_name=fable_name, doc=doc, work = work))
    return records


def get_token_features(token: Token) -> dict:
    return dict(
        token=token.orth_,
        lemma=token.lemma_,
        norm=token.norm_,
        is_stop=token.is_stop,
        upos_tag=token.pos_,
        fine_grained_tag=token.tag_,
        dependency_relation=token.dep_,
    )

dat_path = Path("/work/gospel-ancient-greek/fables-ancient-greek/data")

out_path = dat_path.joinpath("annotations/")
out_path.mkdir(exist_ok=True, parents=True)

print("Loading data.")
fables = load_works(dat_path)

print("Exporting fable annotations as csv.")
for fable in fables:
    fable_name = fable["fable_name"]
    doc = fable["doc"]
    out_file = out_path.joinpath(f"{fable_name}.csv")
    entries = [get_token_features(token) for token in doc]
    table = pd.DataFrame.from_records(entries)
    table.to_csv(out_file)

print("Done.")
