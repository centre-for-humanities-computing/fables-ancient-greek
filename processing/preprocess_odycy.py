from glob import glob
from pathlib import Path
from typing import Iterable

import spacy
from tqdm import tqdm

nlp = spacy.load("grc_odycy_joint_trf")


dat_path = Path("/work/gospel-ancient-greek/fables-ancient-greek/data/")
out_path = dat_path.joinpath("spacy_objects/")

files = list(dat_path.rglob("*.txt"))

for file in tqdm(files, desc="Going through all fables."):
    file_id = file.stem
    with open(file) as in_file:
        work = str(file.parent).split("/")[-1]
        content = in_file.read()

        out_file_path = out_path.joinpath(work)
        out_file_path.mkdir(exist_ok=True, parents=True)
        
        if not out_file_path.is_file():
            doc = nlp(content)
            doc.to_disk(out_file_path.joinpath(f"{file_id}.spacy"))
