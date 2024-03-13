from ast import literal_eval
from functools import partial
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

SHEET_URL = "https://docs.google.com/spreadsheets/d/15WIzk2aV3vCQLnDihdnNCLxMbDmJZiZKmuiM_xRKbwk/edit#gid=282554525"


def fetch_metadata(url: str) -> pd.DataFrame:
    """Loads metadata from Google Sheets url."""
    url = url.replace("/edit#gid=", "/export?format=csv&gid=")
    metadata = pd.read_csv(url)
    metadata.skal_fjernes = metadata.skal_fjernes == "True"
    return metadata


def find_work(work_id: str, md: pd.DataFrame) -> str:
    md = md.dropna(subset=["work", "document_id"])
    md = md[md["document_id"].str.contains(work_id)]
    return md["work"].iloc[0]


def wrap_text(text: str) -> str:
    if len(text) > 10:
        text = "<br>".join(text.split())
    return "<b>" + text


print("Producing Patterns heatmap.")
data = pd.read_csv("results/upos_patterns.csv", index_col=0)
md = fetch_metadata(SHEET_URL)
data.columns = [find_work(work_id, md) for work_id in data.columns]
rel_freq = data.applymap(lambda elem: literal_eval(elem)[2])
counts = data.applymap(lambda elem: literal_eval(elem)[1])
data = data.applymap(lambda elem: literal_eval(elem)[0])
data = data.applymap(wrap_text)
data = data + "<br> [" + counts.applymap(str) + "]"
trace = go.Heatmap(
    z=rel_freq,
    text=data,
    texttemplate="%{text}",
    textfont=dict(size=14),
    x=data.columns,
    y=data.index,
    colorbar=dict(title="Relative Frequency"),
)
fig = go.Figure(
    data=trace,
)
fig = fig.update_layout(
    width=1000,
    height=1400,
)
fig = fig.update_yaxes(autorange="reversed")
out_path = Path("docs/_static/upos_patterns.html")
out_path.parent.mkdir(exist_ok=True, parents=True)
fig.write_html(out_path)

print("Producing UPOS frequency visualizations.")
data = pd.read_csv("results/upos_tags.csv", index_col=0)
data["work_name"] = data["work_id"].map(partial(find_work, md=md))
data = data.set_index(["work_name", "fable_name"]).drop(columns=["work_id"])
freq = data.to_numpy()
rel_freq = pd.DataFrame(
    (freq.T / freq.sum(axis=1)).T, columns=data.columns, index=data.index
)
# Meaning words
fig = px.scatter_matrix(
    rel_freq.reset_index(),
    dimensions=["noun", "adj", "verb"],
    hover_name="fable_name",
    color="work_name",
)
out_path = Path("docs/_static/upos_scatter_matrix.html")
fig.write_html(out_path)

# Function words
fig = px.scatter_matrix(
    rel_freq.reset_index(),
    dimensions=set(rel_freq.columns) - set(["noun", "adj", "verb", "propn", "num"]),
    hover_name="fable_name",
    color="work_name",
)
out_path = Path("docs/_static/upos_scatter_matrix_function.html")
fig.write_html(out_path)
