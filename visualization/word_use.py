from pathlib import Path

import pandas as pd
import plotly.express as px

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

dat_path = Path("/work/gospel-ancient-greek/fables-ancient-greek/data/results")

data = pd.read_csv(dat_path.joinpath("word_use.csv"))

# md = fetch_metadata(SHEET_URL)
# data["work"] = data["work_id"].map(lambda id: find_work(id, md))
data["group"] = (
    "<b>"
    + data["work"]
    + "</b>"
    + " <br> top c-tf-idf: <i>"
    + data["top_ctf-idf"]
    + "</i> <br> top freq: <i>"
    + data["top_frequency_in_work"]
)

out_path = Path("docs/_static/word_use.html")
out_path.parent.mkdir(exist_ok=True, parents=True)

fig = px.scatter(
    data,
    x="x_umap",
    y="y_umap",
    color="group",
    # text="fable",
    hover_data={
        "top_tf-idf": True,
        "top_frequency": True,
        "x_umap": False,
        "y_umap": False,
        "group": False,
        "fable_name": False,
    },
    hover_name="fable_name",
    template="plotly_white",
    height=1200,
    width=1000,
)
fig.update_layout(
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
)
fig.update_traces(marker=dict(size=14, line=dict(width=2, color="black")))
fig.write_html(out_path)
