"""Read preserved notebook output as data; never execute notebook source."""
import ast
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path


class _Table(HTMLParser):
    def __init__(self):
        super().__init__(); self.rows=[]; self.row=None; self.cell=None
    def handle_starttag(self, tag, attrs):
        if tag == "tr": self.row=[]
        elif tag in ("td", "th") and self.row is not None: self.cell=[]
    def handle_data(self, data):
        if self.cell is not None: self.cell.append(data)
    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append("".join(self.cell).strip()); self.cell=None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row); self.row=None


def text_outputs(cell):
    return "\n".join("".join(o.get("text", [])) for o in cell.get("outputs", []))


def _last_epoch(cell):
    records=[]
    for line in text_outputs(cell).splitlines():
        if line.strip().startswith("{'epoch':"):
            value=ast.literal_eval(line)
            if not isinstance(value, dict): raise ValueError("Invalid epoch record")
            records.append(value)
    if not records: raise ValueError("No saved epoch output found")
    return records[-1]


def _table(cell):
    for output in cell.get("outputs", []):
        html=output.get("data", {}).get("text/html")
        if not html: continue
        parser=_Table(); parser.feed("".join(html))
        if not parser.rows or "sample" not in parser.rows[0]: continue
        headers=parser.rows[0]
        return [{k:v for k,v in zip(headers,row) if k} for row in parser.rows[1:]]
    raise ValueError("No saved sample table found")


def extract(root: Path) -> dict:
    path=root/"results/historical/source_notebook.ipynb"
    nb=json.loads(path.read_text(encoding="utf-8"))
    cells=nb["cells"]
    def find(token):
        matches=[c for c in cells if c["cell_type"]=="code" and token in "".join(c["source"])]
        if len(matches)!=1: raise ValueError(f"Ambiguous or absent cell: {token}")
        return matches[0]
    config_cell=find("SEED = 42")
    config=next(ast.literal_eval(l.partition("config: ")[2]) for l in text_outputs(config_cell).splitlines() if l.startswith("config: "))
    t1=_last_epoch(find("def train_lm("))
    t2=_last_epoch(find("def train_harm("))
    table1=_table(find("# ---- baselines: unigram, bigram Markov, untrained ----"))
    table2=_table(find("def lookup_test_acc("))
    return {"source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "status":"extracted historical outputs; not rerun inference",
            "config":config,"task1_final_logged_epoch":t1,"task2_final_logged_epoch":t2,
            "task1_displayed_table":table1,"task2_displayed_table":table2,
            "notes":["Displayed tables are rounded and stored as strings.",
                     "Execution counters are cleared; outputs remain saved.",
                     "No original checkpoint or per-token prediction log is included."]}
