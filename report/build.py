"""Execute report/report.py (a jupytext notebook) and export it to HTML with the code hidden.

The notebook reads only report/results/*.json and report/figures/*.png, so this runs without
the dataset (also in the Pages workflow).

    uv run python report/build.py              # -> report/index.html
    uv run python report/build.py --out _site
"""

import argparse
import shutil
from pathlib import Path

import jupytext
import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE))
    out = Path(parser.parse_args().out).resolve()
    nb = jupytext.read(HERE / "report.py")
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    ExecutePreprocessor(timeout=300, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(HERE)}})
    nbformat.write(nb, HERE / "report.ipynb")
    exporter = HTMLExporter(template_name="lab")
    exporter.exclude_input = True
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, _ = exporter.from_notebook_node(nb)
    body = body.replace("<title>Notebook</title>", "<title>Criteo uplift: what the ads did</title>")
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(body)
    if out != HERE:
        shutil.copytree(HERE / "figures", out / "figures", dirs_exist_ok=True)
    print(f"wrote {out / 'index.html'}")


if __name__ == "__main__":
    main()
