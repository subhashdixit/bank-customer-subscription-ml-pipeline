"""Download only the selected CSV; never extract arbitrary archive paths."""
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"


def download(output=Path("data/bank-full.csv")):
    with urlopen(URL, timeout=120) as response:
        archive = response.read()
    with ZipFile(io.BytesIO(archive)) as outer:
        # UCI distributes the original bank CSV inside bank.zip.
        nested = next(name for name in outer.namelist() if name.endswith("bank.zip"))
        with ZipFile(io.BytesIO(outer.read(nested))) as inner:
            member = next(name for name in inner.namelist() if name.endswith("bank-full.csv"))
            content = inner.read(member)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)
    metadata = {"source": URL, "member": member, "sha256": hashlib.sha256(content).hexdigest(),
                "license": "CC BY 4.0", "dataset": "UCI Bank Marketing (222)"}
    output.with_suffix(".source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {output}; provenance recorded beside it.")


if __name__ == "__main__":
    download()
