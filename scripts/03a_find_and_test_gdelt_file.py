import io
import zipfile
from pathlib import Path
import pandas as pd

def find_project_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(20):
        if (p / "data").exists() and (p / "scripts").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    raise RuntimeError("Could not locate project root (expected data/ and scripts/).")

def main():
    base = find_project_root(Path(__file__).resolve().parent)
    print("Project root:", base)

    # Search anywhere under the project for the ZIP or extracted CSV
    zips = list(base.rglob("20260222.export.CSV.zip"))
    csvs = list(base.rglob("20260222.export.CSV"))

    print("ZIP matches:", [str(p) for p in zips])
    print("CSV matches:", [str(p) for p in csvs])

    if zips:
        zp = zips[0]
        print("Using ZIP:", zp)
        with zipfile.ZipFile(zp, "r") as z:
            inner = z.namelist()[0]
            raw = z.read(inner)
        df = pd.read_csv(io.BytesIO(raw), sep="\t", header=None, dtype=str, low_memory=False)

    elif csvs:
        # Exclude directories named like the zip base if needed; just take first file match
        cp = next((p for p in csvs if p.is_file()), None)
        if cp is None:
            raise FileNotFoundError("Found CSV name matches but none are files.")
        print("Using CSV:", cp)
        df = pd.read_csv(cp, sep="\t", header=None, dtype=str, low_memory=False)

    else:
        raise FileNotFoundError(
            "Could not find 20260222.export.CSV.zip or 20260222.export.CSV anywhere under the project.\n"
            "This strongly suggests the files are not actually inside this project folder from Python's perspective."
        )

    print("Loaded rows:", len(df))
    print("Loaded columns:", df.shape[1])
    if df.shape[1] == 58:
        print("OK: 58 columns detected.")
    else:
        print(f"WARNING: expected 58 columns, got {df.shape[1]}.")

    print("First 2 rows (first 10 cols):")
    print(df.iloc[:2, :10])

if __name__ == "__main__":
    main()