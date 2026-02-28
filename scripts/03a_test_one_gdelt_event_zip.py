import io
import zipfile
from pathlib import Path
import pandas as pd

def find_project_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(15):
        if (p / "data").exists() and (p / "scripts").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    raise RuntimeError("Could not locate project root (expected folders: data/ and scripts/).")

def main():
    script_path = Path(__file__).resolve()
    base_dir = find_project_root(script_path.parent)

    print("Script path:", script_path)
    print("Project root:", base_dir)

    # Search for either the zip OR the extracted CSV
    gdelt_dir = base_dir / "data" / "raw" / "gdelt" / "manual_test"
    if not gdelt_dir.exists():
        raise FileNotFoundError(f"Missing folder: {gdelt_dir}")

    zips = list(gdelt_dir.rglob("20260222.export.CSV.zip"))
    csvs = list(gdelt_dir.rglob("20260222.export.CSV"))

    print("Found ZIPs:", [str(p) for p in zips])
    print("Found CSVs:", [str(p) for p in csvs])

    if zips:
        zip_path = zips[0]
        print("Using ZIP:", zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            inner = z.namelist()[0]
            raw = z.read(inner)
        df = pd.read_csv(io.BytesIO(raw), sep="\t", header=None, dtype=str, low_memory=False)

    elif csvs:
        csv_path = csvs[0]
        print("Using CSV:", csv_path)
        df = pd.read_csv(csv_path, sep="\t", header=None, dtype=str, low_memory=False)

    else:
        raise FileNotFoundError("Could not find 20260222.export.CSV.zip or 20260222.export.CSV under manual_test/")

    print("Loaded rows:", len(df))
    print("Loaded columns:", df.shape[1])
    print("First 3 rows (first 10 cols):")
    print(df.iloc[:3, :10])

    if df.shape[1] == 58:
        print("OK: 58 columns detected.")
    else:
        print(f"WARNING: expected 58 columns, got {df.shape[1]}.")

if __name__ == "__main__":
    main()