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
    raise RuntimeError("Could not locate project root.")

def main():
    base = find_project_root(Path(__file__).resolve().parent)
    csv_path = base / "data" / "raw" / "gdelt" / "manual_test" / "20260222.export.CSV" / "20260222.export.CSV"

    print("Project root:", base)
    print("CSV path:", csv_path)
    print("Exists:", csv_path.exists())

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing extracted CSV: {csv_path}")

    df = pd.read_csv(csv_path, sep="\t", header=None, dtype=str, low_memory=False)

    print("Loaded rows:", len(df))
    print("Loaded columns:", df.shape[1])

    if df.shape[1] == 58:
        print("OK: 58 columns detected.")
    else:
        print(f"WARNING: expected 58 columns, got {df.shape[1]}.")

    print(df.iloc[:3, :10])

if __name__ == "__main__":
    main()