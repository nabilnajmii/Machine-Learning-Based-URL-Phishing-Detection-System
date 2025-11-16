import os
import argparse
import pandas as pd

# Find project root (one level up from /scripts)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to raw CSV with URLs + labels")
    p.add_argument("--url-col", default="url", help="Name of the URL column in the CSV")
    p.add_argument("--label-col", default="label", help="Name of the label column in the CSV")
    p.add_argument("--phish-label", default="phishing", help="Value in label that means phishing")
    p.add_argument("--legit-label", default="legitimate", help="Value in label that means legitimate/benign")
    p.add_argument("--max-per-class", type=int, default=5000,
                   help="Max samples per class (to keep training manageable)")
    args = p.parse_args()

    print(f"[+] Loading raw data from: {args.input}")
    df = pd.read_csv(args.input)

    # 1) Keep only URL + label columns
    df = df[[args.url_col, args.label_col]].copy()
    df.rename(columns={args.url_col: "url", args.label_col: "label"}, inplace=True)

    # 2) Basic cleaning
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"].notna() & (df["url"] != "")]
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)

    # 3) Split into phishing and legitimate subsets
    phish_df = df[df["label"].astype(str).str.lower() == args.phish_label.lower()].copy()
    legit_df = df[df["label"].astype(str).str.lower() == args.legit_label.lower()].copy()

    print(f"[+] Found {len(phish_df)} phishing URLs and {len(legit_df)} legitimate URLs before sampling.")

    # 4) Downsample if too many
    if len(phish_df) > args.max_per_class:
        phish_df = phish_df.sample(n=args.max_per_class, random_state=42).reset_index(drop=True)
    if len(legit_df) > args.max_per_class:
        legit_df = legit_df.sample(n=args.max_per_class, random_state=42).reset_index(drop=True)

    print(f"[+] Using {len(phish_df)} phishing and {len(legit_df)} legitimate after sampling.")

    # 5) Save to data/
    phish_path = os.path.join(DATA_DIR, "phishing.csv")
    legit_path = os.path.join(DATA_DIR, "legit.csv")
    phish_df[["url"]].to_csv(phish_path, index=False)
    legit_df[["url"]].to_csv(legit_path, index=False)

    print(f"[+] Saved phishing URLs to: {phish_path}")
    print(f"[+] Saved legitimate URLs to: {legit_path}")

if __name__ == "__main__":
    main()
