import os
import gdown
import pickle
import faiss

ASSETS = {
    "df_chunks.pkl": "https://drive.google.com/uc?id=1K80EwD-wQ1ttKcNEAv8ynULOIVMBZiA1",
    "faiss.index": "https://drive.google.com/uc?id=1E52rDQ3k1FtqY-Y_7TWDkec6yMjK4zUL"
}

ASSET_DIR = "assets"

def load_assets():
    os.makedirs(ASSET_DIR, exist_ok=True)

    for filename, url in ASSETS.items():
        path = os.path.join(ASSET_DIR, filename)
        if not os.path.exists(path):
            print(f"Downloading {filename}...")
            gdown.download(url, path, quiet=False)

    with open(os.path.join(ASSET_DIR, "df_chunks.pkl"), "rb") as f:
        df_chunks = pickle.load(f)

    index = faiss.read_index(os.path.join(ASSET_DIR, "faiss.index"))

    return df_chunks, index
