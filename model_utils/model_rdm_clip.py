import os
import pandas as pd
import numpy as np
import torch
import clip
from sklearn.metrics.pairwise import cosine_distances

device = "cuda" if torch.cuda.is_available() else "cpu"
txt_file = "/Users/vant7e/Documents/RRI/rsa_analysis/Tiana_PicNaming_MEG_exps_202301_USETHIS/Overt_naming/lists/overt.txt"
output_csv = f"rdm_clip_overt.csv"
output_npy = f"rdm_clip_overt.npy"

def extract_labels_from_txt(txt_path):
    with open(txt_path, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]
    labels = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
    return labels

stimuli = extract_labels_from_txt(txt_file)
unique_labels = sorted(set(stimuli))  # remove duplicates if any

# === CLIP Loading ===
model, preprocess = clip.load("ViT-B/32", device=device)

# === Text Encoding Only, no visual layer===
with torch.no_grad():
    text_tokens = clip.tokenize(unique_labels).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # L2 normalize

# ===（1 - cosine similarity）===
rdm = cosine_distances(text_features.cpu().numpy())

# === RDM ===
rdm_df = pd.DataFrame(rdm, index=unique_labels, columns=unique_labels)
rdm_df.to_csv(output_csv)
print(f"✅ Saved semantic RDM to: {output_csv}")

# === Save ===
np.save(output_npy, rdm)
print(f"✅ Saved semantic RDM (NPY) to: {output_npy}")
