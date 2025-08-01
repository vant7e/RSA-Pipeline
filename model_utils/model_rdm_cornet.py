import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from thingsvision import get_extractor
from thingsvision.core.rsa import compute_rdm

# === Configuration ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'cornet_rt'
target_modules = ['V1', 'V2', 'V4', 'IT', 'decoder.flatten']
output_dir = '/Users/vant7e/Documents/RRI/rsa_analysis/output_rdm'
os.makedirs(output_dir, exist_ok=True)

# === Step 1: Define image root and file list
image_root = '/Users/vant7e/Documents/RRI/rsa_analysis/Tiana_PicNaming_MEG_exps_202301_USETHIS/Overt_naming/'
file_list_path = '/Users/vant7e/Documents/RRI/rsa_analysis/Tiana_PicNaming_MEG_exps_202301_USETHIS/Overt_naming/lists/list1_overt.txt'

# === Step 2: Read relative paths and convert to absolute
with open(file_list_path, 'r') as f:
    rel_paths = [line.strip() for line in f if line.strip()]
image_paths = [os.path.join(image_root, p) for p in rel_paths]

# === Step 3: Extract labels based on file names (i.e., image category)
labels = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]

for p in image_paths:
    if not os.path.exists(p):
        print(f"[X] Not found: {p}")

# === Load model extractor ===
extractor = get_extractor(
    model_name=model_name,
    source='custom',
    device=device,
    pretrained=True
)

# === Load transformation function ===
transform = extractor.get_transformations()

# === Load and transform images ===
images = []
valid_labels = []
for path, label in zip(image_paths, labels):
    try:
        img = Image.open(path).convert("RGB")
        images.append(transform(img))
        valid_labels.append(label)
    except Exception as e:
        print(f"[!] Failed to load image: {path} | Error: {e}")

# === Loop through CORnet-RT modules ===
for module in target_modules:
    print(f"\n🔍 Extracting features for module: {module}")
    features_list = []

    for img in images:
        feats = extractor.extract_features(
            batches=[img.unsqueeze(0).to(device)],
            module_name=module,
            flatten_acts=True
        )
        features_list.append(np.squeeze(feats, axis=0))  # shape: (D,)

    features_array = np.stack(features_list)  # shape: (N, D)
    print(f"[✓] Feature matrix shape for {module}: {features_array.shape}")

    # === Save features as .npy
    np.save(os.path.join(output_dir, f"features_{model_name}_{module.replace('.', '_')}.npy"), features_array)

    # === Compute RDM
    rdm = compute_rdm(features_array, method='correlation')
    assert rdm.shape[0] == rdm.shape[1] == len(valid_labels), "[✗] RDM shape mismatch with labels"

    # === Save RDM as CSV
    rdm_df = pd.DataFrame(rdm, index=valid_labels, columns=valid_labels)
    save_path = os.path.join(output_dir, f"model_rdm_{model_name}_{module.replace('.', '_')}.csv")
    rdm_df.to_csv(save_path)
    print(f"✅ Saved RDM to: {save_path}")