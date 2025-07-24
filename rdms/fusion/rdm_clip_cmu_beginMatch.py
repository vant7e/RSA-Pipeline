import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import StandardScaler
import os

# === 路径 ===
clip_path = "/Users/vant7e/Documents/RRI/rsa_analysis/output_rdm/beginMatch/semantic/semantic_rdm_clip_target.npy"
phon_path = "/Users/vant7e/Documents/RRI/rsa_analysis/output_rdm/beginMatch/phonological/phonological_rdm_panphon_target.npy"
txt_path = "/Users/vant7e/Documents/RRI/rsa_analysis/Tiana_PicNaming_MEG_exps_202301_USETHIS/Covert_pic_naming_begin_sound/lists/list1/target.txt"
output_path = "/Users/vant7e/Documents/RRI/rsa_analysis/output_rdm/beginMatch/fusion/fused_rdm_clip_panphon_target.csv"

# === 加载特征 ===
clip_feat = np.load(clip_path)       # shape: (n_items, dim_clip)
phon_feat = np.load(phon_path)       # shape: (n_items, dim_phon)

# === 获取标签顺序 ===
def extract_labels(txt_path):
    with open(txt_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    return sorted(set(os.path.splitext(os.path.basename(x))[0] for x in lines))

labels = extract_labels(txt_path)
assert len(labels) == clip_feat.shape[0], "Mismatch between labels and features"

# === 标准化特征 ===
clip_feat = StandardScaler().fit_transform(clip_feat)
phon_feat = StandardScaler().fit_transform(phon_feat)

# === 拼接特征 ===
fused_feat = np.concatenate([clip_feat, phon_feat], axis=1)  # shape: (n_items, dim_clip + dim_phon)

# === 计算 RDM ===
fused_rdm = cosine_distances(fused_feat)

# === 保存为 CSV ===
rdm_df = pd.DataFrame(fused_rdm, index=labels, columns=labels)
rdm_df.to_csv(output_path)
print(f"✅ Saved fused RDM to: {output_path}")