import os
import numpy as np
import pandas as pd
import panphon
from sklearn.metrics.pairwise import cosine_distances
from nltk.corpus import cmudict
import nltk

txt_file = "/Users/vant7e/Documents/RRI/rsa_analysis/Tiana_PicNaming_MEG_exps_202301_USETHIS/Overt_naming/lists/overt.txt"
output_csv = f"rdm_cmu_overt.csv"
output_npy = f"rdm_cmu_overt.npy"

cmu = cmudict.dict()
ft = panphon.FeatureTable()

def extract_labels(txt_path):
    with open(txt_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    return sorted(set(os.path.splitext(os.path.basename(p))[0] for p in lines))

stimuli = extract_labels(txt_file)

# === ARPAbet → IPA===
arpabet2ipa = {
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 'AY': 'aɪ',
    'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ',
    'F': 'f', 'G': 'ɡ', 'HH': 'h', 'IH': 'ɪ', 'IY': 'i', 'JH': 'dʒ', 'K': 'k',
    'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'OW': 'oʊ', 'OY': 'ɔɪ',
    'P': 'p', 'R': 'ɹ', 'S': 's', 'SH': 'ʃ', 'T': 't', 'TH': 'θ', 'UH': 'ʊ',
    'UW': 'u', 'V': 'v', 'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'
}
# === Manual adding - Treasure/beanie/seahorse that can not be found in APPAbet ===
cmu["beanie"] = [["B", "IY1", "N", "IY0"]]
cmu["treature"] = [["T", "R", "EH1", "ZH", "ER0"]]
cmu["seahorse"] = [["S", "IY1", "HH", "AO1", "R", "S"]]

def arpabet_to_ipa(arpabet_phonemes):
    ipa_seq = []
    for phone in arpabet_phonemes:
        phone = ''.join(c for c in phone if not c.isdigit())
        ipa_seq.append(arpabet2ipa.get(phone, ''))  # empty string if not found
    return ''.join(ipa_seq)

# === RDM ===
vectors = []
valid_labels = []

for word in stimuli:
    word = word.lower()
    if word not in cmu:
        print(f"[!] Word not found in CMU: {word}")
        continue
    arp_seq = cmu[word][0]  # 取第一个发音
    ipa = arpabet_to_ipa(arp_seq)
    ipa_feats = ft.word_to_vector_list(ipa, numeric=True)
    if not ipa_feats:
        print(f"[!] No features extracted for: {word} (IPA: {ipa})")
        continue
    mean_vector = np.mean(np.array(ipa_feats), axis=0)
    vectors.append(mean_vector)
    valid_labels.append(word)

if not vectors:
    raise RuntimeError("❌ No valid phonological vectors extracted. Check CMU/IPA mapping.")

vectors = np.stack(vectors)
rdm = cosine_distances(vectors)

# === Save ===
pd.DataFrame(rdm, index=valid_labels, columns=valid_labels).to_csv(output_csv)
np.save(output_npy, rdm)

print(f"✅ Saved phonological RDM to: {output_csv}")
print(f"✅ Saved vector matrix to: {output_npy}")
