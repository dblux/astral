import requests
import time
import pprint

import pandas as pd

def get_uniprot_isoforms(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    r = requests.get(url)
    if r.status_code != 200:
        print(f"[{uniprot_id}] Failed with status {r.status_code}")
        return None
    data = r.json()
    result = {}
    # Canonical sequence
    result["canonical"] = {
        "sequence": data["sequence"]["value"],
        "length": data["sequence"]["length"]
    }
    # Isoforms
    isoforms = []
    for comment in data.get("comments", []):
        if comment.get("commentType") == "ALTERNATIVE PRODUCTS":
            for iso in comment.get("isoforms", []):
                entry = {
                    "isoform_id": iso.get("isoformIds", [""])[0],
                    "name": iso.get("name", ""),
                    "seq_ids": iso.get("sequenceIds", []),
                }
                isoforms.append(entry)
    result["isoforms"] = isoforms
    return result


uid1 = 'P01602'
uid2 = 'P04150'

result = get_uniprot_isoforms(uid2)

pprint.pprint(result)


filepath = 'tmp/astral/lyriks402/biomarkers/biomarkers-ancova.csv'
bm_ancova = pd.read_csv(filepath, index_col=0)

filepath = 'tmp/astral/lyriks402/biomarkers/biomarkers-elasticnet-nestedkfold.csv'
bm_enet = pd.read_csv(filepath, index_col=0)
bm_enet.head()

filepath = 'tmp/astral/lyriks402/biomarkers/mongan-etable5.csv'
mongan = pd.read_csv(filepath, index_col=0)
monganq = mongan[mongan.q < 0.05]

for uid, gene in zip(bm_ancova.index, bm_ancova.Gene):
    result = get_uniprot_isoforms(uid)
    isoforms = result.get('isoforms', [])
    if isoforms:
        print(uid, gene)
        # pprint.pprint(isoforms)
        print()

for uid, gene in zip(bm_enet.index, bm_enet.Gene):
    result = get_uniprot_isoforms(uid)
    isoforms = result.get('isoforms', [])
    if isoforms:
        print(uid, gene)
        # pprint.pprint(isoforms)
        print()

for uid, gene in zip(monganq.index, monganq['Protein name']):
    result = get_uniprot_isoforms(uid)
    isoforms = result.get('isoforms', [])
    if isoforms:
        print(uid, gene)
        pprint.pprint(isoforms)
        print()

# Check metadata
filepath = 'data/astral/metadata/metadata-all.csv'
metadata = pd.read_csv(filepath, index_col=0)
metadata.Study.value_counts()

filepath = 'data/astral/raw/reprocessed-data.csv'
reprocessed = pd.read_csv(filepath, index_col=0)
reprocessed.head()
reprocessed.shape

for id in reprocessed.columns:
    print(id)

# 645 samples (including 5 QC samples): LYRIKS (402), SCZ (196), BP (41)

file = 'data/astral/processed/lyriks402-processed.csv'

