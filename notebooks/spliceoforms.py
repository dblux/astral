import json
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


def get_exon_coordinates(uniprot_id, tax_id=None):
    """
    Retrieves exon genomic coordinates for all splice forms of the given UniProt protein ID.
    Parameters:
      - uniprot_id (str): UniProt accession (e.g., 'P12345')
      - tax_id (str, optional): Taxonomy ID to restrict the search (e.g., '9606' for human)
    """
    base_url = "https://www.ebi.ac.uk/proteins/api/coordinates"
    params = {"accession": uniprot_id}
    if tax_id:
        params["taxid"] = tax_id
    response = requests.get(
        base_url, params=params,
        headers={"Accept": "application/json"}
    )
    response.raise_for_status()
    data = response.json()
    assert isinstance(data, list) and len(data) == 1
    entry = data[0]
    # print(entry.keys())
    acc = entry.get("accession")
    # print(entry.get("gene"))
    sequence = entry.get("sequence")
    gene_coords = entry.get("gnCoordinate", [])
    assert isinstance(gene_coords, list) and len(gene_coords) == 1
    trans = gene_coords[0]
    # pprint.pprint(trans)
    # print(trans.keys())
    transcript_id = trans.get("ensemblTranscriptId")
    g = trans.get("genomicLocation", {})
    # chrom = g.get("chromosome")
    exons = g.get("exon", [])
    result = {}
    result["accession"] = acc
    result["sequence"] = sequence
    exon_info = [] 
    for i, exon in enumerate(exons):
        protein_loc = exon.get("proteinLocation", {})
        exon_info.append({
            "exon": i + 1,
            "exon_id": exon.get("id"),
            "start": protein_loc.get("begin").get("position"),
            "end": protein_loc.get("end").get("position"),
        })
    result["exons"] = exon_info 
    return result


itih1_uid = 'P19827'
itih1_1_uid = 'P19827-1'
itih1_3_uid = 'P19827-3'

result = get_uniprot_isoforms(itih1_uid)
pprint.pprint(result)

# Retrieve sequence and exon coordinates for ITIH1-1
result_itih1_1 = get_exon_coordinates(itih1_1_uid)
result_itih1_3 = get_exon_coordinates(itih1_3_uid)

# Save to JSON
filepath = "data/tmp/spliceoforms/itih1-1.json"
with open(filepath, 'w') as f:
    json.dump(result_itih1_1, f, indent=2)

filepath = "data/tmp/spliceoforms/itih1-3.json"
with open(filepath, 'w') as f:
    json.dump(result_itih1_3, f, indent=2)


# Spliceoforms: Retrieve isoforms and exon coordinates from UniProt
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

### Plot peptides ###

filepath = 'data/astral/raw/report.pr_matrix.csv'
pr_matrix = pd.read_csv(filepath, index_col=0)

