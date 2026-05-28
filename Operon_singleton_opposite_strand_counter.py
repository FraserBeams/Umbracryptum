import os
import glob
import subprocess
import pandas as pd
from collections import defaultdict
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Set directories and set definition of your gene cluster

input_dir = "Genbank files for genomes of interest (.gbk or .gbff)"
hmm_file = "HMM file of genes of interest - needs at least two concatenated .hmm files"
outdir = "Your output directory"

MAX_GAP = 20000
MAX_BRIDGING_CDS = 8
EVALUE_CUTOFF = 1e-5
MIN_HITS = 2
SINGLETON_NEARBY_BP = 20000

os.makedirs(outdir, exist_ok=True)

# Search directory for .gbk or .gbff files

genbank_files = glob.glob(os.path.join(input_dir, "*.gbk")) + \
                glob.glob(os.path.join(input_dir, "*.gbff"))

if not genbank_files:
    raise RuntimeError(f"No .gbk or .gbff files found in {input_dir}")


# Extract protein sequences from genbank records

protein_records = []
feature_map = {}
cds_index = defaultdict(list)

for gbk in genbank_files:
    genome = os.path.splitext(os.path.basename(gbk))[0]

    for record in SeqIO.parse(gbk, "genbank"):
        record_id = f"{genome}|{record.id}"

        for feature in record.features:
            if feature.type != "CDS":
                continue

            start = int(feature.location.start)
            end = int(feature.location.end)
            strand = feature.location.strand
            locus = feature.qualifiers.get("locus_tag", ["unknown"])[0]
            protein = feature.qualifiers.get("translation", [None])[0]

            cds_index[record_id].append({
                "start": start,
                "end": end,
                "strand": strand,
                "locus": locus
            })

            if protein:
                protein_records.append(
                    SeqRecord(Seq(protein), id=locus, description="")
                )

                feature_map[locus] = {
                    "genome": genome,
                    "record_id": record_id,
                    "start": start,
                    "end": end,
                    "strand": strand,
                    "locus": locus
                }

for record_id in cds_index:
    cds_index[record_id] = sorted(cds_index[record_id], key=lambda x: x["start"])

protein_fasta = os.path.join(outdir, "all_proteins.faa")
SeqIO.write(protein_records, protein_fasta, "fasta")

# Detect genes of interest with hmmsearch

hmm_tbl = os.path.join(outdir, "hmm.tbl")

subprocess.run([
    "hmmsearch",
    "--tblout", hmm_tbl,
    hmm_file,
    protein_fasta
], check=True)

# Parse HMM hits

hits = []

with open(hmm_tbl) as f:
    for line in f:
        if line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        target = parts[0]
        hmm_name = parts[2]
        evalue = float(parts[4])

        if evalue <= EVALUE_CUTOFF and target in feature_map:
            hit = feature_map[target].copy()
            hit["hmm_type"] = hmm_name
            hit["evalue"] = evalue
            hits.append(hit)

df_hits = pd.DataFrame(hits)

# Handle genomes with no hits

summary_rows = []

all_genomes = [
    os.path.splitext(os.path.basename(g))[0]
    for g in genbank_files
]

if df_hits.empty:
    for genome in all_genomes:
        summary_rows.append({
            "genome": genome,
            "number_hits": 0,
            "number_operons": 0,
            "number_singletons": 0,
            "number_opposite_strand_associated_clusters": 0,
            "number_isolated_singletons": 0
        })

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(outdir, "escrt_locus_counts.tsv"),
        sep="\t",
        index=False
    )

    print("No hits found in this genome. Added to summary.")
    exit()

df_hits = df_hits.sort_values(["genome", "record_id", "start"]).reset_index(drop=True)

# Count number of bridging CDS features, find nearby hits on the opposite strand and cluster hits into clusters or seperate loci

def count_bridging_cds(record_id, prev_end, next_start):
    count = 0

    for feat in cds_index.get(record_id, []):
        if feat["start"] >= next_start:
            break
        if feat["end"] <= prev_end:
            continue

        if feat["start"] >= prev_end and feat["end"] <= next_start:
            count += 1

    return count


def find_opposite_strand_partner(hit, genome_hits):
    midpoint = (hit["start"] + hit["end"]) / 2

    candidates = genome_hits[
        (genome_hits["record_id"] == hit["record_id"]) &
        (genome_hits["locus"] != hit["locus"]) &
        (genome_hits["strand"] != hit["strand"])
    ].copy()

    if candidates.empty:
        return None

    candidates["midpoint"] = (candidates["start"] + candidates["end"]) / 2
    candidates["distance"] = (candidates["midpoint"] - midpoint).abs()

    nearby = candidates[candidates["distance"] <= SINGLETON_NEARBY_BP]
    if nearby.empty:
        return None

    return nearby.sort_values("distance").iloc[0]


def cluster_genome_hits(genome_hits):
    loci = []
    current = []

    for _, row in genome_hits.iterrows():
        if not current:
            current.append(row)
            continue

        prev = current[-1]
        gap = row["start"] - prev["end"]

        bridging_cds = count_bridging_cds(
            record_id=row["record_id"],
            prev_end=prev["end"],
            next_start=row["start"]
        )

        can_join = (
            row["record_id"] == prev["record_id"] and
            row["strand"] == prev["strand"] and
            gap >= 0 and
            gap <= MAX_GAP and
            bridging_cds <= MAX_BRIDGING_CDS
        )

        if can_join:
            current.append(row)
        else:
            loci.append(current)
            current = [row]

    if current:
        loci.append(current)

    return loci

# Classify clusters as operons, singletons or opposite strand associated for each genome

for genome in all_genomes:
    genome_hits = df_hits[df_hits["genome"] == genome].copy()

    if genome_hits.empty:
        summary_rows.append({
            "genome": genome,
            "number_hits": 0,
            "number_operons": 0,
            "number_singletons": 0,
            "number_opposite_strand_associated_clusters": 0,
            "number_isolated_singletons": 0
        })
        continue

    loci = cluster_genome_hits(genome_hits)

    operons = [l for l in loci if len(l) >= MIN_HITS]
    singletons = [l for l in loci if len(l) == 1]

    associated_pairs = set()
    associated_singleton_tags = set()

    for singleton in singletons:
        hit = singleton[0]
        partner = find_opposite_strand_partner(hit, genome_hits)

        if partner is not None:
            pair = tuple(sorted([hit["locus"], partner["locus"]]))
            associated_pairs.add(pair)
            associated_singleton_tags.add(hit["locus"])
            associated_singleton_tags.add(partner["locus"])

    isolated_singletons = [
        s for s in singletons
        if s[0]["locus"] not in associated_singleton_tags
    ]

    summary_rows.append({
        "genome": genome,
        "number_hits": len(genome_hits),
        "number_operons": len(operons),
        "number_singletons": len(singletons),
        "number_opposite_strand_associated_clusters": len(associated_pairs),
        "number_isolated_singletons": len(isolated_singletons),
        "hmm_types_detected": ";".join(sorted(genome_hits["hmm_type"].astype(str).unique()))
    })

# Save the summary as a .tsv file

summary_df = pd.DataFrame(summary_rows)

out_tsv = os.path.join(outdir, "escrt_locus_counts.tsv")
summary_df.to_csv(out_tsv, sep="\t", index=False)

print(f"Saved summary table: {out_tsv}")
print("Done")
