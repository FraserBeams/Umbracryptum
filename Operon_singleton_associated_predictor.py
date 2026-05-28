import os
import glob
import subprocess
import pandas as pd
from collections import defaultdict
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation
import csv

# Set directories and set definition of your gene cluster

input_dir = "Input directory"
hmm_file = "target_hmms.hmm"
outdir = "operons_singletons_associated_final"

MAX_ESCRT_GAP = 20000
MAX_BRIDGING_CDS = 8
EVALUE_CUTOFF = 1e-5
MIN_ESCRT_HITS = 2
SINGLETON_NEARBY_BP = 20000
PLOT_FLANK = 5000

os.makedirs(outdir, exist_ok=True)

# Search directory for .gbk or .gbff files

genbank_files = glob.glob(os.path.join(input_dir, "*.gbk")) + \
                glob.glob(os.path.join(input_dir, "*.gbff"))

if not genbank_files:
    raise RuntimeError(f"No GenBank files found in: {input_dir}")

print(f"Found {len(genbank_files)} genome files")

# Extract protein sequences from genbank records

protein_records = []
feature_map = {}
record_dict = {}
cds_index = {}

for gbk in genbank_files:
    genome_name = os.path.splitext(os.path.basename(gbk))[0]

    for record in SeqIO.parse(gbk, "genbank"):
        uid = f"{genome_name}|{record.id}"
        record_dict[uid] = record

        cds_features = []

        for f in record.features:
            if f.type != "CDS":
                continue

            start = int(f.location.start)
            end = int(f.location.end)
            strand = f.location.strand
            locus = f.qualifiers.get("locus_tag", ["unknown"])[0]
            product = f.qualifiers.get("product", ["hypothetical protein"])[0]
            prot = f.qualifiers.get("translation", [None])[0]

            cds_features.append({
                "start": start,
                "end": end,
                "strand": strand,
                "locus": locus,
                "product": product,
            })

            if prot:
                protein_records.append(
                    SeqRecord(Seq(prot), id=locus, description="")
                )

                feature_map[locus] = {
                    "record_id": uid,
                    "start": start,
                    "end": end,
                    "strand": strand,
                    "locus": locus,
                    "genome": genome_name,
                    "product": product,
                }

        cds_index[uid] = sorted(cds_features, key=lambda x: x["start"])

protein_fasta = os.path.join(outdir, "proteins.faa")
SeqIO.write(protein_records, protein_fasta, "fasta")
print(f"Wrote {len(protein_records)} proteins to {protein_fasta}")

# Detect genes of interest with hmmsearch

hmm_out = os.path.join(outdir, "hmm.tbl")

subprocess.run([
    "hmmsearch",
    "--tblout", hmm_out,
    hmm_file,
    protein_fasta
], check=True)

# Parse HMM hits

hits = []

with open(hmm_out) as f:
    for line in f:
        if line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        target = parts[0]
        query = parts[2]   # HMM name / ESCRT type
        evalue = float(parts[4])

        if evalue < EVALUE_CUTOFF and target in feature_map:
            hit = feature_map[target].copy()
            hit["evalue"] = evalue
            hit["escrt_type"] = query
            hits.append(hit)

df_hits = pd.DataFrame(hits).sort_values(["record_id", "start"]).reset_index(drop=True)

if df_hits.empty:
    raise RuntimeError("No ESCRT HMM hits found")

print(f"Parsed {len(df_hits)} ESCRT hits")

# Count number of bridging CDS features, find nearby hits on the opposite strand and cluster hits into clusters or seperate loci

def count_bridging_cds(record_id, prev_end, next_start):
    """
    Count all CDS features fully between two ESCRT anchors,
    regardless of strand.
    """
    features = cds_index.get(record_id, [])
    count = 0

    for feat in features:
        if feat["start"] >= next_start:
            break
        if feat["end"] <= prev_end:
            continue

        if feat["start"] >= prev_end and feat["end"] <= next_start:
            count += 1

    return count


def find_opposite_strand_partner(singleton_hit, all_hits_df, nearby_bp):
    record_id = singleton_hit["record_id"]
    genome = singleton_hit["genome"]
    strand = singleton_hit["strand"]
    locus = singleton_hit["locus"]
    midpoint = (singleton_hit["start"] + singleton_hit["end"]) / 2.0

    same_record = all_hits_df[
        (all_hits_df["genome"] == genome) &
        (all_hits_df["record_id"] == record_id) &
        (all_hits_df["locus"] != locus) &
        (all_hits_df["strand"] != strand)
    ].copy()

    if same_record.empty:
        return None

    same_record["midpoint"] = (same_record["start"] + same_record["end"]) / 2.0
    same_record["distance"] = (same_record["midpoint"] - midpoint).abs()

    nearby = same_record[same_record["distance"] <= nearby_bp].sort_values("distance")
    if nearby.empty:
        return None

    return nearby.iloc[0].to_dict()

def annotate_singleton_context(singleton_locus, all_hits_df, nearby_bp):
    hit = singleton_locus[0]
    partner = find_opposite_strand_partner(hit, all_hits_df, nearby_bp)

    diff_record = all_hits_df[
        (all_hits_df["genome"] == hit["genome"]) &
        (all_hits_df["record_id"] != hit["record_id"])
    ].copy()

    other_contig = not diff_record.empty
    different_contig_loci = ""
    different_contig_types = ""

    if other_contig:
        different_contig_loci = ";".join(sorted(diff_record["locus"].astype(str).unique()))
        different_contig_types = ";".join(sorted(diff_record["escrt_type"].astype(str).unique()))

    if partner is None:
        return {
            "near_opposite_strand_escrt": False,
            "near_opposite_strand_distance": "",
            "near_opposite_strand_locus": "",
            "near_opposite_strand_type": "",
            "other_escrt_on_different_contig": other_contig,
            "different_contig_escrt_loci": different_contig_loci,
            "different_contig_escrt_types": different_contig_types,
        }

    return {
        "near_opposite_strand_escrt": True,
        "near_opposite_strand_distance": int(partner["distance"]),
        "near_opposite_strand_locus": partner["locus"],
        "near_opposite_strand_type": partner["escrt_type"],
        "other_escrt_on_different_contig": other_contig,
        "different_contig_escrt_loci": different_contig_loci,
        "different_contig_escrt_types": different_contig_types,
    }

# Cluster gene clusters

all_loci = []
current = []

for _, row in df_hits.iterrows():
    if not current:
        current.append(row)
        continue

    prev = current[-1]
    gap = row["start"] - prev["end"]

    bridging_cds = count_bridging_cds(
        record_id=row["record_id"],
        prev_end=prev["end"],
        next_start=row["start"],
    )

    can_join = (
        row["record_id"] == prev["record_id"] and
        row["strand"] == prev["strand"] and
        gap >= 0 and
        gap <= MAX_ESCRT_GAP and
        bridging_cds <= MAX_BRIDGING_CDS
    )

    if can_join:
        row = row.copy()
        row["bridging_cds_from_prev"] = bridging_cds
        current.append(row)
    else:
        all_loci.append(current)
        current = [row]

if current:
    all_loci.append(current)

operons = [locus for locus in all_loci if len(locus) >= MIN_ESCRT_HITS]
singletons = [locus for locus in all_loci if len(locus) == 1]

print(f"Detected {len(operons)} operons")
print(f"Detected {len(singletons)} singleton loci")


# Find opposite strand clusters

used_singleton_loci = set()
associated_pairs = []

for i, locus in enumerate(singletons):
    singleton_hit = locus[0]
    singleton_locus_tag = singleton_hit["locus"]

    if singleton_locus_tag in used_singleton_loci:
        continue

    partner = find_opposite_strand_partner(singleton_hit, df_hits, SINGLETON_NEARBY_BP)
    if partner is None:
        continue

    partner_locus_tag = partner["locus"]
    pair_key = tuple(sorted([singleton_locus_tag, partner_locus_tag]))

    if pair_key in used_singleton_loci:
        continue

    associated_pairs.append({
        "pair_key": pair_key,
        "genome": singleton_hit["genome"],
        "record_id": singleton_hit["record_id"],
        "hit1": singleton_hit,
        "hit2": partner,
    })

    used_singleton_loci.add(singleton_locus_tag)
    used_singleton_loci.add(partner_locus_tag)

# Split singletons into associated and isolated

associated_singleton_loci = []
isolated_singletons = []

associated_singleton_locus_tags = set()
for pair in associated_pairs:
    associated_singleton_locus_tags.add(pair["hit1"]["locus"])
    associated_singleton_locus_tags.add(pair["hit2"]["locus"])

for locus in singletons:
    locus_tag = locus[0]["locus"]
    if locus_tag in associated_singleton_locus_tags:
        associated_singleton_loci.append(locus)
    else:
        isolated_singletons.append(locus)

print(f"Detected {len(associated_pairs)} opposite-strand-associated loci")
print(f"Detected {len(isolated_singletons)} isolated singleton loci")


# Build genbank files for gene clusters

def build_locus_record_from_hits(hit_list, locus_id, record_dict, plot_flank):
    """
    Build a GenBank record from one or more ESCRT hits.
    Export region spans outermost anchors plus plotting flank.
    """
    record_id = hit_list[0]["record_id"]
    genome = hit_list[0]["genome"]
    record = record_dict[record_id]

    anchor_start = min(g["start"] for g in hit_list)
    anchor_end = max(g["end"] for g in hit_list)

    start = max(0, anchor_start - plot_flank)
    end = min(len(record.seq), anchor_end + plot_flank)

    sub_seq = record.seq[start:end]

    new_record = SeqRecord(
        sub_seq,
        id=locus_id,
        name=locus_id,
        description=f"{record_id}:{start}-{end}"
    )
    new_record.annotations["molecule_type"] = "DNA"

    features = []
    gene_function_rows = []
    total_cds = 0

    for f in record.features:
        if f.type != "CDS":
            continue

        s = int(f.location.start)
        e = int(f.location.end)

        if e > start and s < end:
            rel_s = max(0, s - start)
            rel_e = min(end - start, e - start)

            locus_tag = f.qualifiers.get("locus_tag", [""])[0]
            product = f.qualifiers.get("product", ["hypothetical protein"])[0]

            features.append(
                SeqFeature(
                    FeatureLocation(rel_s, rel_e, strand=f.location.strand),
                    type="CDS",
                    qualifiers={"locus_tag": [locus_tag]}
                )
            )

            gene_function_rows.append((locus_tag, product))
            total_cds += 1

    new_record.features = features

    summary = {
        "locus_id": locus_id,
        "genome": genome,
        "record_id": record_id,
        "region_start": start,
        "region_end": end,
        "anchor_start": anchor_start,
        "anchor_end": anchor_end,
        "n_escrt_hits": len(hit_list),
        "n_total_cds_in_region": total_cds,
        "escrt_types": ";".join(sorted({g["escrt_type"] for g in hit_list})),
        "escrt_loci": ";".join([g["locus"] for g in hit_list]),
    }

    return new_record, gene_function_rows, summary

# Create output files

genome_operons = defaultdict(list)
genome_associated = defaultdict(list)
genome_singletons = defaultdict(list)
genome_figure_loci = defaultdict(list)

gene_function_rows = []
operon_summary_rows = []
associated_summary_rows = []
singleton_summary_rows = []

# Create output for operons

for i, locus in enumerate(operons):
    genome = locus[0]["genome"]
    locus_id = f"{genome}_operon_{i}"

    rec, gf_rows, summary = build_locus_record_from_hits(
        hit_list=locus,
        locus_id=locus_id,
        record_dict=record_dict,
        plot_flank=PLOT_FLANK
    )

    summary["locus_class"] = "operon"

    genome_operons[genome].append(rec)
    genome_figure_loci[genome].append(rec)
    gene_function_rows.extend(gf_rows)
    operon_summary_rows.append(summary)

# Create output for opposite strand clusters

for i, pair in enumerate(associated_pairs):
    genome = pair["genome"]
    locus_id = f"{genome}_opposite_strand_associated_{i}"

    rec, gf_rows, summary = build_locus_record_from_hits(
        hit_list=[pair["hit1"], pair["hit2"]],
        locus_id=locus_id,
        record_dict=record_dict,
        plot_flank=PLOT_FLANK
    )

    midpoint1 = (pair["hit1"]["start"] + pair["hit1"]["end"]) / 2.0
    midpoint2 = (pair["hit2"]["start"] + pair["hit2"]["end"]) / 2.0

    summary["locus_class"] = "opposite_strand_associated"
    summary["association_distance"] = int(abs(midpoint2 - midpoint1))
    summary["hit1_locus"] = pair["hit1"]["locus"]
    summary["hit1_type"] = pair["hit1"]["escrt_type"]
    summary["hit1_strand"] = pair["hit1"]["strand"]
    summary["hit2_locus"] = pair["hit2"]["locus"]
    summary["hit2_type"] = pair["hit2"]["escrt_type"]
    summary["hit2_strand"] = pair["hit2"]["strand"]

    genome_associated[genome].append(rec)
    genome_figure_loci[genome].append(rec)
    gene_function_rows.extend(gf_rows)
    associated_summary_rows.append(summary)

# Isolated singleton loci

for i, locus in enumerate(isolated_singletons):
    genome = locus[0]["genome"]
    locus_id = f"{genome}_singleton_{i}"

    rec, gf_rows, summary = build_locus_record_from_hits(
        hit_list=locus,
        locus_id=locus_id,
        record_dict=record_dict,
        plot_flank=PLOT_FLANK
    )

    context = annotate_singleton_context(
        singleton_locus=locus,
        all_hits_df=df_hits,
        nearby_bp=SINGLETON_NEARBY_BP
    )

    summary.update(context)
    summary["locus_class"] = "singleton_isolated"

    genome_singletons[genome].append(rec)
    gene_function_rows.extend(gf_rows)
    singleton_summary_rows.append(summary)


# Save a genbank record for operons, singletons and strand associated clusters

for genome, recs in genome_operons.items():
    path = os.path.join(outdir, f"{genome}_operons.gbk")
    SeqIO.write(recs, path, "genbank")
    print(f"Saved {genome}: {len(recs)} operons")

for genome, recs in genome_associated.items():
    path = os.path.join(outdir, f"{genome}_opposite_strand_associated.gbk")
    SeqIO.write(recs, path, "genbank")
    print(f"Saved {genome}: {len(recs)} opposite-strand-associated loci")

for genome, recs in genome_singletons.items():
    path = os.path.join(outdir, f"{genome}_singletons.gbk")
    SeqIO.write(recs, path, "genbank")
    print(f"Saved {genome}: {len(recs)} isolated singleton loci")

for genome, recs in genome_figure_loci.items():
    path = os.path.join(outdir, f"{genome}_figure_loci.gbk")
    SeqIO.write(recs, path, "genbank")
    print(f"Saved {genome}: {len(recs)} figure loci")


# Save a gene function table for clinker

gf_path = os.path.join(outdir, "gene_functions.csv")

with open(gf_path, "w", newline="") as f:
    writer = csv.writer(f)
    seen = set()

    for locus_tag, annot in gene_function_rows:
        if locus_tag in seen:
            continue
        seen.add(locus_tag)
        writer.writerow([locus_tag, annot])

print(f"Saved gene functions: {gf_path}")


# Save summary tables

operon_summary_path = os.path.join(outdir, "operon_summary.csv")
associated_summary_path = os.path.join(outdir, "opposite_strand_associated_summary.csv")
singleton_summary_path = os.path.join(outdir, "singleton_summary.csv")

pd.DataFrame(operon_summary_rows).to_csv(operon_summary_path, index=False)
pd.DataFrame(associated_summary_rows).to_csv(associated_summary_path, index=False)
pd.DataFrame(singleton_summary_rows).to_csv(singleton_summary_path, index=False)
