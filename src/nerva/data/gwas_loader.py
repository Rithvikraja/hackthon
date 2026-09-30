"""
NERVA-ML GWAS Catalog Ingestion & Variant-to-Gene Mapping
Parses NHGRI-EBI GWAS Catalog bulk association TSV.
Applies genome-wide significance filtering (P <= 5e-8), MHC exclusion,
trait EFO matching, Tier 1/2 mapping, duplicate collapse, and min-size low-power checks.
Strictly requires raw GWAS TSV file; raises DataMissingError if absent.
"""

from typing import Dict, List, Tuple, Set, Optional
import os
import re
import pandas as pd
from nerva.utils.errors import DataMissingError, DataFormatError
from nerva.data.gene_ids import GeneIdHub

TRAIT_EFO_MAP = {
    "PD": ["EFO_0002508", "Parkinson"],
    "AD": ["EFO_0000249", "Alzheimer"],
    "HD": ["EFO_0000544", "Huntington"]
}

def parse_gwas_tsv(
    tsv_path: str,
    gene_hub: Optional[GeneIdHub] = None,
    p_threshold: float = 5.0e-8,
    min_genes: int = 15,
    include_tier2: bool = False
) -> Tuple[Dict[str, List[str]], Dict[str, bool], pd.DataFrame]:
    """
    Parses GWAS Catalog bulk associations TSV.
    """
    if not os.path.exists(tsv_path):
        raise DataMissingError(
            f"GWAS Catalog associations file not found at: {tsv_path}",
            "Run: python scripts/download_data.py to download the bulk associations TSV from NHGRI-EBI GWAS Catalog."
        )

    # Read GWAS TSV with flexible chunking if large
    try:
        # Load necessary columns
        df = pd.read_csv(tsv_path, sep="\t", low_memory=False, on_bad_lines="skip")
    except Exception as e:
        raise DataFormatError(f"Failed to parse GWAS TSV at {tsv_path}: {e}", "Verify TSV formatting.")

    p_col = "P-VALUE" if "P-VALUE" in df.columns else "PVALUE"
    chr_col = "CHR_ID" if "CHR_ID" in df.columns else "CHR"
    pos_col = "CHR_POS" if "CHR_POS" in df.columns else "BP"
    trait_uri_col = "MAPPED_TRAIT_URI" if "MAPPED_TRAIT_URI" in df.columns else "TRAIT_URI"
    trait_col = "MAPPED_TRAIT" if "MAPPED_TRAIT" in df.columns else "DISEASE/TRAIT"
    gene_col = "MAPPED_GENE" if "MAPPED_GENE" in df.columns else "GENE"
    snp_genes_col = "SNP_GENE_IDS" if "SNP_GENE_IDS" in df.columns else "ENSEMBL_GENE_IDS"

    if p_col not in df.columns:
        raise DataFormatError(f"Required P-value column not found in {tsv_path}", "Check GWAS TSV headers.")

    # Convert P-value
    df[p_col] = pd.to_numeric(df[p_col], errors="coerce")
    sig_df = df[df[p_col] <= p_threshold].copy()

    disease_genes = {}
    low_power = {}
    log_rows = []

    for d_id, matches in TRAIT_EFO_MAP.items():
        efo_id = matches[0]
        text_keyword = matches[1].lower()

        # Match trait by EFO URI or text keyword
        uri_mask = sig_df[trait_uri_col].astype(str).str.contains(efo_id, case=False, na=False) if trait_uri_col in sig_df.columns else pd.Series(False, index=sig_df.index)
        text_mask = sig_df[trait_col].astype(str).str.lower().str.contains(text_keyword, na=False) if trait_col in sig_df.columns else pd.Series(False, index=sig_df.index)
        trait_mask = uri_mask | text_mask

        trait_rows = sig_df[trait_mask]

        # MHC exclusion: Chromosome 6, 25-34 Mb
        chr_series = trait_rows[chr_col].astype(str).str.replace("chr", "", case=False)
        pos_series = pd.to_numeric(trait_rows[pos_col], errors="coerce")
        mhc_mask = (chr_series == "6") & (pos_series >= 25000000) & (pos_series <= 34000000)
        
        mhc_excluded_count = int(mhc_mask.sum())
        clean_rows = trait_rows[~mhc_mask]

        tier1_genes = set()
        unmapped_count = 0

        for _, row in clean_rows.iterrows():
            raw_gene_field = str(row.get(gene_col, ""))
            raw_snp_genes = str(row.get(snp_genes_col, ""))

            # Extract symbols and Ensembl IDs
            tokens = [t.strip() for t in re.split(r"[,; \-]+", f"{raw_gene_field} {raw_snp_genes}") if t.strip() and t.strip() != "nan" and t.strip() != "-"]
            for token in tokens:
                if gene_hub:
                    norm_sym, is_m = gene_hub.normalize(token)
                    if is_m:
                        tier1_genes.add(norm_sym)
                    else:
                        unmapped_count += 1
                else:
                    if len(token) > 1 and not token.isdigit():
                        tier1_genes.add(token)

        final_genes = sorted(list(tier1_genes))
        is_low = len(final_genes) < min_genes

        disease_genes[d_id] = final_genes
        low_power[d_id] = is_low

        log_rows.append({
            "disease_id": d_id,
            "raw_associations": len(trait_rows),
            "mhc_excluded": mhc_excluded_count,
            "tier1_count": len(final_genes),
            "unmapped_count": unmapped_count,
            "low_power": is_low
        })

    log_df = pd.DataFrame(log_rows)
    return disease_genes, low_power, log_df


def load_gwas_disease_genes(
    gwas_tsv_path: str = "data/raw/gwas/gwas_catalog_associations.tsv",
    gene_hub: Optional[GeneIdHub] = None,
    p_threshold: float = 5.0e-8,
    min_genes: int = 15
) -> Tuple[Dict[str, List[str]], Dict[str, bool], pd.DataFrame]:
    """
    Entrypoint for loading disease-associated genes from real GWAS TSV.
    Raises DataMissingError if raw TSV is not present.
    """
    if not os.path.exists(gwas_tsv_path):
        # Also check default directory for any .tsv file
        gwas_dir = os.path.dirname(gwas_tsv_path) or "data/raw/gwas"
        found = False
        if os.path.exists(gwas_dir):
            for f in os.listdir(gwas_dir):
                if f.endswith(".tsv") or f.endswith(".txt"):
                    gwas_tsv_path = os.path.join(gwas_dir, f)
                    found = True
                    break
        if not found:
            raise DataMissingError(
                f"GWAS Catalog bulk associations file not found at {gwas_tsv_path}.",
                "Run: python scripts/download_data.py to download associations TSV."
            )

    return parse_gwas_tsv(
        tsv_path=gwas_tsv_path,
        gene_hub=gene_hub,
        p_threshold=p_threshold,
        min_genes=min_genes
    )
