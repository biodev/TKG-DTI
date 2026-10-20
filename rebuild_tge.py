#!/usr/bin/env python
"""Download the Figshare Targetome tables and write the CSVs step 01 reads.

Default (retrain on the current release):

    python rebuild_tge.py --out ./path/to/data/dir/

Drops predicted DTI rows, keeps every other released row, and writes the
column names used by kg_construction_01__drug_interacts_protein.py.

Byte-identical January 2025 training snapshot:

    unzip patch_01-23-25.zip
    python rebuild_tge.py --out ./path/to/data/dir/ --bit-identical 01-23-25 \
        --patch_path ./patch_01-23-25

Download patch_01-23-25.zip from the Targetome Figshare record. That patch
is not in this repository. Filtering Figshare alone cannot rebuild the
snapshot: some January rows were removed, some cells changed, and the drug
table lost four columns.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
KNOWN_SNAPSHOTS = ("01-23-25",)

FIGSHARE = {
    "dti": (
        "https://ndownloader.figshare.com/files/69960483",
        "targetome_expanded_03-26-26.tsv.gz",
    ),
    "drugs": (
        "https://ndownloader.figshare.com/files/69960504",
        "targetome_expanded_drugs_03-26-26.tsv.gz",
    ),
}

DTI_OUT_NAME = "targetome_extended-01-23-25.csv"
DRUGS_OUT_NAME = "targetome_extended_drugs-01-23-25.csv"

DTI_COLS = [
    "pubchem_cid",
    "inchi_key",
    "uniprot_id",
    "pubmed_id",
    "database",
    "assay_type",
    "assay_relation",
    "assay_value",
]
DTI_HEADER = ",".join(DTI_COLS) + "\n"

# March 2026 drug columns. January also had the four annotation columns below.
DRUG_MARCH_COLS = [
    "pubchem_cid",
    "inchi_key",
    "clinical_phase",
    "drug_name",
    "chembl_id",
    "therapeutic_flag",
    "molecule_type",
    "withdrawn_flag",
    "chemical_probe",
    "natural_product",
    "atc",
    "mesh",
    "pharm_mesh",
    "nci",
    "inchi",
]
DRUG_EXTRA_COLS = [
    "drugbank_status",
    "drugbank_type",
    "iuphar_type",
    "iuphar_status",
]
DRUG_JAN_COLS = [
    "pubchem_cid",
    "inchi_key",
    "clinical_phase",
    "drug_name",
    "chembl_id",
    "therapeutic_flag",
    "molecule_type",
    "withdrawn_flag",
    "chemical_probe",
    "natural_product",
    "drugbank_status",
    "drugbank_type",
    "iuphar_type",
    "iuphar_status",
    "atc",
    "mesh",
    "pharm_mesh",
    "nci",
    "inchi",
]
DRUG_INT_COLS = {
    "pubchem_cid",
    "therapeutic_flag",
    "withdrawn_flag",
    "chemical_probe",
    "natural_product",
}
DRUG_HEADER = ",".join(f'"{c}"' for c in DRUG_JAN_COLS) + "\n"


def format_assay(value) -> str:
    """January DTI assay tokens: shorter of unique fixed vs scientific form.

    100000 is written ``1e+05``; 133000 stays ``133000``; 0.0006 is ``6e-04``.
    Ties keep the fixed form. This matches every token in the 01-23-25 file.
    """
    x = float(value)
    if x == 0.0:
        return "0"
    fixed = np.format_float_positional(x, unique=True, trim="k")
    if fixed.endswith("."):
        fixed = fixed[:-1]
    sci = np.format_float_scientific(
        x, unique=True, trim="k", exp_digits=2, sign=False
    ).replace(".e", "e")
    if "e" in sci:
        mant, exp = sci.split("e")
        sign = exp[0] if exp[:1] in "+-" else "+"
        digits = exp[1:] if exp[:1] in "+-" else exp
        sci = f"{mant}e{sign}{int(digits):02d}"
    if len(sci) < len(fixed):
        return sci
    return fixed


def _quote(text: str) -> str:
    return '"' + text.replace('"', '""') + '"'


def _blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        return False
    return False


def _as_int_token(value) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text.endswith(".0"):
            text = text[:-2]
        return str(int(text))
    return str(int(value))


def render_dti_fields(cid, inchi, uniprot, pubmed, database, assay_type, relation, assay_value) -> str:
    def cell(value) -> str:
        if _blank(value):
            return ""
        text = str(value)
        if any(ch in text for ch in ',"\r\n'):
            return _quote(text)
        return text

    if _blank(pubmed):
        pubmed_token = ""
    else:
        pubmed_token = _as_int_token(pubmed)
    if isinstance(assay_value, str) and assay_value != "" and not _is_number(assay_value):
        assay_token = assay_value
    elif _blank(assay_value):
        assay_token = ""
    else:
        assay_token = format_assay(assay_value)
    parts = [
        _as_int_token(cid),
        cell(inchi),
        cell(uniprot),
        pubmed_token,
        cell(database),
        cell(assay_type),
        "" if _blank(relation) else cell(relation),
        assay_token,
    ]
    return ",".join(parts) + "\n"


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def render_drug_fields(values: dict) -> str:
    """January drug dialect: numbers bare, strings quoted, missing fields empty.

    An empty ``iuphar_status`` is written as ``\"\"`` when ``iuphar_type`` is
    present. That is how the 01-23-25 file distinguishes those cells.
    """
    parts = []
    iuphar_type = values.get("iuphar_type", "")
    for col in DRUG_JAN_COLS:
        value = values.get(col, "")
        if _blank(value):
            if col == "iuphar_status" and not _blank(iuphar_type):
                parts.append('""')
            else:
                parts.append("")
        elif col in DRUG_INT_COLS:
            parts.append(_as_int_token(value))
        else:
            parts.append(_quote(str(value)))
    return ",".join(parts) + "\n"


def is_predicted(dti: pd.DataFrame) -> pd.Series:
    assay = dti["assay_type"].astype(str).str.strip().eq("Predicted") if "assay_type" in dti.columns else False
    if "evidence_level" not in dti.columns:
        return assay
    level = dti["evidence_level"].astype(str).str.strip().isin(["0", "0.0"])
    return assay | level


def read_table(path: Path, kind: str) -> pd.DataFrame:
    path = Path(path)
    sep = "\t" if ".tsv" in path.name else ","
    if kind == "dti":
        return pd.read_csv(path, sep=sep, low_memory=False)
    # Keep the literal mesh token "NA". Default NA parsing would drop it.
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, low_memory=False)


def read_lines(path: Path) -> list[str]:
    with Path(path).open(newline="") as handle:
        header = handle.readline()
        if not header:
            raise ValueError(f"Empty file: {path}")
        lines = handle.readlines()
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return lines


def download_figshare(cache_dir: Path) -> tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for key in ("dti", "drugs"):
        url, name = FIGSHARE[key]
        dest = cache_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"using cached {dest}")
        else:
            print(f"downloading {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "tkgdti-rebuild-tge"})
            with urllib.request.urlopen(req) as resp, dest.open("wb") as handle:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
        out.append(dest)
    return out[0], out[1]


def write_default(dti: pd.DataFrame, drugs: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = dti.loc[~is_predicted(dti), DTI_COLS]
    keep.to_csv(out_dir / DTI_OUT_NAME, index=False)
    drugs.loc[:, DRUG_MARCH_COLS].to_csv(out_dir / DRUGS_OUT_NAME, index=False)
    print(
        f"wrote {len(keep)} DTI rows (dropped {int(is_predicted(dti).sum())} predicted) "
        f"and {len(drugs)} drug rows to {out_dir}"
    )


def _norm_token(value) -> str:
    if _blank(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and _is_number(text):
        number = float(text)
        if number.is_integer():
            return str(int(number))
    return text


def _dti_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    assay = df["assay_value"].map(lambda v: "" if _blank(v) else format_assay(v))
    return pd.DataFrame(
        {
            "cid": df["pubchem_cid"].map(_norm_token),
            "inchi": df["inchi_key"].map(_norm_token),
            "uni": df["uniprot_id"].map(_norm_token),
            "pmid": df["pubmed_id"].map(_norm_token),
            "db": df["database"].map(_norm_token),
            "atype": df["assay_type"].map(_norm_token),
            "rel": df["assay_relation"].map(_norm_token),
            "val": assay,
        }
    )


def _drug_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({col: df[col].map(_norm_token) if col in df.columns else "" for col in DRUG_MARCH_COLS})


def _pair_indices(left_keys: pd.DataFrame, right_keys: pd.DataFrame) -> np.ndarray:
    """Row in ``right`` for each row of ``left``, or -1 when unmatched."""
    left = left_keys.copy()
    right = right_keys.copy()
    left["_k"] = pd.util.hash_pandas_object(left, index=False).astype("uint64")
    right["_k"] = pd.util.hash_pandas_object(right, index=False).astype("uint64")
    left["_occ"] = left.groupby("_k", sort=False).cumcount()
    right["_occ"] = right.groupby("_k", sort=False).cumcount()
    right["_i"] = np.arange(len(right), dtype=np.int64)
    left["_row"] = np.arange(len(left), dtype=np.int64)
    merged = left[["_k", "_occ", "_row"]].merge(
        right[["_k", "_occ", "_i"]], on=["_k", "_occ"], how="left"
    )
    merged = merged.sort_values("_row")
    idx = merged["_i"].to_numpy()
    out = np.full(len(left), -1, dtype=np.int32)
    ok = ~pd.isna(idx)
    out[ok] = idx[ok].astype(np.int32)
    return out


def precompute_dti_lines(df: pd.DataFrame) -> list[str]:
    cols = [df[name].to_numpy() for name in DTI_COLS]
    return [render_dti_fields(*(col[i] for col in cols)) for i in range(len(df))]


def precompute_drug_lines(df: pd.DataFrame) -> list[str]:
    """Render March drug rows with the four January-only columns left blank."""
    arrays = {col: df[col].to_numpy() if col in df.columns else None for col in DRUG_MARCH_COLS}
    lines = []
    for i in range(len(df)):
        values = {col: "" for col in DRUG_JAN_COLS}
        for col, arr in arrays.items():
            values[col] = "" if arr is None else arr[i]
        lines.append(render_drug_fields(values))
    return lines


def build_patch(
    january_dti: Path,
    january_drugs: Path,
    march_dti: Path,
    march_drugs: Path,
    patch_dir: Path,
) -> None:
    patch_dir.mkdir(parents=True, exist_ok=True)
    print("reading January and March tables")
    jan_dti = read_table(january_dti, "dti")
    mar_dti = read_table(march_dti, "dti")
    jan_drugs = read_table(january_drugs, "drugs")
    mar_drugs = read_table(march_drugs, "drugs")
    jan_dti_lines = read_lines(january_dti)
    jan_drug_lines = read_lines(january_drugs)
    if len(jan_dti_lines) != len(jan_dti):
        raise SystemExit(
            f"DTI line count {len(jan_dti_lines)} != rows {len(jan_dti)}; refusing to build a patch"
        )
    if len(jan_drug_lines) != len(jan_drugs):
        raise SystemExit(
            f"drug line count {len(jan_drug_lines)} != rows {len(jan_drugs)}; refusing to build a patch"
        )

    print("pairing DTI rows")
    dti_rendered = precompute_dti_lines(mar_dti)
    dti_src = _pair_indices(_dti_key_frame(jan_dti), _dti_key_frame(mar_dti))
    dti_idx, dti_verbatim = _keep_only_exact_lines(dti_src, jan_dti_lines, dti_rendered)
    print("pairing drug rows")
    drug_rendered = precompute_drug_lines(mar_drugs)
    drug_src = _pair_indices(_drug_key_frame(jan_drugs), _drug_key_frame(mar_drugs))
    drug_idx, drug_verbatim = _keep_only_exact_lines(drug_src, jan_drug_lines, drug_rendered)
    np.save(patch_dir / "dti_march_iloc.npy", dti_idx)
    np.save(patch_dir / "drugs_march_iloc.npy", drug_idx)
    (patch_dir / "dti_verbatim_lines.txt").write_text("".join(dti_verbatim))
    (patch_dir / "drugs_verbatim_lines.txt").write_text("".join(drug_verbatim))
    print(
        f"DTI: {len(dti_idx)} January rows, {(dti_idx >= 0).sum()} regenerated, "
        f"{len(dti_verbatim)} verbatim, {len(mar_dti) - len(set(dti_idx[dti_idx >= 0]))} March rows unused"
    )
    print(
        f"drugs: {len(drug_idx)} January rows, {(drug_idx >= 0).sum()} regenerated, "
        f"{len(drug_verbatim)} verbatim, {len(mar_drugs) - len(set(drug_idx[drug_idx >= 0]))} March rows unused"
    )
    print(f"wrote patch to {patch_dir}")


def _keep_only_exact_lines(
    src: np.ndarray, raw_lines: list[str], rendered: list[str]
) -> tuple[np.ndarray, list[str]]:
    out = np.empty(len(raw_lines), dtype=np.int32)
    verbatim = []
    for i, raw in enumerate(raw_lines):
        src_i = int(src[i])
        if src_i >= 0 and rendered[src_i] == raw:
            out[i] = src_i
        else:
            out[i] = -1
            verbatim.append(raw)
    return out, verbatim


def _load_verbatim(path: Path) -> list[str]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    lines = path.read_text().splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return lines


def write_bit_identical(march_dti: Path, march_drugs: Path, patch_dir: Path, out_dir: Path) -> None:
    dti_idx_path = patch_dir / "dti_march_iloc.npy"
    drug_idx_path = patch_dir / "drugs_march_iloc.npy"
    if not dti_idx_path.exists() or not drug_idx_path.exists():
        raise SystemExit(
            f"Missing patch in {patch_dir}. Download patch_01-23-25.zip from Figshare, "
            "unzip it, and pass that directory as --patch_path."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    print("reading March tables")
    mar_dti = read_table(march_dti, "dti")
    mar_drugs = read_table(march_drugs, "drugs")
    dti_idx = np.load(dti_idx_path)
    drug_idx = np.load(drug_idx_path)
    dti_verbatim = _load_verbatim(patch_dir / "dti_verbatim_lines.txt")
    drug_verbatim = _load_verbatim(patch_dir / "drugs_verbatim_lines.txt")
    print("writing DTI")
    _write_indexed(
        out_dir / DTI_OUT_NAME,
        DTI_HEADER,
        dti_idx,
        dti_verbatim,
        precompute_dti_lines(mar_dti),
    )
    print("writing drugs")
    _write_indexed(
        out_dir / DRUGS_OUT_NAME,
        DRUG_HEADER,
        drug_idx,
        drug_verbatim,
        precompute_drug_lines(mar_drugs),
    )
    print(f"wrote bit-identical tables to {out_dir}")


def _write_indexed(path: Path, header: str, indices: np.ndarray, verbatim: list[str], rendered: list[str]) -> None:
    cursor = 0
    with path.open("w", newline="") as handle:
        handle.write(header)
        for src in indices:
            src_i = int(src)
            if src_i < 0:
                if cursor >= len(verbatim):
                    raise SystemExit(f"Verbatim lines exhausted while writing {path}")
                handle.write(verbatim[cursor])
                cursor += 1
            else:
                handle.write(rendered[src_i])
    if cursor != len(verbatim):
        raise SystemExit(f"Unused verbatim lines ({len(verbatim) - cursor}) for {path}")


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_patch_dir(snapshot: str, patch_path: str | None) -> Path:
    if snapshot not in KNOWN_SNAPSHOTS:
        known = ", ".join(KNOWN_SNAPSHOTS)
        raise SystemExit(f"Unknown --bit-identical {snapshot!r}. Known snapshots: {known}")
    if not patch_path:
        raise SystemExit(
            "--bit-identical requires --patch_path, the unzipped patch_01-23-25 directory from Figshare."
        )
    return Path(patch_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=str, help="Directory for the two CSV files")
    parser.add_argument(
        "--bit-identical",
        dest="bit_identical",
        default=None,
        help="Rebuild a training snapshot byte for byte. Currently: 01-23-25",
    )
    parser.add_argument(
        "--build-patch",
        action="store_true",
        help="Write a patch directory from local January CSVs and the March tables",
    )
    parser.add_argument("--march-dti", type=str, help="Local March DTI .tsv or .tsv.gz")
    parser.add_argument("--march-drugs", type=str, help="Local March drug .tsv or .tsv.gz")
    parser.add_argument("--january-dti", type=str, help="January DTI CSV used only with --build-patch")
    parser.add_argument("--january-drugs", type=str, help="January drug CSV used only with --build-patch")
    parser.add_argument(
        "--patch_path",
        type=str,
        help="Unzipped patch directory (patch_01-23-25/). Required with --bit-identical",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Where Figshare downloads are stored (default: <out>/_figshare_cache)",
    )
    return parser.parse_args(argv)


def resolve_march(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.march_dti and args.march_drugs:
        return Path(args.march_dti), Path(args.march_drugs)
    if args.march_dti or args.march_drugs:
        raise SystemExit("Pass both --march-dti and --march-drugs, or neither to download Figshare.")
    if args.cache_dir:
        cache = Path(args.cache_dir)
    elif args.out:
        cache = Path(args.out) / "_figshare_cache"
    else:
        cache = REPO_ROOT / "extdata" / "figshare_cache"
    return download_figshare(cache)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.build_patch:
        if not args.bit_identical:
            raise SystemExit("--build-patch requires --bit-identical 01-23-25")
        if not args.january_dti or not args.january_drugs:
            raise SystemExit("--build-patch requires --january-dti and --january-drugs")
        march_dti, march_drugs = resolve_march(args)
        build_patch(
            Path(args.january_dti),
            Path(args.january_drugs),
            march_dti,
            march_drugs,
            snapshot_patch_dir(args.bit_identical, args.patch_path),
        )
        return

    if not args.out:
        raise SystemExit("--out is required unless --build-patch is set")
    march_dti, march_drugs = resolve_march(args)
    out_dir = Path(args.out)
    if args.bit_identical:
        write_bit_identical(
            march_dti,
            march_drugs,
            snapshot_patch_dir(args.bit_identical, args.patch_path),
            out_dir,
        )
        return

    dti = read_table(march_dti, "dti")
    drugs = read_table(march_drugs, "drugs")
    missing = [col for col in DTI_COLS + ["evidence_level"] if col not in dti.columns]
    if missing:
        raise SystemExit(f"March DTI table is missing columns: {missing}")
    missing_drugs = [col for col in DRUG_MARCH_COLS if col not in drugs.columns]
    if missing_drugs:
        raise SystemExit(f"March drug table is missing columns: {missing_drugs}")
    write_default(dti, drugs, out_dir)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
