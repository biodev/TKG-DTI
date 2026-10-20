import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rebuild_tge  # noqa: E402

JAN_DTI = Path(
    "/home/exacloud/gscratch/mcweeney_lab/evans/data/targetome_expanded/archive/targetome_extended-01-23-25.csv"
)
JAN_DRUGS = Path(
    "/home/exacloud/gscratch/mcweeney_lab/evans/data/targetome_expanded/archive/targetome_extended_drugs-01-23-25.csv"
)
MARCH_DTI = Path(
    "/home/exacloud/gscratch/mcweeney_lab/evans/data/targetome_expanded/targetome_expanded_03-26-26.tsv"
)
MARCH_DRUGS = Path(
    "/home/exacloud/gscratch/mcweeney_lab/evans/data/targetome_expanded/targetome_expanded_drugs_03-26-26.tsv"
)

MARCH_DTI_HEADER = (
    "pubchem_cid\tinchi_key\tdrug_name\tclinical_phase\tuniprot_id\tgene_id\tsymbol\t"
    "pubmed_id\tdatabase\tassay_type\tassay_relation\tassay_value\tevidence_level\n"
)
MARCH_DRUG_HEADER = (
    "pubchem_cid\tinchi_key\tclinical_phase\tdrug_name\tchembl_id\ttherapeutic_flag\t"
    "molecule_type\twithdrawn_flag\tchemical_probe\tnatural_product\tatc\tmesh\t"
    "pharm_mesh\tnci\tinchi\n"
)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class RebuildTgeFixtureTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.jan_dti = self.root / "january_dti.csv"
        self.jan_drugs = self.root / "january_drugs.csv"
        self.mar_dti = self.root / "march_dti.tsv"
        self.mar_drugs = self.root / "march_drugs.tsv"
        self.jan_dti.write_text(
            "pubchem_cid,inchi_key,uniprot_id,pubmed_id,database,assay_type,assay_relation,assay_value\n"
            "1,AAAA,P11111,10,bindingdb,Ki,=,1.5\n"
            "2,BBBB,P22222,20,iuphar,IC50,,1e+05\n"
            "3,CCCC,P33333,30,bindingdb,Kd,<,0.016\n"
        )
        self.jan_drugs.write_text(
            '"pubchem_cid","inchi_key","clinical_phase","drug_name","chembl_id","therapeutic_flag",'
            '"molecule_type","withdrawn_flag","chemical_probe","natural_product","drugbank_status",'
            '"drugbank_type","iuphar_type","iuphar_status","atc","mesh","pharm_mesh","nci","inchi"\n'
            '1,"AAAA","Approved","OldName","CHEMBL1",0,"Small molecule",0,0,0,"approved","small molecule",,,,"mesh","pharm","nci","InChI=1S/old"\n'
            '2,"BBBB","Approved","Same","CHEMBL2",1,"Small molecule",0,0,0,,,,,,"m2","p2","n2","InChI=1S/same"\n'
        )
        self.mar_dti.write_text(
            MARCH_DTI_HEADER
            + "1\tAAAA\tDrugA\tApproved\tP11111\t1\tGENEA\t10\tbindingdb\tKi\t=\t1.5\tIII\n"
            + "2\tBBBB\tDrugB\tApproved\tP22222\t2\tGENEB\t20\tiuphar\tIC50\t\t100000\tIII\n"
            + "9\tDDDD\tDrugD\t\tP44444\t4\tGENED\t40\tbindingdb\tKd\t=\t5\tIII\n"
            + "8\tEEEE\tDrugE\t\tP55555\t5\tGENEE\t\t\tPredicted\t<=\t100\t0\n"
        )
        self.mar_drugs.write_text(
            MARCH_DRUG_HEADER
            + "1\tAAAA\tApproved\tNewName\tCHEMBL1\t0\tSmall molecule\t0\t0\t0\t\tmesh\tpharm\tnci\tInChI=1S/old\n"
            + "2\tBBBB\tApproved\tSame\tCHEMBL2\t1\tSmall molecule\t0\t0\t0\t\tm2\tp2\tn2\tInChI=1S/same\n"
            + "3\tCCCC\tApproved\tExtra\tCHEMBL3\t0\tSmall molecule\t0\t0\t0\t\tm3\tp3\tn3\tInChI=1S/extra\n"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_drops_predictions_and_keeps_released_rows(self):
        out = self.root / "default_out"
        rebuild_tge.main(
            [
                "--out",
                str(out),
                "--march-dti",
                str(self.mar_dti),
                "--march-drugs",
                str(self.mar_drugs),
            ]
        )
        dti = pd.read_csv(out / rebuild_tge.DTI_OUT_NAME)
        drugs = pd.read_csv(out / rebuild_tge.DRUGS_OUT_NAME, dtype=str, keep_default_na=False)
        self.assertEqual(list(dti.columns), rebuild_tge.DTI_COLS)
        self.assertNotIn("Predicted", set(dti.assay_type))
        self.assertIn("DDDD", set(dti.inchi_key))
        self.assertNotIn("CCCC", set(dti.inchi_key))
        self.assertEqual(set(drugs.drug_name), {"NewName", "Same", "Extra"})
        self.assertNotIn("drugbank_status", drugs.columns)
        self.assertNotEqual((out / rebuild_tge.DTI_OUT_NAME).read_bytes(), self.jan_dti.read_bytes())

    def test_bit_identical_fixture_matches_january_bytes(self):
        patch = self.root / "patch"
        out = self.root / "bit_out"
        rebuild_tge.main(
            [
                "--build-patch",
                "--bit-identical",
                "01-23-25",
                "--january-dti",
                str(self.jan_dti),
                "--january-drugs",
                str(self.jan_drugs),
                "--march-dti",
                str(self.mar_dti),
                "--march-drugs",
                str(self.mar_drugs),
                "--patch_path",
                str(patch),
            ]
        )
        rebuild_tge.main(
            [
                "--out",
                str(out),
                "--bit-identical",
                "01-23-25",
                "--march-dti",
                str(self.mar_dti),
                "--march-drugs",
                str(self.mar_drugs),
                "--patch_path",
                str(patch),
            ]
        )
        self.assertEqual((out / rebuild_tge.DTI_OUT_NAME).read_bytes(), self.jan_dti.read_bytes())
        self.assertEqual((out / rebuild_tge.DRUGS_OUT_NAME).read_bytes(), self.jan_drugs.read_bytes())
        dti_idx = __import__("numpy").load(patch / "dti_march_iloc.npy")
        self.assertTrue((dti_idx >= 0).any())
        self.assertIn(b"OldName", (patch / "drugs_verbatim_lines.txt").read_bytes())
        self.assertIn(b"approved", (patch / "drugs_verbatim_lines.txt").read_bytes())

    def test_unknown_snapshot_is_rejected(self):
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "rebuild_tge.py"),
                "--out",
                str(self.root / "out"),
                "--bit-identical",
                "not-a-snapshot",
                "--march-dti",
                str(self.mar_dti),
                "--march-drugs",
                str(self.mar_drugs),
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Unknown --bit-identical", proc.stderr + proc.stdout)

    def test_bit_identical_requires_patch_path(self):
        with self.assertRaises(SystemExit) as caught:
            rebuild_tge.main(
                [
                    "--out",
                    str(self.root / "out"),
                    "--bit-identical",
                    "01-23-25",
                    "--march-dti",
                    str(self.mar_dti),
                    "--march-drugs",
                    str(self.mar_drugs),
                ]
            )
        self.assertIn("--patch_path", str(caught.exception))


class FormatAssayTest(unittest.TestCase):
    def test_january_token_examples(self):
        cases = {
            100000: "1e+05",
            133000: "133000",
            250000: "250000",
            0.0006: "6e-04",
            0.016: "0.016",
            737.5: "737.5",
            0: "0",
            1.5: "1.5",
            9e5: "9e+05",
        }
        for value, token in cases.items():
            self.assertEqual(rebuild_tge.format_assay(value), token)


@unittest.skipUnless(
    JAN_DTI.exists() and JAN_DRUGS.exists() and MARCH_DTI.exists() and MARCH_DRUGS.exists(),
    "January and March Targetome archives are not on this machine",
)
class RebuildTgeArchiveTest(unittest.TestCase):
    def test_bit_identical_matches_january_archives(self):
        patch = ROOT / "extdata" / "tge_bit_identical" / "01-23-25"
        if not (patch / "dti_march_iloc.npy").exists():
            self.skipTest("patch has not been built")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            rebuild_tge.main(
                [
                    "--out",
                    str(out),
                    "--bit-identical",
                    "01-23-25",
                    "--march-dti",
                    str(MARCH_DTI),
                    "--march-drugs",
                    str(MARCH_DRUGS),
                    "--patch_path",
                    str(patch),
                ]
            )
            self.assertEqual(_md5(out / rebuild_tge.DTI_OUT_NAME), _md5(JAN_DTI))
            self.assertEqual(_md5(out / rebuild_tge.DRUGS_OUT_NAME), _md5(JAN_DRUGS))


if __name__ == "__main__":
    unittest.main()
