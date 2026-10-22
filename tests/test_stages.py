"""Synthetic perturbations test software behavior, not biochemical activity."""

import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gemmi
import numpy as np

from mcsa_benchmark import evaluate, evaluate_general, evaluate_mechanism, list_cases
from mcsa_benchmark.evaluate import load_case
from mcsa_benchmark.mechanisms.adenylate import inside

FIXTURE = os.environ.get("MCSA_1QQ5_TEST_CIF")
FIRST = os.environ.get("MCSA_1ZIO_TEST_CIF")
api_module = importlib.import_module("mcsa_benchmark.evaluate")


class ConfigurationTests(unittest.TestCase):
    def test_cases(self):
        self.assertEqual(list_cases(), ["1B6B", "1DHP", "1ET0", "1GSA", "1I8T", "1IDT", "1IGS", "1IM5", "1KQP", "1L7N", "1LDM", "1NDP", "1OTH", "1QFE", "1QQ5", "1XEL", "1XVA", "1ZIO", "2GSA", "5ICD"])

    def test_limits_and_authorship(self):
        case = load_case("1QQ5")
        criterion = case["mechanism_criteria"]
        self.assertTrue(criterion["retrospective"])
        self.assertIn("Author operational", criterion["main_axis"]["distance_lower_basis"])
        self.assertEqual(criterion["main_axis"]["distance_A"], {"min": 2.5, "max": 3.5, "inclusive": True})
        self.assertTrue(inside(2.5, criterion["main_axis"]["distance_A"]))
        self.assertTrue(inside(3.5, criterion["main_axis"]["distance_A"]))
        self.assertFalse(inside(3.50001, criterion["main_axis"]["distance_A"]))
        self.assertTrue(inside(150.0, criterion["main_axis"]["angle_deg"]))
        self.assertTrue(inside(180.0, criterion["main_axis"]["angle_deg"]))
        self.assertFalse(inside(149.999, criterion["main_axis"]["angle_deg"]))

    def test_unknown_case_not_fallback(self):
        with self.assertRaises(ValueError):
            load_case("UNKNOWN")


@unittest.skipUnless(FIXTURE, "Set MCSA_1QQ5_TEST_CIF to the AF3 s0 fixture")
class HaloacidTests(unittest.TestCase):
    def variant(self, modify, stage="all", mapping=None):
        structure = gemmi.read_structure(FIXTURE)
        modify(structure)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "new_prediction.cif"
            structure.make_mmcif_document().write_file(str(path))
            result = evaluate(path, "1QQ5", stage, mapping)
            json.dumps(result, allow_nan=False)
            return result

    def test_baseline(self):
        result = evaluate(FIXTURE, "1QQ5")
        self.assertEqual(result["mechanism_status"], "PASS")
        self.assertEqual(result["general_status"], "PASS")
        self.assertFalse(result["historical_results_used"])

    def test_mechanism_only_never_runs_general(self):
        with patch.object(api_module, "run_general", side_effect=AssertionError("must not run")):
            result = evaluate_mechanism(FIXTURE, "1QQ5")
        self.assertEqual(result["mechanism_status"], "PASS")
        self.assertEqual(result["general_status"], "NOT_RUN")
        self.assertEqual(result["joint_status"], "NOT_RUN")

    def test_general_only_never_runs_mechanism(self):
        with patch.dict(api_module.MECHANISMS, {"haloacid_sn2": lambda *_: (_ for _ in ()).throw(AssertionError("must not run"))}):
            result = evaluate_general(FIXTURE, "1QQ5")
        self.assertEqual(result["general_status"], "PASS")
        self.assertEqual(result["mechanism_status"], "NOT_RUN")

    def test_general_error_does_not_erase_mechanism(self):
        with patch.object(api_module, "run_general", side_effect=ValueError("synthetic general error")):
            result = evaluate(FIXTURE, "1QQ5")
        self.assertEqual(result["mechanism_status"], "PASS")
        self.assertEqual(result["general_status"], "ERROR")

    def test_mechanism_error_does_not_stop_general(self):
        with patch.dict(api_module.MECHANISMS, {"haloacid_sn2": lambda *_: (_ for _ in ()).throw(ValueError("synthetic mechanism error"))}):
            result = evaluate(FIXTURE, "1QQ5")
        self.assertEqual(result["mechanism_status"], "ERROR")
        self.assertEqual(result["general_status"], "PASS")

    def test_stereo_flip_independent_of_reaction_axis(self):
        def change(s):
            atoms = {a.name: a for a in s[0]["C"][0]}
            c = np.array(atoms["C2"].pos.tolist())
            u = np.array(atoms["CL1"].pos.tolist()) - c
            v = np.array(atoms["C3"].pos.tolist()) - c
            normal = np.cross(u, v)
            normal /= np.linalg.norm(normal)
            x = np.array(atoms["C1"].pos.tolist())
            reflected = x - 2 * np.dot(x-c, normal) * normal
            atoms["C1"].pos = gemmi.Position(*reflected)
        result = self.variant(change)
        self.assertEqual(result["mechanism_status"], "PASS")
        self.assertEqual(result["general_status"], "FAIL")
        self.assertIn("INPUT_STEREOCHEMISTRY_MISMATCH", result["general"]["failure_tags"])

    def test_missing_remote_sidechain_does_not_block_mechanism(self):
        result = self.variant(lambda s: s[0]["A"][0].remove_atom("CE", "\x00"))
        self.assertEqual(result["mechanism_status"], "PASS")
        self.assertEqual(result["general_status"], "FAIL")

    def test_missing_reactive_atom_still_runs_general(self):
        result = self.variant(lambda s: s[0]["A"][7].remove_atom("OD1", "\x00"))
        self.assertEqual(result["mechanism_status"], "ERROR")
        self.assertEqual(result["general_status"], "FAIL")

    def test_renaming_reordering_and_residue_renumbering(self):
        def change(s):
            for chain in s[0]:
                if chain.name == "C":
                    atoms = list(chain[0].clone())
                    for i in range(len(chain[0])-1, -1, -1):
                        del chain[0][i]
                    for atom in reversed(atoms):
                        chain[0].add_atom(atom)
                for i, residue in enumerate(chain):
                    residue.seqid = gemmi.SeqId(1000+i, " ")
                chain.name = "NEW_" + chain.name
        result = self.variant(change)
        self.assertEqual(result["joint_status"], "PASS")
        self.assertEqual(result["mechanism"]["site_assignment"]["protein_chain"], "NEW_A")

    def test_far_ligand(self):
        def change(s):
            for atom in s[0]["C"][0]:
                atom.pos.x += 200
        result = self.variant(change)
        self.assertEqual(result["mechanism_status"], "FAIL")
        self.assertIn("LIGAND_TOO_FAR_FROM_PROTEIN", result["general"]["failure_tags"])

    def test_unknown_names_need_explicit_map(self):
        def change(s):
            for i, atom in enumerate(s[0]["C"][0]):
                atom.name = f"X{i}"
        result = self.variant(change)
        self.assertEqual(result["mechanism_status"], "ERROR")
        result = self.variant(change, mapping={"substrate": {"atom_names": [f"X{i}" for i in range(6)]}})
        self.assertEqual(result["joint_status"], "PASS")

    def test_explicit_chemistry_mismatch_is_general_only(self):
        document = gemmi.cif.read_file(FIXTURE)
        block = document.sole_block()
        ids = list(block.find_values("_chem_comp.id"))
        smiles = block.find_values("_chem_comp.pdbx_smiles")
        smiles[ids.index("LIG_C")] = gemmi.cif.quote("CC")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"different_declared_graph.cif"
            document.write_file(str(path))
            result = evaluate(path, "1QQ5")
        self.assertEqual(result["mechanism_status"], "PASS")
        self.assertEqual(result["general_status"], "FAIL")

    def test_cannot_choose_better_unoccupied_site(self):
        def change(s):
            sub = {a.name: a for a in s[0]["C"][0]}
            center = np.array(sub["C2"].pos.tolist())
            direction = np.array(sub["CL1"].pos.tolist()) - center
            direction /= np.linalg.norm(direction)
            for atom in s[0]["A"][7]:
                if atom.name in {"OD1", "OD2"}:
                    atom.pos.x += 100
            for atom in s[0]["B"][7]:
                if atom.name == "OD1":
                    atom.pos = gemmi.Position(*(center-3*direction))
        result = self.variant(change, stage="mechanism")
        self.assertEqual(result["mechanism"]["site_assignment"]["protein_chain"], "A")
        self.assertEqual(result["mechanism_status"], "FAIL")
        self.assertTrue(any(c["tuple_status"] == "PASS" for c in result["mechanism"]["unoccupied_site_diagnostics"]))

    def test_cannot_mix_oxygen_distance_and_angle(self):
        def change(s):
            sub = {a.name: a for a in s[0]["C"][0]}
            center = np.array(sub["C2"].pos.tolist())
            direction = np.array(sub["CL1"].pos.tolist()) - center
            direction /= np.linalg.norm(direction)
            perpendicular = np.cross(direction, [1., 0., 0.])
            perpendicular /= np.linalg.norm(perpendicular)
            for atom in s[0]["A"][7]:
                if atom.name == "OD1":
                    atom.pos = gemmi.Position(*(center+3*perpendicular))
                elif atom.name == "OD2":
                    atom.pos = gemmi.Position(*(center-5*direction))
        result = self.variant(change, stage="mechanism")
        self.assertEqual(result["mechanism_status"], "FAIL")
        self.assertIn("REACTION_AXIS_JOINT_MISMATCH", result["mechanism"]["failure_tags"])

    def test_equivalent_oxygen_names_can_swap(self):
        def change(s):
            oxygens = {a.name: a for a in s[0]["A"][7] if a.name in {"OD1", "OD2"}}
            oxygens["OD1"].name = "OD2"
            oxygens["OD2"].name = "OD1"
        result = self.variant(change, stage="mechanism")
        self.assertEqual(result["mechanism_status"], "PASS")


@unittest.skipUnless(FIRST, "Set MCSA_1ZIO_TEST_CIF to the AF3 s0 fixture")
class FirstCaseTests(unittest.TestCase):
    def test_original_case_still_passes(self):
        result = evaluate(FIRST, "1ZIO")
        self.assertEqual(result["joint_status"], "PASS")

    def test_missing_mg_reports_two_separate_outcomes(self):
        s = gemmi.read_structure(FIRST)
        s[0].remove_chain("D")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"missing_mg.cif"
            s.make_mmcif_document().write_file(str(path))
            result = evaluate(path, "1ZIO")
        self.assertEqual(result["mechanism_status"], "ERROR")
        self.assertEqual(result["general_status"], "FAIL")

    def test_missing_structural_zinc_not_mechanism_failure(self):
        s = gemmi.read_structure(FIRST)
        s[0].remove_chain("E")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"missing_zn.cif"
            s.make_mmcif_document().write_file(str(path))
            result = evaluate(path, "1ZIO")
        self.assertEqual(result["mechanism_status"], "PASS")
        self.assertEqual(result["general_status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
