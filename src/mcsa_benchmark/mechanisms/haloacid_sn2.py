"""Asp8 same-oxygen backside approach. No stereo or clash verdicts here."""

import numpy as np

from ..general import atom_id, distance
from ..structure import InputProblem
from .adenylate import angle, inside


def evaluate_mechanism(context, case):
    role = case["reaction_atoms"]
    substrate = context["ligands"]["substrate"]
    center = substrate[role["substrate_center_index"]]
    leaving = substrate[role["substrate_leaving_index"]]
    ligand_anchor = np.mean([substrate[i]["xyz"] for i in role["substrate_nonreactive_indices"]], axis=0)
    sites, costs = {}, {}
    for chain, residues in context["protein_chains"].items():
        residue = residues[role["protein_sequence_position_1based"] - 1]
        atoms = {a["atom_name"]: a for a in residue["atoms"]}
        if residue["name"] != role["protein_residue_name"]:
            raise InputProblem("MECHANISM_RESIDUE_MISMATCH", f"{chain}: expected Asp at sequence position 8")
        for name in role["anchor_names"] + role["nucleophile_names"]:
            expected = "O" if name.startswith("O") else "C"
            if name not in atoms or atoms[name]["element"] != expected:
                raise InputProblem("MECHANISM_REQUIRED_ATOM_MISSING", f"{chain}: Asp8 {name}")
        anchor = np.mean([atoms[name]["xyz"] for name in role["anchor_names"]], axis=0)
        costs[chain] = float(np.linalg.norm(ligand_anchor - anchor))
        sites[chain] = atoms
    ranked = sorted(costs, key=lambda chain: costs[chain])
    margin = costs[ranked[1]] - costs[ranked[0]] if len(ranked) > 1 else None
    tolerance = case["mechanism_criteria"]["occupied_site_assignment"]["historical_tie_tolerance_A"]
    if margin is not None and margin <= tolerance:
        raise InputProblem("OCCUPIED_SITE_ASSIGNMENT_AMBIGUOUS", f"Anchor margin {margin} A is not unique.")
    occupied = ranked[0]
    rules = case["mechanism_criteria"]["main_axis"]
    candidates, unoccupied = [], []
    for chain, atoms in sites.items():
        for name in role["nucleophile_names"]:
            oxygen = atoms[name]
            d, theta = distance(oxygen, center), angle(oxygen, center, leaving)
            row = {"protein_chain": chain, "oxygen_name": name,
                   "atoms": [atom_id(a) for a in (oxygen, center, leaving)],
                   "distance_A": d, "angle_deg": theta,
                   "distance_status": "PASS" if inside(d, rules["distance_A"]) else "FAIL",
                   "angle_status": "PASS" if inside(theta, rules["angle_deg"]) else "FAIL"}
            row["tuple_status"] = "PASS" if row["distance_status"] == row["angle_status"] == "PASS" else "FAIL"
            (candidates if chain == occupied else unoccupied).append(row)
    passing = [c for c in candidates if c["tuple_status"] == "PASS"]
    tags = []
    if not passing:
        if all(c["distance_A"] > rules["distance_A"]["max"] + 1e-9 for c in candidates):
            tags.append("REACTION_DISTANCE_TOO_LONG")
        if all(c["distance_A"] < rules["distance_A"]["min"] - 1e-9 for c in candidates):
            tags.append("REACTION_DISTANCE_TOO_SHORT_FOR_REACTANT_TASK")
        if not any(c["angle_status"] == "PASS" for c in candidates):
            tags.append("ATTACK_DIRECTION_MISMATCH")
        tags.append("REACTION_AXIS_JOINT_MISMATCH")
    return {
        "criteria_id": case["mechanism_criteria"]["criteria_id"],
        "status": "PASS" if passing else "FAIL", "failure_tags": tags,
        "candidates": candidates, "passing_candidates": passing,
        "unoccupied_site_diagnostics": unoccupied,
        "site_assignment": {"protein_chain": occupied, "costs_A": costs, "margin_A": margin,
                            "reactive_distance_used": False, "substrate_anchor_indices": role["substrate_nonreactive_indices"]},
        "original_C_Cl_bond_A_diagnostic": distance(center, leaving),
        "scope": "Only the selected first-step O-C-Cl geometry. Input stereochemistry, bond integrity, clashes and volume are evaluated independently by general checks. No requirement for a second occupied site or attacking water.",
        "cutoffs": rules,
    }
