"""Six His122-to-ATP axes and same-Mg ATP tridentate contacts."""

from collections import Counter, defaultdict
import numpy as np

from ..general import atom_id, distance
from ..structure import InputProblem
from .adenylate import angle, inside, EPS


def combined_status(*statuses):
    return "ERROR" if "ERROR" in statuses else "FAIL" if "FAIL" in statuses else "PASS"


def evaluate_magnesium(context, case):
    rules = case["mechanism_criteria"]["magnesium"]
    limits = rules["distance_A"]
    ligands = context["ligands"]
    metals = context["metal_copies"].get("MG", [])
    anchor_indices = rules["assignment"]["phosphorus_indices"]
    anchors = {name: np.mean([atoms[i]["xyz"] for i in anchor_indices], axis=0)
               for name, atoms in ligands.items()}
    assigned = defaultdict(list)
    assignments, tags = [], []
    ambiguous_targets = set()
    if len(metals) != case["metals"]["MG"]:
        tags.append("MECHANISM_MG_COPY_COUNT_MISMATCH")
    metals = sorted(metals, key=lambda a: (a["chain"], a["residue_number"], a["insertion_code"], a["atom_name"]))
    costs_by_mg = [{name: float(np.linalg.norm(mg["xyz"] - anchor)) for name, anchor in anchors.items()}
                   for mg in metals]
    nearest_metals = {}
    for name in ligands:
        order = sorted(range(len(metals)), key=lambda i: costs_by_mg[i][name])
        nearest_metals[name] = [i for i in order if costs_by_mg[i][name] - costs_by_mg[order[0]][name]
                               <= rules["assignment"]["tie_tolerance_A"]] if order else []
    # Reciprocal nearest P-centroids avoid assigning a remote, unmatched Mg
    # to an already restored site. No scoring oxygen or pass cutoff is used.
    for metal_index, mg in enumerate(metals):
        costs = costs_by_mg[metal_index]
        ordered = sorted(costs, key=costs.get)
        margin = costs[ordered[1]] - costs[ordered[0]]
        ambiguous = margin <= rules["assignment"]["tie_tolerance_A"]
        preferred = ordered[0]
        if len(nearest_metals[preferred]) > 1 and metal_index in nearest_metals[preferred]:
            ambiguous = True
        target = preferred if not ambiguous and nearest_metals[preferred] == [metal_index] else None
        if ambiguous:
            ambiguous_targets.update(name for name in costs
                                     if costs[name] - costs[ordered[0]] <= rules["assignment"]["tie_tolerance_A"])
            tags.append("MG_ATP_ASSIGNMENT_AMBIGUOUS")
        elif target is not None:
            assigned[target].append(mg)
        else:
            tags.append("MG_NOT_MATCHED_TO_ATP_SITE")
        assignments.append({"metal": atom_id(mg), "ligand_instance": target,
                            "ligand_chain": ligands[target][0]["chain"] if target else None,
                            "nearest_ATP_instance": preferred,
                            "nearest_ATP_centroid_distance_A": costs[preferred],
                            "anchor_costs_A": costs, "margin_A": margin,
                            "status": "AMBIGUOUS" if ambiguous else "ASSIGNED" if target else "UNMATCHED",
                            "reciprocal_nearest": target is not None,
                            "oxygen_distances_or_pass_status_used": False})
    site_results = []
    for instance, atp in ligands.items():
        ions = assigned[instance]
        row_tags, groups = [], {}
        uncertain = instance in ambiguous_targets
        if not ions:
            row_tags.append("MG_NOT_UNAMBIGUOUSLY_ASSIGNED_TO_ATP" if uncertain else "MG_ATP_SITE_UNRECOVERED")
        elif len(ions) > 1:
            row_tags.append("MULTIPLE_MG_ASSIGNED_TO_ONE_ATP")
        if len(ions) == 1:
            mg = ions[0]
            for group in rules["groups"]:
                candidates = [{"oxygen_index": i, "oxygen": atom_id(atp[i]),
                               "distance_A": distance(mg, atp[i])}
                              for i in case["reaction_atoms"]["Mg_donor_indices"][group]]
                nearest = min(candidates, key=lambda row: row["distance_A"])
                d = nearest["distance_A"]
                group_tags = []
                if d < limits["min"] - EPS:
                    group_tags.append(f"MG_ATP_{group.upper()}_CONTACT_TOO_SHORT")
                if d > limits["max"] + EPS:
                    group_tags.append(f"MG_ATP_{group.upper()}_CONTACT_TOO_LONG")
                groups[group] = {"status": "PASS" if inside(d, limits) else "FAIL",
                                 "metal": atom_id(mg), "nearest": nearest,
                                 "all_donor_distances": candidates, "distance_window_A": limits,
                                 "margin_to_window_A": min(d-limits["min"], limits["max"]-d),
                                 "failure_tags": group_tags}
                row_tags.extend(group_tags)
        if uncertain:
            row_tags.append("MG_ATP_ASSIGNMENT_AMBIGUOUS")
        status = "ERROR" if uncertain else "FAIL" if row_tags else "PASS"
        site_results.append({"ligand_instance": instance, "ligand_chain": atp[0]["chain"],
                             "assigned_Mg_count": len(ions), "assigned_metals": [atom_id(mg) for mg in ions],
                             "nearest_Mg_centroid_distance_A_diagnostic": min((c[instance] for c in costs_by_mg), default=None),
                             "status": status, "failure_tags": sorted(set(row_tags)), "groups": groups})
        tags.extend(row_tags)
    tags = sorted(set(tags))
    return {"status": "ERROR" if ambiguous_targets else "FAIL" if tags else "PASS",
            "failure_tags": tags, "site_results": site_results, "assignments": assignments,
            "expected_Mg_count": case["metals"]["MG"], "observed_Mg_count": len(metals),
            "passing_ATP_count": sum(row["status"] == "PASS" for row in site_results),
            "distance_window_A": limits, "scope": "Same Mg and ATP alpha/beta/gamma nonbridging oxygen contacts; not a full water shell or a universal coordination mode."}


def evaluate_mechanism(context, case):
    roles, criteria = case["reaction_atoms"], case["mechanism_criteria"]
    limits = criteria["main_axis"]
    sites = {}
    for chain, residues in sorted(context["protein_chains"].items()):
        residue = residues[roles["protein_sequence_position_1based"] - 1]
        if residue["name"] != roles["protein_residue_name"]:
            raise InputProblem("MECHANISM_RESIDUE_MISMATCH", f"{chain}: expected His122")
        atoms = {a["atom_name"]: a for a in residue["atoms"]}
        for name in roles["anchor_names"] + [roles["nucleophile_name"]]:
            expected = "N" if name == roles["nucleophile_name"] else "C"
            if name not in atoms or atoms[name]["element"] != expected:
                raise InputProblem("MECHANISM_REQUIRED_ATOM_MISSING", f"{chain}: His122 {name}")
        sites[chain] = {"nucleophile": atoms[roles["nucleophile_name"]],
                        "anchor": np.mean([atoms[n]["xyz"] for n in roles["anchor_names"]], axis=0)}
    assignments, candidates, selected = [], [], []
    for instance, atp in context["ligands"].items():
        anchor = np.mean([atp[i]["xyz"] for i in roles["adenosine_indices"]], axis=0)
        costs = {chain: float(np.linalg.norm(anchor - site["anchor"])) for chain, site in sites.items()}
        ordered = sorted(costs, key=costs.get)
        margin = costs[ordered[1]] - costs[ordered[0]]
        ambiguous = margin <= criteria["site_assignment"]["tie_tolerance_A"]
        assigned = None if ambiguous else ordered[0]
        assignments.append({"ligand_instance": instance, "ligand_chain": atp[0]["chain"],
                            "protein_chain": assigned, "anchor_costs_A": costs, "margin_A": margin,
                            "status": "AMBIGUOUS" if ambiguous else "ASSIGNED",
                            "reactive_distance_or_angle_used": False})
        center, leaving = atp[roles["phosphorus_index"]], atp[roles["leaving_oxygen_index"]]
        for chain, site in sites.items():
            nu = site["nucleophile"]
            d, theta = distance(nu, center), angle(nu, center, leaving)
            tags = []
            if d < limits["distance_A"]["min"] - EPS:
                tags.append("REACTION_DISTANCE_TOO_SHORT_FOR_REACTANT_TASK")
            if d > limits["distance_A"]["max"] + EPS:
                tags.append("REACTION_DISTANCE_TOO_LONG")
            if not inside(theta, limits["angle_deg"]):
                tags.append("DEGENERATE_REACTION_AXIS" if theta is None else "ATTACK_DIRECTION_MISMATCH")
            row = {"protein_chain": chain, "ligand_instance": instance, "ligand_chain": atp[0]["chain"],
                   "atoms": [atom_id(a) for a in (nu, center, leaving)],
                   "distance_A": d, "angle_deg": theta, "Pgamma_LG_bond_A_diagnostic": distance(center, leaving),
                   "distance_status": "PASS" if inside(d, limits["distance_A"]) else "FAIL",
                   "angle_status": "PASS" if inside(theta, limits["angle_deg"]) else "FAIL",
                   "status": "FAIL" if tags else "PASS", "failure_tags": tags,
                   "assigned": chain == assigned}
            candidates.append(row)
            if row["assigned"]:
                selected.append(row)
    occupancy = Counter(a["protein_chain"] for a in assignments if a["protein_chain"] is not None)
    site_results = []
    for chain in sites:
        rows = [r for r in selected if r["protein_chain"] == chain]
        tags = sorted({tag for r in rows for tag in r["failure_tags"]})
        if not rows:
            tags.append("CATALYTIC_SITE_UNOCCUPIED")
        elif len(rows) > 1:
            tags.append("MULTIPLE_ATP_ASSIGNED_TO_ONE_SITE")
        site_results.append({"protein_chain": chain, "ATP_count": len(rows),
                             "status": "FAIL" if tags else "PASS", "failure_tags": tags,
                             "ligand_instances": [r["ligand_instance"] for r in rows]})
    tags = sorted({t for r in site_results for t in r["failure_tags"]})
    ambiguous = any(a["status"] == "AMBIGUOUS" for a in assignments)
    if ambiguous:
        tags.append("OCCUPIED_SITE_ASSIGNMENT_AMBIGUOUS")
    axis_status = "ERROR" if ambiguous else "FAIL" if tags else "PASS"
    axis_tags = tags.copy()
    axis_site_results = site_results
    magnesium = evaluate_magnesium(context, case)
    metal_sites = {r["ligand_instance"]: r for r in magnesium["site_results"]}
    site_results = []
    for axis_site in axis_site_results:
        instances = axis_site["ligand_instances"]
        if len(instances) == 1:
            metal_site = metal_sites[instances[0]]
            metal_status, metal_tags = metal_site["status"], metal_site["failure_tags"]
        else:
            metal_status, metal_tags = "FAIL", ["MG_ATP_SITE_CORRESPONDENCE_INCOMPLETE"]
        site_results.append({**axis_site, "axis_status": axis_site["status"], "mg_status": metal_status,
                             "status": combined_status(axis_site["status"], metal_status),
                             "failure_tags": sorted(set(axis_site["failure_tags"] + metal_tags))})
    tags = sorted(set(axis_tags + magnesium["failure_tags"]))
    mg_rows = []
    for mg in context["metal_copies"].get("MG", []):
        for instance, atp in context["ligands"].items():
            for i in roles["phosphate_oxygen_indices"]:
                mg_rows.append({"metal": atom_id(mg), "ligand_instance": instance,
                                "oxygen": atom_id(atp[i]), "oxygen_index": i,
                                "distance_A": distance(mg, atp[i]), "affects_pass": False})
    return {
        "criteria_id": criteria["criteria_id"],
        "status": combined_status(axis_status, magnesium["status"]),
        "axis_status": axis_status, "mg_status": magnesium["status"],
        "axis_failure_tags": axis_tags, "mg_failure_tags": magnesium["failure_tags"],
        "checks": {"reaction_axis": axis_status, "mg_atp_coordination": magnesium["status"]},
        "failure_tags": tags, "cutoffs": limits, "site_assignments": assignments,
        "site_results": site_results, "occupied_site_counts": dict(occupancy),
        "declared_site_count": len(sites), "passing_site_count": sum(s["status"] == "PASS" for s in site_results),
        "axis_site_results": axis_site_results,
        "axis_passing_site_count": sum(s["status"] == "PASS" for s in axis_site_results),
        "mg_passing_site_count": sum(s["mg_status"] == "PASS" for s in site_results),
        "magnesium": magnesium,
        "selected_axes": selected, "all_candidate_axes": candidates,
        "metal_diagnostics": {"observed_Mg": len(context["metal_copies"].get("MG", [])),
                              "phosphate_oxygen_distances": mg_rows, "affects_pass": False},
        "scope": "Six-site His122 ND1 -> ATP Pgamma -> beta-gamma bridge O approach AND same-Mg ATP alpha/beta/gamma contacts for proposal 1 step 1. Axis and Mg outcomes are retained separately. No second half-reaction, protonation or complete water-shell claim. General chemistry checks are independent.",
    }
