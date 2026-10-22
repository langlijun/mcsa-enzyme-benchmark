"""Case-01 V1 reaction axis and Mg-ATP checks, without historical verdicts."""

import math
import numpy as np

from ..general import atom_id, distance
from ..structure import InputProblem

EPS = 1e-9


def angle(a, b, c):
    u, v = a["xyz"] - b["xyz"], c["xyz"] - b["xyz"]
    denominator = np.linalg.norm(u) * np.linalg.norm(v)
    if denominator == 0:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / denominator, -1, 1))))


def inside(value, limits):
    return value is not None and math.isfinite(value) and limits["min"] - EPS <= value <= limits["max"] + EPS


def evaluate_mechanism(context, case):
    ligands, metals = context["ligands"], context["metals"]
    if "Mg" not in metals:
        raise InputProblem("MECHANISM_MAGNESIUM_MISSING_OR_AMBIGUOUS", "One unambiguous Mg is required for the mechanism contacts.")
    atp, amp, mg = ligands["ATP"], ligands["AMP"], metals["Mg"]
    ids, criteria = case["reaction_atoms"], case["mechanism_criteria"]
    cut = criteria["cutoffs"]
    axes = [{
        "AMP_oxygen_index": i, "AMP_oxygen_name": amp[i]["atom_name"],
        "distance_A": distance(amp[i], atp[ids["ATP_Pgamma"]]),
        "angle_deg": angle(amp[i], atp[ids["ATP_Pgamma"]], atp[ids["ATP_LG"]]),
    } for i in ids["AMP_Nu"]]
    too_short = any(x["distance_A"] < cut["reaction_distance_A"]["min"] - EPS for x in axes)
    good = [x for x in axes if inside(x["distance_A"], cut["reaction_distance_A"])
            and inside(x["angle_deg"], cut["attack_angle_deg"])]
    tags = []
    if too_short:
        tags.append("REACTION_DISTANCE_TOO_SHORT_FOR_REACTANT_TASK")
    if all(x["distance_A"] > cut["reaction_distance_A"]["max"] + EPS for x in axes):
        tags.append("REACTION_DISTANCE_TOO_LONG")
    if not any(inside(x["angle_deg"], cut["attack_angle_deg"]) for x in axes):
        tags.append("ATTACK_DIRECTION_MISMATCH")
    if not good and not too_short:
        tags.append("REACTION_AXIS_JOINT_MISMATCH")
    if any(x["angle_deg"] is None for x in axes):
        tags.append("DEGENERATE_REACTION_AXIS")
    checks = {"reaction_axis": bool(good) and not too_short}
    measurements = {"axes": axes}
    for name in ("beta", "gamma"):
        values = [{"ATP_oxygen_index": i, "ATP_oxygen_name": atp[i]["atom_name"],
                   "distance_A": distance(mg, atp[i])} for i in ids[f"ATP_{name}_O"]]
        measurements[f"Mg_{name}"] = values
        witness = min(values, key=lambda v: v["distance_A"])
        limits = cut[f"Mg_{name}_distance_A"]
        checks[f"Mg_{name}"] = inside(witness["distance_A"], limits)
        if not checks[f"Mg_{name}"]:
            direction = "SHORT" if witness["distance_A"] < limits["min"] else "LONG"
            tags.append(f"MG_ATP_{name.upper()}_CONTACT_TOO_{direction}")
    measurements["Mg_AMP_diagnostic"] = [
        {"AMP_oxygen_index": i, "distance_A": distance(mg, amp[i])} for i in ids["AMP_Nu"]
    ]
    measurements["ATP_old_Pgamma_LG_bond_A"] = distance(atp[ids["ATP_Pgamma"]], atp[ids["ATP_LG"]])
    witness = good[0] if good else min(axes, key=lambda v: v["distance_A"])
    return {
        "criteria_id": criteria["criteria_id"], "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": {k: "PASS" if v else "FAIL" for k, v in checks.items()},
        "failure_tags": tags, "axis_witness": witness, "measurements": measurements,
        "metal_atom": atom_id(mg),
        "scope": "Preorganized ATP/AMP axis and Mg-to-ATP beta/gamma oxygen contacts; not a full water shell, Zn site, reaction rate, or entire mechanism.",
    }
