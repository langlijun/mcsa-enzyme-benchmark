"""Fresh geometry checks against the declared input chemical graphs."""

import numpy as np
from itertools import combinations
from rdkit import Chem
from scipy.spatial import cKDTree

from . import _kernels as k
from .structure import check_declared_chemistry, InputProblem


def atom_id(atom):
    return {key: value for key, value in atom.items() if key != "xyz"}


def distance(first, second):
    return float(np.linalg.norm(first["xyz"] - second["xyz"]))


def atom_key(atom):
    return tuple(atom[k] for k in ('chain', 'residue_name', 'residue_number', 'insertion_code', 'atom_name'))


def reaction_center_exemptions(context, case):
    policy = case.get('reaction_pair_policy')
    if not policy:
        return {}
    result = {}
    for first in policy['first_roles']:
        for second in policy['second_roles']:
            i, j = policy['first_index'], policy['second_index']
            a, b = context['ligands'][first], context['ligands'][second]
            result[frozenset([atom_key(a[i]), atom_key(b[j])])] = 'FORMING_PAIR'
            if policy['include_prospective_1_3']:
                ma = Chem.MolFromSmiles(case['ligands'][first]['smiles'])
                mb = Chem.MolFromSmiles(case['ligands'][second]['smiles'])
                for neighbor in ma.GetAtomWithIdx(i).GetNeighbors():
                    result[frozenset([atom_key(a[neighbor.GetIdx()]), atom_key(b[j])])] = 'PROSPECTIVE_1_3_PAIR'
                for neighbor in mb.GetAtomWithIdx(j).GetNeighbors():
                    result[frozenset([atom_key(a[i]), atom_key(b[neighbor.GetIdx()])])] = 'PROSPECTIVE_1_3_PAIR'
    return result


def short_pairs(first, second, category):
    first = [atom for atom in first if atom["element"] not in k.METAL_ELEMENTS]
    second = [atom for atom in second if atom["element"] not in k.METAL_ELEMENTS]
    hits = []
    if not first or not second:
        return hits
    tree = cKDTree(np.array([a["xyz"] for a in second]))
    for a in first:
        for index in sorted(tree.query_ball_point(a["xyz"], 4.5)):
            b = second[index]
            d = distance(a, b)
            radius_sum = k.vdw_radius(a["element"]) + k.vdw_radius(b["element"])
            limit = 0.75 * radius_sum
            if d <= 4.5 and d < limit:
                hits.append({
                    "category": category, "first": atom_id(a), "second": atom_id(b),
                    "distance_A": d, "lower_bound_A": limit,
                    "vdw_ratio": d / radius_sum, "below_bound_A": limit - d,
                    "within_0p01_A_of_boundary": limit - d <= 0.01,
                })
    return hits


def metal_coordination_check(context, case):
    """Retain legacy distance evidence without contributing a general hard gate."""
    policy = case.get('metal_distance_diagnostic_policy') or case.get('general_metal_policy')
    if not policy:
        return {'status': 'NOT_APPLICABLE', 'pairs': [], 'affects_general_pass': False}
    metals = [atom for element, atoms in context['metal_copies'].items()
              if element.upper() == policy['element'].upper() for atom in atoms]
    oxygen = [a for a in context['protein_atoms'] + [a for atoms in context['ligands'].values() for a in atoms]
              if a['element'] == policy['donor_element']]
    lower = policy['minimum_distance_A']
    if not np.isfinite(lower) or lower <= 0:
        raise ValueError('Invalid metal coordination lower bound')
    pairs = []
    if oxygen:
        tree = cKDTree(np.array([a['xyz'] for a in oxygen]))
        for metal in metals:
            for index in sorted(tree.query_ball_point(metal['xyz'], lower)):
                donor = oxygen[index]
                d = distance(metal, donor)
                if d < lower - 1e-9:
                    pairs.append({'tag': policy.get('failure_tag', 'METAL_DISTANCE_BELOW_DIAGNOSTIC_REFERENCE'), 'metal': atom_id(metal),
                                  'donor': atom_id(donor), 'distance_A': d,
                                  'minimum_distance_A': lower, 'below_bound_A': lower-d,
                                  'affects_general_pass': False})
    return {'status': 'NOT_SCORED',
            'diagnostic_status': 'BELOW_LEGACY_BOUND' if pairs else 'NO_SHORT_CONTACT' if metals else 'NO_OBSERVED_METAL',
            'affects_general_pass': False,
            'policy_id': policy['id'], 'minimum_distance_A': lower,
            'method': 'Legacy metal-donor distance diagnostic only; no general distance lower bound',
            'observed_metals': len(metals), 'pairs': pairs}


CHECK_FIELDS = {
    "bond_length_valid": "BOND_LENGTH_OUT_OF_RANGE",
    "bond_angle_valid": "BOND_ANGLE_GEOMETRY_OUT_OF_RANGE",
    "internal_nonbonded_valid": "INTRALIGAND_SHORT_CONTACT",
    "tetrahedral_stereochemistry_valid": "INPUT_STEREOCHEMISTRY_MISMATCH",
    "double_bond_stereochemistry_valid": "INPUT_EZ_MISMATCH",
    "aromatic_planarity_valid": "AROMATIC_RING_NONPLANAR",
    "double_bond_planarity_valid": "DOUBLE_BOND_NONPLANAR",
}


def evaluate_general(context, case, path):
    protein, ligands = context["protein_atoms"], context["ligands"]
    component_metrics, contacts, failures, diagnostics = {}, [], [], []
    checks = {}
    exemptions = reaction_center_exemptions(context, case)
    reactive_contacts = []
    excluded_metals = {element.upper() for element in case.get('general_excluded_metals', [])}
    excluded_input_issues = [issue for issue in context['general_input_issues']
                            if issue.get('tag') == 'METAL_COMPOSITION_OR_COPY_MISMATCH'
                            and issue.get('element', '').upper() in excluded_metals]
    failures.extend(issue for issue in context['general_input_issues'] if issue not in excluded_input_issues)
    checks["protein_and_component_integrity"] = "FAIL" if failures else "PASS"
    chemistry_evidence = {}
    try:
        chemistry_evidence = check_declared_chemistry(path, ligands, case)
        has_explicit_metadata = any("MATCHES_INPUT" in str(value) for value in chemistry_evidence.values())
        checks["declared_chemistry_when_provided"] = "PASS" if has_explicit_metadata else "NOT_APPLICABLE"
    except InputProblem as exc:
        failures.append({"tag": exc.code, "detail": exc.detail})
        checks["declared_chemistry_when_provided"] = "FAIL"
    for role, atoms in ligands.items():
        molecule = Chem.RemoveHs(Chem.MolFromSmiles(case["ligands"][role]["smiles"]))
        graph = {
            "mol": molecule, "source_chain": role,
            "canonical_smiles": Chem.MolToSmiles(molecule, isomericSmiles=True),
            "excluded_stereochemistry_elements": case.get("case_general_requirements", {}).get("excluded_stereochemistry_elements", []),
        }
        mapping = {i: i for i in range(len(atoms))}
        component = {"atoms": atoms, "chain": atoms[0]["chain"],
                     "comp": atoms[0]["residue_name"], "heavy_atom_count": len(atoms)}
        inferred = k.inferred_adjacency(atoms)
        metrics = k.component_chemistry(component, graph, mapping, "DECLARED_ATOM_NAMES", inferred)
        metrics['input_stereochemistry_evidence'] = {
            'method': 'RDKit AssignStereochemistry; coordinates labeled after removing input stereo',
            'affects_pass': False,
            'scope': 'Shared-kernel evidence only; case-specific stereo review remains authoritative when configured.',
            'centers': [
                {'input_atom_index': row['index'], 'atom': atom_id(atoms[row['index']]),
                 'expected_CIP': row['expected'], 'coordinate_CIP': row['observed'],
                 'matches': row['match'], 'counted_by_shared_kernel': row['counted']}
                for row in metrics['tetrahedral_center_records']
            ],
        }
        if case.get("general_extension") == "gale_reactant":
            from .gale_general import review_stereochemistry
            metrics = review_stereochemistry(role, atoms, molecule, metrics, case)
        elif case.get("general_extension") == "ldh_reactant":
            from .ldh_general import review_stereochemistry
            metrics = review_stereochemistry(role, atoms, molecule, metrics, case)
        component_metrics[role] = metrics
        for field, label in CHECK_FIELDS.items():
            value = metrics[field]
            checks[f"{role}.{field}"] = (
                "NOT_APPLICABLE" if value == "NA" else "PASS" if value else "FAIL"
            )
            if value is False:
                failures.append({"tag": label, "component": role, "metric": field})
        # Proximity-perceived extra bonds are not chemical connectivity evidence.
        expected_edges = {
            tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in molecule.GetBonds()
        }
        extra = {(i, j) for i in range(len(inferred)) for j in inferred[i] if i < j} - expected_edges
        if extra:
            diagnostics.append({
                "component": role, "tag": "DISTANCE_INFERRED_EXTRA_BONDS_DIAGNOSTIC_ONLY",
                "affects_pass": False,
                "pairs": [{"indices": [i, j], "first": atom_id(atoms[i]),
                           "second": atom_id(atoms[j]), "distance_A": distance(atoms[i], atoms[j])}
                          for i, j in sorted(extra)],
            })
        contact = k.intergroup_contact(atoms, protein)
        overlap = k.protein_volume_overlap(atoms, protein)
        contacts.append({"component": role, "partner": "protein", **contact, **overlap})
        for key, passed, tag in (
            ("protein_short_contact", contact["clash_pair_count_ratio_lt_0.75"] == 0, "F05_NONCOVALENT_SHORT_CONTACT"),
            ("protein_volume_overlap", overlap["protein_volume_overlap_valid"], "PROTEIN_LIGAND_VOLUME_OVERLAP"),
            ("within_5A_of_protein", contact["within_5A"], "LIGAND_TOO_FAR_FROM_PROTEIN"),
        ):
            checks[f"{role}.{key}"] = "PASS" if passed else "FAIL"
            if not passed:
                failures.append({"tag": tag, "component": role})
    pairs = []
    for role, atoms in ligands.items():
        pairs.extend(short_pairs(atoms, protein, f"{role}_PROTEIN"))
    for (role1, first), (role2, second) in combinations(ligands.items(), 2):
        interligand = k.intergroup_contact(first, second)
        pair_name = f"{role1}_{role2}"
        hits = short_pairs(first, second, pair_name)
        nonreactive = []
        for hit in hits:
            relationship = exemptions.get(frozenset([atom_key(hit['first']), atom_key(hit['second'])]))
            if relationship:
                reactive_contacts.append({**hit, 'relationship':relationship, 'affects_general_pass':False})
            else:
                nonreactive.append(hit)
        checks[f"{pair_name}.short_contact"] = "FAIL" if nonreactive else "PASS"
        if checks[f"{pair_name}.short_contact"] == "FAIL":
            failures.append({"tag": "F05_NONCOVALENT_SHORT_CONTACT", "component": pair_name})
        contacts.append({"component": role1, "partner": role2, **interligand,
                         'nonreactive_clash_pair_count':len(nonreactive), 'raw_contact_statistics_retained':True})
        pairs.extend(nonreactive)
    metal_check = metal_coordination_check(context, case)
    if case.get('metal_distance_diagnostic_policy') or case.get('general_metal_policy'):
        checks['metal_coordination_short_contacts'] = 'NOT_APPLICABLE'
        diagnostics.append({'tag': 'METAL_DISTANCE_DIAGNOSTIC_ONLY',
                            'affects_pass': False, 'evidence': metal_check})
    extension = None
    if case.get("general_extension") == "sam_reactant":
        from .sam_general import evaluate_sam_checks
        extension = evaluate_sam_checks(context, case)
    elif case.get("general_extension") == "ugm_reactant":
        from .ugm_general import evaluate_ugm_checks
        extension = evaluate_ugm_checks(context, case)
    elif case.get("general_extension") == "plp_internal_aldimine":
        from .plp_general import evaluate_plp_checks
        extension = evaluate_plp_checks(context, case, path)
    elif case.get('general_extension') == 'aanat_reactant':
        from .aanat_general import evaluate_aanat_checks
        extension = evaluate_aanat_checks(context, case, path)
    if extension is not None:
        checks.update(extension['checks'])
        failures.extend(extension['failures'])
        diagnostics.extend(extension['diagnostics'])
    result = {
        "criteria_id": case["general_policy"]["id"],
        "status": "FAIL" if failures else "PASS",
        "checks": checks, "failure_tags": sorted({item["tag"] for item in failures}),
        "failures": failures, "components": component_metrics,
        "contacts": contacts, "short_contact_atom_pairs": pairs,
        "diagnostics": diagnostics,
        "metal_coordination_check": metal_check,
        "reaction_center_short_contacts": reactive_contacts,
        "reaction_pair_policy": case.get('reaction_pair_policy'),
        "strict_unexempted_general_status": "FAIL" if failures or reactive_contacts else "PASS",
        "declared_chemistry_evidence": chemistry_evidence,
        "chemical_graph_policy": "Input graph defines atom chemistry; coordinates test its geometry and stereochemistry. Distance-perceived extra edges are not treated as predicted covalent bonds.",
        "scope": "Declared small-molecule geometry, input stereochemistry, protein-ligand and interligand contacts. No mechanism classification, energy or protein RMSD. Metal distance and coordination geometry are diagnostic only, never general gates. Metal count/composition remains checked unless explicitly excluded by the case. Water is excluded.",
    }
    if extension is not None:
        result['case_specific_checks'] = extension
    if case.get('general_extension') == 'plp_internal_aldimine':
        result['scope'] += ' LLP internal geometry and adjacent peptide links are checked with a covalently capped local graph; ADC protein contacts include LLP. No whole-protein conformational validation or LLP-as-free-ligand volume test.'
    if excluded_metals:
        result['excluded_metals'] = sorted(excluded_metals)
        result['excluded_metal_input_diagnostics'] = excluded_input_issues
        result['scope'] += ' Explicitly excluded metal counts: ' + ', '.join(sorted(excluded_metals)) + '.'
    return result
