"""Numerical kernels extracted unchanged from the project basic evaluator.

No historical inputs, labels, paths, or graph-matching fallbacks are used here.
The caller supplies a declared input graph and independently named atom mapping.
Distance-inferred connectivity is diagnostic only, never a binary validity gate.
"""
import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem.rdShapeHelpers import ShapeTverskyIndex
from rdkit.Geometry import Point3D
from scipy.spatial.distance import cdist

PERIODIC_TABLE = Chem.GetPeriodicTable()
METAL_ELEMENTS = {"MG", "ZN"}

def covalent_radius(element):
    try:
        return float(PERIODIC_TABLE.GetRcovalent(PERIODIC_TABLE.GetAtomicNumber(element.title())))
    except Exception:
        return 0.77


def vdw_radius(element):
    try:
        return float(PERIODIC_TABLE.GetRvdw(PERIODIC_TABLE.GetAtomicNumber(element.title())))
    except Exception:
        return 1.70


def inferred_adjacency(atoms, scale=1.35):
    adjacency = [set() for _ in atoms]
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            distance = float(np.linalg.norm(atoms[i]["xyz"] - atoms[j]["xyz"]))
            upper = scale * (covalent_radius(atoms[i]["element"]) + covalent_radius(atoms[j]["element"]))
            if 0.4 < distance <= upper:
                adjacency[i].add(j)
                adjacency[j].add(i)
    return adjacency


def molecule_adjacency(molecule):
    adjacency = [set() for _ in range(molecule.GetNumAtoms())]
    for bond in molecule.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[i].add(j)
        adjacency[j].add(i)
    return adjacency


def plane_max_deviation(points):
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return None
    centered = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    return float(np.max(np.abs(centered.dot(normal))))


def angle_degrees(first, center, third):
    first_vector = np.asarray(first, dtype=float) - np.asarray(center, dtype=float)
    third_vector = np.asarray(third, dtype=float) - np.asarray(center, dtype=float)
    denominator = float(np.linalg.norm(first_vector) * np.linalg.norm(third_vector))
    if denominator == 0.0:
        return float("nan")
    cosine = float(np.dot(first_vector, third_vector) / denominator)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def atom_cloud_molecule(atoms):
    editable = Chem.RWMol()
    for atom in atoms:
        editable.AddAtom(Chem.Atom(atom["element"].title()))
    molecule = editable.GetMol()
    conformer = Chem.Conformer(len(atoms))
    for index, atom in enumerate(atoms):
        xyz = atom["xyz"]
        conformer.SetAtomPosition(index, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    molecule.AddConformer(conformer)
    return molecule


def protein_volume_overlap(ligand_atoms, protein_atoms, cutoff=0.075, vdw_scale=0.8):
    if not ligand_atoms or not protein_atoms:
        return {
            "protein_volume_overlap_fraction": "UNRESOLVED",
            "protein_volume_overlap_valid": "UNRESOLVED",
        }
    ligand_xyz = np.asarray([atom["xyz"] for atom in ligand_atoms], dtype=float)
    protein_xyz = np.asarray([atom["xyz"] for atom in protein_atoms], dtype=float)
    distances = cdist(ligand_xyz, protein_xyz)
    local_mask = distances.min(axis=0) <= 6.0 * vdw_scale
    if not np.any(local_mask):
        overlap = 0.0
    else:
        ligand = atom_cloud_molecule(ligand_atoms)
        local_protein = atom_cloud_molecule(
            [atom for atom, keep in zip(protein_atoms, local_mask) if keep]
        )
        overlap = float(
            ShapeTverskyIndex(
                ligand, local_protein, alpha=1.0, beta=0.0,
                vdwScale=vdw_scale, ignoreHs=True,
            )
        )
    return {
        "protein_volume_overlap_fraction": overlap,
        "protein_volume_overlap_valid": overlap <= cutoff,
    }


def component_chemistry(component, graph_def, mapping, mapping_status, strict_adj):
    molecule = graph_def["mol"]
    atoms = component["atoms"]
    query_adj = molecule_adjacency(molecule)
    target_to_query = {target: query for query, target in mapping.items()}
    inferred_edges = {
        tuple(sorted((target_to_query[i], target_to_query[j])))
        for i in range(len(strict_adj)) for j in strict_adj[i]
        if i < j and i in target_to_query and j in target_to_query
    }
    expected_edges = {
        tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        for bond in molecule.GetBonds()
    }
    missing_edges = expected_edges - inferred_edges
    unexpected_edges = inferred_edges - expected_edges

    bounds = rdDistGeom.GetMoleculeBoundsMatrix(molecule)
    bond_outliers = 0
    worst_bond_ratio = 0.0
    for bond in molecule.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        distance = float(np.linalg.norm(atoms[mapping[i]]["xyz"] - atoms[mapping[j]]["xyz"]))
        low = float(bounds[max(i, j), min(i, j)])
        high = float(bounds[min(i, j), max(i, j)])
        if distance < 0.75 * low or distance > 1.25 * high:
            bond_outliers += 1
        ratio = max((0.75 * low) / distance if distance else float("inf"), distance / (1.25 * high) if high else float("inf"))
        worst_bond_ratio = max(worst_bond_ratio, ratio)

    angle_triplets = []
    angle_end_pairs = set()
    for center in range(molecule.GetNumAtoms()):
        neighbors = sorted(query_adj[center])
        for first_position in range(len(neighbors)):
            for third_position in range(first_position + 1, len(neighbors)):
                first = neighbors[first_position]
                third = neighbors[third_position]
                angle_triplets.append((first, center, third))
                angle_end_pairs.add(tuple(sorted((first, third))))
    angle_outliers = 0
    worst_angle_tolerance_ratio = 0.0
    minimum_angle = float("inf")
    maximum_angle = float("-inf")
    for first, center, third in angle_triplets:
        outer_distance = float(
            np.linalg.norm(atoms[mapping[first]]["xyz"] - atoms[mapping[third]]["xyz"])
        )
        low = float(bounds[max(first, third), min(first, third)])
        high = float(bounds[min(first, third), max(first, third)])
        below = outer_distance < 0.75 * low
        above = outer_distance > 1.25 * high
        if below or above:
            angle_outliers += 1
        ratio = max(
            (0.75 * low) / outer_distance if outer_distance else float("inf"),
            outer_distance / (1.25 * high) if high else float("inf"),
        )
        worst_angle_tolerance_ratio = max(worst_angle_tolerance_ratio, ratio)
        observed_angle = angle_degrees(
            atoms[mapping[first]]["xyz"], atoms[mapping[center]]["xyz"],
            atoms[mapping[third]]["xyz"],
        )
        if math.isfinite(observed_angle):
            minimum_angle = min(minimum_angle, observed_angle)
            maximum_angle = max(maximum_angle, observed_angle)

    internal_pairs = internal_outliers = 0
    worst_internal_ratio = float("inf")
    for i in range(molecule.GetNumAtoms()):
        for j in range(i + 1, molecule.GetNumAtoms()):
            if j in query_adj[i] or tuple(sorted((i, j))) in angle_end_pairs:
                continue
            low = float(bounds[j, i])
            if low <= 0:
                continue
            internal_pairs += 1
            distance = float(np.linalg.norm(atoms[mapping[i]]["xyz"] - atoms[mapping[j]]["xyz"]))
            ratio = distance / low
            worst_internal_ratio = min(worst_internal_ratio, ratio)
            if ratio < 0.70:
                internal_outliers += 1

    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for query_index in range(molecule.GetNumAtoms()):
        xyz = atoms[mapping[query_index]]["xyz"]
        conformer.SetAtomPosition(query_index, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    expected_mol = Chem.Mol(molecule)
    Chem.AssignStereochemistry(expected_mol, force=True, cleanIt=True)
    expected_centers = {
        atom.GetIdx(): atom.GetProp("_CIPCode")
        for atom in expected_mol.GetAtoms() if atom.HasProp("_CIPCode")
    }
    observed_mol = Chem.Mol(molecule)
    observed_mol.RemoveAllConformers()
    observed_mol.AddConformer(conformer)
    Chem.RemoveStereochemistry(observed_mol)
    Chem.AssignAtomChiralTagsFromStructure(observed_mol, confId=0, replaceExistingTags=True)
    Chem.AssignStereochemistryFrom3D(observed_mol, confId=0, replaceExistingTags=True)
    Chem.AssignStereochemistry(observed_mol, force=True, cleanIt=True)
    observed_centers = {
        atom.GetIdx(): atom.GetProp("_CIPCode")
        for atom in observed_mol.GetAtoms() if atom.HasProp("_CIPCode")
    }
    stereo_matches = sum(observed_centers.get(index) == code for index, code in expected_centers.items())
    excluded_elements = set(graph_def.get("excluded_stereochemistry_elements", []))
    center_records = [
        {"index": index, "element": molecule.GetAtomWithIdx(index).GetSymbol(),
         "expected": code, "observed": observed_centers.get(index, "UNASSIGNED"),
         "match": observed_centers.get(index) == code,
         "counted": molecule.GetAtomWithIdx(index).GetSymbol() not in excluded_elements}
        for index, code in expected_centers.items()
    ]
    counted_centers = [row for row in center_records if row["counted"]]
    specified_double_bonds = {}
    observed_double_bonds = {}
    for bond in expected_mol.GetBonds():
        if bond.GetStereo() in {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ}:
            key = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
            specified_double_bonds[key] = str(bond.GetStereo())
    for bond in observed_mol.GetBonds():
        if bond.GetStereo() in {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ}:
            key = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
            observed_double_bonds[key] = str(bond.GetStereo())
    matching_double_bonds = sum(
        observed_double_bonds.get(key) == code for key, code in specified_double_bonds.items()
    )

    aromatic_deviations = []
    for ring in Chem.GetSymmSSSR(molecule):
        indices = list(ring)
        if len(indices) not in {5, 6} or not all(molecule.GetAtomWithIdx(i).GetIsAromatic() for i in indices):
            continue
        aromatic_deviations.append(plane_max_deviation([atoms[mapping[i]]["xyz"] for i in indices]))

    double_deviations = []
    for bond in molecule.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE or bond.IsInRing():
            continue
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if molecule.GetAtomWithIdx(i).GetSymbol() != "C" or molecule.GetAtomWithIdx(j).GetSymbol() != "C":
            continue
        group = sorted({i, j} | query_adj[i] | query_adj[j])
        if len(group) >= 4:
            double_deviations.append(plane_max_deviation([atoms[mapping[index]]["xyz"] for index in group]))

    return {
        "chain": component["chain"],
        "comp_id": component["comp"],
        "heavy_atom_count": component["heavy_atom_count"],
        "graph_source_chain": graph_def["source_chain"],
        "canonical_smiles": graph_def["canonical_smiles"],
        "atom_mapping_status": mapping_status,
        "input_bond_count": len(expected_edges),
        "missing_inferred_bond_count": len(missing_edges),
        "unexpected_inferred_bond_count": len(unexpected_edges),
        "connectivity_geometry_consistent": len(missing_edges) == 0 and len(unexpected_edges) == 0,
        "bond_length_outlier_count": bond_outliers,
        "bond_length_valid": bond_outliers == 0,
        "worst_bond_tolerance_ratio": worst_bond_ratio if math.isfinite(worst_bond_ratio) else "UNBOUNDED_ZERO_DISTANCE",
        "input_bond_angle_count": len(angle_triplets),
        "bond_angle_outlier_count": angle_outliers,
        "bond_angle_valid": angle_outliers == 0,
        "worst_bond_angle_tolerance_ratio": worst_angle_tolerance_ratio if math.isfinite(worst_angle_tolerance_ratio) else "UNBOUNDED_ZERO_DISTANCE",
        "minimum_observed_bond_angle_degrees": minimum_angle if math.isfinite(minimum_angle) else "",
        "maximum_observed_bond_angle_degrees": maximum_angle if math.isfinite(maximum_angle) else "",
        "internal_nonbonded_pair_count": internal_pairs,
        "internal_nonbonded_outlier_count": internal_outliers,
        "internal_nonbonded_valid": internal_outliers == 0,
        "minimum_internal_distance_to_DG_lower_bound_ratio": worst_internal_ratio if internal_pairs else "",
        "specified_tetrahedral_center_count": len(expected_centers),
        "matching_tetrahedral_center_count": stereo_matches,
        "tetrahedral_stereochemistry_valid": all(row["match"] for row in counted_centers) if counted_centers else "NA",
        "counted_tetrahedral_center_count": len(counted_centers),
        "tetrahedral_center_records": center_records,
        "specified_double_bond_stereochemistry_count": len(specified_double_bonds),
        "matching_double_bond_stereochemistry_count": matching_double_bonds,
        "double_bond_stereochemistry_valid": matching_double_bonds == len(specified_double_bonds) if specified_double_bonds else "NA",
        "aromatic_ring_count": len(aromatic_deviations),
        "aromatic_planarity_max_deviation_A": max(aromatic_deviations) if aromatic_deviations else "",
        "aromatic_planarity_valid": max(aromatic_deviations) <= 0.25 if aromatic_deviations else "NA",
        "evaluable_aliphatic_CC_double_bond_count": len(double_deviations),
        "double_bond_planarity_max_deviation_A": max(double_deviations) if double_deviations else "",
        "double_bond_planarity_valid": max(double_deviations) <= 0.25 if double_deviations else "NA",
    }


def intergroup_contact(first, second, threshold=0.75):
    first = [atom for atom in first if atom["element"] not in METAL_ELEMENTS]
    second = [atom for atom in second if atom["element"] not in METAL_ELEMENTS]
    if not first or not second:
        return {
            "minimum_vdw_ratio": "NO_NONMETAL_PAIR",
            "minimum_distance_A": "NO_NONMETAL_PAIR",
            "clash_pair_count_ratio_lt_0.75": 0,
            "severe_clash_pair_count_ratio_lt_0.50": 0,
            "within_5A": "UNRESOLVED",
        }
    second_xyz = np.asarray([atom["xyz"] for atom in second], dtype=float)
    second_radii = np.asarray([vdw_radius(atom["element"]) for atom in second], dtype=float)
    minimum_ratio = float("inf")
    minimum_distance = float("inf")
    violating = 0
    severe = 0
    for start in range(0, len(first), 128):
        chunk = first[start:start + 128]
        chunk_xyz = np.asarray([atom["xyz"] for atom in chunk], dtype=float)
        chunk_radii = np.asarray([vdw_radius(atom["element"]) for atom in chunk], dtype=float)
        distances = cdist(chunk_xyz, second_xyz)
        denominators = chunk_radii[:, None] + second_radii[None, :]
        ratios = distances / denominators
        minimum_ratio = min(minimum_ratio, float(np.min(ratios)))
        minimum_distance = min(minimum_distance, float(np.min(distances)))
        local = distances <= 4.5
        if np.any(local):
            violating += int(np.sum(local & (ratios < threshold)))
            severe += int(np.sum(local & (ratios < 0.50)))
    return {
        "minimum_vdw_ratio": minimum_ratio if math.isfinite(minimum_ratio) else ">4.5A_NO_LOCAL_PAIR",
        "minimum_distance_A": minimum_distance if math.isfinite(minimum_distance) else ">4.5",
        "clash_pair_count_ratio_lt_0.75": violating,
        "severe_clash_pair_count_ratio_lt_0.50": severe,
        "within_5A": minimum_distance <= 5.0 if math.isfinite(minimum_distance) else False,
    }
