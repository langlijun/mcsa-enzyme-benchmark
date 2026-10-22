"""Read a complete declared case, with atom-name mapping independent of pose."""

from collections import Counter
from dataclasses import dataclass
import math

import gemmi
import numpy as np
from rdkit import Chem

AA = dict(zip(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
    "ARNDCQEGHILKMFPSTWYV",
))
SIDECHAINS = {
    "ALA": "CB", "ARG": "CB CG CD NE CZ NH1 NH2", "ASN": "CB CG OD1 ND2",
    "ASP": "CB CG OD1 OD2", "CYS": "CB SG", "GLN": "CB CG CD OE1 NE2",
    "GLU": "CB CG CD OE1 OE2", "GLY": "", "HIS": "CB CG ND1 CD2 CE1 NE2",
    "ILE": "CB CG1 CG2 CD1", "LEU": "CB CG CD1 CD2", "LYS": "CB CG CD CE NZ",
    "MET": "CB CG SD CE", "PHE": "CB CG CD1 CD2 CE1 CE2 CZ",
    "PRO": "CB CG CD", "SER": "CB OG", "THR": "CB OG1 CG2",
    "TRP": "CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2",
    "TYR": "CB CG CD1 CD2 CE1 CE2 CZ OH", "VAL": "CB CG1 CG2",
}


@dataclass
class InputProblem(Exception):
    code: str
    detail: str
    is_model_failure: bool = False

    def __str__(self):
        return f"{self.code}: {self.detail}"


def atom_record(chain, residue, atom):
    return {
        "chain": chain.name, "residue_name": residue.name,
        "residue_number": residue.seqid.num,
        "insertion_code": residue.seqid.icode.strip(),
        "atom_name": atom.name, "element": atom.element.name.upper(),
        "xyz": np.array([atom.pos.x, atom.pos.y, atom.pos.z]),
    }


def selector(residue):
    return {key: residue["atoms"][0][key] for key in
            ("chain", "residue_name", "residue_number", "insertion_code")}


def check_declared_chemistry(path, ligands, case):
    """Check explicit CIF metadata when present; never infer bonds from distances."""
    evidence = {}
    def identity_smiles(molecule):
        molecule = Chem.Mol(molecule)
        excluded = case.get("case_general_requirements", {}).get("excluded_stereochemistry_elements", [])
        for atom in molecule.GetAtoms():
            if atom.GetSymbol() in excluded:
                atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
        return Chem.MolToSmiles(molecule, isomericSmiles=True)
    if not str(path).lower().endswith((".cif", ".mmcif", ".cif.gz", ".mmcif.gz")):
        return {role: "CASE_INPUT_GRAPH_NO_CIF_CHEMISTRY_METADATA" for role in ligands}
    block = gemmi.cif.read_file(str(path)).sole_block()
    def column(name):
        return [gemmi.cif.as_string(value) for value in block.find_values(name)]
    comp_ids = column("_chem_comp.id")
    comp_smiles = column("_chem_comp.pdbx_smiles")
    smiles_by_id = dict(zip((x.upper() for x in comp_ids), comp_smiles)) if len(comp_ids) == len(comp_smiles) else {}
    bond_comp = column("_chem_comp_bond.comp_id")
    bond_a = column("_chem_comp_bond.atom_id_1")
    bond_b = column("_chem_comp_bond.atom_id_2")
    for role, atoms in ligands.items():
        component_name = atoms[0]["residue_name"].upper()
        expected = Chem.MolFromSmiles(case["ligands"][role]["smiles"])
        smiles = smiles_by_id.get(component_name, "")
        checked = []
        if smiles and smiles not in {"?", "."}:
            supplied = Chem.MolFromSmiles(smiles)
            if supplied is None or identity_smiles(supplied) != identity_smiles(expected):
                raise InputProblem("DECLARED_LIGAND_CHEMISTRY_MISMATCH", f"{role}: CIF SMILES differs from fixed input.", True)
            checked.append("CIF_SMILES_MATCHES_INPUT")
        if len(bond_comp) == len(bond_a) == len(bond_b):
            names = {a["atom_name"] for a in atoms}
            explicit = {tuple(sorted((a, b))) for comp, a, b in zip(bond_comp, bond_a, bond_b)
                        if comp.upper() == component_name and a in names and b in names}
            if explicit:
                expected_edges = {
                    tuple(sorted((atoms[b.GetBeginAtomIdx()]["atom_name"], atoms[b.GetEndAtomIdx()]["atom_name"])))
                    for b in expected.GetBonds()
                }
                if explicit != expected_edges:
                    raise InputProblem("DECLARED_LIGAND_CONNECTIVITY_MISMATCH", f"{role}: explicit CIF bonds differ from input.", True)
                checked.append("CIF_BOND_CONNECTIVITY_MATCHES_INPUT")
        evidence[role] = checked or ["CASE_INPUT_GRAPH_NO_EXPLICIT_CIF_GRAPH"]
    return evidence


def read_case(path, case, user_map=None):
    modification_defs = {m['residue_name']: m for m in case.get('polymer_modifications', [])}
    expected_modifications = {m['position']: m for m in case.get('polymer_modifications', [])}
    if user_map is not None:
        allowed = {"chain", "residue_name", "residue_number", "insertion_code", "atom_names"}
        if not isinstance(user_map, dict) or set(user_map) - set(case["ligands"]):
            raise InputProblem("INVALID_ATOM_MAP", f"Map keys must be declared roles: {list(case['ligands'])}.")
        for role, spec in user_map.items():
            if not isinstance(spec, dict) or set(spec) - allowed:
                raise InputProblem("INVALID_ATOM_MAP", f"{role}: unsupported selector fields.")
            if "atom_names" not in spec or not isinstance(spec["atom_names"], list) or not all(
                isinstance(name, str) for name in spec["atom_names"]
            ):
                raise InputProblem("INVALID_ATOM_MAP", f"{role}: atom_names must be a list of names.")
        for definition in case["ligands"].values():
            group = definition.get("copy_group")
            if group:
                roles = {r for r, d in case["ligands"].items() if d.get("copy_group") == group}
                if set(user_map) & roles and not roles <= set(user_map):
                    raise InputProblem("INVALID_ATOM_MAP", f"Provide every {group} copy when explicitly mapping identical components.")
    try:
        structure = gemmi.read_structure(str(path))
    except (RuntimeError, ValueError) as exc:
        raise InputProblem("STRUCTURE_PARSE_ERROR", str(exc)) from exc
    if len(structure) != 1:
        raise InputProblem("MULTIPLE_OR_MISSING_MODELS", "Provide one selected coordinate model.")
    protein_by_chain, residues, ignored_waters = {}, [], 0
    for chain in structure[0]:
        for residue in chain:
            atoms = []
            for atom in residue:
                if not all(math.isfinite(v) for v in (atom.pos.x, atom.pos.y, atom.pos.z)):
                    raise InputProblem("NONFINITE_COORDINATE", f"{chain.name}/{residue.seqid}/{atom.name}", True)
                if atom.element.name in {"H", "D"}:
                    continue
                if atom.altloc not in {"", "\x00", " "}:
                    raise InputProblem("ALTLOC_REQUIRES_SELECTION", "Select one compatible conformer before evaluation.")
                if not math.isfinite(atom.occ) or atom.occ <= 0:
                    raise InputProblem("NONPOSITIVE_OCCUPANCY", f"{chain.name}/{residue.seqid}/{atom.name}", True)
                atoms.append(atom_record(chain, residue, atom))
            if not atoms:
                continue
            if len({a["atom_name"] for a in atoms}) != len(atoms):
                raise InputProblem("DUPLICATE_ATOM_NAME", f"{chain.name}/{residue.seqid}")
            if residue.name.upper() in {"HOH", "WAT", "DOD"}:
                ignored_waters += 1
                continue
            record = {"name": residue.name.upper(), "atoms": atoms}
            free_amino_acid = record["name"] in case.get("nonpolymer_amino_acid_ligands", []) and (
                residue.entity_type == gemmi.EntityType.NonPolymer or
                (residue.entity_type == gemmi.EntityType.Unknown and residue.het_flag == "H")
            )
            if (record["name"] in AA or record["name"] in modification_defs) and not free_amino_acid:
                protein_by_chain.setdefault(chain.name, []).append(record)
            else:
                residues.append(record)
    if len(protein_by_chain) != case["protein_copies"]:
        raise InputProblem("PROTEIN_COPY_COUNT_MISMATCH", f"Require {case['protein_copies']} protein chains.", True)
    general_issues = []
    for chain, protein_residues in protein_by_chain.items():
        sequence = "".join(AA[modification_defs[r['name']]['parent_residue_name']] if r['name'] in modification_defs
                           else AA[r['name']] for r in protein_residues)
        if sequence != case["protein_sequence"]:
            raise InputProblem("PROTEIN_SEQUENCE_OR_COMPLETENESS_MISMATCH", f"Chain {chain} does not match the declared sequence; a verified sequence crosswalk is needed.", True)
        for i, residue in enumerate(protein_residues):
            modification = modification_defs.get(residue['name'])
            expected = expected_modifications.get(i+1)
            if expected and residue['name'] != expected['residue_name']:
                general_issues.append({'tag':'REQUIRED_POLYMER_MODIFICATION_MISSING', 'location':selector(residue),
                                       'sequence_position':i+1, 'expected':expected['residue_name']})
            if modification and (expected is None or expected['residue_name'] != residue['name']):
                general_issues.append({'tag':'POLYMER_MODIFICATION_WRONG_POSITION', 'location':selector(residue),
                                       'sequence_position':i+1})
            required = ({a['name'] for a in modification['atoms']} if modification else
                        set(("N CA C O " + SIDECHAINS[residue["name"]]).split()))
            observed = {a["atom_name"] for a in residue["atoms"]}
            allowed = required | ({"OXT"} if i == len(protein_residues) - 1 else set())
            if not required <= observed or not observed <= allowed:
                general_issues.append({"tag": "PROTEIN_HEAVY_ATOM_MISMATCH", "location": selector(residue),
                                       "missing": sorted(required-observed), "extra": sorted(observed-allowed)})
            for atom in residue["atoms"]:
                expected_element = (next((a['element'] for a in modification['atoms'] if a['name']==atom['atom_name']), atom['atom_name'][0])
                                    if modification else atom['atom_name'][0])
                if atom["element"] != expected_element:
                    general_issues.append({"tag": "PROTEIN_ATOM_ELEMENT_MISMATCH", "location": selector(residue)})
    proteins = [a for rows in protein_by_chain.values() for r in rows for a in r["atoms"]]
    ligands, metals, metal_copies, mapping_output, consumed = {}, {}, {}, {}, set()
    group_used = Counter()
    for role, definition in case["ligands"].items():
        expected = Counter(s.upper() for s in definition["element_order"])
        candidates = [(i, r) for i, r in enumerate(residues)
                      if Counter(a["element"] for a in r["atoms"]) == expected]
        group = definition.get("copy_group", role)
        if case.get("copy_aware"):
            required_copies = sum(Counter(s.upper() for s in d["element_order"]) == expected
                                  for d in case["ligands"].values())
            if len(candidates) != required_copies:
                detail = f"{role}: require {required_copies} complete components, found {len(candidates)}."
                if definition.get("mechanism_required", True):
                    raise InputProblem("LIGAND_COMPOSITION_OR_COPY_MISMATCH", detail, True)
                general_issues.append({"tag": "LIGAND_COMPOSITION_OR_COPY_MISMATCH", "detail": detail})
            candidates = [(i, r) for i, r in candidates if i not in consumed]
            candidates.sort(key=lambda item: tuple(str(v) for v in selector(item[1]).values()))
            if not candidates and not definition.get("mechanism_required", True):
                continue
        else:
            candidates = [(i, r) for i, r in candidates if i not in consumed]
            remaining = definition.get("group_copies", 1) - group_used[group]
            if len(candidates) != remaining:
                raise InputProblem("LIGAND_COMPOSITION_OR_COPY_MISMATCH",
                                   f"{group}: expected {remaining} remaining complete components; found {len(candidates)}.", True)
        custom = (user_map or {}).get(role)
        if custom:
            candidates = [(i, r) for i, r in candidates
                          if all(selector(r).get(k) == v for k, v in custom.items() if k != "atom_names")]
        elif case.get("copy_aware"):
            candidates = candidates[:1]
        elif definition.get("group_copies", 1) > 1:
            # These IDs enumerate input copies only; catalytic site assignment
            # is a separate, score-independent operation in the mechanism module.
            candidates = sorted(candidates, key=lambda item: tuple(selector(item[1]).values()))[:1]
        if len(candidates) != 1:
            raise InputProblem("LIGAND_COMPOSITION_OR_COPY_MISMATCH",
                               f"{role}: expected one complete component; found {len(candidates)}.", True)
        index, residue = candidates[0]
        consumed.add(index)
        group_used[group] += 1
        by_name = {a["atom_name"]: a for a in residue["atoms"]}
        matches = ([custom["atom_names"]] if custom and "atom_names" in custom else
                   [names for names in definition["atom_name_templates"] if set(names) == set(by_name)])
        if len(matches) != 1:
            raise InputProblem("ATOM_MAPPING_REQUIRED", f"{role}: supply --atom-map with names in the declared SMILES order.")
        names = matches[0]
        if len(names) != len(set(names)) or set(names) != set(by_name):
            raise InputProblem("INVALID_ATOM_MAP", f"{role}: atom map must cover every heavy atom exactly once.")
        ordered = [by_name[name] for name in names]
        if [a["element"] for a in ordered] != [s.upper() for s in definition["element_order"]]:
            raise InputProblem("ATOM_MAP_ELEMENT_MISMATCH", role)
        ligands[role] = ordered
        mapping_output[role] = {**selector(residue), "atom_names": names,
                               "mapping_method": "USER_DECLARED" if custom else "REGISTERED_ATOM_NAME_TEMPLATE"}
    for element, copies in case["metals"].items():
        candidates = [(i, r["atoms"][0]) for i, r in enumerate(residues)
                      if len(r["atoms"]) == 1 and r["atoms"][0]["element"] == element.upper()]
        if len(candidates) != copies:
            general_issues.append({"tag": "METAL_COMPOSITION_OR_COPY_MISMATCH", "element": element,
                                   "expected": copies, "observed": len(candidates)})
        consumed.update(i for i, _ in candidates)
        metal_copies[element] = [a for _, a in candidates]
        if len(candidates) == copies == 1:
            metals[element] = candidates[0][1]
    if len(consumed) != len(residues):
        general_issues.append({"tag": "UNDECLARED_COMPONENT", "detail": "Extra non-water components are outside the declared case."})
    return {"protein_atoms": proteins, "protein_chains": protein_by_chain,
            "ligands": ligands, "metals": metals, "metal_copies": metal_copies, "atom_mapping": mapping_output,
            "ignored_water_count": ignored_waters, "general_input_issues": general_issues}
