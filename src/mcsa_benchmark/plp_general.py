"""Chemical checks for an internal polymer LLP, separate from ADC approach."""

import gemmi
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from . import _kernels as k
from .general import CHECK_FIELDS, atom_id, distance, short_pairs
from .mechanisms.adenylate import inside


def identity(atom):
    return tuple(atom[key] for key in ('chain','residue_number','insertion_code','atom_name'))


def local_fragment(context, chain, definition):
    """Use real neighboring backbone coordinates as caps, not disconnected LLP."""
    residues = context['protein_chains'][chain]
    pos = definition['position']-1
    residue = residues[pos]
    if residue['name'] != definition['residue_name']:
        return None, [], ['REQUIRED_LLP_MODIFICATION_MISSING']
    by_name = {a['atom_name']:a for a in residue['atoms']}
    names = [a['name'] for a in definition['atoms']]
    missing = [name for name in names if name not in by_name]
    caps = []
    if pos == 0 or pos+1 >= len(residues):
        return None, [], ['INTERNAL_LLP_PEPTIDE_NEIGHBOR_MISSING']
    for offset, required in [(-1,['CA','C','O']),(1,['N','CA'])]:
        available = {a['atom_name']:a for a in residues[pos+offset]['atoms']}
        missing.extend(f'{offset:+d}:{name}' for name in required if name not in available)
        caps.extend(available[name] for name in required if name in available)
    if missing:
        return None, [], missing
    atoms = [by_name[name] for name in names] + caps
    editable = Chem.RWMol()
    for spec in definition['atoms']:
        atom = Chem.Atom(spec['element'].title())
        atom.SetFormalCharge(spec['formal_charge'])
        editable.AddAtom(atom)
    for element in ['C','C','O','N','C']:
        editable.AddAtom(Chem.Atom(element))
    index = {name:i for i,name in enumerate(names)}
    orders = {'SING':Chem.BondType.SINGLE,'DOUB':Chem.BondType.DOUBLE,'TRIP':Chem.BondType.TRIPLE}
    for bond in definition['bonds']:
        a,b = bond['atoms']
        editable.AddBond(index[a],index[b],orders[bond['order']])
    start = len(names)
    for a,b,order in [(start,start+1,'SING'),(start+1,start+2,'DOUB'),
                      (start+1,index['N'],'SING'),(index['C'],start+3,'SING'),(start+3,start+4,'SING')]:
        editable.AddBond(a,b,orders[order])
    mol = editable.GetMol()
    Chem.SanitizeMol(mol)
    for spec in definition['atoms']:
        if spec.get('CIP'):
            atom = mol.GetAtomWithIdx(index[spec['name']])
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
            if not atom.HasProp('_CIPCode') or atom.GetProp('_CIPCode') != spec['CIP']:
                atom.InvertChirality()
                Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
            if not atom.HasProp('_CIPCode') or atom.GetProp('_CIPCode') != spec['CIP']:
                raise ValueError('Cannot construct the declared LLP stereochemistry')
    return mol, atoms, []


def stereo_details(molecule, atoms):
    expected = Chem.Mol(molecule)
    Chem.AssignStereochemistry(expected,force=True,cleanIt=True)
    observed = Chem.Mol(molecule)
    observed.RemoveAllConformers()
    Chem.RemoveStereochemistry(observed)
    conf = Chem.Conformer(len(atoms))
    conf.Set3D(True)
    for i, atom in enumerate(atoms):
        conf.SetAtomPosition(i, atom['xyz'])
    observed.AddConformer(conf)
    Chem.AssignStereochemistryFrom3D(observed,replaceExistingTags=True)
    Chem.AssignStereochemistry(observed,force=True,cleanIt=True)
    rows = []
    for atom in expected.GetAtoms():
        if not atom.HasProp('_CIPCode'):
            continue
        i = atom.GetIdx()
        wanted = atom.GetProp('_CIPCode')
        measured = observed.GetAtomWithIdx(i)
        found = measured.GetProp('_CIPCode') if measured.HasProp('_CIPCode') else None
        rows.append({'atom':atom_id(atoms[i]), 'expected_CIP':wanted, 'observed_CIP':found,
                     'status':'MATCH' if found==wanted else 'UNASSIGNED_FROM_3D' if found is None else 'INVERTED'})
    return rows


def explicit_llp_graph(path, definition):
    if not str(path).lower().endswith(('.cif','.mmcif','.cif.gz','.mmcif.gz')):
        return {'status':'NOT_PROVIDED','basis':'Registered CCD graph'}
    block = gemmi.cif.read_file(str(path)).sole_block()
    names = {a['name'] for a in definition['atoms']}
    edges = {}
    for r in block.find('_chem_comp_bond.', ['comp_id','atom_id_1','atom_id_2','value_order']):
        a,b = gemmi.cif.as_string(r[1]),gemmi.cif.as_string(r[2])
        if r[0].upper() == 'LLP' and a in names and b in names:
            edges[tuple(sorted((a,b)))] = r[3].upper()
    if not edges:
        return {'status':'NOT_PROVIDED','basis':'Registered CCD graph'}
    expected = {tuple(sorted(b['atoms'])) for b in definition['bonds']}
    imine = tuple(sorted(("C4'",'NZ')))
    valid = set(edges)==expected and edges.get(imine) in {'DOUB','DOUBLE','2'}
    return {'status':'PASS' if valid else 'FAIL','edge_count':len(edges),
            'missing_edges':[list(x) for x in sorted(expected-set(edges))],
            'unexpected_edges':[list(x) for x in sorted(set(edges)-expected)],
            'declared_C4prime_NZ_order':edges.get(imine)}


def substrate_crosslinks(path, context):
    """Reject explicitly declared cross-component substrate bonds, not guessed bonds."""
    if not str(path).lower().endswith(('.cif','.mmcif','.cif.gz','.mmcif.gz')):
        return []
    block = gemmi.cif.read_file(str(path)).sole_block()
    def tokens(tag):
        return [gemmi.cif.as_string(v) if v not in {'.','?'} else '' for v in block.find_values(tag)]
    types = tokens('_struct_conn.conn_type_id')
    if not types:
        return []
    fields = ['label_asym_id','label_comp_id','label_seq_id','label_atom_id',
              'auth_asym_id','auth_seq_id','pdbx_PDB_ins_code','auth_atom_id']
    cols = {f:tokens('_atom_site.'+f) for f in fields}
    lookup = {}
    for i in range(len(cols['label_asym_id'])):
        def value(field):
            return cols[field][i] if i<len(cols[field]) else ''
        key = tuple(value(f) for f in fields[:4])
        seq = value('auth_seq_id') or value('label_seq_id')
        if seq:
            lookup[key] = (value('auth_asym_id') or value('label_asym_id'),int(seq),
                           value('pdbx_PDB_ins_code'),value('auth_atom_id') or value('label_atom_id'))
    substrate = {identity(a) for atoms in context['ligands'].values() for a in atoms}
    columns = {f'ptnr{n}_{f}':tokens(f'_struct_conn.ptnr{n}_{f}') for n in [1,2]
               for f in fields[:4]+['auth_asym_id','auth_seq_id']}
    result = []
    for i,kind in enumerate(types):
        if not kind.lower().startswith('covale'):
            continue
        endpoints = []
        for n in [1,2]:
            def value(field):
                values = columns[f'ptnr{n}_{field}']
                return values[i] if i<len(values) else ''
            key = tuple(value(f) for f in fields[:4])
            resolved = lookup.get(key)
            if resolved is None and value('auth_asym_id') and value('auth_seq_id'):
                resolved = (value('auth_asym_id'),int(value('auth_seq_id')),'',value('label_atom_id'))
            endpoints.append(resolved)
        if any(e in substrate for e in endpoints if e is not None):
            result.append({'row':i+1,'connection_type':kind,'endpoints':endpoints})
    return result


def evaluate_plp_checks(context, case, path):
    definition = case['polymer_modifications'][0]
    limits = case['case_general_requirements']['existing_C4prime_NZ_A']
    checks, failures, diagnostics, components = {}, [], [], []
    explicit = explicit_llp_graph(path,definition)
    checks['LLP.explicit_CCD_graph_when_provided'] = 'NOT_APPLICABLE' if explicit['status']=='NOT_PROVIDED' else explicit['status']
    if explicit['status']=='FAIL':
        failures.append({'tag':'DECLARED_LLP_CHEMISTRY_MISMATCH','detail':explicit})
    crosslinks = substrate_crosslinks(path,context)
    checks['ADC.no_declared_crosscomponent_covalent_bond'] = 'FAIL' if crosslinks else 'PASS'
    if crosslinks:
        failures.append({'tag':'UNDECLARED_SUBSTRATE_COVALENT_LINK','connections':crosslinks})
    stereo = []
    for role, atoms in context['ligands'].items():
        mol = Chem.MolFromSmiles(case['ligands'][role]['smiles'])
        stereo.extend({'component':role, **r} for r in stereo_details(mol,atoms))
    for chain, residues in sorted(context['protein_chains'].items()):
        role = f'{chain}.LLP{definition["position"]}'
        residue = residues[definition['position']-1]
        by_name = {a['atom_name']:a for a in residue['atoms']}
        pair = [by_name[n] for n in ["C4'",'NZ'] if n in by_name]
        d = distance(*pair) if len(pair)==2 else None
        bond_pass = inside(d,limits)
        checks[f'{role}.existing_C4prime_NZ'] = 'PASS' if bond_pass else 'FAIL'
        bond = {'atoms':[atom_id(a) for a in pair],'distance_A':d,'bounds_A':limits,'status':checks[f'{role}.existing_C4prime_NZ']}
        if not bond_pass:
            failures.append({'tag':'LLP_INTERNAL_ALDIMINE_BOND_OUT_OF_RANGE' if d is not None else 'LLP_INTERNAL_ALDIMINE_BOND_ATOM_MISSING',
                             'component':role, **bond})
        mol, atoms, missing = local_fragment(context,chain,definition)
        if missing:
            checks[f'{role}.local_graph_completeness'] = 'FAIL'
            failures.append({'tag':'LLP_LOCAL_CHEMISTRY_INCOMPLETE','component':role,'missing':missing})
            components.append({'component':role,'existing_bond':bond,'missing':missing})
            continue
        checks[f'{role}.local_graph_completeness'] = 'PASS'
        graph = {'mol':mol,'source_chain':role,'canonical_smiles':Chem.MolToSmiles(mol)}
        component = {'atoms':atoms,'chain':chain,'comp':'CAPPED_POLYMER_LLP','heavy_atom_count':len(atoms)}
        metrics = k.component_chemistry(component,graph,{i:i for i in range(len(atoms))},'CCD_ATOM_NAMES_AND_SEQUENCE_ADJACENCY',k.inferred_adjacency(atoms))
        for field, tag in CHECK_FIELDS.items():
            value = metrics[field]
            checks[f'{role}.{field}'] = 'NOT_APPLICABLE' if value=='NA' else 'PASS' if value else 'FAIL'
            if value is False:
                failures.append({'tag':'LLP_'+tag,'component':role,'metric':field})
        stereo.extend({'component':role, **r} for r in stereo_details(mol,atoms))
        bounds = rdDistGeom.GetMoleculeBoundsMatrix(mol)
        bond_rows = []
        for edge in mol.GetBonds():
            i,j = edge.GetBeginAtomIdx(),edge.GetEndAtomIdx()
            value = distance(atoms[i],atoms[j])
            lower,upper = .75*float(bounds[max(i,j),min(i,j)]),1.25*float(bounds[min(i,j),max(i,j)])
            passed = lower <= value <= upper
            peptide = atoms[i]['residue_number'] != atoms[j]['residue_number']
            row = {'atoms':[atom_id(atoms[i]),atom_id(atoms[j])], 'distance_A':value,
                   'lower_A':lower,'upper_A':upper,'status':'PASS' if passed else 'FAIL',
                   'peptide_link':peptide}
            bond_rows.append(row)
            if peptide and not passed:
                failures.append({'tag':'LLP_PEPTIDE_LINK_OUT_OF_RANGE','component':role,**row})
        checks[f'{role}.peptide_continuity'] = 'PASS' if all(r['status']=='PASS' for r in bond_rows if r['peptide_link']) else 'FAIL'
        # The C=N local plane is distinct from torsion of the imine versus the PLP ring.
        plane = k.plane_max_deviation([by_name[n]['xyz'] for n in ['C4',"C4'",'NZ','CE']])
        checks[f'{role}.C_equals_N_local_planarity'] = 'PASS' if plane <= case['general_policy']['flatness_max_A'] else 'FAIL'
        if checks[f'{role}.C_equals_N_local_planarity']=='FAIL':
            failures.append({'tag':'LLP_IMINE_NONPLANAR','component':role,'max_deviation_A':plane})
        included = {identity(a) for a in atoms + residue['atoms']}
        outside = [a for a in context['protein_atoms'] if identity(a) not in included]
        pairs = short_pairs(residue['atoms'],outside,'POLYMER_LLP_OTHER_PROTEIN')
        checks[f'{role}.nonlocal_protein_short_contact'] = 'FAIL' if pairs else 'PASS'
        if pairs:
            failures.append({'tag':'LLP_NONLOCAL_PROTEIN_SHORT_CONTACT','component':role,'pairs':pairs})
        components.append({'component':role,'existing_bond':bond,'metrics':metrics,
                           'local_graph_atom_map':[atom_id(a) for a in atoms], 'bond_evidence':bond_rows,
                           'C_equals_N_plane_max_deviation_A':plane,'nonlocal_protein_short_pairs':pairs})
    for r in stereo:
        if r['status'] != 'MATCH':
            failures.append({'tag':'STEREOCENTER_INVERTED' if r['status']=='INVERTED' else 'SPECIFIED_STEREOCENTER_NOT_RECOVERED_3D',**r})
    for role in case['reaction_atoms']['ADC_roles']:
        n = context['ligands'][role][case['reaction_atoms']['ADC_N_index']]
        for chain,residues in context['protein_chains'].items():
            c = next((a for a in residues[definition['position']-1]['atoms'] if a['atom_name']=="C4'"),None)
            if c is not None and distance(n,c)<1.8:
                diagnostics.append({'tag':'COVALENT_LIKE_REACTIVE_CONTACT','atoms':[atom_id(n),atom_id(c)],
                                    'distance_A':distance(n,c),'affects_pass':False,
                                    'note':'Not proof of new bonding. Standard noncovalent short-contact checks remain active.'})
    return {'checks':checks,'failures':failures,'diagnostics':diagnostics,'LLP_components':components,
            'stereochemistry_details':stereo,'explicit_LLP_chemistry':explicit,
            'scope':'Declared ADC chemistry plus capped polymer LLP geometry and peptide continuity. Local graph handles covalent-neighbor exclusions; remaining LLP/protein short contacts use the common 0.75 vdW rule. No LLP-as-free-ligand volume fraction or whole-protein geometry claim.'}
