"""Unified dispatcher; biological case rules remain separate from general QC."""
import hashlib
from importlib import import_module
from importlib.resources import files
import json
from pathlib import Path
import platform

import gemmi
import numpy
import rdkit
import scipy

from .general import evaluate_general as run_general
from .structure import InputProblem, read_case

MODULES = ["adenylate", "haloacid_sn2", "ndpk_histidine", "igps_nac", "lys_carbonyl_multisite",
           "sam_n_methyl", "ornithine_carbamoyl", "dhqd_lys_carbonyl", "nitroreductase_hydride",
           "aanat_acyl", "ugm_n5", "plp_transaldimination", "gale_hydride", "ldh_tetramer",
           "nicotinamidase", "isocitrate", "nad_synthetase", "glutathione_synthesis",
           "pmp_carbonyl", "phosphoserine_phosphatase"]
MECHANISMS = {name:import_module('.mechanisms.'+name, __package__).evaluate_mechanism for name in MODULES}


def data_file(name):
    return files('mcsa_benchmark').joinpath('data',name)


def list_cases():
    registry=json.loads(data_file('registry.json').read_text())
    return sorted(key for key in registry if len(key)==4)


def load_case(case_id):
    registry=json.loads(data_file('registry.json').read_text());key=case_id.upper()
    if key not in registry:
        raise ValueError(f'Unknown case {case_id}; available: {list_cases()}')
    text=data_file(registry[key]).read_text();case=json.loads(text)
    case['general_policy']=json.loads(data_file('general_policy.json').read_text())
    supported={'bond_and_angle_bounds_factors':[0.75,1.25],'internal_nonbonded_lower_factor':0.70,
               'intermolecular_vdw_lower_factor':0.75,'flatness_max_A':0.25,'volume_overlap_max':0.075,
               'shape_vdw_scale':0.8,'ligand_protein_max_distance_A':5.0}
    if any(case['general_policy'].get(k)!=v for k,v in supported.items()):
        raise ValueError('General policy and numerical kernels disagree')
    if case.get('general_metal_policy'):
        raise ValueError('Unified general does not accept a metal-distance hard gate')
    case['configuration_sha256']=hashlib.sha256(text.encode()).hexdigest()
    return case


def error_stage(exc):
    return {'status':'ERROR','failure_tags':[exc.code if isinstance(exc,InputProblem) else 'EVALUATION_ERROR'],'detail':str(exc)}


def check_finite_output(stage):
    try: json.dumps(stage,allow_nan=False)
    except ValueError as exc:
        raise InputProblem('NONFINITE_MEASUREMENT','Nonfinite metric; no binary success inferred.') from exc


def evaluate(cif_path,case_id,stage='all',atom_map=None,atom_profile=None,site_map=None):
    if stage not in {'mechanism','general','all'}:
        raise ValueError('stage must be mechanism, general or all')
    case,path=load_case(case_id),Path(cif_path)
    result={'case_id':case['case_id'],'variant':case.get('variant','primary'),'structure':str(path),
            'requested_stage':stage,'configuration_sha256':case['configuration_sha256'],'historical_results_used':False,
            'mechanism':{'status':'NOT_RUN'},'general':{'status':'NOT_RUN'},
            'software':{'package':'mcsa-enzyme-benchmark','version':'0.7.0','python':platform.python_version(),
                        'gemmi':gemmi.__version__,'numpy':numpy.__version__,'rdkit':rdkit.__version__,'scipy':scipy.__version__}}
    if 'input_state_contract' in case:result['input_state_contract']=case['input_state_contract']
    try:
        result['structure_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        if case.get('atom_profile_required'):
            if atom_profile is None and (not atom_map or set(atom_map)!=set(case['ligands'])):
                raise InputProblem('ATOM_PROFILE_REQUIRED','This case requires an explicit model naming profile or complete atom map.')
            if atom_profile is not None:
                profile=next((p for p in case['atom_profiles'] if p.lower()==atom_profile.lower()),None)
                if profile is None:raise InputProblem('UNKNOWN_ATOM_PROFILE',str(atom_profile))
                for definition in case['ligands'].values():
                    definition['atom_name_templates']=[case['atom_profiles'][profile][definition['copy_group']]]
                result['atom_profile']=profile
        elif atom_profile is not None:
            raise InputProblem('ATOM_PROFILE_NOT_APPLICABLE','Use atom_map for this case; a naming profile is only defined for2GSA.')
        context=read_case(path,case,atom_map)
        result.update(input_status='PASS',atom_mapping=context['atom_mapping'],ignored_water_count=context['ignored_water_count'])
    except (InputProblem,OSError,ValueError,RuntimeError) as exc:
        result['input_status']='FAIL' if isinstance(exc,InputProblem) and exc.is_model_failure else 'ERROR'
        result['input_problem']=str(exc)
        for name in ('mechanism','general'):
            if stage in {name,'all'}:result[name]=error_stage(exc)
    else:
        if stage in {'mechanism','all'}:
            try:
                if site_map is not None:
                    if case['mechanism_type']!='ldh_tetramer':
                        raise InputProblem('SITE_MAP_NOT_SUPPORTED','Explicit site_map is defined only for1LDM.')
                    context['explicit_site_map']=site_map
                result['mechanism']=MECHANISMS[case['mechanism_type']](context,case)
                check_finite_output(result['mechanism']);result['mechanism']['recomputed']=True
            except (InputProblem,ValueError,RuntimeError,KeyError,IndexError,TypeError) as exc:
                result['mechanism']=error_stage(exc)
        if stage in {'general','all'}:
            try:
                result['general']=run_general(context,case,path)
                check_finite_output(result['general']);result['general']['recomputed']=True
            except (InputProblem,ValueError,RuntimeError,KeyError,IndexError,TypeError) as exc:
                result['general']=error_stage(exc)
    m,g=result['mechanism']['status'],result['general']['status']
    result.update(mechanism_status=m,general_status=g,
                  joint_status='NOT_RUN' if stage!='all' else 'ERROR' if 'ERROR' in {m,g} else 'PASS' if m==g=='PASS' else 'FAIL')
    result['joint_note']='Optional same-sample intersection; never substitute it for either independent stage.'
    return result


def evaluate_mechanism(cif_path,case_id,atom_map=None,atom_profile=None,site_map=None):
    return evaluate(cif_path,case_id,'mechanism',atom_map,atom_profile,site_map)


def evaluate_general(cif_path,case_id,atom_map=None,atom_profile=None):
    return evaluate(cif_path,case_id,'general',atom_map,atom_profile)
