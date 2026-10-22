"""Evaluate new structures with an explicit case and independent stages."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from .evaluate import evaluate, list_cases, load_case


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('list-cases')
    p=sub.add_parser('criteria');p.add_argument('--case',required=True)
    p=sub.add_parser('evaluate');p.add_argument('cifs',nargs='+',type=Path);p.add_argument('--case',required=True)
    p.add_argument('--stage',choices=['mechanism','general','all'],default='all')
    p.add_argument('--atom-map',type=Path);p.add_argument('--atom-profile');p.add_argument('--site-map',type=Path)
    p.add_argument('--json',type=Path);p.add_argument('--csv',type=Path);p.add_argument('--export-map',type=Path)
    args=parser.parse_args()
    if args.command=='list-cases':print(json.dumps(list_cases()));return 0
    try:case=load_case(args.case)
    except ValueError as exc:parser.error(str(exc))
    if args.command=='criteria':
        print(json.dumps({'case_id':case['case_id'],'mechanism':case['mechanism_criteria'],'general':case['general_policy'],
                          'scope':case.get('input_state_contract',case.get('input_scope',''))},ensure_ascii=False,indent=2));return 0
    outputs=[p.resolve() for p in [args.json,args.csv,args.export_map] if p]
    inputs={p.resolve() for p in args.cifs}|{p.resolve() for p in [args.atom_map,args.site_map] if p}
    if len(outputs)!=len(set(outputs)) or inputs.intersection(outputs):parser.error('Outputs must be distinct from inputs and each other')
    if args.export_map and len(args.cifs)!=1:parser.error('--export-map requires one input')
    try:
        atom_map=json.loads(args.atom_map.read_text()) if args.atom_map else None
        site_map=json.loads(args.site_map.read_text()) if args.site_map else None
    except (OSError,ValueError) as exc:parser.error(str(exc))
    results=[evaluate(p,args.case,args.stage,atom_map,args.atom_profile,site_map) for p in args.cifs]
    report={'package_version':'0.7.0','evaluations':results,
            'summary':{s:dict(Counter(r[s+'_status'] for r in results)) for s in ['mechanism','general','joint']}}
    if args.json:
        args.json.parent.mkdir(parents=True,exist_ok=True);args.json.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if args.csv:
        rows=[{k:r.get(k,'') for k in ['case_id','structure','input_status','mechanism_status','general_status','joint_status','structure_sha256']}|
              {'mechanism_failure_tags':';'.join(r['mechanism'].get('failure_tags',[])),
               'general_failure_tags':';'.join(r['general'].get('failure_tags',[]))} for r in results]
        args.csv.parent.mkdir(parents=True,exist_ok=True)
        with args.csv.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if args.export_map:
        if 'atom_mapping' not in results[0]:parser.error('Cannot export a map from an unmapped structure')
        mapping={k:{field:value for field,value in v.items() if field!='mapping_method'} for k,v in results[0]['atom_mapping'].items()}
        args.export_map.parent.mkdir(parents=True,exist_ok=True);args.export_map.write_text(json.dumps(mapping,indent=2)+'\n')
    print(json.dumps(report['summary'],ensure_ascii=False))
    return 2 if any('ERROR' in [r['mechanism_status'],r['general_status']] for r in results) else 0


if __name__=='__main__':
    raise SystemExit(main())
