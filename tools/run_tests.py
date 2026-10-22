"""Run installed-package tests, optionally with an external reviewed archive."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.json.resolve() if args.json else None
    for name in list(os.environ):
        if name.startswith('MCSA_') and (name.endswith(('_CIF', '_DIR')) or name == 'MCSA_PANEL_MANIFEST'):
            os.environ.pop(name)
    if args.fixtures:
        data = args.fixtures.resolve()
        settings = json.loads((data/'validation/test_environment.json').read_text())
        for key, relative in settings.items():
            path = (data/relative).resolve()
            if not key.startswith('MCSA_') or not path.is_relative_to(data) or not path.exists():
                parser.error(f'Invalid or missing reviewed fixture: {key}')
            os.environ[key] = str(path)
        manifest = data/'validation/primary400.json'
        if not manifest.is_file():
            parser.error('Missing reviewed primary400 manifest')
        os.environ['MCSA_PANEL_MANIFEST'] = str(manifest)
    suite = unittest.defaultTestLoader.discover(str(root/'tests'))
    old_cwd = Path.cwd()
    try:
        with tempfile.TemporaryDirectory() as temp:
            os.chdir(temp)
            result = unittest.TextTestRunner(verbosity=2).run(suite)
    finally:
        os.chdir(old_cwd)
    report = {'tests_run': result.testsRun, 'passed': result.testsRun-len(result.errors)-len(result.failures)-len(result.skipped),
              'failures': len(result.failures), 'errors': len(result.errors),
              'skipped': [{'test': str(t), 'reason': why} for t, why in result.skipped],
              'external_fixtures_supplied': bool(args.fixtures),
              'success': result.wasSuccessful() and not (args.fixtures and result.skipped)}
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'skipped'} | {'skipped_count':len(result.skipped)}))
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

