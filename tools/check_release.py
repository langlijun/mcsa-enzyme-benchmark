"""Check staged publish scope and optionally preserve all frozen evaluator files."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frozen-source', type=Path)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    staged = subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0')
    staged = [p for p in staged if p]
    if not staged:
        raise SystemExit('No staged/tracked files to audit')
    problems = []
    patterns = [r'/Users/[A-Za-z0-9._-]+/', r'/home/[A-Za-z0-9._-]+/', r'-----BEGIN [A-Z ]*PRIVATE KEY-----',
                r'gh[pousr]_[A-Za-z0-9]{30,}', r'github_pat_[A-Za-z0-9_]{30,}']
    for relative in staged:
        path = root/relative
        if path.is_symlink() or not path.is_file():
            problems.append(relative + ': not a regular file')
            continue
        if any(part in {'results','validation','data_private','.venv','.local','__pycache__'} for part in path.relative_to(root).parts):
            problems.append(relative + ': private/generated directory')
        if path.suffix.lower() in {'.cif','.pdb','.gz','.pdf','.xlsx','.zip','.pyc','.pem','.key'}:
            problems.append(relative + ': excluded artifact type')
        try:
            content = path.read_text(encoding='utf-8')
        except UnicodeError:
            problems.append(relative + ': non-text artifact')
            continue
        if any(re.search(pattern, content) for pattern in patterns):
            # Keep scanner rule literals out of their own match results.
            if relative != 'tools/check_release.py':
                problems.append(relative + ': local path or credential-like string')
    preserved = []
    if args.frozen_source:
        old = args.frozen_source/'src/mcsa_benchmark'
        if not old.is_dir():
            raise SystemExit('Frozen evaluator source directory missing')
        for path in sorted(old.rglob('*')):
            if path.suffix not in {'.py','.json'} or '__pycache__' in path.parts:
                continue
            dest = root/'src/mcsa_benchmark'/path.relative_to(old)
            same = dest.is_file() and path.read_bytes() == dest.read_bytes()
            if not same:
                problems.append(str(path.relative_to(old)) + ': changed frozen evaluator')
            preserved.append({'file':str(path.relative_to(old)), 'identical':same,
                              'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    report = {'staged_files':len(staged),'preserved_source_files':preserved,'problems':problems}
    output = root/'.local/release_check.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'staged_files':len(staged),'frozen_files_checked':len(preserved),'problems':problems}))
    return int(bool(problems))


if __name__ == '__main__':
    raise SystemExit(main())
