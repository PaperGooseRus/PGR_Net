#!/usr/bin/env python3
"""Одна команда для полного воспроизводимого Stage 4 (подготовка + baseline)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

from prepare_stage4 import ROOT, prepare, verify_snapshot
from evaluate_stage4 import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--min-df', type=int, default=2)
    parser.add_argument('--max-features', type=int, default=12000)
    args = parser.parse_args()
    try:
        encoding = prepare(args.root, args.min_df, args.max_features)
        evaluation = evaluate(args.root)
        result = {
            'stage': 4, 'status': 'passed',
            'features': encoding['features'],
            'classes': encoding['classes'],
            'train_rows': evaluation['train_samples'],
            'val': evaluation['results']['val']['both'],
            'test': evaluation['results']['test']['both'],
            'snapshot': verify_snapshot(args.root),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        sys.exit('STAGE4 ERROR: ' + str(exc))


if __name__ == '__main__':
    main()
