#!/usr/bin/env python3
from pathlib import Path
import argparse, sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import solver

if __name__ == '__main__':
    ap=argparse.ArgumentParser(description='Regenerate the primary FEM/AFEM/local-correction CSV archive.')
    ap.add_argument('--outdir', default=str(Path(__file__).resolve().parents[1]/'regenerated_primary'))
    args=ap.parse_args()
    solver.main(args.outdir)
