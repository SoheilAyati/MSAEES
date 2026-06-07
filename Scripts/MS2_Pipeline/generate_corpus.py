#!/usr/bin/env python3
"""
generate_corpus.py  --  build a multi-seed synthetic corpus for MS2
===================================================================

Calls the Milestone-1 generator + aggregator to produce, for each seed: 9
per-appliance traces and one aggregated mixed scenario (with ground truth).
Different seeds = different appliance *instances* (compressor sizes, EV rates,
PV clouds, ...), so you can train on some seeds and test on held-out ones.

Run it from this folder (Scripts/MS2_Pipeline):

    python generate_corpus.py --seeds 101 102 103 --outdir corpus
    python generate_corpus.py --seeds 104 105 106 --outdir corpus   # add more

Each seed produces:
    <outdir>/seed_<NNN>/<appliance>.h5     (9 single-appliance files)
    <outdir>/scenario_seed<NNN>.h5         (aggregate + ground truth)

Then:
    python train.py --task disaggregate --data "corpus/scenario_*.h5" --model lgbm
    python train.py --task identify     --data "corpus/seed_*"         --features full
"""
from __future__ import annotations
import argparse, os, subprocess, sys, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))          # Scripts/MS2_Pipeline -> repo root
GEN = os.path.join(REPO, "Scripts", "Synthetic_data_generator", "Appliance_generator.py")
AGG = os.path.join(REPO, "Scripts", "Aggregator", "aggregator.py")

# fixed phase assignment (instance variety comes from the seed, not the phase)
PHASE = {"fridge": "L1", "resistive": "L2", "hair_dryer": "L3", "pc": "L2",
         "washing_machine": "L1", "ev": "L1", "baseload": "L3",
         "pv": "all", "synchronous": "all"}
APPLIANCES = list(PHASE)


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("ERROR running:", " ".join(str(c) for c in cmd))
        print(r.stdout[-800:]); print(r.stderr[-800:]); sys.exit(1)
    return r


def main():
    ap = argparse.ArgumentParser(description="Generate a multi-seed synthetic corpus")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--duration", type=int, default=86400, help="seconds (default 24h)")
    ap.add_argument("--base-date", default="2024-03-01")
    args = ap.parse_args()
    base = datetime.date.fromisoformat(args.base_date)

    for s in args.seeds:
        sdir = os.path.join(args.outdir, f"seed_{s:03d}")
        os.makedirs(sdir, exist_ok=True)
        anchor = (base + datetime.timedelta(days=s % 350)).isoformat()
        for app in APPLIANCES:
            out = os.path.join(sdir, f"{app}.h5")
            run([sys.executable, GEN, "--appliance", app, "--seed", str(s),
                 "--phase", PHASE[app], "--anchor-date", anchor,
                 "--duration", str(args.duration), "--output", out])
        scen = os.path.join(args.outdir, f"scenario_seed{s:03d}.h5")
        run([sys.executable, AGG, "--input-dir", sdir, "--output", scen,
             "--tier", "train", "--seed", str(s)])
        print(f"seed {s:3d}: 9 appliances + scenario  ->  {scen}")
    print("corpus done:", args.outdir)


if __name__ == "__main__":
    main()
