#!/usr/bin/env python3
"""Standalone eval runner — import-safe from the package root."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
from e3_only.configs.prism import PrismConfig, resolve
from e3_only.evaluate_prism import evaluate

def run_eval(checkpoint, save_preds, log_path, tta=False, region_vote=False,
             limit=None, which="teacher", presence_gate=None,
             logit_adjust=None, logit_prior=None):
    cfg = PrismConfig()
    cfg.device = "cuda"
    cfg.num_workers = 0
    
    ckpt_path = resolve(checkpoint)
    if not Path(ckpt_path).exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", flush=True)
        print("You need to train first. Run:", flush=True)
        print("  python -m e3_only.train_prism", flush=True)
        sys.exit(1)
    
    log = open(log_path, "w") if log_path else None
    try:
        evaluate(cfg, ckpt_path, log=log, save_preds=save_preds,
                 tta=tta, region_vote=region_vote, which=which, limit=limit,
                 presence_gate=presence_gate, logit_adjust=logit_adjust,
                 logit_prior=logit_prior)
    finally:
        if log:
            log.close()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Evaluate a PRISM checkpoint and save predictions")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--save-preds", required=True, help="Directory to save prediction images")
    p.add_argument("--log", default="/dev/null")
    p.add_argument("--tta", action="store_true", help="Test-time augmentation (4x flip/mirror)")
    p.add_argument("--region-vote", action="store_true", help="Pool posterior over SAM regions")
    p.add_argument("--presence-gate", type=float, default=None,
                   help="soft inventory prior from the presence head (default off)")
    p.add_argument("--logit-adjust", type=float, default=None,
                   help="Stage 2 class-prior term z_c - tau*log pi_c (tau>0 = balanced rule)")
    p.add_argument("--logit-prior", default=None, choices=["presence", "point_share"])
    p.add_argument("--which", default="teacher", choices=["teacher", "student"],
                   help="Which weights to load from checkpoint")
    p.add_argument("--limit", type=int, default=None, help="Max images to evaluate")
    args = p.parse_args()
    
    run_eval(args.checkpoint, args.save_preds, args.log,
             tta=args.tta, region_vote=args.region_vote,
             limit=args.limit, which=args.which,
             presence_gate=args.presence_gate,
             logit_adjust=args.logit_adjust, logit_prior=args.logit_prior)
    print("DONE", flush=True)
