#!/usr/bin/env python3
"""Standalone eval runner — import-safe from the package root."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from PRISM.e3_only.configs.base import Config, resolve
from PRISM.e3_only.evaluate import evaluate

def run_eval(checkpoint, save_preds, log_path):
    cfg = Config()
    cfg.device = "cuda"
    cfg.num_workers = 0
    
    log = open(log_path, "w") if log_path else None
    try:
        evaluate(cfg, resolve(checkpoint), log=log, save_preds=save_preds)
    finally:
        if log:
            log.close()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--save-preds", required=True)
    p.add_argument("--log", default="/dev/null")
    args = p.parse_args()
    
    run_eval(args.checkpoint, args.save_preds, args.log)
    print("DONE", flush=True)
