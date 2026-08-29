"""Run an experiment from a config file.

Examples (from the repo root; `point_only_sam_rs_Es_5pt/` is a package):

  python -m point_only_sam_rs_Es_5pt.run_experiment \
      --config point_only_sam_rs_Es_5pt/configs/e1_point_only.py
  python -m point_only_sam_rs_Es_5pt.run_experiment \
      --config point_only_sam_rs_Es_5pt/configs/e2_prototypes.py --epochs 50
  python -m point_only_sam_rs_Es_5pt.run_experiment --evaluate \
      --config point_only_sam_rs_Es_5pt/configs/e2_prototypes.py \
      --checkpoint point_only_sam_rs_Es_5pt/runs/checkpoints/E2_epoch_0010.pt \
      --save-preds /abs/path/point_only_sam_rs_Es_5pt/runs/predictions/E2_ep10 \
      --log /abs/path/point_only_sam_rs_Es_5pt/runs/logs/E2_ep10_eval.log
"""
import argparse
import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path

from .configs.base import Config, resolve
from .evaluate import evaluate
from .train import train


def load_config(path: str) -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    sys.path.insert(0, str(p.parent))          # so configs can `from base import Config`
    spec = importlib.util.spec_from_file_location("exp_config", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CONFIG


def _print_config(cfg: Config, log=None):
    def _out(m, log=None):
        print(m, flush=True)
        if log is not None:
            log.write(m + "\n")
            log.flush()
    _out("===== " + cfg.experiment + " =====", log)
    for k, v in asdict(cfg).items():
        _out(f"  {k}: {v}", log)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a point-only SAM experiment (v2)")
    parser.add_argument("--config", required=True, help="path to a configs/eX_*.py file")
    parser.add_argument("--evaluate", action="store_true",
                        help="evaluate a checkpoint instead of training")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--save-preds", default=None,
                        help="(with --evaluate) write per-image coloured segmentation "
                             "maps to this directory")
    parser.add_argument("--use-refine", action="store_true",
                        help="(with --evaluate) enable eval-time prototype refinement "
                             "(default OFF)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda | cpu (default: from config)")
    parser.add_argument("--log", default=None, help="also append log output to this file")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.train_manifest:
        cfg.train_manifest = args.train_manifest
    if args.val_manifest:
        cfg.val_manifest = args.val_manifest
    if args.save_dir:
        cfg.save_dir = args.save_dir
    if args.save_every is not None:
        cfg.save_every = args.save_every
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.device is not None:
        cfg.device = args.device
    if args.use_refine:
        cfg.proto_use_refine_at_eval = True

    log = None
    if args.log:
        p = Path(resolve(args.log))
        p.parent.mkdir(parents=True, exist_ok=True)
        log = open(p, "a")
    _print_config(cfg, log)

    if args.evaluate:
        if not args.checkpoint:
            parser.error("--evaluate requires --checkpoint")
        evaluate(cfg, args.checkpoint, log=log, save_preds=args.save_preds)
    else:
        train(cfg, log=log)
    if log is not None:
        log.close()


if __name__ == "__main__":
    main()
