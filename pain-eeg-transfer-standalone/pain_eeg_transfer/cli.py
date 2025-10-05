
import argparse
from .convert_bids_to_npz import convert_bids
from .riemann_features import make_riemann_features
from .train_dann import run_training

def main():
    parser = argparse.ArgumentParser(prog="pet")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("convert-bids")
    p1.add_argument("--bids_root", required=True)
    p1.add_argument("--task", default="lep")
    p1.add_argument("--event_map", required=True)
    p1.add_argument("--tmin", type=float, default=-0.2)
    p1.add_argument("--tmax", type=float, default=0.8)
    p1.add_argument("--baseline", nargs=2, type=float, default=None)
    p1.add_argument("--sfreq", type=float, default=250)
    p1.add_argument("--picks", default="eeg")
    p1.add_argument("--max_trials", type=int, default=None)
    p1.add_argument("--out", required=True)

    p2 = sub.add_parser("riemann")
    p2.add_argument("--in_npz", required=True)
    p2.add_argument("--out_npz", required=True)

    p3 = sub.add_parser("train-dann")
    p3.add_argument("--features", required=True)
    p3.add_argument("--scheme", choices=["loso","lodo"], default="loso")
    p3.add_argument("--test_domain", default=None)
    p3.add_argument("--epochs", type=int, default=30)
    p3.add_argument("--batch_size", type=int, default=128)
    p3.add_argument("--lr", type=float, default=1e-3)
    p3.add_argument("--lambda", dest="lam", type=float, default=0.1)
    p3.add_argument("--hidden", type=int, default=128)
    p3.add_argument("--seed", type=int, default=42)
    p3.add_argument("--out_dir", required=True)

    args = parser.parse_args()
    if args.cmd == "convert-bids":
        convert_bids(args)
    elif args.cmd == "riemann":
        make_riemann_features(args)
    elif args.cmd == "train-dann":
        run_training(args)
