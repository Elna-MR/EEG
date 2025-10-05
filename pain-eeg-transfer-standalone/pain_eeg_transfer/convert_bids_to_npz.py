
import numpy as np, pandas as pd
from pathlib import Path
import mne
from mne_bids import BIDSPath, read_raw_bids
from tqdm import tqdm

def _parse_event_map(s):
    out={}
    for tok in s.split(","):
        k,v = tok.split(":")
        out[k.strip()] = int(v)
    return out

def convert_bids(args):
    bids_root = Path(args.bids_root)
    event_map = _parse_event_map(args.event_map)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    X_list,y_list,subj_list,dom_list = [],[],[],[]
    participants = pd.read_csv(bids_root/"participants.tsv", sep="\t")
    for _,row in tqdm(participants.iterrows(), total=len(participants), desc="Subjects"):
        sub = str(row["participant_id"]).replace("sub-","").strip()
        bids_path = BIDSPath(root=bids_root, subject=sub, task=args.task, datatype="eeg")
        try:
            raw = read_raw_bids(bids_path=bids_path, verbose="ERROR")
        except Exception as e:
            print(f"[WARN] skip {sub}: {e}"); continue
        raw.load_data(); raw.pick(args.picks)
        if args.sfreq: raw.resample(args.sfreq)
        try:
            events, event_id = mne.events_from_annotations(raw)
        except Exception as e:
            print(f"[WARN] no annotations {sub}: {e}"); continue
        id_map={}
        for name,code in event_id.items():
            for k,v in event_map.items():
                if k.lower() in name.lower():
                    id_map[code]=v
        sel = np.isin(events[:,2], list(id_map.keys()))
        events = events[sel]
        if len(events)==0:
            print(f"[WARN] no matching events {sub}"); continue
        y = np.vectorize(id_map.get)(events[:,2])
        epochs = mne.Epochs(raw, events, event_id=None, tmin=args.tmin, tmax=args.tmax, baseline=tuple(args.baseline) if args.baseline else None, preload=True)
        X = epochs.get_data().astype("float32")
        if args.max_trials: X, y = X[:args.max_trials], y[:args.max_trials]
        domain = bids_root.name
        X_list.append(X); y_list.append(y.astype("int64"))
        subj_list.append(np.array([sub]*len(y))); dom_list.append(np.array([domain]*len(y)))
    if not X_list: raise RuntimeError("no data extracted")
    X = np.concatenate(X_list,0); y = np.concatenate(y_list,0)
    subjects = np.concatenate(subj_list,0); domain = np.concatenate(dom_list,0)
    np.savez_compressed(out/"raw_epochs.npz", X=X, y=y, subject=subjects, domain=domain)
    print(f"[OK] saved {X.shape} to {out/'raw_epochs.npz'}")
