
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

def make_riemann_features(args):
    data = np.load(args.in_npz, allow_pickle=True)
    X = data["X"]; y=data["y"]; subject=data["subject"]; domain=data["domain"]
    cov = Covariances(estimator='oas').fit_transform(X)
    ts = TangentSpace().fit(cov)
    X_ts = ts.transform(cov).astype("float32")
    np.savez_compressed(args.out_npz, X=X_ts, y=y, subject=subject, domain=domain)
    print(f"[OK] features {X_ts.shape} -> {args.out_npz}")
