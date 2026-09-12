import numpy as np
from scipy.spatial.transform import Rotation

LAGS = np.arange(-15,16)


def poses(tensor, transpose=False):
    out = np.tile(np.eye(4),(len(tensor),1,1))
    r = Rotation.from_quat(tensor[:,201:205]).as_matrix()
    out[:,:3,:3] = r.transpose(0,2,1) if transpose else r
    out[:,:3,3] = tensor[:,198:201]
    return out


def rotation_error(a,b):
    relative = a[:,:3,:3].transpose(0,2,1) @ b[:,:3,:3]
    return np.degrees(np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1)))


def translation_offset(a,b):
    delta = a[:,:3,3]-b[:,:3,3]
    bias = delta.mean(0)
    return bias,np.linalg.norm(delta-bias,axis=-1)*1000


def point_epe(a,b):
    return np.linalg.norm(a-b,axis=-1).mean(-1)


def lag_audit(actual,reference,eligible,lags=LAGS):
    """actual[t] 对 reference[t+lag]；共同支持、前半选lag、后半评估。"""
    margin = int(np.abs(lags).max())
    t = np.flatnonzero(eligible)
    t = t[(t>=margin)&(t<len(reference)-margin)]
    errors = np.stack([point_epe(actual[t],reference[t+lag]) for lag in lags]) if len(t) else np.empty((len(lags),0))
    if len(t)<4:
        return dict(support=len(t),selected_lag=None,oracle_lag=None),t,errors
    split = len(t)//2
    # Prefer zero/near-zero when losses tie; then the negative lag.
    def best(values):
        return int(np.lexsort((lags,np.abs(lags),values))[0])
    train_best,oracle = best(errors[:,:split].mean(-1)),best(errors.mean(-1))
    zero = int(np.flatnonzero(lags==0)[0])
    return dict(support=len(t),selected_lag=int(lags[train_best]),oracle_lag=int(lags[oracle]),
        baseline_mean_mm=float(errors[zero].mean()),oracle_mean_mm=float(errors[oracle].mean()),
        heldout_count=len(t)-split,heldout_baseline_mm=float(errors[zero,split:].mean()),
        heldout_selected_mm=float(errors[train_best,split:].mean())),t,errors
