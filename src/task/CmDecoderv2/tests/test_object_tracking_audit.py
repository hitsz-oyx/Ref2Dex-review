import numpy as np
from scipy.spatial.transform import Rotation
from src.task.CmDecoderv2.research.object_tracking_audit.diagnostics import poses,translation_offset,lag_audit
from src.task.CmDecoderv2.research.cross_hand_cm_swap.diagnostics import local_object_flow


def test_native_xyzw_and_explicit_inverse():
    tensor = np.zeros((1,205))
    tensor[:,201:205] = Rotation.from_euler('z',90,degrees=True).as_quat()
    np.testing.assert_allclose(poses(tensor)[0,:3,:3]@np.array([1,0,0]),[0,1,0],atol=1e-10)
    np.testing.assert_allclose(poses(tensor,True)[0,:3,:3]@np.array([1,0,0]),[0,-1,0],atol=1e-10)


def test_fixed_world_translation_does_not_change_local_flow():
    p = np.tile(np.eye(4),(3,1,1));p[:,0,3] = [0,.1,.3]
    q = p.copy();q[:,:3,3] += [.2,-.4,1.]
    bias,residual = translation_offset(q,p)
    np.testing.assert_allclose(bias,[.2,-.4,1.])
    np.testing.assert_allclose(residual,0,atol=1e-10)
    np.testing.assert_allclose(local_object_flow(np.ones((2,3)),p),local_object_flow(np.ones((2,3)),q),atol=1e-10)


def test_lag_sign_common_support_and_heldout():
    rng = np.random.default_rng(4)
    reference = rng.normal(size=(90,2,3))
    actual = np.roll(reference,3,axis=0)
    result,t,errors = lag_audit(actual,reference,np.ones(90,bool))
    assert result['selected_lag']==result['oracle_lag']==-3
    assert result['heldout_selected_mm']==0
    assert errors.shape==(31,60)
    assert (t[[0,-1]]==[15,74]).all()


def test_insufficient_support_is_missing():
    result,t,e = lag_audit(np.zeros((20,1,3)),np.zeros((20,1,3)),np.ones(20,bool))
    assert result['selected_lag'] is None and e.shape==(31,0)
