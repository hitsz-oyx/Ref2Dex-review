import numpy as np
from src.task.CmDecoderv2.research.cross_hand_cm_swap.diagnostics import (
    gates, local_object_flow, contact_comparison, shifted_donors, paired_statistics,
)


def test_object_flow_uses_current_rotated_frame_and_mm():
    pose = np.tile(np.eye(4), (2,1,1))
    pose[:,:3,:3] = [[0,-1,0],[1,0,0],[0,0,1]]
    pose[1,1,3] = .002
    flow = local_object_flow(np.array([[.1,.2,.3]]), pose)
    np.testing.assert_allclose(flow, [[[2,0,0]]], atol=1e-10)


def test_empty_contact_is_not_match():
    points = np.array([[0.,0.,0.],[.03,0.,0.]])
    a = np.array([[0,0],[1,0],[1,0]],bool)
    b = np.array([[0,0],[1,0],[0,1]],bool)
    iou, gap = contact_comparison(points,a,b)
    np.testing.assert_allclose(iou,[0,1,0])
    np.testing.assert_allclose(gap,[1e6,0,30])


def test_gate_requires_every_transition_and_moving_effect():
    bank = {k:np.ones((5,4)) for k in ('actual_rms_mm','reference_rms_mm','contact_iou')}
    bank.update(target_valid=np.ones((5,4),bool), mano_valid=np.ones((5,4),bool),
                effect_epe_mm=np.zeros((5,4)), contact_centroid_mm=np.zeros((5,4)))
    bank['mano_valid'][1,3] = False
    bank['effect_epe_mm'][2,2] = 1.001
    bank['contact_iou'][3,0] = .249
    bank['actual_rms_mm'][4] = .999
    mask = gates(bank)
    assert mask['strict_moving'].tolist() == [True,False,False,False,False]
    assert mask['strict'].tolist() == [True,False,False,False,True]


def test_shift_is_same_parent_valid_and_far_or_missing():
    sequence = np.array([0,0,0,1])
    frame = np.array([0,10,30,100])
    valid = np.array([True,False,True,True])
    donor = shifted_donors(sequence,frame,valid)
    assert donor.tolist() == [2,2,0,-1]
    np.testing.assert_array_equal(donor,shifted_donors(sequence,frame,valid))


def test_parent_macro_and_paired_ci():
    result = paired_statistics(np.array([2.,2.,5.]), np.array([1.,1.,1.]),np.array([0,0,1]))
    assert result['micro'] == 3.
    assert result['macro'] == 3.5
    assert result['delta'] == 2.
    assert result['delta_ci95'] == [1.,4.]
