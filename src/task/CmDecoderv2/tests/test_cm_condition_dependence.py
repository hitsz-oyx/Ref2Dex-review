from types import SimpleNamespace

import numpy as np
import torch

from src.task.CmDecoderv2.research.cm_condition_dependence.diagnostics import (
    continuous_starts, paired_summary, select_donors, spread_starts,
)
from src.task.CmDecoderv2.research.cm_condition_dependence.run import decode_core, error_values


def make_bank(count=4):
    bank = dict(start_frame=np.arange(count)*30, sequence_number=np.zeros(count,dtype=int),
                cm_valid=np.ones((count,4),dtype=bool), object_pose=np.tile(np.eye(4),(count,1,1)),
                current_wrist=np.tile(np.eye(4),(count,1,1)), current_q=np.zeros((count,6)),
                active_fraction=np.full(count,.3), flow_rms_mm=np.full(count,10.),
                flow_mean_mm=np.tile([10.,0.,0.],(count,1)))
    bank['flow_mean_mm'][1] = [-10.,0.,0.]
    return bank


def test_donors_use_current_conditions_and_input_motion_only():
    bank=make_bank()
    bank['current_wrist'][2,0,3]=.031
    bank['sequence_number'][3]=1
    selected=select_donors(bank)
    assert selected['matched_swap'][0]==1
    assert selected['matched_swap'][2]==-1
    assert selected['swap'][3]==-1
    bank['target_q']=np.full((4,6),np.nan)
    bank['prediction_epe']=np.full(4,-1e9)
    after=select_donors(bank)
    for key in selected:
        np.testing.assert_array_equal(selected[key],after[key])
    bank['cm_valid'][1,2]=False
    assert select_donors(bank)['matched_swap'][0]==-1


def test_no_overlap_or_relaxed_threshold_fallback():
    bank=make_bank(2)
    bank['start_frame'][1]=19
    assert select_donors(bank)['swap'][0]==-1
    bank['start_frame'][1]=20
    assert select_donors(bank)['matched_swap'][0]==1
    bank['flow_rms_mm'][1]=21
    assert select_donors(bank)['matched_swap'][0]==-1
    assert select_donors(bank)['swap'][0]==1


def test_continuous_rollout_windows_never_bridge_missing_or_invalid_frames():
    bank=make_bank(40)
    bank['start_frame']=np.arange(40)
    bank['cm_valid'][20,0]=False
    eligible,lookup=continuous_starts(bank)
    assert eligible[0] and eligible[21]
    assert not eligible[6] and not eligible[25]
    selected=spread_starts(bank,eligible)
    assert all(eligible[i] for i in selected)
    assert all(abs(a-b)>=16 for a,b in zip(selected,selected[1:]))
    assert lookup[0,21]==21


def test_cluster_bootstrap_uses_paired_recipients():
    rows=[]
    for i,seq in enumerate(['a','a','b']):
        for condition,value in [('correct',1.),('identity',5.),('swap',3.)]:
            rows.append(dict(sample_id=i,sequence_id=seq,condition=condition,hand_epe_mm=value))
    rows.append(dict(sample_id=3,sequence_id='c',condition='correct',hand_epe_mm=100.))
    rows.append(dict(sample_id=3,sequence_id='c',condition='identity',hand_epe_mm=200.))
    stats=paired_summary(rows,repeats=100)['swap']
    assert stats['samples']==3 and stats['sequences']==2
    assert stats['frame_micro']['penalty_vs_correct_mm']==2.
    assert stats['micro_ci95']['penalty_vs_correct_mm']==[2.,2.]
    assert stats['frame_micro']['paired_correct_epe_mm']==1.


class CaptureCore:
    def __call__(self,tokens,anchors,normals,state,links):
        self.inputs=[v.clone() for v in (tokens,anchors,normals,state,links)]
        b,k=tokens.shape[:2]
        return dict(pred_q_delta=torch.zeros(b,k,6),pred_wrist_rotvec=torch.zeros(b,k,3),
                    pred_wrist_translation=torch.zeros(b,k,3))


def test_complete_cm_swap_and_zero_preserve_recipient_state():
    bank=make_bank(2)
    bank['cm_tokens']=np.arange(2*4*2*3).reshape(2,4,2,3).astype(float)
    bank['anchor_pos']=bank['cm_tokens']+100
    bank['anchor_normal']=bank['cm_tokens']+200
    bank['current_state']=np.stack([np.zeros(15),np.ones(15)])
    bank['link_features']=np.stack([np.zeros((18,10)),np.ones((18,10))])
    model=SimpleNamespace(core=CaptureCore())
    q,w=decode_core(model,bank,[0],'swap',[1],'cpu')
    for position,key in enumerate(['cm_tokens','anchor_pos','anchor_normal']):
        np.testing.assert_array_equal(model.core.inputs[position],bank[key][[1]])
    assert not model.core.inputs[3].any() and not model.core.inputs[4].any()
    np.testing.assert_array_equal(q,bank['current_q'][[0]])
    np.testing.assert_array_equal(w,bank['current_wrist'][[0]])
    decode_core(model,bank,[0],'zero_all',[0],'cpu')
    assert all(not value.any() for value in model.core.inputs[:3])


def test_vector_epe_uses_mm_and_is_not_coordinate_rmse():
    points=torch.tensor([[[.003,.004,0.],[.003,.004,0.]]])
    q=torch.zeros(1,6)
    wrist=torch.eye(4)[None]
    values=error_values(points,torch.zeros_like(points),q,q,wrist,wrist)
    np.testing.assert_allclose(values['hand_epe_mm'],5.,atol=1e-6)
    np.testing.assert_allclose(values['wrist_rotation_deg'],0.,atol=1e-6)


def test_recursive_state_is_fed_back_and_wrist_delta_is_right_multiplied():
    class MoveCore(CaptureCore):
        def __call__(self,*args):
            result=super().__call__(*args)
            result['pred_q_delta'][:]=.1
            result['pred_wrist_translation'][...,0]=.01
            return result
    bank=make_bank(2)
    for key in ('cm_tokens','anchor_pos','anchor_normal'):
        bank[key]=np.zeros((2,4,2,3))
    bank['current_wrist'][0,:3,:3]=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    model=SimpleNamespace(core=MoveCore())
    kinematics=SimpleNamespace(query_features=lambda *args:np.zeros((18,10)))
    q=torch.zeros(1,6)
    wrist=torch.tensor(bank['current_wrist'][[0]],dtype=torch.float32)
    for _ in range(2):
        q,wrist=decode_core(model,bank,[0],'correct',[0],'cpu',q=q,wrist=wrist,kinematics=kinematics)
    torch.testing.assert_close(q,torch.full((1,6),.2))
    torch.testing.assert_close(wrist[0,:3,3],torch.tensor([0,.02,0]))
