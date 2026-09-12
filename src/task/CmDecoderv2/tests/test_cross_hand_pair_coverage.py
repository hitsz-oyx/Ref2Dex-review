import numpy as np
from src.task.CmDecoderv2.research.cross_hand_pair_coverage.matching import (
    exact_effect_pairs,contact_filter,selection_stats,allowed,
)


def test_jensen_pruning_matches_brute_force_including_rotation():
    rng = np.random.default_rng(42)
    a = rng.normal(size=(8,20,3))
    b = a.copy()
    b[3:] += .4
    # Equal means alone must not imply equivalent pointwise flow.
    b[0] = -a[0]+2*a[0].mean(0)
    starts = np.arange(5)
    actual,edges = exact_effect_pairs(a,b,starts,np.ones(5,bool),np.ones(5,bool),'cpu')
    expected = np.zeros((5,5),bool)
    for i in starts:
        for j in starts:
            errors = np.linalg.norm(a[i:i+4]-b[j:j+4],axis=-1).mean(-1)
            rms = np.maximum(np.sqrt((a[i:i+4]**2).sum(-1).mean(-1)),np.sqrt((b[j:j+4]**2).sum(-1).mean(-1)))
            expected[i,j] = (errors<=np.maximum(1,.25*rms)).all()
    np.testing.assert_array_equal(actual,expected)
    assert not actual[0,0]


def test_invalid_source_and_recipient_are_excluded():
    flows = np.zeros((6,2,3))
    grid,_ = exact_effect_pairs(flows,flows,np.arange(3),np.array([1,0,1],bool),np.array([0,1,1],bool),'cpu')
    assert not grid[1].any()
    assert not grid[:,0].any()
    assert grid[0,2]


def test_source_reuse_and_temporal_overlap_do_not_inflate_capacity():
    starts = np.array([0,1,8])
    edges = dict(target=np.array([0,1,2]),source=np.array([0,0,0]),ratio=np.zeros((3,4)))
    result = selection_stats(edges,starts,np.ones(3,bool),'any_time')
    assert result['windows']==3
    assert result['unique_selected_sources']==result['one_to_one_capacity']==result['nonoverlap_greedy']==1
    assert result==selection_stats(edges,starts,np.ones(3,bool),'any_time')


def test_contact_empty_frames_fail_and_all_four_frames_required():
    edges = dict(target=np.array([0]),source=np.array([0]),ratio=np.zeros((1,4)),epe_mm=np.zeros((1,4)))
    mask = np.ones((4,2),bool)
    source = mask.copy()
    source[3] = False
    out = contact_filter(edges,mask,source,np.zeros((2,3)),np.array([0]))
    assert len(out['target'])==0


def test_nonoverlap_policy_reserves_shared_endpoint():
    assert allowed('nonoverlap',np.array([4,5,-5])).tolist()==[False,True,True]
