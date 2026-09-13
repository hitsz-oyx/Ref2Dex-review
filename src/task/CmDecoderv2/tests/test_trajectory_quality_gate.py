import numpy as np
import pytest
from src.task.CmDecoderv2.research.trajectory_quality_gate.gates import quality_flags


def inputs(n=1):
    return [np.ones((n,4)),np.ones((n,4)),np.ones((n,4)),np.full((n,5),20.),np.full((n,5),15.)]


def test_inclusive_threshold_and_last_endpoint():
    x=inputs(2);x[3][1,4]=20.001
    flags=quality_flags(*x)
    assert flags['combined20'].tolist()==[True,False]
    assert flags['pose20_reasons'].tolist()==[0,2]


def test_effect_and_pose_are_independent_and_all_steps_required():
    x=inputs(2);x[0][0,3]=1.001;x[3][1,0]=21
    flags=quality_flags(*x)
    assert flags['effect'].tolist()==[False,True]
    assert flags['pose20'].tolist()==[True,False]
    assert flags['combined20'].tolist()==[False,False]


def test_relative_effect_limit_and_reason_bits():
    x=inputs();x[1][:]=8;x[0][:]=2
    assert quality_flags(*x)['effect'][0]
    x[0][0,0]=2.1;x[3][0,0]=21;x[4][0,0]=16
    assert quality_flags(*x)['pose20_reasons'][0]==7


def test_nonfinite_and_wrong_window_are_rejected():
    x=inputs();x[0][0,0]=np.nan
    with pytest.raises(ValueError):quality_flags(*x)
    x=inputs();x[3]=x[3][:,:4]
    with pytest.raises(ValueError):quality_flags(*x)
