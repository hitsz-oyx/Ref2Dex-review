import numpy as np

PROFILES = {'pose20':(20.,15.),'pose40':(40.,30.),'pose80':(80.,60.)}


def quality_flags(effect_epe,actual_rms,reference_rms,position,rotation):
    """K4效果与K+1姿态各自判定；返回失败原因而非训练接受决定。"""
    n=len(effect_epe)
    for value,width in ((effect_epe,4),(actual_rms,4),(reference_rms,4),(position,5),(rotation,5)):
        if value.shape!=(n,width) or not np.isfinite(value).all() or (value<0).any():
            raise ValueError('Expected finite nonnegative K4 effect / K+1 pose arrays')
    ratio=effect_epe/np.maximum(1.,.25*np.maximum(actual_rms,reference_rms))
    flags=dict(effect=(ratio<=1).all(-1),effect_ratio_max=ratio.max(-1),
               position_max_mm=position.max(-1),rotation_max_deg=rotation.max(-1))
    for name,(mm,deg) in PROFILES.items():
        bad_position=(position>mm).any(-1)
        bad_rotation=(rotation>deg).any(-1)
        flags[name]=~(bad_position|bad_rotation)
        flags['combined'+name[4:]]=flags['effect']&flags[name]
        flags[name+'_reasons']=(~flags['effect']).astype(np.uint8)+2*bad_position.astype(np.uint8)+4*bad_rotation.astype(np.uint8)
    return flags
