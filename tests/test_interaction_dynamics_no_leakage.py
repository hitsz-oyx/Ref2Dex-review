import inspect


def test_future_targets_are_not_model_inputs_when_model_exists():
    try:
        from src.task.InteractionDynamics.model import InteractionDynamicsModel
    except ModuleNotFoundError:
        return
    forbidden = {"obj_future", "future_obj_points", "obj_disp_gt", "effect_obj_disp_gt"}
    assert forbidden.isdisjoint(inspect.signature(InteractionDynamicsModel.forward).parameters)
