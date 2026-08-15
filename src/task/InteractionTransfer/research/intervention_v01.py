"""V0.1 action intervention on a fixed synthetic current state."""
import torch
from src.task.InteractionTransfer.model import InteractionTransfer


@torch.no_grad()
def run():
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b, no, nh = (1, 512, 1538) if device.type == "cuda" else (2, 12, 20)
    o = torch.randn(b, no, 3, device=device) * .1
    h = torch.randn(b, nh, 3, device=device) * .1
    on = torch.nn.functional.normalize(torch.randn_like(o), dim=-1)
    hn = torch.nn.functional.normalize(torch.randn_like(h), dim=-1)
    action = torch.randn_like(h) * .01
    model = InteractionTransfer(k=16, radius=.05, dense_checkpoint=None if device.type == "cuda" else "synthetic").to(device).eval()
    variants = {"gt": action, "zero": torch.zeros_like(action),
                "reverse": -action, "shuffle": action.flip(0)}
    out = {name: model(o, on, h, hn, value) for name, value in variants.items()}
    message_norm = {name: float(value["edge_message"].norm()) for name, value in out.items()}
    effect_delta = {name: float((value["object_flow"] - out["zero"]["object_flow"]).norm()) for name, value in out.items()}
    assert message_norm["zero"] == 0.0
    assert message_norm["gt"] > 0.0 and message_norm["reverse"] > 0.0
    print({"message_norm": message_norm, "effect_delta_from_zero": effect_delta})


if __name__ == "__main__":
    run()
