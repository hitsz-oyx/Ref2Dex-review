"""V0 最小可复查 sanity：python -m ...diag_forward"""
import torch
from src.task.InteractionTransfer.model import InteractionTransfer


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b, no, nh = (1, 512, 1538) if device.type == "cuda" else (2, 12, 20)
    o = torch.randn(b, no, 3, device=device) * .1
    h = torch.randn(b, nh, 3, device=device) * .1
    on = torch.nn.functional.normalize(torch.randn_like(o), dim=-1)
    hn = torch.nn.functional.normalize(torch.randn_like(h), dim=-1)
    model = InteractionTransfer(k=16, radius=.05, dense_checkpoint=None if device.type == "cuda" else "synthetic").to(device)
    out = model(o, on, h, hn, torch.randn_like(h) * .01)
    zero = model(o, on, h, hn, torch.zeros_like(h))
    assert out["object_flow"].shape == o.shape
    assert torch.isfinite(out["object_flow"]).all()
    max_zero = float(zero["edge_message"].abs().max())
    assert max_zero < 1e-6, max_zero
    print({"object_flow_shape": tuple(out["object_flow"].shape), "zero_message_max": max_zero,
           "valid_edges": int(out["edge_valid"].sum())})


if __name__ == "__main__":
    main()
