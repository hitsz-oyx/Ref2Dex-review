import argparse

from src.base import cleanup_distributed, load_config
from src.task.Actiontoken.config import validate_config
from src.task.Actiontoken.runner import ActionTokenRunner, ActionTokenV2Runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src.task.Actiontoken.config:Config")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--device")
    args = parser.parse_args()
    cfg = load_config(args.config, overrides=args.set)
    if args.device:
        cfg.train.device = args.device
    validate_config(cfg)
    try:
        runner_class = (ActionTokenV2Runner if str(cfg.model.class_path).endswith("ActionTokenV2Model")
                        else ActionTokenRunner)
        runner_class(cfg, mode="train").run()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
