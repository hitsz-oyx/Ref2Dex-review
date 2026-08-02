from __future__ import annotations

from types import SimpleNamespace

from src.base.base_runner import BaseRunner


def _validation_runner(*, patience: int = 2) -> tuple[BaseRunner, list[tuple[int, bool]]]:
    runner = BaseRunner.__new__(BaseRunner)
    runner.cfg = SimpleNamespace(
        train=SimpleNamespace(
            metric_for_best="val/loss",
            lower_is_better=True,
            early_stopping_patience=patience,
            early_stopping_threshold=0.0,
        )
    )
    runner.best_metric = None
    runner.early_stopping_metric = None
    runner.epochs_without_improvement = 0
    runner.evals_without_improvement = 0
    runner._early_stopping_triggered = False
    runner._last_validation_epoch = None
    runner.is_primary = False
    runner._record_metrics = lambda metrics, epoch: None
    saved: list[tuple[int, bool]] = []
    runner.save = lambda *, epoch, is_best: saved.append((epoch, is_best))
    return runner, saved


def test_validation_keeps_checkpoint_best_and_early_stop_reference_separate() -> None:
    runner, saved = _validation_runner()

    assert not runner._handle_validation({"val/loss": 0.5}, epoch=1)
    assert runner.best_metric == 0.5
    assert runner.early_stopping_metric == 0.5
    assert runner.evals_without_improvement == 0

    # This was the previous bug: checkpoint selection updated best_metric
    # first, then the same value was compared against itself and counted as a
    # non-improvement.  A real improvement must reset patience instead.
    assert not runner._handle_validation({"val/loss": 0.4}, epoch=2)
    assert runner.best_metric == 0.4
    assert runner.early_stopping_metric == 0.4
    assert runner.evals_without_improvement == 0
    assert saved == [(1, True), (2, True)]

    assert not runner._handle_validation({"val/loss": 0.4}, epoch=3)
    assert runner.evals_without_improvement == 1
    assert runner._handle_validation({"val/loss": 0.4}, epoch=4)
    assert runner._early_stopping_triggered


def test_step_validation_uses_the_same_stop_path() -> None:
    runner = BaseRunner.__new__(BaseRunner)
    runner.cfg = SimpleNamespace(
        train=SimpleNamespace(
            log_every_steps=100,
            eval_every_steps=1,
            save_every_steps=None,
        )
    )
    runner.train_loader = [{"id": 0}, {"id": 1}]
    runner.val_loaders = {"val/": object()}
    runner.global_step = 0
    runner.total_steps = 10
    runner._require_train_ready = lambda: None
    runner._set_train_epoch = lambda epoch: None
    runner.train_mode = lambda: None
    runner.prepare_batch = lambda batch: batch
    runner.train_step = lambda batch: {"loss": 1.0}
    runner.batch_size = lambda batch: 1
    runner.evaluate_all = lambda: {"val/loss": 1.0}
    runner._record_metrics = lambda metrics, epoch: None
    runner._compute_averager_metrics = lambda averager, prefix: {}

    handled: list[tuple[dict[str, float], int]] = []

    def stop_after_first_validation(metrics: dict[str, float], epoch: int) -> bool:
        handled.append((metrics, epoch))
        return True

    runner._handle_validation = stop_after_first_validation

    runner.train_epoch(0)

    assert handled == [({"val/loss": 1.0}, 1)]
    assert runner.global_step == 1


def test_test_loader_bundle_is_separate_from_validation() -> None:
    runner = BaseRunner.__new__(BaseRunner)
    train = object()
    val = object()
    test = object()
    unpacked = runner._unpack_dataloader_bundle(
        (train, val, test, {"source": "unit-test"}, {"val/": val}, {"test/": test})
    )

    assert unpacked == (
        train, val, test, {"source": "unit-test"}, {"val/": val}, {"test/": test}
    )


def test_eval_mode_prefers_test_metrics_when_available() -> None:
    runner = BaseRunner.__new__(BaseRunner)
    runner.mode = "eval"
    runner.test_loaders = {"test/": object()}
    runner.test_loader = None
    runner.is_primary = False
    runner.evaluate_test_all = lambda: {"test/loss": 0.1}
    runner.evaluate_all = lambda: {"val/loss": 0.2}

    assert runner.run() == {"test/loss": 0.1}
