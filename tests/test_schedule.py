from __future__ import annotations

from experiments.unitscope.torch_training import _stable_schedule


def test_frozen_slot_schedule_never_refills_after_deletion() -> None:
    universe = tuple(f"record-{index}" for index in range(20))
    positions = _stable_schedule(universe, round_index=3, seed=99, count=8)
    scheduled = {universe[position] for position in positions}
    removed = next(iter(scheduled))
    active = set(universe) - {removed}
    selected_after_deletion = scheduled & active
    assert len(selected_after_deletion) == 7
    assert selected_after_deletion.issubset(scheduled)
    assert not ((set(universe) - scheduled) & selected_after_deletion)
