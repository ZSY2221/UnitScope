from experiments.unitscope.model import GroundTruth, LocalKind, PublicRecord
from experiments.unitscope.run_db_workload import (
    distinct_person_count,
    stable_local_count,
)


def test_person_count_deduplicates_one_person_across_silos() -> None:
    records = (
        PublicRecord("r1", "silo-0", "person-a-s0", LocalKind.ATOM, 1.0),
        PublicRecord("r2", "silo-1", "person-a-s1", LocalKind.ATOM, 1.0),
        PublicRecord("r3", "silo-1", "person-b-s1", LocalKind.ATOM, 1.0),
    )
    truth = GroundTruth((("r1", "person-a"), ("r2", "person-a"), ("r3", "person-b")))
    assert (
        distinct_person_count(
            [record.record_id for record in records], truth, {"person-a"}
        )
        == 1
    )


def test_stable_local_exposes_cross_silo_duplicate_bias() -> None:
    records = (
        PublicRecord("r1", "silo-0", "person-a-s0", LocalKind.ATOM, 1.0),
        PublicRecord("r2", "silo-1", "person-a-s1", LocalKind.ATOM, 1.0),
    )
    truth = GroundTruth((("r1", "person-a"), ("r2", "person-a")))
    assert stable_local_count(records, truth, {"person-a"}) == 2
