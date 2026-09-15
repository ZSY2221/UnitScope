"""Dataset adapters used by the UnitScope experiments."""

from .femnist import (
    FEMNISTPartition,
    load_leaf_femnist,
    load_tff_federated_emnist,
    partition_femnist_writers,
)
from .sent140 import Sent140Partition, load_and_partition_sent140
from .movielens import (
    MovieLensPartition,
    balance_movielens_partition,
    load_and_partition_movielens1m,
)
from .synthea import (
    SyntheaTask,
    build_synthea_future_inpatient_task,
    generate_health_card_policy_evidence,
)

__all__ = [
    "FEMNISTPartition",
    "Sent140Partition",
    "MovieLensPartition",
    "balance_movielens_partition",
    "SyntheaTask",
    "build_synthea_future_inpatient_task",
    "generate_health_card_policy_evidence",
    "load_leaf_femnist",
    "load_tff_federated_emnist",
    "load_and_partition_sent140",
    "load_and_partition_movielens1m",
    "partition_femnist_writers",
]
