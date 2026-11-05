"""Data utilities for DiffJev (AGENTS.md section 5):

- ``synthetic``     : synthetic choice / noul / score generators for mock tests
- ``splits``        : deterministic grouped 90/5/5 splitting & overlap quarantine
- ``augmentation``  : K-proposal sampling and 15% certified null augmentation
- ``mixture``       : fixed-budget 4-bucket sampling trace (8 per bucket = 32)
- ``sources``       : OpenJevData-140k, ToolRet-Training-20w, ARC/BoolQ/HelpSteer2
                      fallback mappers, JevBench mapper, and manifest builder
"""

from .augmentation import DEFAULT_K_PROPOSAL, AugmentedMenu, augment_menu, sample_target_k
from .mixture import BUCKETS, build_trace, save_trace, trace_stats
from .sources import (
    AUDITED_NULL_PHRASES,
    SOURCES,
    PreparedDatasetManifest,
    inspect_schema,
    load_local_dataset,
    map_arc_record,
    map_boolq_record,
    map_helpsteer2_record,
    map_jevbench_record,
    map_openjev_record,
    map_toolret_record,
    materialize_example,
    prepare_training_manifest,
    record_dedup_keys,
)
from .splits import SPLITS, assign_split, group_records, normalize_text, split_records
from .synthetic import synthetic_choice, synthetic_mixture, synthetic_noul, synthetic_score

__all__ = [
    "DEFAULT_K_PROPOSAL",
    "AugmentedMenu",
    "augment_menu",
    "sample_target_k",
    "BUCKETS",
    "build_trace",
    "save_trace",
    "trace_stats",
    "AUDITED_NULL_PHRASES",
    "SOURCES",
    "PreparedDatasetManifest",
    "inspect_schema",
    "load_local_dataset",
    "map_arc_record",
    "map_boolq_record",
    "map_helpsteer2_record",
    "map_jevbench_record",
    "map_openjev_record",
    "map_toolret_record",
    "materialize_example",
    "prepare_training_manifest",
    "record_dedup_keys",
    "SPLITS",
    "assign_split",
    "group_records",
    "normalize_text",
    "split_records",
    "synthetic_choice",
    "synthetic_mixture",
    "synthetic_noul",
    "synthetic_score",
]
