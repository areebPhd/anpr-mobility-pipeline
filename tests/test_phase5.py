import pandas as pd
import pytest

from ch3_pipeline.phase5 import (
    chronological_partition,
    combined_partition,
    direct_pair_corridor,
    leakage_overlaps,
    od_aggregate,
    sequence_partition,
    suppress_rare_cells,
)


def test_od_aggregation():
    frame = pd.DataFrame({"origin_camera":["1","1","2"],"destination_camera":["2","2","1"],
        "duration_seconds":[10,20,30],"route_distance_m":[100,100,100],"speed_kph":[36,18,12]})
    result = od_aggregate(frame)
    assert result.leg_count.sum() == 3
    assert result.loc[(result.origin_camera=="1") & (result.destination_camera=="2"),"median_duration_seconds"].iloc[0] == 15


def test_direct_pair_corridor_is_directed_and_rejects_same_camera():
    assert direct_pair_corridor("1","2") != direct_pair_corridor("2","1")
    with pytest.raises(ValueError): direct_pair_corridor("1","1")


def test_rare_cell_suppression_boundary():
    frame=pd.DataFrame({"cell":[1,2,3],"n":[9,10,11]})
    assert suppress_rare_cells(frame,"n",10).cell.tolist() == [2,3]


def test_sequence_disjoint_split_is_deterministic():
    values=[sequence_partition(f"key-{index}") for index in range(1000)]
    assert values == [sequence_partition(f"key-{index}") for index in range(1000)]
    assert set(values) == {"train","validation","test"}


def test_chronological_boundaries():
    assert chronological_partition(25)=="train"
    assert chronological_partition(26)=="validation"
    assert chronological_partition(30)=="validation"
    assert chronological_partition(31)=="test"


def test_combined_split_quarantines_boundary_spanning_sequences():
    assert combined_partition(1,25)=="train"
    assert combined_partition(26,30)=="validation"
    assert combined_partition(31,35)=="test"
    assert combined_partition(24,26)=="boundary_spanning_quarantine"


def test_leakage_detection():
    frame=pd.DataFrame({"sequence_key":["a","b","b","c"],"split":["train","train","test","test"]})
    assert leakage_overlaps(frame)["train_test"] == 1


def test_partition_reconciliation_and_privacy_safe_columns():
    frame=pd.DataFrame({"sequence_key":[f"k{i}" for i in range(20)]})
    frame["split"]=frame.sequence_key.map(sequence_partition)
    assert frame.groupby("split").size().sum()==len(frame)
    report=frame.groupby("split").size().reset_index(name="rows")
    assert "sequence_key" not in report.columns
