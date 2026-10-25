"""Small fictional demonstration; no external input or network access."""
from datetime import datetime
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from ch3_pipeline.ingestion import Observation, consecutive_pairs
from ch3_pipeline.phase4a import TransitionFacts, primary_classification, derived_average_speed_kph
from ch3_pipeline.phase4b import EligibleLeg, primary_rule
from ch3_pipeline.phase5 import od_aggregate
from ch3_pipeline.phase8 import AcceptedLeg, chain_legs

def main():
    # All labels, times, distances and reference speeds below are invented.
    observations = [
        Observation("FICTIONAL_VEHICLE_A", "FICTIONAL_CAMERA_A", "2000-01-03T09:00:00", .9),
        Observation("FICTIONAL_VEHICLE_A", "FICTIONAL_CAMERA_B", "2000-01-03T09:10:00", .9),
        Observation("FICTIONAL_VEHICLE_A", "FICTIONAL_CAMERA_C", "2000-01-03T09:20:00", .9),
        Observation("FICTIONAL_VEHICLE_B", "FICTIONAL_CAMERA_A", "2000-01-03T10:00:00", .9),
        Observation("FICTIONAL_VEHICLE_B", "FICTIONAL_CAMERA_B", "2000-01-03T12:00:00", .9),
    ]
    pairs = list(consecutive_pairs(iter(observations)))
    accepted = []
    chain_input = []
    for pair in pairs:
        start = datetime.fromisoformat(pair.origin_timestamp)
        end = datetime.fromisoformat(pair.destination_timestamp)
        duration = (end - start).total_seconds()
        facts = TransitionFacts(duration, pair.origin_camera == pair.destination_camera,
                                duration == 0, True, True, True)
        if primary_classification(facts) != "route_valid_intercamera_le_240m_candidate":
            continue
        distance = 3000.0
        speed = derived_average_speed_kph(distance, duration, True)
        if not primary_rule(EligibleLeg(duration, speed, 30.0, 1.0)):
            continue
        accepted.append(dict(origin_camera=pair.origin_camera, destination_camera=pair.destination_camera,
                             duration_seconds=duration, route_distance_m=distance, speed_kph=speed))
        chain_input.append(AcceptedLeg(pair.vehicle, 1, pair.origin_camera, pair.destination_camera,
                                      start, end, distance))
    od = od_aggregate(pd.DataFrame(accepted))
    result = {"fixture": "entirely fictional", "observations": len(observations),
              "consecutive_candidates": len(pairs), "accepted_legs": len(accepted),
              "od_cells": len(od), "chain_units": len(chain_legs(chain_input, 30)),
              "note": "Zero inter-leg gap starts a new chain under the retained rule."}
    assert result["consecutive_candidates"] == 3 and result["accepted_legs"] == 2
    assert result["od_cells"] == 2 and result["chain_units"] == 2
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
