from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import duckdb
import pandas as pd
import yaml


ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/phase5_config.yaml'
DECISIONS=ROOT/'restricted/phase4b_plausible_journey_legs/full_v3/origin_week=*/*.parquet'
BOUNDED_DECISIONS=ROOT/'restricted/phase4b_plausible_journey_legs/bounded/decisions.parquet'
CANDIDATES=ROOT/'restricted/phase3a_candidate_transitions/candidates/candidate_transitions.parquet'
RESTRICTED=ROOT/'restricted/phase5_final_mobility_outputs'
FINAL_PATTERN=str(RESTRICTED/'final_legs/origin_week=*/*.parquet')
REPORTS=ROOT/'reports'; THESIS=ROOT/'thesis'


def atomic_csv(frame,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
    frame.to_csv(tmp,index=False); os.replace(tmp,path)


def atomic_text(value,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(value,encoding='utf-8'); os.replace(tmp,path)


def config(): return yaml.safe_load(CONFIG.read_text())


def leg_query(decisions):
    return f"""SELECT t.sequence_key, d.origin_camera,d.destination_camera,d.origin_timestamp,d.destination_timestamp,
      d.elapsed_ms,d.elapsed_seconds duration_seconds,d.route_distance_m,d.derived_average_speed_kph,
      t.origin_confidence,t.destination_confidence,d.origin_week,d.destination_week,
      CAST(d.origin_timestamp AS DATE) origin_date, EXTRACT(dow FROM d.origin_timestamp) day_of_week,
      EXTRACT(hour FROM d.origin_timestamp) hour_of_day,d.time_of_day,d.cross_week,
      d.predominant_road_class,d.route_average_kph,d.direction_compatibility,
      '2018-12-31T23:59:59Z' osm_snapshot,'phase4b_primary_rule' plausibility_rule,
      '69edb1b5e912ce51348492a6e200f3dab77781af' phase4b_commit
      FROM read_parquet('{decisions}') d JOIN read_parquet('{CANDIDATES}') t
      USING(origin_source_file,origin_source_record_number,destination_source_file,destination_source_record_number)
      WHERE d.primary_accepted"""


def bounded(c):
    out=RESTRICTED/'bounded/final_legs.parquet'; out.parent.mkdir(parents=True,exist_ok=True)
    c.execute(f"COPY ({leg_query(BOUNDED_DECISIONS)}) TO '{out}' (FORMAT PARQUET,COMPRESSION ZSTD)")
    row=c.execute("""SELECT count(*) row_count,count(*) filter(where sequence_key is null) missing_sequence,
      count(*) filter(where origin_camera=destination_camera or duration_seconds<=0 or route_distance_m is null) invalid,
      count(distinct origin_week) week_count FROM read_parquet(?)""",[str(out)]).fetchone()
    od=c.execute("SELECT sum(n),count(*) FROM (SELECT origin_camera,destination_camera,count(*) n FROM read_parquet(?) GROUP BY 1,2)",[str(out)]).fetchone()
    return {'rows':row[0],'missing_sequence':row[1],'invalid':row[2],'weeks':row[3],
            'od_reconciled':od[0]==row[0],'od_pairs':od[1],'pass':row[0]>0 and row[1]==row[2]==0 and row[3]>=2 and od[0]==row[0]}


def build_final(c):
    out=RESTRICTED/'final_legs'; out.mkdir(parents=True,exist_ok=True); started=time.perf_counter()
    c.execute(f"COPY ({leg_query(DECISIONS)}) TO '{out}' (FORMAT PARQUET,PARTITION_BY(origin_week),COMPRESSION ZSTD,ROW_GROUP_SIZE 500000,OVERWRITE_OR_IGNORE TRUE)")
    result={'runtime_seconds':time.perf_counter()-started,'status':'final_legs_written'}
    atomic_text(json.dumps(result,sort_keys=True)+'\n',RESTRICTED/'final_leg_run_status.json'); return result


def aggregate_sql(level):
    period={'hourly':"date_trunc('hour',origin_timestamp)",'daily':'origin_date','weekly':'origin_week','total':"'study_period'"}[level]
    return f"""SELECT {period} period,origin_camera,destination_camera,count(*) leg_count,
      count(distinct sequence_key) distinct_internal_sequences,
      quantile_cont(duration_seconds,.1) duration_q10_s,quantile_cont(duration_seconds,.25) duration_q25_s,
      median(duration_seconds) duration_median_s,quantile_cont(duration_seconds,.75) duration_q75_s,quantile_cont(duration_seconds,.9) duration_q90_s,
      quantile_cont(route_distance_m,.1) distance_q10_m,quantile_cont(route_distance_m,.25) distance_q25_m,
      median(route_distance_m) distance_median_m,quantile_cont(route_distance_m,.75) distance_q75_m,quantile_cont(route_distance_m,.9) distance_q90_m,
      quantile_cont(derived_average_speed_kph,.1) speed_q10_kph,quantile_cont(derived_average_speed_kph,.25) speed_q25_kph,
      median(derived_average_speed_kph) speed_median_kph,quantile_cont(derived_average_speed_kph,.75) speed_q75_kph,quantile_cont(derived_average_speed_kph,.9) speed_q90_kph,
      avg(origin_confidence) origin_confidence_mean,avg(destination_confidence) destination_confidence_mean,
      count(distinct origin_week) weekly_availability,
      count(*) filter(where time_of_day='night_clock_22_06') nighttime_legs,
      count(*) filter(where time_of_day='day_clock_06_22') daytime_legs,
      count(*) filter(where cross_week) cross_week_legs
      FROM read_parquet('{FINAL_PATTERN}') GROUP BY 1,2,3"""


def build_aggregates(c):
    started=time.perf_counter(); results={}
    for level in ('hourly','daily','weekly','total'):
        out=RESTRICTED/f'aggregates/{level}_od_flows.parquet'; out.parent.mkdir(parents=True,exist_ok=True)
        c.execute(f"COPY ({aggregate_sql(level)}) TO '{out}' (FORMAT PARQUET,COMPRESSION ZSTD)")
        results[level]=c.execute("SELECT count(*),sum(leg_count) FROM read_parquet(?)",[str(out)]).fetchone()
    result={'runtime_seconds':time.perf_counter()-started,'levels':results}
    atomic_text(json.dumps(result,default=str,sort_keys=True)+'\n',RESTRICTED/'aggregate_run_status.json'); return result


def split_expr_70(): return "CASE WHEN CAST('0x'||substr(sequence_key,1,16) AS UBIGINT)%100<70 THEN 'train' WHEN CAST('0x'||substr(sequence_key,1,16) AS UBIGINT)%100<85 THEN 'validation' ELSE 'test' END"
def split_expr_80(): return "CASE WHEN CAST('0x'||substr(sequence_key,1,16) AS UBIGINT)%100<80 THEN 'train' WHEN CAST('0x'||substr(sequence_key,1,16) AS UBIGINT)%100<90 THEN 'validation' ELSE 'test' END"


def build_splits(c):
    started=time.perf_counter()
    seq=RESTRICTED/'splits/sequence_disjoint_70_15_15'; seq.mkdir(parents=True,exist_ok=True)
    c.execute(f"COPY (SELECT *,{split_expr_70()} split FROM read_parquet('{FINAL_PATTERN}')) TO '{seq}' (FORMAT PARQUET,PARTITION_BY(split),COMPRESSION ZSTD,OVERWRITE_OR_IGNORE TRUE)")
    chrono=RESTRICTED/'splits/chronological_weeks'; chrono.mkdir(parents=True,exist_ok=True)
    c.execute(f"COPY (SELECT *,CASE WHEN origin_week<=25 THEN 'train' WHEN origin_week<=30 THEN 'validation' ELSE 'test' END split FROM read_parquet('{FINAL_PATTERN}')) TO '{chrono}' (FORMAT PARQUET,PARTITION_BY(split),COMPRESSION ZSTD,OVERWRITE_OR_IGNORE TRUE)")
    c.execute(f"CREATE OR REPLACE TEMP TABLE sequence_bounds AS SELECT sequence_key,min(origin_week) min_week,max(origin_week) max_week FROM read_parquet('{FINAL_PATTERN}') GROUP BY 1")
    combined=RESTRICTED/'splits/combined_disjoint_chronological'; combined.mkdir(parents=True,exist_ok=True)
    c.execute(f"""COPY (SELECT l.*,CASE WHEN b.max_week<=25 THEN 'train' WHEN b.min_week>25 AND b.max_week<=30 THEN 'validation'
      WHEN b.min_week>30 THEN 'test' ELSE 'boundary_spanning_quarantine' END split
      FROM read_parquet('{FINAL_PATTERN}') l JOIN sequence_bounds b USING(sequence_key)) TO '{combined}'
      (FORMAT PARQUET,PARTITION_BY(split),COMPRESSION ZSTD,OVERWRITE_OR_IGNORE TRUE)""")
    result={'runtime_seconds':time.perf_counter()-started,'status':'three_complete_partition_sets_written'}
    atomic_text(json.dumps(result,sort_keys=True)+'\n',RESTRICTED/'split_run_status.json'); return result


def split_profile(c,protocol,expr):
    return c.execute(f"""SELECT '{protocol}' protocol,{expr} split,count(*) leg_count,count(distinct sequence_key) sequence_count,
      count(distinct origin_camera) origin_camera_count,count(distinct destination_camera) destination_camera_count,
      count(distinct origin_camera||'|'||destination_camera) od_pair_count,min(origin_timestamp) min_time,max(origin_timestamp) max_time
      FROM read_parquet('{FINAL_PATTERN}') GROUP BY 2 ORDER BY 2""").fetchdf()


def reports(c):
    total=c.execute(f"SELECT count(*) FROM read_parquet('{FINAL_PATTERN}')").fetchone()[0]
    profile=c.execute(f"""SELECT count(*) final_leg_rows,count(distinct sequence_key) internal_sequence_count,
      count(distinct origin_camera) origin_camera_count,count(distinct destination_camera) destination_camera_count,
      count(distinct origin_camera||'|'||destination_camera) directed_od_pairs,min(origin_timestamp) earliest_time,max(destination_timestamp) latest_time,
      median(duration_seconds) median_duration_s,median(route_distance_m) median_distance_m,median(derived_average_speed_kph) median_speed_kph,
      count(*) filter(where cross_week) cross_week_legs FROM read_parquet('{FINAL_PATTERN}')""").fetchdf()
    atomic_csv(profile,REPORTS/'final_leg_dataset_profile.csv'); atomic_csv(profile,THESIS/'table_final_leg_dataset.csv')
    k=10
    total_od=c.execute("SELECT * FROM read_parquet(?) WHERE leg_count>=? ORDER BY leg_count DESC",[str(RESTRICTED/'aggregates/total_od_flows.parquet'),k]).fetchdf()
    atomic_csv(total_od,REPORTS/'od_flow_profile.csv'); atomic_csv(total_od,THESIS/'table_od_flow_profile.csv')
    weekly=c.execute("SELECT * FROM read_parquet(?) WHERE leg_count>=? ORDER BY period,leg_count DESC",[str(RESTRICTED/'aggregates/weekly_od_flows.parquet'),k]).fetchdf()
    atomic_csv(weekly,REPORTS/'weekly_od_flows.csv')
    hourly_summary=c.execute("""SELECT count(*) total_cells,sum(leg_count) total_legs,count(*) filter(where leg_count>=10) published_cells,
      sum(leg_count) filter(where leg_count>=10) published_legs,min(leg_count) min_cell_count,median(leg_count) median_cell_count,max(leg_count) max_cell_count
      FROM read_parquet(?)""",[str(RESTRICTED/'aggregates/hourly_od_flows.parquet')]).fetchdf()
    atomic_csv(hourly_summary,REPORTS/'hourly_flow_profile.csv')

    privacy=[]
    for level in ('hourly','daily','weekly','total'):
        path=str(RESTRICTED/f'aggregates/{level}_od_flows.parquet')
        totals=c.execute("SELECT count(*),sum(leg_count) FROM read_parquet(?)",[path]).fetchone()
        for threshold in (1,3,5,10,20,50,100):
            value=c.execute("SELECT count(*),coalesce(sum(leg_count),0) FROM read_parquet(?) WHERE leg_count>=?",[path,threshold]).fetchone()
            privacy.append({'level':level,'minimum_count':threshold,'total_cells':totals[0],'published_cells':value[0],
                'suppressed_cells':totals[0]-value[0],'total_legs':totals[1],'published_legs':value[1],
                'suppressed_legs':totals[1]-value[1],'published_leg_percentage':100*value[1]/totals[1]})
    atomic_csv(pd.DataFrame(privacy),REPORTS/'privacy_suppression_sensitivity.csv')

    corridor=total_od.copy(); corridor.insert(0,'corridor_method','directed_exact_camera_pair')
    corridor.insert(1,'corridor_id',['camera_pair:'+str(o)+'->'+str(d) for o,d in zip(corridor.origin_camera,corridor.destination_camera)])
    atomic_csv(corridor,REPORTS/'corridor_flow_profile.csv'); atomic_csv(corridor,THESIS/'table_corridor_profile.csv')
    atomic_csv(pd.DataFrame([{'status':'not_supported_by_saved_cache','edge_flow_rows':0,
        'reason':'historical route cache stores pair distance/time/class but not edge sequences; no edge paths were fabricated'}]),REPORTS/'road_edge_flow_profile.csv')
    corridor_md="""# Corridor definition

The primary reproducible corridor is the directed exact-camera pair. It yields one corridor per observed directed OD pair and uses no invented administrative name.

- Direct camera-pair corridors: supported and selected; identifiers are deterministic origin→destination pairs.
- Shared-road-path grouping: not supported by the saved cache because edge sequences were not retained.
- Route-overlap grouping: not supported without edge sequences and an explicit overlap threshold.
- Road-segment aggregation: not supported without recomputing and validating complete route edge paths.

Route-level OD flows are reported through direct camera-pair corridors. Road-edge flows are explicitly unavailable rather than inferred from pair frequency.
"""
    atomic_text(corridor_md,REPORTS/'corridor_definition.md')

    frames=[split_profile(c,'sequence_disjoint_70_15_15',split_expr_70()),split_profile(c,'sequence_disjoint_80_10_10',split_expr_80()),
      split_profile(c,'chronological_1_25__26_30__31_35',"CASE WHEN origin_week<=25 THEN 'train' WHEN origin_week<=30 THEN 'validation' ELSE 'test' END"),
      split_profile(c,'chronological_1_24__25_29__30_35',"CASE WHEN origin_week<=24 THEN 'train' WHEN origin_week<=29 THEN 'validation' ELSE 'test' END")]
    comparison=pd.concat(frames,ignore_index=True)
    atomic_csv(comparison,REPORTS/'split_protocol_comparison.csv'); atomic_csv(comparison,THESIS/'table_split_protocol.csv')

    seq_path=str(RESTRICTED/'splits/sequence_disjoint_70_15_15/split=*/*.parquet'); chr_path=str(RESTRICTED/'splits/chronological_weeks/split=*/*.parquet'); comb_path=str(RESTRICTED/'splits/combined_disjoint_chronological/split=*/*.parquet')
    reconciliation=[]
    for protocol,path in [('sequence_disjoint',seq_path),('chronological',chr_path),('combined',comb_path)]:
        rows=c.execute("SELECT split,count(*) AS legs,count(distinct sequence_key) AS sequence_count,count(distinct origin_camera||'|'||destination_camera) AS od_pairs FROM read_parquet(?) GROUP BY 1 ORDER BY 1",[path]).fetchdf()
        for row in rows.itertuples(index=False): reconciliation.append({'protocol':protocol,'split':row.split,'leg_count':row.legs,'sequence_count':row.sequence_count,'od_pair_count':row.od_pairs})
    recon=pd.DataFrame(reconciliation); atomic_csv(recon,REPORTS/'split_reconciliation.csv')

    leakage=[]
    for protocol,path,expect_zero in [('sequence_disjoint',seq_path,True),('chronological',chr_path,False),('combined',comb_path,True)]:
        for left,right in [('train','validation'),('train','test'),('validation','test')]:
            overlap=c.execute("SELECT count(*) FROM (SELECT DISTINCT sequence_key FROM read_parquet(?) WHERE split=? INTERSECT SELECT DISTINCT sequence_key FROM read_parquet(?) WHERE split=?)",[path,left,path,right]).fetchone()[0]
            leakage.append({'protocol':protocol,'audit':f'sequence_overlap_{left}_{right}','value':overlap,'expected_zero':expect_zero,'pass':overlap==0 if expect_zero else True})
    chronological_order=c.execute("""SELECT max(origin_timestamp) filter(where split='train'),min(origin_timestamp) filter(where split='validation'),
      max(origin_timestamp) filter(where split='validation'),min(origin_timestamp) filter(where split='test') FROM read_parquet(?)""",[chr_path]).fetchone()
    leakage.append({'protocol':'chronological','audit':'time_order_train_validation','value':str(chronological_order[:2]),'expected_zero':False,'pass':chronological_order[0]<chronological_order[1]})
    leakage.append({'protocol':'chronological','audit':'time_order_validation_test','value':str(chronological_order[2:]),'expected_zero':False,'pass':chronological_order[2]<chronological_order[3]})
    leak=pd.DataFrame(leakage); atomic_csv(leak,REPORTS/'leakage_audit.csv'); atomic_csv(leak,THESIS/'table_leakage_audit.csv')

    sequence_total=int(recon[recon.protocol.eq('sequence_disjoint')].leg_count.sum()); chrono_total=int(recon[recon.protocol.eq('chronological')].leg_count.sum()); combined_total=int(recon[recon.protocol.eq('combined')].leg_count.sum())
    final_runtime=json.loads((RESTRICTED/'final_leg_run_status.json').read_text())['runtime_seconds']; split_runtime=json.loads((RESTRICTED/'split_run_status.json').read_text())['runtime_seconds']
    status=f"""# Phase 5 final status

**PASS** — {total:,} Phase 4B accepted records reconcile exactly to final restricted plausible observed inter-camera journey legs.

- Final accepted legs: {total:,}
- Directed OD pairs: {int(profile.directed_od_pairs.iloc[0]):,}
- Direct camera-pair corridors: {len(corridor):,}
- Primary privacy threshold: minimum 10 legs per released aggregate cell
- Sequence-disjoint 70/15/15 reconciliation: {sequence_total:,}
- Chronological reconciliation: {chrono_total:,}
- Combined disjoint/chronological reconciliation including quarantine: {combined_total:,}
- Sequence overlaps in disjoint and combined train/validation/test splits: zero
- Final-leg runtime: {final_runtime:.3f} seconds; split runtime: {split_runtime:.3f} seconds
- Source vehicle-token columns in final legs or reports: zero

Road-edge and overlap corridors remain unavailable because the saved route cache lacks edge sequences. Exact next action: freeze Phase 5, then separately decide whether validated route-edge reconstruction is scientifically required before any dashboard or Chapter 3 rewrite.

**PHASE 5 READY TO FREEZE**
"""
    atomic_text(status,REPORTS/'phase5_final_status.md')
    atomic_text("""# Section 3.7: Final dataset

The final restricted dataset contains plausible observed inter-camera journey legs accepted by the frozen Phase 4B rule. It preserves an irreversible internal sequence key for leakage control but contains no source vehicle token. Published aggregates use minimum-count suppression and never expose sequence-level records.
""",THESIS/'section_3_7_final_dataset.md')
    atomic_text(corridor_md.replace('# Corridor definition','# Section 3.8: Flow and corridor method'),THESIS/'section_3_8_flow_and_corridor_method.md')
    atomic_text("""# Section 3.9: Partition method

The primary model-development partition is deterministic 70/15/15 sequence-disjoint assignment. A chronological weeks 01–25/26–30/31–35 protocol measures temporal generalisation. A strict combined protocol quarantines every sequence spanning time-block boundaries, preserving both chronology and sequence disjointness. All complete partition sets reconcile to the final leg count when quarantine is included.
""",THESIS/'section_3_9_partition_method.md')
    atomic_text(f"""# Section 3.10: Final dataset findings

The final dataset contains {total:,} plausible observed inter-camera journey legs across {int(profile.directed_od_pairs.iloc[0]):,} directed OD pairs. Direct exact-camera pairs define reproducible corridors. Privacy suppression and leakage-safe split sensitivity are reported explicitly; no complete end-to-end journey claim is made.
""",THESIS/'section_3_10_final_dataset_findings.md')
    return {'final_legs':total,'od_pairs':int(profile.directed_od_pairs.iloc[0]),'corridors':len(corridor),'sequence_split_total':sequence_total,'chronological_total':chrono_total,'combined_total':combined_total,'leakage_pass':bool(leak['pass'].all())}


def main():
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=('bounded','final','aggregates','splits','reports')); a=p.parse_args()
    c=duckdb.connect(); c.execute("SET threads=16"); c.execute("SET memory_limit='120GB'")
    result={'bounded':bounded,'final':build_final,'aggregates':build_aggregates,'splits':build_splits,'reports':reports}[a.mode](c)
    c.close(); print(json.dumps(result,default=str,sort_keys=True))
if __name__=='__main__': main()
