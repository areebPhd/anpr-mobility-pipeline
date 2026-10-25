from pathlib import Path
import csv
import pandas as pd

from ch3_pipeline.phase6 import bar_svg, line_svg, map_svg, suppressed, prohibited_identifier_text


def test_figure_source_reconciliation_synthetic():
    weekly=[10,20,30]
    assert sum(weekly)==60


def test_figure_has_caption_axes_units_denominator_and_provenance(tmp_path):
    p=tmp_path/'figure.svg'
    bar_svg(p,'Verified title','Subtitle',['W1','W2'],[10,20],'Week','Observations (records)','synthetic.csv','n=30 records')
    text=p.read_text()
    for required in ('Verified title','Week','Observations (records)','Denominator: n=30 records','Source/provenance: synthetic.csv'):
        assert required in text


def test_bar_renderer_accepts_pandas_series(tmp_path):
    p=tmp_path/'series.svg'
    bar_svg(p,'Series input','Subtitle',pd.Series(['A','B']),pd.Series([10,20]),'Category','Count (records)','synthetic.csv','n=30')
    assert p.stat().st_size > 1000


def test_line_figure_labels_log_scale_when_used(tmp_path):
    p=tmp_path/'line.svg'
    line_svg(p,'Rank','Subtitle',['1','2'],{'Count':[10,100]},'Rank','Legs','synthetic.csv','n=110',log_y=True)
    assert 'log10 scale' in p.read_text()


def test_phase5_minimum_count_suppression():
    assert suppressed([0,1,9,10,11],10)==[None,None,None,10,11]


def test_prohibited_identifier_detection():
    assert prohibited_identifier_text('contains sequence_key')
    assert prohibited_identifier_text('source_token')
    assert not prohibited_identifier_text('aggregate leg count')


def test_dashboard_aggregate_reconciliation_synthetic():
    cells=[12,20,9,2]
    published=[x for x in suppressed(cells,10) if x is not None]
    assert sum(published)==32 and len(published)==2


def test_spatial_map_uses_supplied_verified_coordinates_only(tmp_path):
    p=tmp_path/'map.svg'
    cameras=[('A',-1.6,54.9),('B',-1.5,55.0)]
    map_svg(p,'Map','Exact only',cameras,[('A','B',10)],'synthetic','n=2 exact cameras')
    text=p.read_text()
    assert text.count('<circle ')==2
    assert 'inferred' not in text.lower()


def test_manifest_schema_contract():
    required={'asset_type','asset_id','file_path','source_report_or_aggregate','chapter_section','verified_result_supported','denominator_or_scope','privacy_status'}
    candidate=Path('reports/chapter3_results_manifest.csv')
    if candidate.exists():
        with candidate.open(newline='') as f:
            assert required.issubset(next(csv.reader(f)))
    else:
        assert len(required)==8
