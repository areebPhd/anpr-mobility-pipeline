# Reproducibility and release scope

This candidate packages retained Chapter 3 software for inspection and synthetic
testing. Source algorithms are preserved; machine-specific input roots are
made relative. The original research archive remains separate.

## Important partition distinction

`src/ch3_pipeline/phase5.py:sequence_partition` hashes the supplied key again
with a seed. The historical executed primary split in `scripts/run_phase5.py`
uses the first 16 hexadecimal digits of the existing sequence key modulo 100,
with buckets 0–69 / 70–84 / 85–99. These are different procedures. The helper
must not be cited as replaying the historical split. No historical split has
been changed for this release.

## Inputs and limits

Full research replay requires authorised observations, camera metadata, network
assets, configurations and fitted/derived states appropriate to each stage.
The historical manual camera-assignment ledger, full route-edge sequences and
raw routing cache are not supplied. Some modules retain exploratory or
provisional stages; their comments and configuration labels must be respected.
Original empirical outputs and confidential data are not bundled.

The public tests cover synthetic boundary cases. The four excluded test modules
are `test_chapter3_mathematics.py`, `test_phase7a.py`, `test_phase7b.py` and
`test_phase8d.py`; they require historical thesis documents or restricted
research assets. Their historical pass counts are not represented as public
release test results. The old thesis-figure generator is also excluded.

The release includes no Git history, research CSV/Parquet outputs, source
observations, real vehicle/sequence values, camera-coordinate tables, model
weights, logs, caches, thesis draft or private backup. Identifiers appearing
as schema names and deliberately fictional test literals describe the API.
