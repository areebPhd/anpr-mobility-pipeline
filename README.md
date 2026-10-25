# ANPR mobility construction pipeline

Research software supporting contribution C1 of Areeb Alshoshan's PhD thesis.
The implementation provides ingestion and reconciliation, consecutive transition
construction, camera/network helpers, rule-based leg selection, aggregation,
partitioning, and qualified within-week journey chaining.

**Version:** 0.1.0. Released under the [MIT licence](LICENSE).

## Run the fictional example

Use Python 3.11 or later in a virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,geo]'
python examples/fictional_pipeline.py
python -m pytest
```

The example generates its own fictional observations and route assumptions in
memory. It prints aggregate counts only, needs no research data and makes no
network requests. It exercises pairing, classification, primary leg selection,
OD aggregation and within-week chaining. It does not demonstrate camera matching
or reproduce the thesis's full-data results.

## Package contents

- `src/ch3_pipeline/`: retained research implementation.
- `scripts/`: research processing and diagnostic drivers; review their inputs
  before use. Some require authorised observations or a supplied road network.
- `configs/`: research configuration templates, including historical choices.
  Original machine-specific input roots are replaced with `data/input`.
- `tests/`: synthetic software tests. Historical document-content and
  restricted-data replay tests are excluded from this public release.
- `docs/REPRODUCIBILITY.md`: scope, prerequisites and known historical differences.

The configuration templates include large research-machine resource settings;
they are not the configuration used by the small example. Supply appropriate
paths and resources before using a research driver. No driver runs on install.

## Scientific scope

Accepted inter-camera legs and inferred chains are rule-defined analytical
objects, not verified complete journeys. Sequence separation does not prove
physical-vehicle independence. Pseudonymous hashes and small-cell suppression
do not establish anonymity or formal privacy. See the important partition
helper difference in `docs/REPRODUCIBILITY.md`.

## Citation and related software

Use `CITATION.cff` to identify this software. Thesis DOI and final deposit
metadata will be added when available. The companion transport evaluation
implementation is available at
[transport-synthetic-data-evaluation](https://github.com/areebPhd/transport-synthetic-data-evaluation).
