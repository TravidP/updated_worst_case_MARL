# CB-WCE standalone · protocol 8.1

Independent Grid/Monaco experiments with MA2C, PPO and target-network DQL.
Original code and results are preserved in the original project. This directory has
its own Git history and can be moved without the original workspace.

- [中文训练与保存方案](docs/revision_plan_v8_zh.md)
- [English training and storage plan](docs/revision_plan_v8_en.md)
- [Implementation and verification status](docs/implementation_status.md)
- [Extraction provenance](provenance/extraction.json)

## Inputs still required

Publication validation scenarios are intentionally unspecified. Supply one frozen
manifest per network. The `pilot-manifest` command creates an explicitly **unfrozen**
Uniform correctness fixture, never a publication validation default.

Manifest structure:

```json
{
  "version": "8.1", "network": "grid", "split": "validation",
  "frozen": false,
  "scenarios": [{
    "id": "development_example", "horizon": 3600,
    "generation_seed": 9001,
    "blocks": [{"start": 0, "duration": 3600, "profile": "Uniform"}]
  }]
}
```

Use actual approved scenarios before freezing. Blocks support profile names or
eleven normalized mixture weights, `multiplier`, and spatial `sigma`. Intervals
must cover exactly 3600 seconds. IDs must be unique. Final-test manifests use
`split: test`; never select training parameters with them.

## Local correctness checks

Use Python 3.10 with the dependencies in `packaging/requirements.in`, preferably
the locked container. SUMO and TraCI must be available. Set `PYTHONPATH` to your
SUMO tools directory only if TraCI is not installed as a package.

```bash
export TF_ENABLE_ONEDNN_OPTS=0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export CBWCE_DATA_ROOT=/tmp/cbwce-inputs
export CBWCE_CACHE_ROOT=/tmp/cbwce-cache
python -m unittest discover -s tests -v
python main.py check --network grid
python main.py check --network monaco
python main.py pilot-manifest --network grid --output /tmp/grid-pilot.json
python main.py parent --network grid --controller dql --seed 9001 --pilot \
  --validation /tmp/grid-pilot.json --output /tmp/dql-parent-pilot \
  --max-episodes 1 --validate-every 1 --validation-repeats 1 \
  --pilot-block-seconds 10
python scripts/verify.py --output /tmp/cbwce-verification-new
```

Output directories must be new. Short pilot blocks deliberately change the
simulation horizon and are **not performance evidence**. Publication runs reject
all pilot budget/configuration overrides and reject stale verification gates.

## Initial training, WCE and continuation

Run `python main.py <stage> --help`. All stages require `--network`, `--controller`,
`--seed`, and `--validation`; `--output` defaults to a unique directory under
`CBWCE_RUN_ROOT` and must be explicit for resume. Publication additionally requires a
current `--gate`, the exact locked runtime, and an approved frozen validation set.

Use `references.json` to locate the best full parent/WCE. Never guess checkpoint
filenames. WCE requires `--parent`; continuation requires `--parent --method`;
fixed/online WCE also require `--wce`. Publication ancestry must be plateau-confirmed.

Resume an interrupted stage with the same arguments/output and `--resume`.
One owner holds `.run.lock`; duplicate jobs fail. Stage budget and exploration
are not reset. A segment's `interrupted` result is not convergence.

`python main.py tuning-configs --output /tmp/cbwce-candidates` generates the
24 prescribed development candidates without starting simulations.

## Storage and portability

`CBWCE_ASSETS_ROOT` defaults to bundled assets. `CBWCE_DATA_ROOT`,
`CBWCE_RUN_ROOT` and `CBWCE_CACHE_ROOT` identify external storage. Explicit
`--output` paths are authoritative for single runs; the Slurm matrix builder uses
the run root. Runtime simulation files live in temporary directories.

`references.json` tracks full best/latest/final states and light validation/common
budget snapshots. Validation histories and episode/block summaries are compressed
JSONL shards. Full lane traces require audit mode. Artifact hashes detect
corruption; only load trusted locally generated pickle training states.

```bash
python scripts/package_code.py --output /tmp/cbwce-code.tar.gz
python main.py package --kind inputs --source assets --output /tmp/cbwce-assets.tar.gz
python main.py archive --source /scratch/ACCOUNT/cbwce/completed-run \
  --destination /projects/ACCOUNT/cbwce/completed-run
```

Archive transfer never deletes the source. Check archival status before manually
removing inactive scratch copies. Scratch and node-local storage are not backups.

## Results and audit exports

`scripts/export_history.py --run RUN --output episodes.csv` exports only durable
complete episodes and removes discarded resume branches from the view; raw
attempt journals remain available for auditing.

`scripts/evaluate_suite.py --run RUN --manifest TEST.json --output EVALUATION`
evaluates the selected final endpoint. Add `--selection best_at_common_budget`
for the common-budget endpoint. It caches identical selections, writes compressed
result shards and retains the prescribed Uniform/peak audit traces.
`scripts/report.py --input EVALUATION --input ... --output REPORT` checks traffic
pairing and the three independent training seeds before computing comparisons.

## Container and Hábrók

`packaging/build.sh` uses the committed hash-locked Python dependency set and builds an
Apptainer image from a **digest-pinned Python 3.10 base** supplied through
`CBWCE_BASE_IMAGE`. The script records exact runtime and SIF checksum. Apptainer
must be installed on the build machine. Run the six-case verifier inside the
image before publication. Do not build/install dependencies inside training jobs.
Set `CBWCE_UPDATE_LOCK=1` only when intentionally refreshing the dependency lock.

`scripts/slurm/prepare_jobs.py --help` creates parent/WCE/continuation matrices
after checking prerequisites. It does not submit them. Run the next stage only
after application-level plateau completion and successful `sacct` accounting.
For example, from this source folder:

```bash
sbatch --array=0-17%4 scripts/slurm/train.sbatch /scratch/ACCOUNT/cbwce/parents.json
```

Use the actual matrix size printed by the generator. The job copies and verifies
the code release on `$TMPDIR`, runs the image with explicit binds, and writes
durable checkpoints directly under the scratch run root. In-container paths for
validation and gate files must be inside the bound roots. Use 8 GiB only as an
initial resource pilot; set publication memory from measured peak RSS × 1.3.

The launcher forwards termination signals and preserves the previous durable
checkpoint. Archive from a login node; compute nodes must not assume `/projects`
is mounted. See the bilingual plans for the full stopping and storage contract.

## License and attribution

The original MIT license and authorship headers are retained. Exact imported
working-tree contents and source commit are recorded in `provenance/extraction.json`.
