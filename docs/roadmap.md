# iaq4j Roadmap

Requirements catalog for future implementation. Items are grouped by domain and
marked with priority: **P0** (next up), **P1** (soon), **P2** (later).

---

## Implementation Order

**Canonical sequence — ordered by impact, effort, and hard dependencies.**
Items must be completed in this order. Do not start an item until all items
above it are fully complete (all checkboxes ticked).

| # | Item | Priority | Rationale |
|---|---|---|---|
| 1 | ~~**Temporal Feature Engineering**~~ ✅ | P0 | Retrain is already required (window_size change broke all artifacts). Fully implement staleness detection, SyntheticSource timestamps, and cyclical time features before retraining. Starting any other work on top of stale artifacts is wasteful. |
| 2 | ~~**Security Hardening**~~ ✅ (HTTPS, rate limiting, headers) | P0 | Production gaps that exist right now. Cannot responsibly expose new endpoints from LLM Readiness without first hardening the surface. |
| 3 | ~~**Pytest Suite — Tier 1**~~ ✅ (168 tests) | P0 | Pure core: quantities, profiles, windowing, merkle, schemas. No mocking needed. |
| 4 | ~~**Training Checkpoint & Resume**~~ ✅ | P1 | Training on real data takes 6–16h per model. Graceful SIGINT/SIGTERM, periodic checkpoint, `--resume` CLI flag. |
| 5 | ~~**Pytest Suite — Tier 2**~~ ✅ (98 tests, 266 total) | P1 | Model forward passes, config merging, pipeline e2e with SyntheticSource, profile IO resolution. No bugs uncovered — all application code behaved correctly. |
| 6 | ~~**DAG Merkle Tree**~~ ✅ | P1 | Small + self-contained (merkle.py + tests only). Prerequisite for multi-source pipeline. Done while structure is simple — delaying means any multi-source work has to retrofit. |
| 7 | **Domain Exception Hierarchy** (R1) | P1 | Unifies 3 competing error patterns (structured exception, error dict, boolean flag) into one `IAQError` hierarchy with `DomainErrorCode` attached at origin. Small, self-contained, no pipeline.py changes. Prerequisite for Phase 1 typed exceptions. See `docs/architecture_appraisal.md`. |
| 8 | **Ingestion Consistency Validation** (datetime & units) | P1 | Pipeline ingestion has no timezone normalization, no unit detection/conversion, and no sampling interval checks. Mixed timestamp formats crashed InfluxDB DataFrameClient; kOhm vs Ohm VOC data passes range validation silently. Must be settled before multi-source pipeline adds more data sources with varying formats. Uses existing `quantities.py:convert_to_canonical()` and `SensorProfile.feature_units`. |
| 9 | ~~**Config Decomposition**~~ (R2) ✅ | P1 | Split `Settings` god object into `APIConfig`, `ModelConfig`, `DatabaseConfig`, `KANSidecarConfig` sub-configs. Fixed DI in `profiles.py` (explicit params), `schemas.py` (module-level field mapping), `orchestrator.py` (explicit types). Facade pattern preserves backward compat. |
| 10 | **Service Extraction** (R3) | P1 | Extract `PredictionService` from `main.py` to own 4 mutable globals (`predictors`, `inference_engines`, `active_model`, `_pending_mappings`). Extract `SensorRegistrationService` from 90+ lines of route-level field mapping logic. Enables thread safety and testability. See `docs/architecture_appraisal.md`. |
| 11 | **LLM Readiness — Phase 1** (structured internals) | P1 | Plumbing prerequisite for all agent work: config cache, InfluxDB reads, typed exceptions (now uses R1 hierarchy), `StructuredResponse`. No pipeline.py changes. Unblocks MCP. |
| 12 | **Multi-source Pipeline** | P1 | Depends on DAG Merkle tree (#6). Touches pipeline.py ingestion path once — doing it now while pipeline.py is stable means one rework, not two. |
| 13 | **Label Studio DB Integration** | P1 | Export InfluxDB data into Label Studio for annotation. Uses existing `InfluxDBSource` reads + Label Studio import API. Self-contained CLI command + new module, no pipeline.py changes. |
| 14 | **MCP Server** | P1 | Expose all agent-facing capabilities as MCP tools. Depends on Phase 1 (#11). Pipeline is now stable (ingestion settled by #12). |
| 15 | **Model Factory Registry** (R4) | P2 | Replace `build_model()` if/elif chain with declarative factory dict. Small, self-contained. See `docs/architecture_appraisal.md`. |
| 16 | **Data Source Registry** (R5) | P2 | Add registry + factory for data sources (mirrors sensor registry pattern). Remove CLI string matching. Enables plugin-style data source addition. See `docs/architecture_appraisal.md`. |
| 17 | **MLflow Integration** (tracing + remaining) | P2 | Adds training callbacks to the *final* pipeline shape — no rework needed. Demoted: observability is valuable but not blocking any downstream features. |
| 18 | **Pytest Suite — Tier 3** (IO/mocked) | P2 | Covers multi-source ingestion, Label Studio export, MCP tools, MLflow hooks, inference engine. More valuable after those features exist. ~100 tests, needs mocking. |
| 19 | **MCP Pipeline Design Tools** | P2 | Can reason about multiple sources (depends on #12). MCP tools for sensor inventory, pipeline spec design/validate/commit. Replaces LLM Readiness Phase 3. |
| 20 | **LLM-Driven Pipeline Design** | P2 | Capstone. Depends on MCP Server + Pipeline Design Tools + multi-source. LLM as full pipeline designer (feature engineering + model selection). |
| 21 | **Running Envelope Tracker** | P2 | `InferenceEngine` maintains long-running history buffer (24h+), computes rolling envelope baselines in real-time. Transitions from saved training-time baselines on cold start. Improves inference accuracy for long-running deployments where sensor drift is significant. |
| 22 | **Pipeline Data Enrichment** | P2 | New ENRICHMENT stage between INGESTION and FEATURE_ENGINEERING. `EnrichmentSource` ABC with `enrich(df) -> df` method joins external data by timestamp. Candidate sources: outdoor weather (OpenMeteo API), outdoor AQ (OpenAQ), derived rate-of-change columns (dVOC/dt, dT/dt), heater profile params (if added to firmware). Enriched columns become available to `SensorProfile.engineer_features()`. Depends on multi-source pipeline (#12) for consistent ingestion. |
| 23 | **Apple Silicon Training Optimization** | P1 | Optimize training loop for MPS (M4). Currently only device detection — no MPS-specific tuning. Changes: `torch.compile(backend="aot_eager")` on model forward pass (20-40% speedup), mixed precision via `torch.autocast("mps")` (15-30% + lower memory), DataLoader `num_workers=4` + `pin_memory=True` + `persistent_workers=True`, `non_blocking=True` on `.to(device)` transfers, increase batch size from 32 to 64-128 (M4 unified memory can handle it). All changes in `training/utils.py:train_model()` and `training/pipeline/orchestrator.py`. Validate with benchmark before/after on MLP 10-epoch run. |
| 24 | **Training Early Stopping** | P1 | Stop training after best epoch + patience window instead of running all epochs. Currently trains for the full epoch count even when val loss diverges (BNN diverged at epoch 19 but ran to 50). Add `EarlyStopping` callback: monitor val_loss, patience=15, restore best weights on stop. Saves hours on long runs and prevents overfitting degradation (MLP peaks at epoch 5, CNN at epoch 12). Config: `training.early_stopping_patience` in `model_config.yaml`. Changes in `training/utils.py:train_model()`. |
| 26 | **Cross-Training: Self-Supervised Pre-training** | P1 | Use 3 years of unlabeled BME680 data (raw sensor readings without BSEC IAQ scores) to pre-train feature representations via autoencoder, then fine-tune on recent labeled data. Warm-up filtering (600s) cleans power-cycle artifacts from historical data. Phase 1: masked autoencoder on (temperature, humidity, VOC resistance) windows learns drift-invariant representations. Phase 2: freeze encoder, attach prediction head, fine-tune on BSEC-labeled data. Phase 3: pseudo-labeling — use fine-tuned model to label high-confidence old data, retrain on combined set. Depends on warm-up filtering (done) and early stopping (#24). Target: R² > 0.8 on real data. |
| 25 | **Hyperparameter Sweep Framework** | P2 | Systematic parameter search integrated into CLI. `python -m iaq4j sweep --model bnn --grid kl_weight=0.00005,0.0001,0.0003 --grid learning_rate=0.0003,0.001 --sweep-epochs 30 --final-epochs 100`. Grid search with optional Bayesian optimization (optuna). Results table ranked by R², best config auto-retrained. Currently done via ad-hoc scripts (sweep_bnn.py). Formalize into `training/sweep.py` + CLI subcommand. Depends on early stopping (#24) to avoid wasting epochs. |

**Dependency graph:**
```
DAG Merkle Tree (done) ─┐
                        ├──→ Ingestion Consistency (#8) ──→ Multi-source Pipeline (#12) ─┐
                        │                                                                 │
Domain Exception Hierarchy (#7) ─┐                                                       │
                                 │                                                        │
Config Decomposition (#9, done) ─┤                                                        │
                                 │                                                        │
Service Extraction (#10) ────────┤                                                        │
                                 │                                                        │
                                 └──→ LLM Readiness Phase 1 (#11)                         │
                                              │                                           │
                                              ├──→ MCP Server (#14)                       │
                                              │         │                                 │
                                              │         ├──→ MCP Pipeline Design (#19) ←──┘
                                              │         │              │
                                              │         │              └──→ LLM-Driven Design (#20)
                                              │         │
Label Studio DB Integration (#13) ← independent
MLflow Integration (#17)          ← independent, P2
Model Factory Registry (#15)      ← independent, P2
Data Source Registry (#16)        ← before Multi-source Pipeline (#12) ideally
Pytest Tier 3 (#18) (covers all above)
```

**Key ordering rationale:**
- **Exception hierarchy before Phase 1**: Phase 1 specifies "typed exceptions" — R1 provides the hierarchy that Phase 1 populates. Small, self-contained, no conflicts.
- **Config decomposition before Phase 1**: Phase 1 adds config cache invalidation. Refactoring Settings first means Phase 1 caches are added to the right classes, not retrofitted later.
- **Service extraction before Phase 1**: Phase 1 adds InfluxDB read queries. Extracting services first means the queries are added to a service, not another global singleton.
- **R1→R2→R3 sequence**: Exception hierarchy is leaf (no deps). Config decomposition depends on knowing the error types. Service extraction uses the decomposed config.
- **Ingestion consistency before multi-source**: Multi-source pipeline adds CSV, Label Studio, and external API sources — each with different timestamp formats and unit conventions. Timezone normalization and unit detection must be in the ingestion path *before* more sources are wired in.
- **Phase 1 before multi-source**: Phase 1 doesn't touch pipeline.py. Multi-source does. No conflict, and Phase 1 can proceed in parallel conceptually.
- **Multi-source before MLflow**: Both touch pipeline.py but different areas (ingestion vs training callbacks). Doing multi-source first settles the pipeline shape. MLflow then writes hooks against the final structure — zero rework.
- **Label Studio DB after multi-source**: Can leverage any multi-source infrastructure but is otherwise self-contained. No hard dependency.
- **MLflow demoted to P2**: Not blocking any downstream features. Adding callbacks to a stable pipeline is additive work whenever it's picked up.
- **R4/R5 at P2**: `build_model()` if/elif and data source string matching work fine today. Closing them is clean-up, not urgent.

**Superseded items (replaced by MCP):**
- ~~LLM Readiness — Phase 2~~ → replaced by MCP Server (item 14)
- ~~LLM Agent~~ → replaced by MCP Server (item 14) — no custom agent loop needed; MCP clients (Claude Desktop, Claude Code) provide the orchestration layer
- ~~LLM Readiness — Phase 3~~ → replaced by MCP Pipeline Design Tools (item 19)

**Parked (no fixed order):**
- Artifact Versioning remaining (dataset semver, migration guide) — pick up opportunistically
- Telemetry Integrity remaining — pick up alongside LLM Readiness Phase 1
- Sensor Onboarding remaining (background tasks) — pick up alongside MCP Server

---

## Pytest Suite

### Automated test coverage for core business logic — P0 (Tier 1), P1 (Tier 2), P2 (Tier 3)

**Current state:** 266 tests across 10 modules (Tier 1 + Tier 2 complete).
Tier 3 (IO/mocked) remaining.

**Goal:** ~340 tests across 12 modules, rolled out in three tiers ordered by
effort-to-value ratio. 47% of testable functions are pure (no mocking needed).

**Infrastructure:** ✅

- [x] `tests/` directory structure: `tests/unit/`, `tests/integration/`
- [x] `tests/conftest.py` with fixtures: `bme680_profile`, `bsec_standard`, `sample_raw_data`, `sample_reading`, `sample_timestamps`, `patched_models_base`, `model_artifact_dir`, `fast_pipeline_kwargs`
- [x] `pytest.ini`: `testpaths = tests`, `pythonpath = .`
- [x] Dev dependencies: `pytest`, `pytest-cov`

---

### Tier 1: Pure Core — ✅ DONE (168 tests)

| Module | File | Tests |
|--------|------|-------|
| `app/quantities.py` | `tests/unit/test_quantities.py` | ~30 |
| `app/builtin_profiles.py` | `tests/unit/test_builtin_profiles.py` | ~30 |
| `training/utils.py` (pure fns) | `tests/unit/test_training_utils.py` | ~25 |
| `training/merkle.py` | `tests/unit/test_merkle.py` | ~35 |
| `app/schemas.py` | `tests/unit/test_schemas.py` | ~18 |
| `app/profiles.py` (pure parts) | `tests/unit/test_profiles.py` | ~20 |

---

### Tier 2: Stateful Core — ✅ DONE (98 tests, 266 cumulative)

No bugs uncovered — all application code behaved correctly.

| Module | File | Tests |
|--------|------|-------|
| `app/models.py` | `tests/unit/test_models.py` | 40 (build_model 8, forward pass 10, BNN 4, KAN 3, IAQPredictor 15) |
| `app/config.py` | `tests/unit/test_config.py` | 25 (load 8, get_model_config 10, training 7) |
| `training/pipeline.py` | `tests/integration/test_pipeline.py` | 25 (PreprocessingReport 7, validation 5, e2e 13) |
| `app/profiles.py` (IO) | `tests/unit/test_profiles_io.py` | 8 (sensor 4, standard 4) |

---

### Tier 3: IO/Mocked (~100 tests, ~3-4 days) — P2

Requires mocking external services (InfluxDB, Ollama, HTTP).

| Module | What to test | Tests |
|--------|-------------|-------|
| `app/inference.py` | Staleness detection, sequence validation, prior computation, Bayesian update, drift analysis, mock predictor | 35-45 |
| `app/field_mapper.py` | Tier 1/2/3 mapping strategies, `_normalize`, `_detect_timestamp`, mock Ollama | 25-32 |
| `training/data_sources.py` | `SyntheticSource` shape/range, `CSVDataSource` with `tmp_path`, mock InfluxDB/LabelStudio | 35-45 |

**Requirements (Tier 3):**

- [ ] `tests/unit/test_inference.py`: staleness flag on gap > 60s; sequence regression detected; prior from history; drift analysis warnings; mock `IAQPredictor.predict` returns known values
- [ ] `tests/unit/test_field_mapper.py`: exact match resolves, fuzzy match with `rapidfuzz`, mock Ollama HTTP response, `_normalize` strips/lowercases
- [ ] `tests/unit/test_data_sources.py`: `SyntheticSource.fetch` returns correct shape/columns/ranges; `CSVDataSource` reads test CSV from `tmp_path`; mock `DataFrameClient` for InfluxDB
- [ ] Additional dev dependencies: `responses` or `respx` (for HTTP mocking), `rapidfuzz` (optional, for fuzzy tests)

---

## Artifact Versioning

### Semantic versioning for models and datasets — P1

**Current state:** Per-model incrementing versions (`mlp-v1`, `mlp-v2`) with
SHA256 data fingerprints and git commit tracking in `MANIFEST.json`.

**Goal:** Layer proper semver (`MAJOR.MINOR.PATCH`) on top of the existing
lineage system so that artifact consumers can reason about compatibility.

**Proposed semantics:**

| Bump  | Trigger |
|-------|---------|
| MAJOR | Breaking change — input schema change (new/removed features), incompatible scaler format, sensor profile change |
| MINOR | Retrained model — same schema, new data or hyperparameters, improved metrics |
| PATCH | Metadata-only — config tweak, documentation, no weight changes |

**Requirements:**

- [x] Version string in `config.json` and `MANIFEST.json` follows `{model_type}-{MAJOR}.{MINOR}.{PATCH}` (e.g. `mlp-2.1.0`)
- [ ] Dataset artifacts get independent semver tied to data fingerprint and preprocessing pipeline version
- [x] Auto-detect bump level: compare current vs previous config.json schema, scaler shape, and data fingerprint
- [x] CLI support: `python -m iaq4j version` to show current active model versions
- [x] Warn (not reject) loading a model whose schema fingerprint doesn't match the serving code's expected schema
- [ ] Migration guide when MAJOR bump occurs (what changed, how to retrain)

---

## DAG Merkle Tree — ✅ DONE

### Directed acyclic graph provenance for multi-source training — P1

**Previous state:** Linear 6-level Merkle chain (Sensor → RawData → CleansedData → ... → TrainedModel). Only one data source per training run.

**Goal:** Support multiple data sources (sensor + external AQ APIs + weather APIs) feeding into one training run with faithful provenance tracking.

**Changes (all in `training/merkle.py`):**

- [x] `build_cleansed_data_node` accepts `Union[MerkleNode, List[MerkleNode]]` — single node auto-wrapped in list; identical hash for single-source trees (backward compat)
- [x] `build_external_source_node` — new leaf node for non-sensor sources (`source_name`, `source_url`, `api_version`, optional `extra` metadata dict)
- [x] `_diff_recursive` — child matching by hash first, then node_type; reports `+type`/`-type` for added/removed children
- [x] Module docstring — updated ASCII art to show DAG structure
- [x] Tests — `TestBuildExternalSourceNode`, `TestMultiSourceDAG`, `TestDiffDAG` in `tests/unit/test_merkle.py` (46 tests total, all passing)

**Not changed:** MerkleNode, other builders, verify, serialization, pipeline.py.

---

## Multi-source Pipeline

### Pipeline support for multiple data sources in one training run — P1

**Depends on:** DAG Merkle Tree (done). Done before MLflow so pipeline.py ingestion is settled before training callbacks are added — avoids rework.

**Goal:** Allow `TrainingPipeline` to ingest data from multiple sources (e.g., BME680 sensor + OpenWeather API + PurpleAir API) and merge them into a single training dataset with full provenance tracking via the DAG Merkle tree.

**Requirements:**

- [ ] `DataSource` ABC extended with `source_type` property for Merkle node building
- [ ] `TrainingPipeline` accepts `List[DataSource]` — fetches each, merges DataFrames by timestamp
- [ ] Merge strategy: outer join on DatetimeIndex, forward-fill external sources (lower sample rate)
- [ ] Each source gets its own `build_raw_data_node` subtree; all feed into `build_cleansed_data_node`
- [ ] `ExternalAPISource` — new `DataSource` subclass for generic REST API data (URL, auth, field mapping)
- [ ] Config: `data_sources` list in `model_config.yaml` replaces single `data_source` key
- [ ] CLI: `python -m iaq4j train --model mlp --data-source influxdb,openweather`

---

## Label Studio DB Integration

### Export InfluxDB data to Label Studio for annotation — P1

**Depends on:** None (uses existing `InfluxDBSource`). Complements existing `LabelStudioDataSource` (which imports *from* Label Studio).

**Goal:** Push raw sensor data from InfluxDB into Label Studio as annotation tasks, optionally pre-annotated with model predictions. Enables human-in-the-loop data curation: annotators review IAQ readings, correct labels, flag anomalies. Corrected data flows back via the existing `LabelStudioDataSource` for retraining.

**Data flow:**
```
InfluxDB → export tool → Label Studio tasks (with optional pre-annotations)
                                    ↓
                          Human annotators review/correct
                                    ↓
                    LabelStudioDataSource (existing) → retrain
```

**Requirements:**

- [ ] `training/label_studio_export.py` — new module: query InfluxDB, format as Label Studio import JSON, push via `/api/projects/{id}/import`
- [ ] Pre-annotation support: if trained model exists, run inference on exported data and include predictions as pre-annotations (annotators correct rather than label from scratch)
- [ ] Configurable export window: time range, measurement, filters (e.g. `iaq_accuracy >= 2`)
- [ ] Label Studio project template: define labeling interface (IAQ value correction, anomaly flagging, reject/approve)
- [ ] CLI: `python -m iaq4j export --target labelstudio --project-id <id> [--hours-back 24] [--pre-annotate]`
- [ ] Idempotent: skip tasks already imported (match by timestamp or task ID)

---

## Training Run Entity

### Unified lifecycle container for training runs — P2

**Current state:** A single training run produces several disconnected artifacts:
`PipelineResult` (in-memory, discarded after CLI prints), `data_manifest.json`
(provenance), `config.json` (model config + version), `MANIFEST.json` entry
(central registry), `training_history.json` (per-epoch losses), and binary
artifacts (`model.pt`, scaler `.pkl` files). No single object ties these
together as "one run." Each model type shares one directory
(`trained_models/mlp/`) so each run overwrites the last.

**Goal:** A `TrainingRun` entity that captures the full lifecycle — from data
extraction through evaluation — in a single addressable container. Enables
run comparison, reproduction, and rollback.

**What it holds:**

| Field | Source today |
|-------|-------------|
| Run ID | Not assigned — would be `{model_type}-{semver}` or a UUID |
| Input config snapshot | `data_manifest.json → config` |
| Data source metadata | `data_manifest.json → data_source` |
| Data fingerprint | `MANIFEST.json` entry |
| Preprocessing report | `data_manifest.json → preprocessing_issues` |
| Stage timings | `data_manifest.json → stages` |
| Training history | `training_history.json` |
| Evaluation metrics | `config.json` + `MANIFEST.json` |
| Model artifacts | `model.pt`, `feature_scaler.pkl`, `target_scaler.pkl` |
| Version + schema fingerprint | `config.json` + `MANIFEST.json` |

**Key capabilities:**

- **Compare runs** — diff two runs by config, data, metrics without reading
  multiple JSON files
- **Reproduce a run** — config snapshot + data fingerprint + `random_state`
  (now seeded) is sufficient to replay
- **Roll back** — runs stored in versioned directories
  (`trained_models/mlp/1.2.0/`) instead of overwriting a single directory
- **Query** — `python -m iaq4j runs --model mlp` lists all runs with metrics

**Requirements:**

- [ ] `TrainingRun` dataclass that unifies all per-run state into one object
- [ ] Versioned artifact directories (`trained_models/mlp/1.2.0/`) — each run preserved, not overwritten
- [ ] `TrainingRun.save()` / `TrainingRun.load()` for serialization to/from a run directory
- [ ] `PipelineResult` replaced by or wraps `TrainingRun`
- [ ] `python -m iaq4j runs` CLI command to list, compare, and diff runs
- [ ] `IAQPredictor.load_model()` updated to load from versioned directories
- [ ] Rollback support: `python -m iaq4j activate --model mlp --version 1.1.0`

---

## Training Checkpoint & Resume

### Pause/resume long training runs without losing progress — P1

**Current state:** The training loop (`training/utils.py:train_model()`) keeps
`best_val_loss` and model weights in memory only. No checkpoint is written to
disk until the SAVING pipeline stage after all epochs complete. Killing a
training run (intentionally or due to crash) loses all epoch progress.

**Problem:** Training on real data from InfluxDB (430k+ windows, 200 epochs)
takes 6–16+ hours per model depending on architecture (KAN is slowest). Five
models trained sequentially means a full experiment run can exceed 40 hours.
There is no way to pause and resume without restarting from epoch 0.

**Goal:** Periodic checkpointing during training so that runs can be interrupted
and resumed from the last checkpoint.

**Requirements:**

- [ ] Save checkpoint every N epochs (default: 20) to `trained_models/{model_type}/checkpoint.pt`
- [ ] Checkpoint includes: `model.state_dict()`, `optimizer.state_dict()`, `scheduler.state_dict()`, `epoch`, `best_val_loss`, `best_model_state_dict`, `train_losses`, `val_losses`
- [ ] `train_model()` accepts optional `resume_from` path — loads checkpoint and continues from saved epoch
- [ ] `TrainingPipeline` detects existing checkpoint for current model type and offers resume
- [ ] CLI support: `python -m iaq4j train --model kan --resume`
- [ ] Track `best_model_state_dict` in memory (not just `best_val_loss`) so the best weights are preserved even if later epochs regress
- [ ] Clean up checkpoint file after successful SAVING stage completion
- [ ] Graceful interrupt handler (SIGINT/SIGTERM) — save checkpoint before exit

**Nice-to-have:**

- [ ] `python -m iaq4j train --model kan --checkpoint-freq 10` to configure frequency
- [ ] Progress reporting: log estimated time remaining based on epoch durations

---

## MLflow Integration

### Experiment tracking and model registry — P1

**Current state:** Training runs produce disconnected artifacts (model weights,
scalers, config, manifest, training history) stitched together by custom code.
The Training Run Entity (P2 above) was designed to unify these into a single
addressable container. MLflow does all of this out of the box — and adds
experiment comparison, model registry, artifact versioning, and a web UI.

**Relationship to Training Run Entity:** MLflow would **subsume the Training
Run Entity entirely**. Every field in the proposed `TrainingRun` dataclass
(config snapshot, data fingerprint, metrics, artifacts, stage timings) maps
directly to MLflow's native concepts (params, metrics, artifacts, tags). Rather
than building a custom run container, we adopt the industry standard.

**What MLflow replaces:**

| Current custom component | MLflow equivalent |
|--------------------------|-------------------|
| `MANIFEST.json` central registry | MLflow experiment + run list |
| `training_history.json` per-epoch losses | `mlflow.log_metric()` per step |
| `config.json` metadata (version, fingerprint) | Run params + tags |
| Versioned artifact directories (proposed) | MLflow artifact store |
| `python -m iaq4j runs` (proposed) | MLflow UI + `mlflow.search_runs()` |
| Run comparison (proposed) | MLflow UI compare view |

**What stays custom (MLflow doesn't replace these):**

| Component | Why it stays |
|-----------|-------------|
| Merkle tree data provenance (`training/merkle.py`) | MLflow tracks artifacts but not content-addressable data lineage |
| Sensor profiles / IAQ standards (`app/profiles.py`) | Domain-specific ABCs — no MLflow equivalent |
| Schema fingerprint + semver (`training/utils.py`) | Compatibility detection is domain logic; logged as MLflow tags |
| `TrainingPipeline` FSM (`training/pipeline.py`) | Orchestration logic — MLflow tracks results, not execution |
| Feature engineering code | Per-profile methods — code, not config |

**Integration surface — what gets logged:**

```python
# In training/pipeline.py or training/train.py

with mlflow.start_run(run_name=f"{model_type}-{version}"):
    # Params (static per run)
    mlflow.log_params({
        "model_type": model_type,
        "sensor_type": sensor_profile.name,
        "iaq_standard": iaq_standard.name,
        "window_size": config["window_size"],
        "num_features": num_features,
        "learning_rate": config["learning_rate"],
        "epochs": config["epochs"],
        "schema_fingerprint": schema_fingerprint,
        "data_fingerprint": data_fingerprint,
        "git_commit": git_commit,
    })

    # Metrics (per epoch)
    for epoch in range(epochs):
        train_loss, val_loss = train_one_epoch(...)
        mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

    # Final evaluation metrics
    mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})

    # Artifacts
    mlflow.pytorch.log_model(model, "model")
    mlflow.log_artifact("feature_scaler.pkl")
    mlflow.log_artifact("target_scaler.pkl")
    mlflow.log_artifact("data_manifest.json")

    # Tags
    mlflow.set_tags({
        "version": version,
        "schema_fingerprint": schema_fingerprint,
        "merkle_root": merkle_root,
    })
```

**MLflow Tracing — MCP tool call observability:**

When the MCP server exposes training, prediction, and query tools to LLM agents,
traces provide full visibility into what the agent triggered and what happened
inside the system. MLflow Tracing captures parent-child span trees with timing,
inputs/outputs, and status — ideal for multi-step MCP tool chains.

Key design constraint: **MLflow is a separate observability domain.** No MLflow
imports in `app/`, `training/pipeline.py`, or the MCP server layer. The core
application emits events/callbacks; a dedicated MLflow adapter in
`training/mlflow_tracer.py` consumes them and produces traces.

**Architecture:**

```
MCP tool call                          MLflow
───────────                            ──────
mcp_server.py                          training/mlflow_tracer.py
  │                                      │
  ├─ calls train_single_model()          │  ← wraps with @mlflow.trace
  │    │                                 │
  │    ├─ pipeline.orchestrate()         │  ← span per FSM stage
  │    │    ├─ _do_ingestion()           │     (SOURCE_ACCESS, INGESTION, ...)
  │    │    ├─ _do_feature_eng()         │     timing + row counts in/out
  │    │    ├─ _do_windowing()           │     gap_info, segment counts
  │    │    ├─ _do_splitting()           │
  │    │    ├─ _do_scaling()             │
  │    │    ├─ _do_training()            │     per-epoch metrics
  │    │    ├─ _do_evaluation()          │     MAE, RMSE, R²
  │    │    └─ _do_saving()              │     version, merkle_root
  │    │                                 │
  │    └─ returns PipelineResult         │  ← span closed
  │                                      │
  └─ returns MCP ToolResult              │  ← root span closed
```

**Boundary rules:**
- `training/pipeline.py` already emits `StageResult` via `on_stage_complete()`
  callback. The tracer subscribes to this — no MLflow awareness in the pipeline.
- `training/train.py` is the only file that imports `mlflow` today. The new
  `training/mlflow_tracer.py` is the only other file that may import it.
- The MCP server layer (`mcp_server.py`) never imports `mlflow`. It calls
  `train_single_model()` which internally handles tracing.
- Inference tracing (optional, later): `InferenceEngine` emits prediction events
  via a callback; the tracer subscribes and creates inference spans. No MLflow
  import in `app/inference.py`.

**Trace hierarchy for a training MCP call:**

```
Trace: "mcp:train_model(mlp)"
├─ Span: source_access        (12ms)
├─ Span: ingestion             (3.2s,  rows_in=1.19M, rows_out=434k)
├─ Span: feature_engineering   (1.1s,  features=10)
├─ Span: windowing             (0.8s,  segments=62, windows=433k)
├─ Span: splitting             (0.1s,  train=346k, val=87k)
├─ Span: scaling               (0.2s)
├─ Span: training              (5.7h,  epochs=200, best_val_loss=0.012)
│   ├─ Metric: train_loss      (per epoch)
│   └─ Metric: val_loss        (per epoch)
├─ Span: evaluation            (2.1s,  mae=27.54, r2=0.20)
└─ Span: saving                (0.4s,  version=mlp-2.1.0, merkle=a3f2...)
```

**Open design questions (to decide before implementation):**

1. **Deployment mode:** Local file-based (`mlruns/` directory) vs MLflow Tracking
   Server (SQLite/Postgres backend, S3/local artifact store). Local is simplest
   to start; server enables remote access and the web UI.

2. **Layer vs replace:** Should MLflow run alongside `MANIFEST.json` (additive
   layer, low risk) or replace it entirely (cleaner, but migration needed)?
   Recommendation: layer first, then deprecate MANIFEST once stable.

3. **Model registry:** Use MLflow Model Registry for promotion
   (Staging → Production) or keep the current `trained_models/` directory
   as the serving source? Registry adds governance but also complexity.

4. **Artifact store:** Local filesystem vs S3-compatible (MinIO on the Pi)?
   Local is fine for single-node; MinIO enables backup and remote access.

5. **IAQPredictor loading:** Should `IAQPredictor.load_model()` load from
   MLflow artifact store or keep loading from `trained_models/`? Could
   support both with a config flag.

6. **Inference tracing granularity:** Trace every prediction request (high
   volume, ~3s intervals) or only on-demand / sampled? Recommendation: off
   by default, enabled via config flag or MCP tool parameter.

**Requirements:**

- [x] Add `mlflow` to dependencies
- [x] Wrap `train_single_model()` with `mlflow.start_run()` context manager
- [x] Log params: model config, sensor type, IAQ standard, schema fingerprint, data fingerprint, git commit
- [x] Log metrics per epoch: train_loss, val_loss
- [x] Log final metrics: MAE, RMSE, R² (or whatever evaluation produces)
- [x] Log artifacts: model.pt, scalers, data_manifest.json
- [x] Log tags: version (semver), schema_fingerprint, merkle_root
- [ ] Create `training/mlflow_tracer.py` — adapter that subscribes to pipeline callbacks and produces MLflow traces/spans
- [ ] Add `@mlflow.trace` wrapper in `train_single_model()` as root span
- [ ] Add per-stage child spans via `on_stage_complete()` callback (no MLflow in pipeline.py)
- [ ] Add inference trace support via callback in `InferenceEngine` (no MLflow in app/inference.py)
- [ ] Decide deployment mode (local file vs tracking server)
- [ ] Decide layer vs replace strategy for MANIFEST.json
- [ ] Update `python -m iaq4j runs` to query MLflow instead of (or in addition to) MANIFEST
- [ ] Evaluate MLflow Model Registry for model promotion workflow
- [ ] Update standalone training scripts (`train_models.py`, `train_all_models.py`) to log to MLflow

---

## Sensor Onboarding

### LLM-driven semantic field mapping from firmware API to sensor profile — P1

**Current state:** `SensorProfile.raw_features` defines canonical internal names
(e.g. `temperature`, `voc_resistance`). Clients must send data using exactly
these names. Adding a new sensor or firmware variant requires manually writing
the field mapping.

**Goal:** Use a small LLM as a one-time onboarding tool that reads a firmware
API spec (OpenAPI doc, header file, example JSON payload) and produces a
semantic mapping from firmware field names to the `SensorProfile`'s internal
feature names. The mapping is persisted in config — no LLM in the hot path.

**Supported input sources:**

**Two primary input sources:**

**Source A: CSV file with sensor readings** — the most common path. User
uploads a CSV file containing column headers from their sensor. The mapper
reads the headers, samples values (to infer units and ranges), and maps each
column to a quantity in `quantities.yaml`.

```
Upload: my_sensor_data.csv
Headers: timestamp, comp_temp, comp_hum, comp_press, gas_res, static_iaq
Sample:  2026-01-15T10:00:00, 23.5, 55.2, 1013.0, 85000, 42

Mapper output:
  comp_temp    → temperature (°C, confidence 0.95, fuzzy match)
  comp_hum     → relative_humidity (%RH, confidence 0.92, fuzzy match)
  comp_press   → barometric_pressure (hPa, confidence 0.90, fuzzy match)
  gas_res      → voc_resistance (Ω, confidence 0.70, LLM match)
  static_iaq   → bsec_iaq (index, confidence 0.85, LLM match)
```

**Source B: Device profile on GitHub** — the firmware repo documents the
sensor's API fields in struct definitions, JSON schemas, README docs, or
example payloads. The mapper crawls the repo to extract field metadata.

| Source | Example | What the mapper extracts |
|--------|---------|--------------------------|
| CSV with headers | `--source data.csv` | Column names, sampled values for unit/range inference |
| GitHub repo | `--source https://github.com/org/firmware` | Struct defs, JSON schemas, API routes, README field docs |
| Example JSON payload | `--source payload.json` | Field names, value types, example values |
| OpenAPI / header file | `--source api.yaml` or `sensor.h` | Field names, types, descriptions |
| Raw URL | `--source https://docs.example.com/api` | Fetches and parses page content |

For CSV files, the mapper:
1. Reads column headers as candidate field names
2. Samples N rows to infer value ranges, types, and likely units
3. Cross-references against `quantities.yaml` (valid ranges, units)
4. Feeds results to the tiered matching pipeline

For GitHub repos, the mapper uses `gh` CLI or the GitHub API to:
1. List repo contents and identify relevant files (`.h`, `.c`, `.json`,
   `.yaml`, `README.md`, API route handlers)
2. Extract struct/typedef definitions and JSON serialization fields
3. Parse any example payloads or test fixtures
4. Feed all discovered field metadata to the matching tiers

**How it works (both sources):**

1. User provides a CSV file or device profile reference (GitHub repo, URL, file)
2. Mapper extracts field names and metadata from the source
3. Tier 1+2 (exact/fuzzy) resolve obvious matches against `quantities.yaml`
   entries — using field names, sampled value ranges, and unit inference
4. Remaining ambiguous fields go to Tier 3 (LLM) with full context from both
   the source and the quantity registry
5. LLM outputs candidate mapping with confidence scores and reasoning
6. User confirms or adjusts interactively
7. Mapping saved to `model_config.yaml` under `sensor.field_mapping`
8. At ingestion time, a lightweight translation layer applies the static mapping
   before the reading hits `SensorReading` validation

**Example mapping output:**

```yaml
sensor:
  type: bme680
  field_mapping:
    comp_temp: temperature
    comp_hum: rel_humidity
    comp_press: pressure
    gas_resistance_ohm: voc_resistance
```

**Impact on existing code:**

| Component | Change |
|-----------|--------|
| `SensorProfile` ABC | Add optional `field_descriptions` property (units, physical meaning) so the LLM has structured context to reason against |
| `SensorReading` schema | Apply `field_mapping` translation in `_build_readings()` validator before feature-name matching |
| `model_config.yaml` | New optional `sensor.field_mapping` section |
| Training data ingestion | Apply same mapping when reading from InfluxDB (column rename at fetch time) |
| Inference hot path | Zero cost — mapping resolved to a static dict at startup, simple key rename |

**Mapping backend — pluggable, tiered strategy:**

The mapper uses a tiered approach: deterministic matching first, LLM only for
ambiguous fields. The LLM backend itself is pluggable (local or cloud).

| Tier | Method | When it fires | Dependency |
|------|--------|---------------|------------|
| 1 | **Exact match** | Firmware field name equals internal name | None |
| 2 | **Fuzzy match** | `rapidfuzz` similarity > 0.85 + unit/range validation against `field_descriptions` | `rapidfuzz` (pip) |
| 3 | **LLM (local)** | Ambiguous fields — Ollama with Phi-3-mini or Llama 3.2 3B | Ollama running locally (optional) |
| 3 | **LLM (cloud)** | Same trigger — Claude Haiku or equivalent | API key (optional) |

- Tier 1+2 handle the common case (trivial renames, casing differences) with
  zero LLM cost
- Tier 3 only fires for genuinely different naming (`comp_gas` → `voc_resistance`)
- User picks backend via `--backend fuzzy|ollama|anthropic` flag (default: `fuzzy`,
  falls back to LLM if any field is unresolved)
- Ollama is called via HTTP (`localhost:11434/api/generate`) — no Python SDK needed,
  just `httpx` which is already a FastAPI dependency

**Done (implemented):**

- [x] `SensorProfile.field_descriptions` property on ABC with default empty dict
- [x] `BME680Profile.field_descriptions` with unit, description, example per feature

**Requirements:**

- [x] CLI command: `python -m iaq4j map-fields --source <file_or_url> [--backend fuzzy|ollama]`
- [ ] GitHub repo input: crawl repo via `gh` CLI for `.h`/`.c` structs, JSON schemas, example payloads, and README field docs
- [x] REST endpoint for sensor registration (see Sensor Registration API below)
- [x] Tier 1: exact name match (case-insensitive, strip underscores/hyphens)
- [x] Tier 2: fuzzy match via `rapidfuzz` + validate against `valid_ranges` and `field_descriptions` units
- [x] Tier 3 (Ollama): prompt template sends `field_descriptions` + firmware fields, expects JSON mapping back
- [ ] Tier 3 (Cloud): same prompt, via Anthropic Messages API with Haiku
- [x] Interactive confirmation: show mapping table with confidence, let user override
- [x] Confirmed mapping persisted in `model_config.yaml` under `sensor.field_mapping`
- [x] `SensorReading._build_readings()` applies `field_mapping` at validation time
- [x] Training data sources apply the same mapping at column-rename time
- [x] Fallback: if no mapping configured, current behavior (exact names) is unchanged
- [ ] `rapidfuzz` added as optional dependency (`pip install iaq4j[mapping]`)
- [ ] Ollama and Anthropic SDK are optional — graceful error if not available and selected

---

## Multi-Device Support

### Central physical quantity registry for units and denomination — P0

**Current state:** Units and valid ranges are defined per `SensorProfile`
subclass (`field_descriptions`, `valid_ranges`). With a single sensor (BME680)
this works, but with multiple devices it creates problems:

- Two sensors measuring the same quantity (e.g. temperature) may use different
  units (°C vs °F) — the model expects one canonical unit
- `valid_ranges` are unit-dependent and duplicated per profile
- Feature engineering formulas (e.g. absolute humidity) silently assume specific
  units
- The semantic field mapper has no way to know two fields represent the same
  physical quantity in different units

**Goal:** A central YAML-based registry of physical quantities that defines
canonical units, valid ranges, and accepted alternate units. Each sensor
profile maps its raw features to quantities in this table. The same table
is used by the semantic field mapper to match firmware API fields by physical
meaning, not just name.

**Design: YAML as the source of truth**

The registry lives in `quantities.yaml` alongside the other config files.
Python code (`app/quantities.py`) loads it at startup and exposes typed
lookup + unit conversion. No physical quantities are hardcoded in Python.

```yaml
# quantities.yaml — central physical quantity registry

quantities:
  temperature:
    canonical_unit: "°C"
    symbol: "T"
    description: "Ambient temperature"
    valid_range: [-40, 85]
    alternate_units:
      "°F":
        convert: "({value} - 32) * 5/9"
      "K":
        convert: "{value} - 273.15"

  relative_humidity:
    canonical_unit: "%RH"
    symbol: "RH"
    description: "Relative humidity"
    valid_range: [0, 100]

  barometric_pressure:
    canonical_unit: "hPa"
    symbol: "P"
    description: "Barometric pressure"
    valid_range: [300, 1100]
    alternate_units:
      "Pa":
        convert: "{value} / 100"
      "kPa":
        convert: "{value} * 10"
      "mbar":
        convert: "{value}"             # 1:1 with hPa
      "inHg":
        convert: "{value} * 33.8639"

  voc_resistance:
    canonical_unit: "Ω"
    symbol: "R_VOC"
    description: "MOX sensor resistance to volatile organic compounds"
    valid_range: [1000, 2000000]
    alternate_units:
      "kΩ":
        convert: "{value} * 1000"
      "MΩ":
        convert: "{value} * 1000000"

  co2:
    canonical_unit: "ppm"
    symbol: "CO2"
    description: "Carbon dioxide concentration"
    valid_range: [400, 5000]

  pm2_5:
    canonical_unit: "µg/m³"
    symbol: "PM2.5"
    description: "Particulate matter ≤ 2.5µm"
    valid_range: [0, 500]

  tvoc:
    canonical_unit: "ppb"
    symbol: "TVOC"
    description: "Total volatile organic compounds"
    valid_range: [0, 60000]

  # --- IAQ indices (computed by sensor firmware / external libraries) ---
  # These are both sensor outputs AND prediction targets.

  bsec_iaq:
    canonical_unit: "index"
    symbol: "IAQ"
    description: "Bosch BSEC Indoor Air Quality index"
    valid_range: [0, 500]
    kind: "iaq_standard"              # marks this as a prediction target, not just a sensor input

  epa_aqi:
    canonical_unit: "index"
    symbol: "AQI"
    description: "US EPA Air Quality Index"
    valid_range: [0, 500]
    kind: "iaq_standard"
```

**IAQ quantities serve dual roles:**

IAQ indices like `bsec_iaq` are both sensor outputs (the firmware computes
them) and prediction targets (the model learns to reproduce them). This means:

- They appear in `quantities.yaml` alongside sensor readings — same unit,
  range, and description metadata
- The `iaq_actual` field on `SensorReading` maps to a quantity in the table
  (e.g. `bsec_iaq`), enabling the mapper to match firmware fields like
  `static_iaq` or `iaq_score` to the right quantity
- `IAQStandard` subclasses derive their `scale_range` from the YAML table
  but still own behavioral logic in Python: category breakpoints, clamping,
  and `categorize()` — these are domain rules, not data
- Comparing predicted vs actual IAQ uses the same quantity definition, ensuring
  consistent scale and units across evaluation

**SensorProfile changes:**

```python
# Each profile maps its raw features to quantities from the YAML table:
class BME680Profile(SensorProfile):
    @property
    def feature_quantities(self) -> Dict[str, str]:
        return {
            "temperature":    "temperature",           # feature name → quantity name
            "rel_humidity":   "relative_humidity",
            "pressure":       "barometric_pressure",
            "voc_resistance": "voc_resistance",
        }

# A sensor reporting Fahrenheit — the quantity registry handles conversion:
class SomeOtherSensor(SensorProfile):
    @property
    def feature_quantities(self) -> Dict[str, str]:
        return {
            "temp_f": "temperature",    # maps to same quantity
        }

    @property
    def feature_units(self) -> Dict[str, str]:
        return {
            "temp_f": "°F",             # not canonical — auto-converted at ingestion
        }
```

**How the mapper uses it:**

The semantic field mapper matches firmware fields against `quantities.yaml`
by description, unit, and valid range — not just field name. When a firmware
field has unit "°F" and range [32, 185], the mapper can confidently match it
to the `temperature` quantity even if the firmware calls it `comp_t`.

**Impact on existing code:**

| Component | Change |
|-----------|--------|
| New `quantities.yaml` | YAML table of all known physical quantities with canonical units, ranges, and conversions |
| New `app/quantities.py` | Loads YAML, exposes `get_quantity()`, `convert_to_canonical()` |
| `SensorProfile` ABC | Add `feature_quantities` property mapping features → quantity names; `valid_ranges` and `field_descriptions` become computed from the YAML registry |
| `BME680Profile` | Replace hardcoded `valid_ranges`/`field_descriptions` with `feature_quantities` mapping |
| Feature engineering | Conversion to canonical units applied before formulas |
| Semantic field mapper | Matches by physical quantity (description + unit + range), not just field name |
| `SensorReading` validation | Range checks derived from YAML registry in canonical units |
| Backward compat | `valid_ranges` and `field_descriptions` still work as computed properties |

**Requirements:**

- [ ] `quantities.yaml` with all known physical quantities, canonical units, valid ranges, and alternate unit conversions
- [ ] `app/quantities.py` loads YAML, exposes `get_quantity()`, `convert_to_canonical(value, from_unit, quantity_name)`
- [ ] Conversion expressions in YAML evaluated safely (no `eval` — use a simple expression parser or lookup table)
- [ ] `SensorProfile.feature_quantities` property maps feature names → quantity names from the YAML table
- [ ] `valid_ranges` and `field_descriptions` become computed from `feature_quantities` + YAML registry (backward compat)
- [ ] Conversion applied transparently at ingestion time (before feature engineering)
- [ ] Semantic mapper uses quantity identity + unit + range for cross-device field matching
- [ ] Registry extensible: add new quantities by editing YAML, no Python changes needed

---

## Sensor Registration API

### REST endpoint for sensor onboarding and field mapping — P1

**Depends on:** Semantic field mapping, Physical quantity registry

**Goal:** Expose the sensor onboarding workflow as REST endpoints so it can be
driven from a web UI, CI pipeline, or external tooling — not only the CLI.

**Proposed endpoints:**

```
POST   /sensors/register
       Accepts a firmware source and returns a proposed field mapping.

       Request body:
       {
         "source_type": "github_repo" | "api_spec_url" | "example_payload",
         "source": "https://github.com/org/firmware",   // or URL, or inline JSON
         "sensor_name": "my_bme680",                     // optional human label
         "backend": "fuzzy"                              // fuzzy | ollama | anthropic
       }

       Response:
       {
         "mapping_id": "abc123",
         "status": "proposed",
         "mapping": {
           "comp_temp":           {"internal": "temperature",    "quantity": "temperature",       "confidence": 0.98, "method": "fuzzy"},
           "comp_hum":            {"internal": "rel_humidity",   "quantity": "relative_humidity",  "confidence": 0.95, "method": "fuzzy"},
           "gas_resistance_ohm":  {"internal": "voc_resistance", "quantity": "voc_resistance",     "confidence": 0.72, "method": "llm"}
         },
         "unresolved": [],
         "source_fields_found": ["comp_temp", "comp_hum", "comp_press", "gas_resistance_ohm"]
       }

POST   /sensors/register/{mapping_id}/confirm
       User confirms or overrides the proposed mapping.

       Request body:
       {
         "overrides": {
           "gas_resistance_ohm": "voc_resistance"        // manual correction if needed
         }
       }

       Response:
       {
         "status": "confirmed",
         "sensor_name": "my_bme680",
         "field_mapping": { ... },                        // final mapping
         "persisted_to": "model_config.yaml"
       }

GET    /sensors
       List registered sensors and their active field mappings.

DELETE /sensors/{sensor_name}
       Remove a registered sensor and its mapping.
```

**Design notes:**

- `POST /sensors/register` is async-friendly: repo crawl + LLM call may take
  seconds, so return a `mapping_id` the client can poll or the response can
  be streamed
- The same tiered matching logic (exact → fuzzy → LLM) is shared between CLI
  and API — extracted into a reusable `FieldMapper` service class
- Confirmed mappings persist to `model_config.yaml` under `sensor.field_mapping`
  (same location the CLI uses)
- Multiple sensors can be registered simultaneously for multi-device deployments

**Requirements:**

- [x] `FieldMapper` service class extracted from CLI logic (shared by CLI + API)
- [x] `POST /sensors/register` endpoint — accepts source, runs tiered mapping, returns proposal
- [x] `POST /sensors/register/{mapping_id}/confirm` — persists confirmed mapping to config
- [x] `GET /sensors` — list active field mapping
- [x] `DELETE /sensors/mapping` — remove field mapping
- [x] Proposed mappings stored in memory until confirmed
- [ ] Background task support for slow operations (repo crawl, LLM inference)

---

## Security Hardening

### Production security improvements — P0

**Current state:** Service is behind nginx reverse proxy but has several gaps.

**Done:**

- [x] Docker port bound to localhost only (`127.0.0.1:8001:8000`) — external
  traffic must go through nginx
- [x] HTTPS via Let's Encrypt / certbot on nginx for `enviro-sensors.uk` — `deploy/nginx-iaq4j.conf`
- [x] API key authentication — `X-API-Key` header enforced on `/predict` and all mutation endpoints
- [x] Rate limiting on nginx — `limit_req_zone` at 10 r/s, burst 20 in `deploy/nginx-iaq4j.conf`
- [x] CORS tightened — explicit allowed origins (`enviro-sensors.uk`, `localhost`) in `app/main.py`
- [x] Security headers — `Strict-Transport-Security`, `X-Content-Type-Options`, `X-Frame-Options`,
  `X-XSS-Protection`, `Referrer-Policy`, `Permissions-Policy` in `deploy/nginx-iaq4j.conf`
- [x] Read-only Docker filesystem — `read_only: true` with `tmpfs: /tmp:size=64m` in `docker-compose.yml`
- [x] Deploy script uses `IAQ4J_SERVER` env var — no hardcoded production credentials in source

**Remaining (ops, not code):**

- [ ] Run `certbot --nginx -d enviro-sensors.uk` on the production server to provision the TLS cert
- [ ] Set `API_KEY` env var in production (generate with `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`)
- [ ] Restrict `/iaq/docs` and `/iaq/redoc` to localhost-only in nginx if public exposure is undesirable

---

## Telemetry Integrity

### Per-sensor sequence numbers for ordering and replay detection — P2

**Background:** Comparison against the IAQSignedTelemetry domain model
identified sequence numbers as a meaningful gap. The domain model requires
a monotonically increasing `sequence_number` per `sensor_id` as a first-class
invariant. iaq4j currently has no ordering or replay protection at the
ingestion boundary.

**What sequence numbers give us:**

| Problem | Without sequence number | With sequence number |
|---------|------------------------|---------------------|
| Dropped readings | Silent — gap in InfluxDB time series is indistinguishable from normal | Detectable — gap in sequence means missed readings |
| Reordered delivery | Silent — out-of-order write lands in InfluxDB at wrong timestamp | Detectable — sequence regresses |
| Replay attack | Identical duplicate is accepted and written | Rejected — sequence already seen for this sensor |
| Sensor restart | No signal | Sequence resets to 0, can be flagged as an event |

**Scope:** This is relevant today only if sensors are networked (MQTT, HTTP
push over the internet). For a local BME680 read over I²C on the same Pi, it
adds no value. Implement when remote or multi-sensor deployments are in scope.

**Proposed design:**

- Add optional `sequence_number: int` field to `SensorReading`
- `InferenceEngine` (or a new ingestion layer) tracks last-seen sequence per
  `sensor_id` in memory (and optionally persists to InfluxDB)
- On receipt: if `sequence_number <= last_seen`, log a warning and optionally
  reject (configurable — warn-only vs strict mode)
- Sequence resets (sensor restart) detected as large backward jump; logged
  as an event rather than an error
- No sequence number present → accepted without check (backward compat,
  local sensor mode)

**Requirements:**

- [ ] Add optional `sequence_number: int` to `SensorReading` schema
- [ ] `InferenceEngine` tracks last-seen sequence per `sensor_id`
- [ ] Configurable enforcement mode: `off` (default) | `warn` | `strict`
- [ ] Sequence gaps and regressions logged as structured warnings in `IAQResponse`
- [ ] Sequence state persisted across restarts (InfluxDB tag or local file)
- [ ] `GET /sensors/{sensor_id}/sequence` endpoint to inspect current state

---

## LLM Readiness

### Infrastructure prerequisites for agentic orchestration — P1

**Current state:** The codebase has strong machine-readable foundations — the
training pipeline emits structured `PipelineResult`/`StageResult`, sensor
profiles are self-describing ABCs, the field mapper returns confidence-scored
proposals, and model artifacts carry version + fingerprint metadata. However,
several gaps prevent an LLM agent (or any external orchestrator) from
reliably driving the system end-to-end.

**Goal:** Close the gap between "ML platform with a CLI" and "ML platform an
agent can operate." Every component that the agent needs must be accessible
via REST, return structured data, and fail with actionable error information.

**Codebase assessment — what's already agent-ready:**

| Component | File | Score | Why |
|-----------|------|-------|-----|
| TrainingPipeline | `training/pipeline.py` | 9/10 | Structured `PipelineResult`, callbacks, `StageResult` with timing |
| Sensor Profiles | `app/profiles.py` | 9/10 | Self-describing ABCs, `feature_quantities`, `valid_ranges` |
| Field Mapper | `app/field_mapper.py` | 9/10 | 3-tier strategy, confidence scores, method provenance |
| Quantities Registry | `quantities.yaml` | 9/10 | Central, aliases, valid_ranges, unit conversions |
| MANIFEST.json | `trained_models/` | 9/10 | Version, fingerprints, metrics, git_commit |
| IAQResponse schema | `app/schemas.py` | 8/10 | Bayesian structure: observation, predicted, prior, uncertainty |

**What needs fixing — 7 gap areas:**

---

#### Phase 1: Machine-Readable Foundation

Close internal gaps so that every component returns structured, actionable
information. No new REST endpoints yet — these are plumbing fixes.

**A. Config cache stale after writes**

`app/config.py` never invalidates `_model_config_cache` after endpoints write
to YAML. An agent (or any caller) that writes config then reads it back gets
stale values.

| Change | File | What |
|--------|------|------|
| Add `invalidate_config_cache()` | `app/config.py` | Clear `_model_config_cache` dict |
| Call after YAML writes | `app/main.py` | Every endpoint that writes to `model_config.yaml` calls invalidate |

**B. No InfluxDB read access**

`app/database.py` only has `write_prediction()`. An agent can't query
prediction history, compare predicted vs actual IAQ, or analyze trends.

| Change | File | What |
|--------|------|------|
| `query_predictions()` | `app/database.py` | Read back predictions by time range, model type |
| `query_raw_readings()` | `app/database.py` | Read raw sensor readings |
| `query_prediction_vs_actual()` | `app/database.py` | Join predicted and actual IAQ for evaluation |

**C. Training exceptions swallowed**

`training/train.py:train_single_model()` catches all exceptions and returns
`None`. An agent can't distinguish "failed because data was empty" from
"failed because GPU OOM" from "succeeded."

| Change | File | What |
|--------|------|------|
| Re-raise as `PipelineError` | `training/train.py` | Typed exception with stage, error code, traceback |
| `FailureInfo` schema | `training/pipeline.py` | Structured failure metadata (stage, cause, suggestion) |

**D. Standardized error model**

API endpoints return ad-hoc error dicts. An agent needs a consistent schema
to parse failures programmatically.

| Change | File | What |
|--------|------|------|
| `StructuredResponse` Pydantic model | `app/schemas.py` | Unified response envelope for MCP tools and REST: `status`, `result`, `warnings`, `error_code`, `detail`, `context`, `next_steps` |
| Exception handlers | `app/main.py` | Map known exceptions to `StructuredResponse` |

**D.1. Severity-graded responses**

Every tool/endpoint response uses `StructuredResponse` with a status field:

| Status | Meaning | LLM action |
|--------|---------|------------|
| `success` | Completed normally | Use result |
| `warning` | Completed with issues | Report to user, maybe adjust |
| `partial` | Some results, some failures | Use what worked, explain gaps |
| `error` | Failed, but recoverable | Try alternative approach using `next_steps` |
| `fatal` | Failed, system-level issue | Escalate to user immediately |

**D.2. Actionable `next_steps` on every non-success response**

Errors and warnings include a list of concrete suggestions the LLM can act on.

**D.3. Warning aggregation from `PreprocessingReport`**

The pipeline already tracks issues via `PreprocessingReport` but they are only
logged. `StructuredResponse` surfaces them as a `warnings` list so that MCP
tools and REST endpoints can return them to the caller.

**D.4. Domain-specific error codes**

| Code | When | Suggested recovery |
|------|------|-------------------|
| `NO_DATA` | InfluxDB query returned empty | Widen time range or check measurement name |
| `INSUFFICIENT_DATA` | Too few samples for window_size | Use smaller window or different model type |
| `SCHEMA_MISMATCH` | Model expects different features | Retrain or check field mapping |
| `INFLUX_UNREACHABLE` | Can't connect to InfluxDB | Check host/port/network |
| `TRAINING_DIVERGED` | Loss went to NaN/inf | Reduce learning rate |
| `NEGATIVE_R2` | Model worse than predicting the mean | Check data quality, try different architecture |
| `STALE_CONFIG` | Config cache out of sync | Invalidate and retry |
| `CHECKPOINT_NOT_FOUND` | Resume requested but no checkpoint exists | Start fresh training |

**Requirements (Phase 1):**

- [ ] `app/config.py`: add `invalidate_config_cache()` method; call from all YAML-writing endpoints
- [ ] `app/database.py`: add `query_predictions(time_range, model_type)` returning list of dicts
- [ ] `app/database.py`: add `query_raw_readings(time_range)` returning list of dicts
- [ ] `app/database.py`: add `query_prediction_vs_actual(time_range, model_type)` for evaluation
- [ ] `training/train.py`: stop catching all exceptions; re-raise as typed `PipelineError`
- [ ] `training/pipeline.py`: add `FailureInfo` dataclass with stage, error_code, suggestion
- [ ] `training/pipeline.py`: `PipelineResult` includes `warnings` list from `PreprocessingReport`
- [ ] `app/schemas.py`: add `StructuredResponse` Pydantic model with `status` (success/warning/partial/error/fatal), `result`, `warnings`, `error_code`, `detail`, `context` dict, `next_steps` list
- [ ] `app/schemas.py`: add `DomainErrorCode` enum with all domain error codes from table above
- [ ] `app/main.py`: add exception handlers that return `StructuredResponse`
- [ ] All existing REST endpoints adopt `StructuredResponse` envelope (backward-compatible: `result` field holds current response body)

---

#### Phase 2: Agent Tool Surface

Expose every component the agent needs as a REST endpoint. Each endpoint
returns structured JSON that maps 1:1 to an agent tool.

**New endpoints:**

| Endpoint | Purpose |
|----------|---------|
| `GET /profiles/sensors` | List registered sensor profiles with features, quantities, ranges |
| `GET /profiles/standards` | List IAQ standards with categories and scale ranges |
| `GET /quantities` | Physical quantity registry (names, units, aliases, ranges) |
| `GET /models/{type}/info` | Full artifact metadata: version, fingerprint, metrics, data lineage |
| `GET /models/manifest` | Central MANIFEST.json with all training runs |
| `GET /history/predictions` | Query historical predictions from InfluxDB (time range, model type) |
| `GET /history/readings` | Query historical raw sensor readings |
| `GET /history/accuracy` | Predicted vs actual IAQ comparison |
| `GET /config` | Active model + database config (secrets redacted) |
| `POST /training/start` | Trigger training as background task, return `job_id` |
| `GET /training/{job_id}` | Poll training job status and results |
| `GET /audit` | Queryable mutation audit trail (model switches, config writes) |

**Consistency fix:**

| Endpoint | Change |
|----------|--------|
| `GET /predict/compare` | Return structured `IAQResponse` per model instead of flat dicts |

**Requirements (Phase 2):**

- [ ] `GET /profiles/sensors` — serialize sensor profile ABCs (features, quantities, ranges, field descriptions)
- [ ] `GET /profiles/standards` — serialize IAQ standards (categories, scale range, breakpoints)
- [ ] `GET /quantities` — serve `quantities.yaml` as structured JSON
- [ ] `GET /models/{type}/info` — load `config.json` + `MANIFEST.json` entry for model type
- [ ] `GET /models/manifest` — serve full `MANIFEST.json`
- [ ] `GET /history/predictions` — query InfluxDB via new `query_predictions()` (Phase 1)
- [ ] `GET /history/readings` — query InfluxDB via new `query_raw_readings()` (Phase 1)
- [ ] `GET /history/accuracy` — query InfluxDB via new `query_prediction_vs_actual()` (Phase 1)
- [ ] `GET /config` — serialize active settings with secrets redacted
- [ ] `POST /training/start` — launch `train_single_model()` in background task, return job ID
- [ ] `GET /training/{job_id}` — return job status (pending/running/completed/failed) and results
- [ ] `GET /audit` — queryable log of mutations with timestamps
- [ ] Fix `/predict/compare` to return `Dict[str, IAQResponse]` instead of flat dicts
- [ ] OpenAPI schema includes all new endpoints with typed request/response models

---

#### Phase 3: Pipeline Design Tool Surface

**Must be complete before LLM-Driven Pipeline Design can be implemented.**

The LLM pipeline designer needs tools that don't yet exist anywhere in the
system. It must be able to inspect what training data is available, read and
write pipeline specs, and validate its own proposals before committing them.

**New endpoints:**

| Endpoint | Purpose |
|----------|---------|
| `GET /sensors/inventory` | Query InfluxDB for what measurements exist, their time range, row count, and sampling interval — the LLM must know what training data it can actually use before reasoning about sensor asymmetry |
| `GET /pipeline/spec/{model_type}` | Read the persisted `PipelineSpec` for a deployed model — LLM inspects existing designs before creating new ones |
| `POST /pipeline/design` | Core endpoint — accepts `ProblemStatement` + `SensorInventory`, invokes LLM, returns a `PipelineSpec` proposal for user review |
| `POST /pipeline/validate` | Validate a `PipelineSpec` before committing — checks formulas are safe (AST evaluator), features are computable from declared sensors, recommended model type exists |
| `POST /pipeline/commit` | Persist a validated `PipelineSpec` as a YAML artifact, write `pipeline_spec_path` to `model_config.yaml`, invalidate config cache |
| `GET /training/history` | All past training runs with metrics, model types, data fingerprints — LLM uses this to assess whether a model recommendation is feasible given data quantity |

**Why `GET /sensors/inventory` is the critical prerequisite:**

The LLM cannot reason about the training vs inference sensor asymmetry
unless it knows what measurements are actually present in InfluxDB, their
time coverage, and whether enough data exists to learn from a privileged
signal (e.g. 1000 hours of presence sensor data alongside VOC). Without
this, the LLM is guessing about data availability.

**New domain objects required before these endpoints can be built:**

| Object | File | Description |
|---|---|---|
| `SensorInventoryEntry` | `app/schemas.py` | Measurement name, time range, row count, median interval, null rate |
| `ProblemStatement` | `app/schemas.py` | Natural language goal + optional constraints |
| `SensorInventory` | `app/schemas.py` | Training sensors, inference sensors (the asymmetry) |
| `FeatureSpec` | `app/schemas.py` | Always-available features, supervision signals, formulas, rationale |
| `ModelSpec` | `app/schemas.py` | Recommended model, reasoning, alternatives, hyperparameter hints |
| `PipelineSpec` | `app/schemas.py` | Composite of `FeatureSpec` + `ModelSpec` |

**Requirements (Phase 3):**

- [ ] `app/database.py`: add `query_sensor_inventory()` — list measurements in InfluxDB with time range, count, median interval
- [ ] `app/schemas.py`: add `SensorInventoryEntry`, `ProblemStatement`, `SensorInventory`, `FeatureSpec`, `ModelSpec`, `PipelineSpec` Pydantic models
- [ ] `GET /sensors/inventory` — calls `query_sensor_inventory()`, returns `List[SensorInventoryEntry]`
- [ ] `GET /pipeline/spec/{model_type}` — reads persisted `PipelineSpec` YAML from `trained_models/{model_type}/`
- [ ] `POST /pipeline/design` — LLM prompt wired to quantities registry + sensor inventory; returns `PipelineSpec`
- [ ] `POST /pipeline/validate` — runs AST safety check on formulas; validates sensor availability; returns validation report
- [ ] `POST /pipeline/commit` — persists `PipelineSpec` YAML, updates `model_config.yaml`, invalidates cache
- [ ] `GET /training/history` — aggregates past run metrics from `MANIFEST.json` files across all model types
- [ ] LLM backend for pipeline design reuses pluggable pattern from `app/field_mapper.py`
- [ ] `pipeline_spec_hash` added to schema fingerprint computation in `training/utils.py`

---

#### Phase 4: ~~Agent Loop~~ → Superseded by MCP Server

~~Phase 4 is the existing **LLM Agent** roadmap item below.~~

**This phase is superseded by the MCP Server.** MCP clients (Claude Desktop,
Claude Code, any MCP-compatible LLM) provide the orchestration layer natively.
No custom agent loop is needed. The tool registry maps 1:1 to MCP tools
instead of REST endpoints.

---

## MCP Server

### Model Context Protocol server for LLM-driven IAQ operations — P1

**Depends on:** LLM Readiness Phase 1 (structured internals)

**Supersedes:** LLM Readiness Phase 2 (REST tool surface), LLM Agent (custom
orchestrator). MCP eliminates the need to build REST endpoints as an
intermediate layer and removes the need for a custom agent loop — MCP clients
handle orchestration natively.

**Current state:** The app has a FastAPI REST API serving real sensor clients
(Node-RED, test_client.py). The roadmap previously planned 12+ REST endpoints
as an agent tool surface, plus a custom LLM agent wrapper to call them.

**Goal:** Expose all agent-facing capabilities as MCP tools via a standalone
MCP server process. The FastAPI REST API remains for sensor clients. Both
import the same core modules (`app.*`, `training.*`).

**Architecture:**

```
Sensor clients ──► FastAPI (REST API) ──► app.* / training.*
                                              ▲
LLM clients ─────► MCP Server (stdio/SSE) ───┘
(Claude Desktop,
 Claude Code,
 any MCP client)
```

- **FastAPI** stays unchanged — serves sensors, predictions, health checks
- **MCP Server** is a separate entry point (`mcp_server.py` or `python -m iaq4j mcp`)
- Both share the same core: models, config, profiles, InfluxDB, training pipeline
- MCP transport: stdio (for Claude Desktop/Code) or SSE (for remote clients)

**MCP tool registry — maps 1:1 to the planned Phase 2 endpoints:**

| MCP Tool | Replaces REST Endpoint | Purpose |
|----------|----------------------|---------|
| `get_sensor_profiles` | `GET /profiles/sensors` | List sensor profiles with features, quantities, ranges |
| `get_iaq_standards` | `GET /profiles/standards` | List IAQ standards with categories and scale ranges |
| `get_quantities` | `GET /quantities` | Physical quantity registry (names, units, aliases, ranges) |
| `get_model_info` | `GET /models/{type}/info` | Artifact metadata: version, fingerprint, metrics, lineage |
| `get_manifest` | `GET /models/manifest` | Central MANIFEST.json with all model versions |
| `query_predictions` | `GET /history/predictions` | Historical predictions from InfluxDB (time range, model type) |
| `query_readings` | `GET /history/readings` | Historical raw sensor readings from InfluxDB |
| `evaluate_accuracy` | `GET /history/accuracy` | Predicted vs actual IAQ comparison |
| `get_config` | `GET /config` | Active model + database config (secrets redacted) |
| `start_training` | `POST /training/start` | Trigger training as background task |
| `get_training_status` | `GET /training/{job_id}` | Poll training job status and results |
| `get_audit_log` | `GET /audit` | Queryable mutation trail |
| `predict` | `POST /predict` | Run prediction with given sensor readings |
| `compare_models` | `GET /predict/compare` | Compare predictions across all model types |
| `select_model` | `POST /model/select` | Switch active model type |
| `map_fields` | `POST /sensors/register` | Semantic field mapping (exact → fuzzy → LLM) |
| `list_sensors` | `GET /sensors` | List registered sensor mappings |

**MCP resources (read-only data the LLM can inspect):**

| Resource | Content |
|----------|---------|
| `iaq4j://config` | Active `model_config.yaml` (secrets redacted) |
| `iaq4j://manifest` | `MANIFEST.json` with all model versions and metrics |
| `iaq4j://quantities` | `quantities.yaml` physical quantity registry |
| `iaq4j://profile/{sensor_type}` | Sensor profile details (features, ranges, engineering) |

**Requirements:**

- [ ] Add `mcp` Python SDK dependency
- [ ] Create `mcp_server.py` entry point with stdio transport
- [ ] Register all tools from tool registry table above with typed Pydantic input schemas
- [ ] Register MCP resources for config, manifest, quantities, profiles
- [ ] Tools call core functions directly (not via HTTP to FastAPI)
- [ ] `start_training` tool runs training in background thread, returns job ID
- [ ] `get_training_status` tool tracks background training jobs
- [ ] All tool responses return structured JSON (reuse existing Pydantic schemas)
- [ ] Secrets redacted from all config/database tool responses
- [ ] Error responses use structured format (reuse `APIError` from Phase 1)
- [ ] CLI support: `python -m iaq4j mcp` starts the MCP server
- [ ] Claude Desktop config example in README or `deploy/` directory
- [ ] SSE transport option for remote MCP clients (optional, can be deferred)

**Why MCP over custom agent:**

| Concern | Custom Agent (old plan) | MCP Server (new plan) |
|---------|----------------------|---------------------|
| Orchestration logic | Must build and maintain | LLM client provides it |
| Tool discovery | Manual tool registry | MCP protocol handles it |
| Multi-turn reasoning | Must implement | Native to LLM client |
| Client compatibility | Only our `/agent/ask` endpoint | Claude Desktop, Claude Code, any MCP client |
| REST endpoints needed | 12+ new endpoints | Zero — tools call core directly |
| Maintenance surface | Agent code + REST endpoints + tool wrappers | MCP server + tool definitions |

---

## ~~LLM Agent~~ (Superseded by MCP Server)

### ~~Agentic orchestrator for data ingestion, training, and insight~~ — ~~P2~~ Superseded

> **This item is superseded by the MCP Server (above).** MCP clients provide
> the orchestration layer natively. The tool registry below is preserved for
> reference — it informed the MCP tool design. No custom agent loop will be
> built.

**~~Goal:~~** ~~An embedded LLM agent that acts as the system's orchestration layer.
Users express intent in natural language; the agent coordinates data ingestion,
field mapping, training, evaluation, and reporting by calling existing
components as tools.~~

**~~Depends on:~~** ~~LLM Readiness Phase 1 + Phase 2 (above), Semantic field mapping~~

**Prerequisites from LLM Readiness (must be complete before starting):**

- Phase 1: config cache invalidation, InfluxDB read access, training exception
  propagation, standardized `APIError` model
- Phase 2: all `GET /profiles/*`, `GET /quantities`, `GET /models/*`,
  `GET /history/*`, `GET /config`, `POST /training/start`,
  `GET /training/{job_id}`, `GET /audit` endpoints operational

**Core capabilities:**

**1. Universal data ingestion**
- Accept sensor data from any source: file upload (CSV, JSON, Excel),
  URL (raw file, API endpoint), or GitHub repo
- LLM inspects the data, identifies columns, infers physical quantities
  and units by cross-referencing `quantities.yaml`
- Proposes a field mapping, user confirms, data is normalized to canonical
  units and available as a `DataSource` for training

**2. Pipeline orchestration**
- User says "train a model on this data" — agent coordinates the full
  workflow: ingest → map → validate → train → evaluate → report
- Long-running operations (training) run async with progress updates
- Agent decides model type, hyperparameters, or asks user when ambiguous
- Compares new model against existing versions and recommends promotion

**3. Diagnostic reasoning**
- Interpret sensor drift, model divergence, preprocessing issues
- "Why did IAQ spike at 3am?" — agent queries InfluxDB history, correlates
  features, explains in natural language
- Compare `iaq_predicted` vs `iaq_actual` and diagnose systematic errors

**4. Conversational query interface**
- `POST /agent/ask` — natural language questions about air quality, model
  performance, data quality, sensor health
- Agent has read access to InfluxDB history, model manifests, training
  reports, and live predictions

**Tool registry — each tool maps 1:1 to a Phase 2 REST endpoint:**

| Tool | REST Endpoint | Purpose |
|------|---------------|---------|
| `inspect_data` | `POST /data/upload`, `POST /data/import` | Read file/URL, sample rows, infer schema and units |
| `map_fields` | `POST /sensors/register` | Semantic field mapper (exact → fuzzy → LLM) |
| `get_profiles` | `GET /profiles/sensors`, `GET /profiles/standards` | List sensor profiles and IAQ standards |
| `get_quantities` | `GET /quantities` | Physical quantity registry |
| `train_model` | `POST /training/start` | Launch training as background task |
| `poll_training` | `GET /training/{job_id}` | Check training job status and results |
| `get_model_info` | `GET /models/{type}/info`, `GET /models/manifest` | Artifact metadata, versions, metrics |
| `query_history` | `GET /history/predictions`, `GET /history/readings` | InfluxDB reads |
| `evaluate_accuracy` | `GET /history/accuracy` | Predicted vs actual IAQ comparison |
| `get_config` | `GET /config` | Active model + database config (redacted) |
| `get_audit` | `GET /audit` | Queryable mutation trail |
| `manage_models` | `POST /model/select`, existing endpoints | Switch active, compare versions |
| `manage_sensors` | `GET /sensors`, `DELETE /sensors/mapping` | List/remove field mappings |

**Proposed endpoints:**

```
POST   /agent/ask
       Natural language query or command.
       { "message": "Train an MLP on this CSV", "file": <upload> }
       Returns: streaming response with reasoning + tool calls + result

POST   /data/upload
       Upload sensor data file for LLM-assisted ingestion.
       Returns: proposed field mapping + data summary

POST   /data/import
       Import from URL (CSV link, API endpoint, GitHub repo).
       { "url": "https://github.com/user/sensor-data", "format": "auto" }
       Returns: proposed field mapping + data summary

GET    /agent/tasks
       List running/completed agent tasks (training jobs, data imports).
```

**Architecture:**

- LLM backend: pluggable (Ollama local / Anthropic cloud), same as field
  mapper — configured via `model_config.yaml`
- Agent runs in a background task (FastAPI `BackgroundTasks` or Celery)
  for long operations
- Tool calls are structured function calls — the LLM decides which tools
  to invoke and in what order based on user intent
- All tool results are logged for audit trail
- Prediction hot path (`/predict`) is unchanged — zero LLM latency on
  real-time inference

**Requirements:**

- [ ] Tool registry: wrap existing components as callable tools with typed input/output schemas
- [ ] Agent loop: LLM receives user message + tool definitions, emits tool calls, receives results, iterates
- [ ] `POST /agent/ask` endpoint with streaming response
- [ ] `POST /data/upload` — file upload + LLM-assisted schema inspection and mapping
- [ ] `POST /data/import` — URL/repo import with auto-detection
- [ ] `ExternalDataSource` class implementing `DataSource` ABC for user-provided data
- [ ] Background task management for long-running agent operations
- [ ] Conversation history (per-session) for multi-turn interactions
- [ ] Ollama / Anthropic backend selection via config
- [ ] Guardrails: agent can read data and trigger training but cannot modify production model without user confirmation

---

## Temporal Feature Engineering

### Cyclical time features for diurnal and weekly IAQ patterns — P0 (implement first)

**Background:** Audit (Mar 2026) found the models are fully blind to when a
reading occurs. Indoor air quality has strong diurnal and weekly patterns
(morning cooking peaks, occupancy cycles, heating/cooling seasons) that are
invisible to the current feature set. All five model types receive only
physical sensor readings + two engineered features (`voc_ratio`,
`abs_humidity`).

**Temporal awareness audit summary:**

| Aspect | Status |
|---|---|
| Timestamp stored on `SensorReading` | ✅ Optional ISO string |
| Chronological train/val split | ✅ No shuffling |
| Gap detection + longest contiguous segment | ✅ `training/pipeline.py` |
| Sequence number replay/regression detection | ✅ `app/inference.py` |
| Timestamps written to InfluxDB | ✅ Every prediction |
| Time-based features (hour, day-of-week, seasonal) | ❌ None |
| Staleness detection (slow/delayed readings) | ❌ None |
| Synthetic data timestamps | ❌ `SyntheticSource` uses `RangeIndex` |

**Proposed changes (~1 hour implementation):**

1. Add 4 cyclical time features to `BME680Profile.engineer_features()` and
   `engineer_features_single()` in `app/builtin_profiles.py`:
   - `hour_sin` = sin(2π × hour / 24)
   - `hour_cos` = cos(2π × hour / 24)
   - `dow_sin`  = sin(2π × day_of_week / 7)
   - `dow_cos`  = cos(2π × day_of_week / 7)
2. Update `SensorProfile` ABC signatures in `app/profiles.py` to accept
   optional timestamp in both `engineer_features()` and
   `engineer_features_single()`.
3. Pass `SensorReading.timestamp` through `IAQPredictor.predict()` →
   `engineer_features_single()` in `app/models.py`.
4. Pass `DatetimeIndex` through FEATURE_ENGINEERING stage in
   `training/pipeline.py`.
5. Update `num_features` 6 → 10 in `model_config.yaml` and `app/config.py`
   (affects `input_dim` for MLP, KAN, BNN).
6. Add synthetic `DatetimeIndex` to `SyntheticSource` in
   `training/data_sources.py` so time features can be exercised without real
   data.

**Breaking changes:**
- `num_features` 6 → 10 changes the schema fingerprint → MAJOR version bump
  on all 5 models. Full retrain required.
- All existing trained artifacts become incompatible.

**Decision deferred because:** LSTM `window_size` 10→60 and CNN 10→30 already
require a full retrain (MAJOR bump). Implement time features in the same
retrain cycle to avoid paying the retrain cost twice.

**Timestamp availability:** All InfluxDB training data has timestamps so
missing-timestamp fallback at inference time is a non-issue in practice.
`SensorReading.timestamp` is `Optional[str]`; if absent at inference time,
fall back to `datetime.utcnow()` (server receive time).

**Requirements:**

- [x] `BME680Profile.engineer_features()` generates `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` from DatetimeIndex
- [x] `BME680Profile.engineer_features_single()` generates same features from optional timestamp (fallback: zeros)
- [x] `SensorProfile` ABC updated — both methods accept `Optional[datetime]` / `Optional[np.ndarray]`
- [x] `IAQPredictor.predict()` extracts and forwards timestamp to profile
- [x] `training/pipeline.py` FEATURE_ENGINEERING stage passes DatetimeIndex to profile
- [x] `SyntheticSource` generates a realistic `DatetimeIndex` (random timestamps spanning 1 week)
- [x] `num_features` updated to 10 everywhere (`model_config.yaml`, `app/config.py` defaults)
- [x] Staleness detection added to `InferenceEngine` — flags readings with gap > 60 s
- [ ] All 5 models retrained after implementation

---

## LLM-Driven Pipeline Design

### LLM as domain reasoner for feature engineering and model selection — P2

**Depends on:** LLM Readiness Phase 1 + Phase 2 + Phase 3 (all must be complete first)

- Phase 1: structured internals (config cache, InfluxDB reads, typed exceptions, `APIError`)
- Phase 2: agent tool surface (profiles, quantities, models, history, training endpoints)
- Phase 3: pipeline design tools (`GET /sensors/inventory`, `POST /pipeline/design`, `POST /pipeline/validate`, `POST /pipeline/commit`, `GET /pipeline/spec/{model_type}`, `GET /training/history`)

**Background:** Current architecture conflates two concerns in `SensorProfile.engineer_features()`:
- *Feature engineering* — the intellectual decision of which features are meaningful
- *Feature extraction* — the deterministic computation of those features from raw data

Both are currently hardcoded Python. This item introduces a principled separation
where a domain-reasoning LLM owns both engineering decisions **and model architecture
selection**, while Python owns deterministic execution.

**Motivating example:**

> Problem: "How does air quality change when a person enters a room?"
>
> Training sensors: VOC, temperature, relative humidity, presence sensor (mm-wave radar)
> Inference sensors: VOC, temperature, relative humidity only (no presence sensor)
>
> The LLM reasons:
> 1. **Features** — Human presence causes VOC to rise (breath, skin emissions),
>    temperature to increase slightly, humidity to increase from breathing. The
>    presence sensor is *privileged information* — available during training as a
>    supervision signal but absent at inference. Therefore engineer proxy features
>    (dVOC/dt, temperature gradient, humidity derivative, interaction terms) that
>    capture the presence signature from always-available sensors alone.
>
> 2. **Model** — We are inferring a latent variable (presence) from indirect
>    observations with prior knowledge from training data. This is a Bayesian
>    inference problem. Recommend BNN — it gives P(IAQ | sensors) with calibrated
>    uncertainty and can be extended to P(presence | sensors) as an auxiliary output.
>    A deterministic MLP or LSTM would underfit the uncertainty in this latent
>    variable problem.

The LLM acts as a **full pipeline designer**: it reasons about the problem structure,
the data asymmetry, and selects the model family that matches the problem's
statistical nature — not just the one that historically scores lowest MAE.

This is the ML concept of **Learning Using Privileged Information (LUPI)** combined
with LLM domain reasoning across the full training pipeline.

**Proposed domain model:**

| Object | Owner | Description |
|---|---|---|
| `ProblemStatement` | User input | Natural language goal |
| `SensorInventory` | User input | Training sensors vs inference sensors — the asymmetry |
| `FeatureSpec` | LLM output | Structured feature definitions split into training-only and always-available |
| `SupervisionSignal` | Part of `FeatureSpec` | Privileged features present only in training (e.g. presence sensor) |
| `ModelSpec` | LLM output | Recommended model architecture + justification |
| `PipelineSpec` | LLM output | Composite of `FeatureSpec` + `ModelSpec` — the full design |
| `FeatureExtractor` | Python | Executes `FeatureSpec` deterministically, path-aware (training vs inference) |

**Architecture:**

```
User provides:
  ProblemStatement + SensorInventory (training ≠ inference)
          │
          ▼
  LLM (domain reasoner)
  - Understands physics (quantities.yaml as grounding)
  - Identifies privileged signals (training-only)
  - Designs proxy features bridging the gap          → FeatureSpec
  - Selects model architecture for problem structure  → ModelSpec
  - Explains reasoning for both decisions             → rationale
          │
          ▼
  PipelineSpec (persisted YAML artifact, version-controlled)
  ├── feature_spec:
  │   ├── always_available: [voc_ratio, abs_humidity, dvoc_dt, ...]
  │   ├── supervision_signals: [presence]        ← training only
  │   ├── feature_formulas: {name: expression}   ← safe AST evaluated
  │   └── rationale: {name: "physical reasoning"}
  └── model_spec:
      ├── recommended: bnn
      ├── reasoning: "latent variable inference with calibrated uncertainty"
      ├── alternatives: [lstm, mlp]
      └── hyperparameter_hints: {prior_sigma: 0.5, kl_weight: 0.8}
          │
    ┌─────┴─────┐
    │           │
Training     Inference
FeatureExtractor  FeatureExtractor
(full set    (always_available
 incl.        features only)
 supervision
 signals)
```

**Model selection reasoning examples:**

| Problem structure | LLM recommendation | Why |
|---|---|---|
| Latent variable inference (presence detection) | BNN | Calibrated uncertainty over hidden state |
| Strong temporal dynamics (slow VOC drift) | LSTM | Sequence memory across long windows |
| Sharp local patterns (cooking spike) | CNN | Convolutional kernels detect local shape |
| Simple regression, lots of data | MLP | Low inductive bias, fast, interpretable |
| Novel sensor, small data, high uncertainty | BNN | Bayesian priors regularise under data scarcity |

**Relationship to existing architecture:**

- `SensorProfile` is retained but scoped to the **sensor contract** only:
  raw feature names, valid ranges, physical units, quantities mapping.
  It no longer owns `engineer_features()` — that moves to `FeatureExtractor`.
- `PipelineSpec` sits above `SensorProfile` as the **reasoning layer**.
- The safe AST evaluator in `app/quantities.py` executes LLM-generated
  feature formulas without `eval()`.
- The LLM backend (Ollama / Anthropic) reuses the same pluggable backend
  pattern as the field mapper (`app/field_mapper.py`).
- `PipelineSpec` is a versioned artifact stored alongside `model.pt` —
  the schema fingerprint includes a `pipeline_spec_hash`.
- The existing BNN (`app/models.py`) is already the correct architecture
  for the motivating example — no new model type needed for first use case.

**What changes to existing code when this is implemented:**

- `SensorProfile` ABC: remove `engineer_features()`, `engineer_features_single()`,
  `engineered_feature_names()` — replaced by `FeatureExtractor`
- `training/pipeline.py` FEATURE_ENGINEERING stage: calls `FeatureExtractor`
  instead of profile method
- `training/train.py`: reads recommended model type from `ModelSpec` if not
  explicitly overridden by `--model` CLI flag
- `app/models.py` inference path: calls `FeatureExtractor` in inference mode
- `model_config.yaml`: add `pipeline_spec_path` pointing to persisted spec
- New CLI: `python -m iaq4j design-pipeline --problem "..." --training-sensors ... --inference-sensors ...`

**Requirements:**

- [ ] `ProblemStatement` + `SensorInventory` value objects (plain dataclasses)
- [ ] `FeatureSpec` domain object: `always_available`, `supervision_signals`, `feature_formulas`, `rationale`
- [ ] `ModelSpec` domain object: `recommended`, `reasoning`, `alternatives`, `hyperparameter_hints`
- [ ] `PipelineSpec` composite: `FeatureSpec` + `ModelSpec`
- [ ] `FeatureExtractor` Python class — executes `FeatureSpec` in training or inference mode
- [ ] LLM prompt template: ProblemStatement + SensorInventory + quantities.yaml context → PipelineSpec
- [ ] Safe formula evaluation wired to existing AST evaluator in `app/quantities.py`
- [ ] `PipelineSpec` persisted as YAML alongside model artifacts
- [ ] `pipeline_spec_hash` added to schema fingerprint
- [ ] `SensorProfile` refactored — sensor contract only, engineering methods removed
- [ ] CLI: `python -m iaq4j design-pipeline` command
- [ ] Existing `BME680Profile` features ported to a reference `PipelineSpec` YAML as first example
- [ ] `--model` CLI flag overrides LLM recommendation when explicitly provided

---

## Architectural Remediation

### Formal appraisal: `docs/architecture_appraisal.md` (2026-03-12)

Full audit across extensibility, DDD adherence, error taxonomy, and state management.
Property-based test suite added (29 Hypothesis tests, 358 total).
Shape assertions added to CNN/LSTM forward passes.

The five remediation items below address the structural weaknesses identified.

---

### R1: Domain Exception Hierarchy — P1

**Problem:** Three competing error patterns — structured exceptions in pipeline,
error dicts in models, boolean flags in database. `DomainErrorCode` enum defined
but only populated in API handlers, not at the point where errors originate.
`_classify_error()` in pipeline reverse-engineers error codes from exception
message strings.

**Goal:** Single `IAQError` base exception with `DomainErrorCode` attached. All
domain/training code raises typed exceptions. API handlers catch and map to
`StructuredResponse`. `_classify_error()` becomes unnecessary.

**Files to modify:**

| File | Change |
|---|---|
| `app/exceptions.py` (new) | `IAQError(Exception)` base + 6 subtypes matching `DomainErrorCode` |
| `training/pipeline.py` | Replace generic `Exception` catches with typed `IAQError` subtypes; remove `_classify_error()` |
| `app/models.py` | `IAQPredictor.predict()` raises `FeatureEngineeringError` instead of returning error dict |
| `app/database.py` | `InfluxDBManager` raises `InfluxUnreachableError` instead of setting boolean flag |
| `app/main.py` | Exception handler catches `IAQError` subtypes, maps `.code` to `StructuredResponse` |
| `app/schemas.py` | No change — `DomainErrorCode` enum already exists |

**Requirements:**

- [ ] `app/exceptions.py`: `IAQError(code, suggestion=None)` base class
- [ ] Subtypes: `NoDataError`, `InsufficientDataError`, `SchemaMismatchError`, `TrainingDivergedError`, `InfluxUnreachableError`, `CheckpointNotFoundError`
- [ ] `training/pipeline.py`: stages raise typed exceptions; `FailureInfo.error_code` populated from `exc.code`
- [ ] `training/pipeline.py`: remove `_classify_error()` static method
- [ ] `app/models.py`: `predict()` raises instead of returning `{"status": "error"}`
- [ ] `app/database.py`: `InfluxDBManager` connection methods raise `InfluxUnreachableError`
- [ ] `app/main.py`: global handler catches `IAQError`, maps to `StructuredResponse`
- [ ] Tests: verify each subtype carries correct `DomainErrorCode`

**Effort:** Small (~2h). Self-contained, no pipeline shape changes.

---

### R2: Config Decomposition — ✅ DONE

**Problem:** `Settings` in `app/config.py` (282 LOC) was a god object owning API
settings, model paths, database config, YAML loading, sensor identity, and caching.
Domain modules (`profiles.py`, `schemas.py`) imported it directly — violating DDD
dependency direction (domain should not depend on infrastructure).

**Solution:** Nested sub-config dataclasses with facade pattern. `Settings` remains
the single import point (`from app.config import settings`) but delegates to typed
sub-objects. Domain factories accept optional explicit parameters.

**Files modified:**

| File | Change |
|---|---|
| `app/config.py` | Added `APIConfig`, `ModelConfig`, `DatabaseConfig`, `KANSidecarConfig` dataclasses. `Settings` facade delegates via `settings.api`, `settings.model`, `settings.database`, `settings.kan_sidecar`. All backward-compat properties preserved. |
| `app/profiles.py` | `get_sensor_profile(sensor_type=None)` — explicit param bypasses config |
| `app/schemas.py` | Module-level `configure_field_mapping()` / `get_field_mapping()` — no import-time config coupling |
| `training/pipeline/orchestrator.py` | Passes explicit `sensor_type`/`standard_type` to factory functions |
| `app/main.py` | Calls `configure_field_mapping()` at startup, on confirm, and on delete |
| `tests/unit/test_config.py` | 30 new tests for sub-config access, delegation, and methods |
| `tests/unit/test_profiles_io.py` | 4 new tests for explicit `sensor_type`/`standard_type` params |

**Requirements:**

- [x] `ModelConfig`: YAML loading, model-specific config, `get_model_config()`, `invalidate_cache()`
- [x] `DatabaseConfig`: InfluxDB connection, YAML loading, `get_database_config()`, `invalidate_cache()`
- [x] `APIConfig`: title, version, host, port, api_key, environment
- [x] `KANSidecarConfig`: remote_url, remote_timeout
- [x] `settings` singleton retained for backward compat — delegates to sub-configs
- [x] `get_sensor_profile(sensor_type: str = None)` — accepts explicit type, falls back to config
- [x] `get_iaq_standard(standard_type: str = None)` — same pattern
- [x] `schemas.py` field mapping decoupled from config import via `configure_field_mapping()`
- [x] `orchestrator.py` passes explicit types to profile/standard factories
- [x] No functional change — all 541 existing tests pass

---

### R3: Service Extraction from main.py — P1

**Problem:** `app/main.py` has 4 module-level mutable variables (`predictors`,
`inference_engines`, `active_model`, `_pending_mappings`). `active_model` is
mutated via `global` keyword in a route handler. Not thread-safe. 90+ lines of
field mapping business logic lives in route handlers instead of a service.

**Goal:** Extract `PredictionService` and `SensorRegistrationService`. Route
handlers become thin adapters that delegate to service methods. State lives in
service instances, not module globals.

**Files to modify:**

| File | Change |
|---|---|
| `app/prediction_service.py` (new) | `PredictionService` — owns `predictors`, `inference_engines`, `active_model` |
| `app/sensor_registration_service.py` (new) | `SensorRegistrationService` — owns `_pending_mappings`, field mapping logic |
| `app/main.py` | Routes delegate to service instances; lifespan creates services |

**Requirements:**

- [ ] `PredictionService.__init__(settings)`: loads models, creates predictors + inference engines
- [ ] `PredictionService.predict(reading, model_type=None)`: uses active or specified model
- [ ] `PredictionService.select_model(model_type)`: validates + switches active model
- [ ] `PredictionService.list_models()`: returns available models with status
- [ ] `SensorRegistrationService.propose_mapping(payload)`: returns mapping proposal with ID
- [ ] `SensorRegistrationService.confirm_mapping(mapping_id)`: persists to config
- [ ] `app/main.py` lifespan: `prediction_svc = PredictionService(settings)`
- [ ] All route handlers: 3-5 lines max (parse request → call service → return response)
- [ ] `_pending_mappings` TTL: proposals expire after configurable timeout (default 30min)

**Effort:** Medium (~4h). Main.py gets shorter, services are independently testable.

---

### R4: Model Factory Registry — P2

**Problem:** `build_model()` in `app/models.py:305` is a 50-line if/elif chain.
Each new model type requires a new branch with model-specific parameter extraction.

**Goal:** Replace with declarative factory dict. Adding a model requires only
adding an entry to the dict.

**Requirements:**

- [ ] `_MODEL_FACTORIES: Dict[str, Callable]` mapping model type to constructor wrapper
- [ ] Each wrapper reads its params from config: `lambda cfg, ws, nf: MLPRegressor(input_dim=ws*nf, ...)`
- [ ] `build_model()` becomes 3 lines: lookup → extract config → call factory
- [ ] `MODEL_REGISTRY` retained for backward compat (used in `IAQPredictor.load_model()`)

**Effort:** Small (~1h). Self-contained change to `app/models.py`.

---

### R5: Data Source Registry — P2

**Problem:** Data source selection in CLI uses string matching:
`if data_source == "synthetic": source = SyntheticSource()`. No registry analogous
to the sensor/standard registries. Adding a new data source requires editing CLI code.

**Goal:** Registry + factory pattern matching the sensor registry. CLI uses registry
lookup. New sources register themselves at import time.

**Requirements:**

- [ ] `_DATA_SOURCE_REGISTRY: Dict[str, type]` in `training/data_sources.py`
- [ ] `register_data_source(name, cls)` and `get_data_source(name, **kwargs)` factory
- [ ] Built-in sources register at module level: `register_data_source("synthetic", SyntheticSource)`
- [ ] CLI: `source = get_data_source(args.data_source, **source_kwargs)`
- [ ] `python -m iaq4j list-sources` command to show registered data sources

**Effort:** Small (~1h). Mirrors existing sensor registry pattern.

---
