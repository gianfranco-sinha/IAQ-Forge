# iaq4j Roadmap

Requirements catalog for future implementation. Items are grouped by domain and
marked with priority: **P0** (next up), **P1** (soon), **P2** (later).

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

- [ ] Version string in `config.json` and `MANIFEST.json` follows `{model_type}-{MAJOR}.{MINOR}.{PATCH}` (e.g. `mlp-2.1.0`)
- [ ] Dataset artifacts get independent semver tied to data fingerprint and preprocessing pipeline version
- [ ] Auto-detect bump level: compare current vs previous config.json schema, scaler shape, and data fingerprint
- [ ] CLI support: `python -m iaq4j version` to show current active model versions
- [ ] Reject loading a model whose MAJOR version doesn't match the serving code's expected schema
- [ ] Migration guide when MAJOR bump occurs (what changed, how to retrain)

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

| Source | Example | What the mapper extracts |
|--------|---------|--------------------------|
| Example JSON payload | `--api-spec payload.json` | Field names, value types, example values |
| OpenAPI / header file | `--api-spec api.yaml` or `sensor.h` | Field names, types, descriptions |
| GitHub repo | `--api-spec https://github.com/org/firmware` | Crawls repo for struct defs, JSON schemas, API routes, README field docs |
| Raw URL | `--api-spec https://docs.example.com/api` | Fetches and parses page content |

For GitHub repos, the mapper uses `gh` CLI or the GitHub API to:
1. List repo contents and identify relevant files (`.h`, `.c`, `.json`,
   `.yaml`, `README.md`, API route handlers)
2. Extract struct/typedef definitions and JSON serialization fields
3. Parse any example payloads or test fixtures
4. Feed all discovered field metadata to the matching tiers

**How it works:**

1. User provides firmware API reference (file, URL, or GitHub repo)
2. Mapper extracts field names and metadata from the source (repo crawl for
   GitHub, direct parse for files)
3. Tier 1+2 (exact/fuzzy) resolve obvious matches against `field_descriptions`
4. Remaining ambiguous fields go to Tier 3 (LLM) with full context from both
   the firmware source and the sensor profile
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

- [ ] CLI command: `python -m iaq4j map-fields --api-spec <file_or_url_or_repo> [--backend fuzzy|ollama|anthropic]`
- [ ] GitHub repo input: crawl repo via `gh` CLI for `.h`/`.c` structs, JSON schemas, example payloads, and README field docs
- [ ] REST endpoint for sensor registration (see Sensor Registration API below)
- [ ] Tier 1: exact name match (case-insensitive, strip underscores/hyphens)
- [ ] Tier 2: fuzzy match via `rapidfuzz` + validate against `valid_ranges` and `field_descriptions` units
- [ ] Tier 3 (Ollama): prompt template sends `field_descriptions` + firmware fields, expects JSON mapping back
- [ ] Tier 3 (Cloud): same prompt, via Anthropic Messages API with Haiku
- [ ] Interactive confirmation: show mapping table with confidence, let user override
- [ ] Confirmed mapping persisted in `model_config.yaml` under `sensor.field_mapping`
- [ ] `SensorReading._build_readings()` applies `field_mapping` at validation time
- [ ] Training data sources apply the same mapping at column-rename time
- [ ] Fallback: if no mapping configured, current behavior (exact names) is unchanged
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

- [ ] `FieldMapper` service class extracted from CLI logic (shared by CLI + API)
- [ ] `POST /sensors/register` endpoint — accepts source, runs tiered mapping, returns proposal
- [ ] `POST /sensors/register/{mapping_id}/confirm` — persists confirmed mapping to config
- [ ] `GET /sensors` — list registered sensors and their field mappings
- [ ] `DELETE /sensors/{sensor_name}` — remove a sensor registration
- [ ] Proposed mappings stored in memory (or temp file) until confirmed
- [ ] Background task support for slow operations (repo crawl, LLM inference)

---

## Security Hardening

### Production security improvements — P1

**Current state:** Service is behind nginx reverse proxy but has several gaps.

**Done:**

- [x] Docker port bound to localhost only (`127.0.0.1:8001:8000`) — external
  traffic must go through nginx

**Requirements:**

- [ ] HTTPS via Let's Encrypt / certbot on nginx for `enviro-sensors.uk`
- [ ] API key authentication — require `X-API-Key` header on `/predict` and mutation endpoints
- [ ] Rate limiting on nginx (`limit_req_zone`) to prevent abuse
- [ ] Tighten CORS — replace `allow_origins=["*"]` with explicit allowed origins
- [ ] Add `X-Content-Type-Options`, `X-Frame-Options`, `Strict-Transport-Security` headers in nginx
- [ ] Read-only filesystem in Docker container (`read_only: true` with tmpfs for `/tmp`)

---

## LLM Agent

### Agentic orchestrator for data ingestion, training, and insight — P2

**Goal:** An embedded LLM agent that acts as the system's orchestration layer.
Users express intent in natural language; the agent coordinates data ingestion,
field mapping, training, evaluation, and reporting by calling existing
components as tools.

**Depends on:** Physical quantity registry, Semantic field mapping

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

**Tool registry — each tool maps to an existing component:**

```
inspect_data     → Read file/URL, sample rows, infer schema and units
map_fields       → Semantic field mapper (exact → fuzzy → LLM)
validate_data    → PreprocessingReport, outlier detection, gap analysis
train_model      → TrainingPipeline.orchestrate()
evaluate_model   → Metrics comparison, category distribution, version diff
query_history    → InfluxDB reads (predictions, actuals, sensor readings)
explain          → Interpret readings + prediction in natural language
manage_models    → List versions, switch active, compare MANIFEST entries
manage_sensors   → List registered sensors, field mappings, quantities
```

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
