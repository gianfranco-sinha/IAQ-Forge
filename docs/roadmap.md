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

**Goal:** A central `PhysicalQuantity` registry that defines canonical units,
valid ranges, and conversion functions. Each `SensorProfile` declares which
quantity each raw feature maps to, plus an optional unit conversion — rather
than hardcoding units and ranges per profile.

**Proposed design:**

```python
# app/quantities.py — central registry

@dataclass
class PhysicalQuantity:
    name: str                          # e.g. "temperature"
    canonical_unit: str                # e.g. "°C"
    symbol: str                        # e.g. "T"
    valid_range: Tuple[float, float]   # in canonical unit
    description: str

QUANTITY_REGISTRY = {
    "temperature":          PhysicalQuantity("temperature", "°C", "T", (-40, 85), "Ambient temperature"),
    "relative_humidity":    PhysicalQuantity("relative_humidity", "%RH", "RH", (0, 100), "Relative humidity"),
    "barometric_pressure":  PhysicalQuantity("barometric_pressure", "hPa", "P", (300, 1100), "Barometric pressure"),
    "voc_resistance":       PhysicalQuantity("voc_resistance", "Ω", "R_VOC", (1000, 2_000_000), "MOX VOC sensor resistance"),
}
```

**SensorProfile changes:**

```python
# Instead of defining valid_ranges and field_descriptions per profile:
class BME680Profile(SensorProfile):
    @property
    def feature_quantities(self) -> Dict[str, FeatureMapping]:
        return {
            "temperature":    FeatureMapping(quantity="temperature", unit="°C"),       # already canonical
            "rel_humidity":   FeatureMapping(quantity="relative_humidity", unit="%RH"),
            "pressure":       FeatureMapping(quantity="barometric_pressure", unit="hPa"),
            "voc_resistance": FeatureMapping(quantity="voc_resistance", unit="Ω"),
        }

# A hypothetical sensor reporting Fahrenheit:
class SomeOtherSensor(SensorProfile):
    @property
    def feature_quantities(self) -> Dict[str, FeatureMapping]:
        return {
            "temp_f": FeatureMapping(quantity="temperature", unit="°F", convert=fahrenheit_to_celsius),
        }
```

**Impact on existing code:**

| Component | Change |
|-----------|--------|
| New `app/quantities.py` | Central `PhysicalQuantity` registry + `FeatureMapping` dataclass |
| `SensorProfile` ABC | Add `feature_quantities` property; derive `valid_ranges`, `field_descriptions` from registry |
| `BME680Profile` | Replace `valid_ranges`/`field_descriptions` with `feature_quantities` mapping |
| Feature engineering | Conversion applied before formulas, guaranteeing canonical units |
| Semantic field mapper | Can match by physical quantity, not just field name — much more robust |
| `SensorReading` validation | Range checks derived from registry in canonical units |
| Backward compat | `valid_ranges` and `field_descriptions` still work as computed properties |

**Requirements:**

- [ ] `app/quantities.py` with `PhysicalQuantity` dataclass and `QUANTITY_REGISTRY`
- [ ] `FeatureMapping` dataclass: maps a raw feature to a quantity + source unit + optional converter
- [ ] `SensorProfile.feature_quantities` abstract property replaces per-profile unit definitions
- [ ] `valid_ranges` and `field_descriptions` become computed from `feature_quantities` + registry (backward compat)
- [ ] Unit conversion functions for common pairs (°F→°C, Pa→hPa, kΩ→Ω, etc.)
- [ ] Conversion applied transparently at ingestion time (before feature engineering)
- [ ] Semantic mapper uses quantity identity for cross-device field matching
- [ ] Registry extensible: new quantities added without modifying existing profiles

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
