# ============================================================================
# File: app/sensor_physics.py
# Sensor hardware constants and physics models.
#
# This module captures physical facts about sensors — heater profiles,
# warm-up characteristics, sampling modes, drift models — that are
# independent of how iaq4j uses them for ML.
#
# Sensors are classified by family:
#   VOC  — MOX gas sensors (heated metal-oxide). Measure volatile organic
#          compounds via gas resistance. Require heater plate warm-up,
#          exhibit long-term drift, sensitive to T/H cross-interference.
#   PM   — Particulate matter sensors (laser scattering). Measure particle
#          counts and mass concentrations across size bins. Fan-driven
#          airflow, no heater, minimal drift.
# ============================================================================
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SensorFamily(Enum):
    """IAQ sensor classification by measurement principle."""
    VOC = "voc"   # MOX gas sensors (heated metal-oxide, gas resistance)
    PM = "pm"     # Particulate matter sensors (laser scattering, mass/count)


@dataclass(frozen=True)
class HeaterProfile:
    """MOX sensor heater plate configuration (VOC family only).

    Attributes:
        temperature_c: Heater plate set-point temperature in °C.
        warmup_seconds: Time for the heater plate to stabilise after power-on.
            Readings during this period are unreliable.
        duty_cycle_ms: Heater-on duration per measurement cycle (ms).
            None = continuous or not applicable.
    """
    temperature_c: float
    warmup_seconds: float
    duty_cycle_ms: Optional[float] = None


@dataclass(frozen=True)
class FanProfile:
    """Particulate matter sensor fan configuration (PM family only).

    Attributes:
        startup_seconds: Time for airflow to stabilise after power-on.
        cleaning_interval_seconds: Auto-clean fan reversal interval.
            None = no auto-clean.
    """
    startup_seconds: float
    cleaning_interval_seconds: Optional[float] = None


@dataclass(frozen=True)
class SamplingMode:
    """Sensor sampling mode configuration.

    Attributes:
        name: Mode identifier (e.g. "lp", "ulp", "continuous").
        interval_seconds: Expected sampling interval in seconds.
        family: Sensor measurement family (VOC or PM).
        heater: Heater profile for VOC sensors. None for PM.
        fan: Fan profile for PM sensors. None for VOC.
    """
    name: str
    interval_seconds: float
    family: SensorFamily = SensorFamily.VOC
    heater: Optional[HeaterProfile] = None
    fan: Optional[FanProfile] = None


# ============================================================================
# VOC: BME680 / BME688 — Bosch MOX environmental sensor
# ============================================================================

BME680_HEATER_LP = HeaterProfile(
    temperature_c=320.0,
    warmup_seconds=600.0,   # 10 min — MOX plate stabilisation
    duty_cycle_ms=100.0,
)

BME680_HEATER_ULP = HeaterProfile(
    temperature_c=320.0,
    warmup_seconds=900.0,   # 15 min — slower stabilisation at lower duty
    duty_cycle_ms=20.0,
)

BME680_LP = SamplingMode(
    name="lp",
    interval_seconds=3.0,
    family=SensorFamily.VOC,
    heater=BME680_HEATER_LP,
)

BME680_ULP = SamplingMode(
    name="ulp",
    interval_seconds=300.0,
    family=SensorFamily.VOC,
    heater=BME680_HEATER_ULP,
)

# ============================================================================
# VOC: SGP40 — Sensirion MOX VOC sensor
# ============================================================================

SGP40_HEATER = HeaterProfile(
    temperature_c=200.0,
    warmup_seconds=60.0,    # 1 min — faster stabilisation than BME680
    duty_cycle_ms=30.0,
)

SGP40_DEFAULT = SamplingMode(
    name="default",
    interval_seconds=1.0,
    family=SensorFamily.VOC,
    heater=SGP40_HEATER,
)

# ============================================================================
# PM: SPS30 — Sensirion laser-scattering particulate matter sensor
# ============================================================================

SPS30_FAN = FanProfile(
    startup_seconds=30.0,               # 30s fan stabilisation
    cleaning_interval_seconds=604800.0,  # 1 week auto-clean interval
)

SPS30_DEFAULT = SamplingMode(
    name="default",
    interval_seconds=1.0,
    family=SensorFamily.PM,
    fan=SPS30_FAN,
)

# ============================================================================
# PM: PMS5003 — Plantower laser-scattering particulate matter sensor
# ============================================================================

PMS5003_FAN = FanProfile(
    startup_seconds=30.0,
    cleaning_interval_seconds=None,  # no auto-clean
)

PMS5003_DEFAULT = SamplingMode(
    name="default",
    interval_seconds=2.3,   # ~2.3s active mode
    family=SensorFamily.PM,
    fan=PMS5003_FAN,
)
