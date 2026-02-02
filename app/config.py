# ============================================================================
# File: app/config.py
# ============================================================================
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API settings
    API_TITLE: str = "BSEC IAQ Reproduction Service"
    API_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Model settings
    DEFAULT_MODEL: str = "mlp"  # 'mlp' or 'kan'
    MLP_MODEL_PATH: str = "trained_models/mlp"
    KAN_MODEL_PATH: str = "trained_models/kan"
    WINDOW_SIZE: int = 10

    # InfluxDB (optional, for logging predictions)
    INFLUX_ENABLED: bool = False
    INFLUX_HOST: str = "87.106.102.14"
    INFLUX_PORT: int = 8086
    INFLUX_DATABASE: str = "home_study_room_iaq"
    INFLUX_USERNAME: str = ""
    INFLUX_PASSWORD: str = ""
    INFLUX_TIMEOUT: int = 60

    class Config:
        env_file = ".env"


settings = Settings()
