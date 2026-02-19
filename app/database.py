"""
Database connectivity and health checks for InfluxDB.
Supports both InfluxDB 1.x and 2.x
"""

from typing import Optional, Dict
import logging
import json
from app.config import settings

# Import appropriate client based on version
try:
    from influxdb import DataFrameClient
    from influxdb import InfluxDBClient

    HAS_INFLUXDB_V1 = True
except ImportError:
    DataFrameClient = None
    InfluxDBClient = None
    HAS_INFLUXDB_V1 = False

try:
    from influxdb_client import InfluxDBClient as InfluxDBClientV3

    HAS_INFLUXDB_V3 = True
except ImportError:
    InfluxDBClientV3 = None
    HAS_INFLUXDB_V3 = False

logger = logging.getLogger(__name__)


class InfluxDBManager:
    """Manage InfluxDB connections with health checks."""

    def __init__(self):
        self.client = None  # Type varies based on version
        self.client_type = None
        self.connected: bool = False
        self.last_error: Optional[str] = None

        # Load database configuration from YAML
        self.db_config = settings.get_database_config()
        self.version = self.db_config.get("version", "1.x")

        if self.db_config.get("enabled", False):
            self._connect()

    def _connect(self) -> bool:
        """Attempt to connect to InfluxDB based on version."""
        try:
            version = self.db_config.get("version", "1.x")

            if version == "2.x":
                self._connect_influxdb_v2()
            else:
                self._connect_influxdb_v1()

            if self.connected:
                logger.info(
                    f"Connected to InfluxDB {version} at {self.db_config.get('host')}:{self.db_config.get('port')}"
                )
            return self.connected

        except Exception as e:
            self.connected = False
            self.last_error = str(e)
            logger.error(f"Failed to connect to InfluxDB: {e}")
            return False

    def _connect_influxdb_v1(self) -> bool:
        """Connect to InfluxDB 1.x using InfluxDBClient."""
        if not HAS_INFLUXDB_V1:
            logger.warning("InfluxDB 1.x client not available")
            self.last_error = "InfluxDB 1.x client not available"
            return False

        self.client = InfluxDBClient(
            host=self.db_config.get("host"),
            port=self.db_config.get("port"),
            username=self.db_config.get("username"),
            password=self.db_config.get("password"),
            database=self.db_config.get("database"),
            timeout=self.db_config.get("timeout", 60),
        )

        # Test connection by pinging
        self.client.ping()

        # Verify database exists
        databases = self.client.get_list_database()
        db_names = [db["name"] for db in databases]
        database_name = self.db_config.get("database")

        if database_name not in db_names:
            logger.warning(
                f"Database '{database_name}' not found. Available: {db_names}"
            )
            self.last_error = f"Database '{database_name}' does not exist"
            self.connected = False
            return False

        self.connected = True
        self.client_type = "InfluxDBClient"
        return True

    def _connect_influxdb_v2(self) -> bool:
        """Connect to InfluxDB 2.x using InfluxDBClient."""
        if not HAS_INFLUXDB_V3:
            logger.warning(
                "InfluxDB 2.x client not available, falling back to 1.x client"
            )
            return self._connect_influxdb_v1()

        token = self.db_config.get("token", "")
        org = self.db_config.get("org", "")

        if not token:
            logger.error("Token is required for InfluxDB 2.x")
            self.last_error = "Token is required for InfluxDB 2.x"
            return False
        if not org:
            logger.error("Organization is required for InfluxDB 2.x")
            self.last_error = "Organization is required for InfluxDB 2.x"
            return False

        self.client = InfluxDBClientV3(
            token=token,
            org=org,
            timeout=self.db_config.get("timeout", 60) * 1000,  # Convert to milliseconds
        )

        # Test connection
        health = self.client.health()
        if health.status == "pass":
            self.connected = True
            self.client_type = "InfluxDBClientV3"
            return True
        else:
            self.last_error = f"InfluxDB 2.x health check failed: {health.message}"
            self.connected = False
            return False

    def health_check(self) -> Dict:
        """Check InfluxDB health status."""
        if not self.db_config.get("enabled", False):
            return {
                "enabled": False,
                "status": "disabled",
                "message": "InfluxDB integration is disabled",
            }

        if not self.connected:
            # Try to reconnect
            self._connect()

        if self.connected:
            try:
                # Perform version-specific health check
                if self.version == "2.x":
                    health = self.client.health()
                    return {
                        "enabled": True,
                        "status": "healthy" if health.status == "pass" else "unhealthy",
                        "connected": True,
                        "host": self.db_config.get("host"),
                        "port": self.db_config.get("port"),
                        "database": self.db_config.get("bucket")
                        or self.db_config.get("database"),
                        "version": self.version,
                        "client_type": self.client_type,
                        "health_check": health.message
                        if health.status != "pass"
                        else "OK",
                    }
                else:
                    # InfluxDB 1.x - ping test
                    self.client.ping()
                    return {
                        "enabled": True,
                        "status": "healthy",
                        "connected": True,
                        "host": self.db_config.get("host"),
                        "port": self.db_config.get("port"),
                        "database": self.db_config.get("database"),
                        "version": self.version,
                        "client_type": self.client_type,
                    }
            except Exception as e:
                self.connected = False
                self.last_error = str(e)
                logger.error(f"InfluxDB health check failed: {e}")

        return {
            "enabled": True,
            "status": "unhealthy",
            "connected": False,
            "host": self.db_config.get("host"),
            "port": self.db_config.get("port"),
            "database": self.db_config.get("database"),
            "version": self.version,
            "client_type": self.client_type,
            "error": self.last_error,
        }

    def write_prediction(
        self,
        timestamp,
        iaq_predicted,
        model_type,
        readings: dict = None,
        iaq_actual: float = None,
        # Legacy kwargs for backward compatibility
        temperature=None,
        rel_humidity=None,
        pressure=None,
        voc_resistance=None,
    ):
        """Write prediction to InfluxDB (if enabled and connected).

        Accepts either a generic ``readings`` dict or the legacy keyword
        arguments.
        """
        # Build fields dict from readings or legacy kwargs
        if readings is None:
            readings = {}
            if temperature is not None:
                readings["temperature"] = temperature
            if rel_humidity is not None:
                readings["rel_humidity"] = rel_humidity
            if pressure is not None:
                readings["pressure"] = pressure
            if voc_resistance is not None:
                readings["voc_resistance"] = voc_resistance

        if not self.db_config.get("enabled", False):
            logger.debug("InfluxDB logging disabled - skipping write")
            return False

        if not self.connected:
            logger.warning("InfluxDB not connected - attempting reconnection")
            self._connect()
            if not self.connected:
                logger.error("InfluxDB reconnection failed - cannot write prediction")
                return False

        try:
            fields = {k: float(v) for k, v in readings.items()}
            fields["iaq_predicted"] = float(iaq_predicted)
            if iaq_actual is not None:
                fields["iaq_actual"] = float(iaq_actual)

            # Attempt write based on version
            if self.version == "2.x":
                from influxdb_client import Point
                from datetime import datetime

                point = Point("iaq_predictions").tag("model", model_type)
                for k, v in fields.items():
                    point = point.field(k, v)
                point = point.time(datetime.fromtimestamp(timestamp))

                write_api = self.client.write_api()
                write_api.write(bucket=self.db_config.get("bucket"), record=point)
                result = True

            else:
                json_body = [
                    {
                        "measurement": "iaq_predictions",
                        "time": timestamp,
                        "tags": {"model": model_type},
                        "fields": fields,
                    }
                ]
                result = self.client.write_points(json_body, time_precision="s")

            if result or result is None:
                logger.info("✓ Write successful!")
                return True
            else:
                logger.error("✗ Write failed - write_points returned False")
                return False

        except Exception as e:
            logger.error(f"✗ InfluxDB write exception: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Full error: {str(e)}")
            self.connected = False
            self.last_error = str(e)
            return False

    def close(self):
        """Close InfluxDB connection."""
        if self.client:
            self.client.close()
            self.connected = False
            logger.info("InfluxDB connection closed")


# Global instance
influx_manager = InfluxDBManager()
