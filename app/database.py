"""
Database connectivity and health checks for InfluxDB.
"""

from influxdb import DataFrameClient
from typing import Optional, Dict
import logging
import json
from app.config import settings

logger = logging.getLogger(__name__)


class InfluxDBManager:
    """Manage InfluxDB connections with health checks."""

    def __init__(self):
        self.client: Optional[DataFrameClient] = None
        self.connected: bool = False
        self.last_error: Optional[str] = None

        if settings.INFLUX_ENABLED:
            self._connect()

    def _connect(self) -> bool:
        """Attempt to connect to InfluxDB."""
        try:
            self.client = DataFrameClient(
                host=settings.INFLUX_HOST,
                port=settings.INFLUX_PORT,
                username=settings.INFLUX_USERNAME,
                password=settings.INFLUX_PASSWORD,
                database=settings.INFLUX_DATABASE,
                timeout=settings.INFLUX_TIMEOUT
            )

            # Test connection by pinging
            self.client.ping()

            # Verify database exists
            databases = self.client.get_list_database()
            db_names = [db['name'] for db in databases]

            if settings.INFLUX_DATABASE not in db_names:
                logger.warning(
                    f"Database '{settings.INFLUX_DATABASE}' not found. "
                    f"Available: {db_names}"
                )
                self.last_error = f"Database '{settings.INFLUX_DATABASE}' does not exist"
                self.connected = False
                return False

            self.connected = True
            self.last_error = None
            logger.info(
                f"Connected to InfluxDB at {settings.INFLUX_HOST}:{settings.INFLUX_PORT}"
            )
            return True

        except Exception as e:
            self.connected = False
            self.last_error = str(e)
            logger.error(f"Failed to connect to InfluxDB: {e}")
            return False

    def health_check(self) -> Dict:
        """Check InfluxDB health status."""
        if not settings.INFLUX_ENABLED:
            return {
                'enabled': False,
                'status': 'disabled',
                'message': 'InfluxDB integration is disabled'
            }

        if not self.connected:
            # Try to reconnect
            self._connect()

        if self.connected:
            try:
                # Perform a simple query to verify connectivity
                self.client.ping()

                return {
                    'enabled': True,
                    'status': 'healthy',
                    'connected': True,
                    'host': settings.INFLUX_HOST,
                    'port': settings.INFLUX_PORT,
                    'database': settings.INFLUX_DATABASE
                }
            except Exception as e:
                self.connected = False
                self.last_error = str(e)
                logger.error(f"InfluxDB health check failed: {e}")

        return {
            'enabled': True,
            'status': 'unhealthy',
            'connected': False,
            'host': settings.INFLUX_HOST,
            'port': settings.INFLUX_PORT,
            'database': settings.INFLUX_DATABASE,
            'error': self.last_error
        }

    def write_prediction(self, timestamp, temperature, humidity, pressure,
                         resistance, iaq_predicted, model_type):
        """Write prediction to InfluxDB (if enabled and connected)."""
        if not settings.INFLUX_ENABLED:
            logger.debug("InfluxDB logging disabled - skipping write")
            return False

        if not self.connected:
            logger.warning("InfluxDB not connected - attempting reconnection")
            self._connect()
            if not self.connected:
                logger.error("InfluxDB reconnection failed - cannot write prediction")
                return False

        try:
            json_body = [
                {
                    "measurement": "iaq_predictions",
                    "time": timestamp,
                    "tags": {
                        "model": model_type
                    },
                    "fields": {
                        "temperature": float(temperature),
                        "humidity": float(humidity),
                        "pressure": float(pressure),
                        "resistance": float(resistance),
                        "iaq_predicted": float(iaq_predicted)
                    }
                }
            ]

            # DEBUG: Log what we're writing
            logger.info("=" * 70)
            logger.info("INFLUXDB WRITE ATTEMPT")
            logger.info("-" * 70)
            logger.info(f"Database: {settings.INFLUX_DATABASE}")
            logger.info(f"Measurement: iaq_predictions")
            logger.info(f"Timestamp: {timestamp}")
            logger.info(f"Tags: model={model_type}")
            logger.info(f"Fields:")
            logger.info(f"  - temperature: {temperature}")
            logger.info(f"  - humidity: {humidity}")
            logger.info(f"  - pressure: {pressure}")
            logger.info(f"  - resistance: {resistance}")
            logger.info(f"  - iaq_predicted: {iaq_predicted}")
            logger.info(f"JSON Body: {json.dumps(json_body, indent=2)}")

            # Attempt write
            result = self.client.write_points(json_body, time_precision='s')

            if result:
                logger.info("✓ Write successful!")
                logger.info("=" * 70)
                return True
            else:
                logger.error("✗ Write failed - write_points returned False")
                logger.info("=" * 70)
                return False

        except Exception as e:
            logger.error(f"✗ InfluxDB write exception: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Full error: {str(e)}")
            logger.info("=" * 70)
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