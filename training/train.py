# training/train.py
from typing import Optional

from training.data_sources import DataSource, SyntheticSource
from training.pipeline import TrainingPipeline, PipelineError, PipelineResult


def train_single_model(
    model_type: str,
    epochs: int = 200,
    window_size: int = 10,
    num_records: int = None,
    data_source: DataSource = None,
) -> Optional[PipelineResult]:
    """Train a single model using the TrainingPipeline.

    Args:
        model_type: One of "mlp", "kan", "lstm", "cnn".
        epochs: Number of training epochs.
        window_size: Sliding window size.
        num_records: Number of synthetic samples (ignored when data_source is provided).
        data_source: Data source to use. Defaults to SyntheticSource.

    Returns:
        PipelineResult on success, None on failure.
    """
    try:
        if data_source is None:
            num_samples = num_records if num_records else 1000
            data_source = SyntheticSource(num_samples=num_samples)

        pipeline = TrainingPipeline(
            source=data_source,
            model_type=model_type,
            epochs=epochs,
            window_size=window_size,
        )
        return pipeline.orchestrate()

    except PipelineError as e:
        print(f"Error training {model_type}: {e}")
        return None
    except Exception as e:
        print(f"Error training {model_type}: {e}")
        return None
