"""Label Studio integration — data source, exporter, and launcher."""


def __getattr__(name):
    """Lazy imports to avoid circular dependency with training.data_sources."""
    if name == "LabelStudioDataSource":
        from integrations.label_studio.data_source import LabelStudioDataSource
        return LabelStudioDataSource
    if name == "LabelStudioExporter":
        from integrations.label_studio.exporter import LabelStudioExporter
        return LabelStudioExporter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["LabelStudioDataSource", "LabelStudioExporter"]
