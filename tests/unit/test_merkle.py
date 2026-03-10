"""Tests for training/merkle.py — MerkleNode, builder functions, diff, hashing."""
import json

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from training.merkle import (
    MerkleNode,
    _hash_dict,
    build_cleansed_data_node,
    build_preprocessed_data_node,
    build_raw_data_node,
    build_sensor_node,
    build_split_data_node,
    build_trained_model_node,
    compute_scaler_hash,
    diff_merkle_trees,
)


# ── MerkleNode ─────────────────────────────────────────────────────────────


class TestMerkleNode:
    def test_compute_hash_deterministic(self):
        node = MerkleNode("test", {"key": "value"})
        h1 = node.compute_hash()
        node2 = MerkleNode("test", {"key": "value"})
        h2 = node2.compute_hash()
        assert h1 == h2

    def test_compute_hash_is_sha256_hex(self):
        node = MerkleNode("test", {"key": "value"})
        h = node.compute_hash()
        assert len(h) == 64
        int(h, 16)  # valid hex

    def test_different_content_different_hash(self):
        n1 = MerkleNode("test", {"key": "a"})
        n2 = MerkleNode("test", {"key": "b"})
        assert n1.compute_hash() != n2.compute_hash()

    def test_different_type_different_hash(self):
        n1 = MerkleNode("type_a", {"key": "value"})
        n2 = MerkleNode("type_b", {"key": "value"})
        assert n1.compute_hash() != n2.compute_hash()

    def test_child_change_propagates(self):
        child1 = MerkleNode("child", {"v": "1"})
        child2 = MerkleNode("child", {"v": "2"})
        parent1 = MerkleNode("parent", {"p": "x"}, children=[child1])
        parent2 = MerkleNode("parent", {"p": "x"}, children=[child2])
        assert parent1.compute_hash() != parent2.compute_hash()

    def test_to_dict(self):
        node = MerkleNode("test", {"k": "v"})
        node.compute_hash()
        d = node.to_dict()
        assert d["node_type"] == "test"
        assert d["content_inputs"] == {"k": "v"}
        assert "content_hash" in d
        assert isinstance(d["children"], list)

    def test_from_dict_round_trip(self):
        child = MerkleNode("child", {"c": 1})
        parent = MerkleNode("parent", {"p": 2}, children=[child])
        parent.compute_hash()
        d = parent.to_dict()
        restored = MerkleNode.from_dict(d)
        assert restored.node_type == "parent"
        assert len(restored.children) == 1
        assert restored.children[0].node_type == "child"
        assert restored.content_hash == parent.content_hash

    def test_from_dict_missing_content_hash(self):
        d = {"node_type": "test", "content_inputs": {"k": "v"}, "children": []}
        node = MerkleNode.from_dict(d)
        assert node.content_hash == ""

    def test_from_dict_no_children_key(self):
        d = {"node_type": "test", "content_inputs": {"k": "v"}}
        node = MerkleNode.from_dict(d)
        assert node.children == []


# ── Builder functions ──────────────────────────────────────────────────────


class TestBuilderFunctions:
    def _build_full_tree(self):
        """Build a complete 6-level tree and return (root, sensor_node)."""
        sensor = build_sensor_node("sensor-001", "fw-1.0", "bme680")
        raw = build_raw_data_node(sensor, {"source_type": "influxdb"}, 1000)
        cleansed = build_cleansed_data_node(raw, "fp123", 950, 50, "50 NaN rows")
        preprocessed = build_preprocessed_data_node(
            cleansed, "schema_fp", {"temp": {"mean": 22}},
            {"voc_resistance": 100000}, "scaler_hash_abc", 10, 10, 900,
        )
        split = build_split_data_node(preprocessed, 0.2, 720, 180)
        root = build_trained_model_node(
            split, "weights_hash_xyz",
            {"mae": 5.0, "rmse": 8.0, "r2": 0.9},
            {"epochs": 100, "learning_rate": 0.001, "batch_size": 32},
            "mlp", "mlp-1.0.0",
        )
        return root, sensor

    def test_sensor_node(self):
        node = build_sensor_node("s1", "fw1", "bme680")
        assert node.node_type == "sensor"
        assert node.content_inputs["sensor_id"] == "s1"
        assert node.content_inputs["sensor_type"] == "bme680"
        assert node.children == []

    def test_sensor_node_none_id(self):
        node = build_sensor_node(None, None, "bme680")
        assert node.content_inputs["sensor_id"] == "unknown"
        assert node.content_inputs["firmware_version"] == "unknown"

    def test_raw_data_node(self):
        sensor = build_sensor_node("s1", "fw1", "bme680")
        raw = build_raw_data_node(sensor, {"source_type": "influxdb"}, 1000)
        assert raw.node_type == "raw_data"
        assert raw.content_inputs["raw_row_count"] == 1000
        assert len(raw.children) == 1
        assert raw.children[0].node_type == "sensor"

    def test_cleansed_data_node(self):
        sensor = build_sensor_node("s1", "fw1", "bme680")
        raw = build_raw_data_node(sensor, {}, 1000)
        cleansed = build_cleansed_data_node(raw, "fp123", 950, 50, "issues")
        assert cleansed.node_type == "cleansed_data"
        assert cleansed.content_inputs["data_fingerprint"] == "fp123"
        assert len(cleansed.children) == 1
        assert cleansed.children[0].node_type == "raw_data"

    def test_preprocessed_data_node(self):
        sensor = build_sensor_node("s1", "fw1", "bme680")
        raw = build_raw_data_node(sensor, {}, 1000)
        cleansed = build_cleansed_data_node(raw, "fp", 950, 50, "ok")
        prep = build_preprocessed_data_node(
            cleansed, "sfp", {"t": {"mean": 22}}, {}, "sh", 10, 10, 900,
        )
        assert prep.node_type == "preprocessed_data"
        assert prep.content_inputs["schema_fingerprint"] == "sfp"
        assert len(prep.children) == 1

    def test_split_data_node(self):
        sensor = build_sensor_node("s1", "fw1", "bme680")
        raw = build_raw_data_node(sensor, {}, 100)
        cleansed = build_cleansed_data_node(raw, "fp", 90, 10, "ok")
        prep = build_preprocessed_data_node(cleansed, "sfp", {}, {}, "sh", 10, 10, 80)
        split = build_split_data_node(prep, 0.2, 64, 16)
        assert split.node_type == "split_data"
        assert split.content_inputs["train_samples"] == 64
        assert split.content_inputs["val_samples"] == 16

    def test_trained_model_node(self):
        root, _ = self._build_full_tree()
        assert root.node_type == "trained_model"
        assert root.content_inputs["model_type"] == "mlp"
        assert root.content_inputs["version"] == "mlp-1.0.0"

    def test_full_tree_hash(self):
        root, _ = self._build_full_tree()
        h = root.compute_hash()
        assert len(h) == 64

    def test_full_tree_serialize_deserialize(self):
        root, _ = self._build_full_tree()
        root.compute_hash()
        d = root.to_dict()
        restored = MerkleNode.from_dict(d)
        restored.compute_hash()
        assert restored.content_hash == root.content_hash

    def test_leaf_change_changes_root(self):
        root1, _ = self._build_full_tree()
        root1.compute_hash()

        # Build tree with different sensor_id
        sensor2 = build_sensor_node("sensor-002", "fw-1.0", "bme680")
        raw2 = build_raw_data_node(sensor2, {"source_type": "influxdb"}, 1000)
        cleansed2 = build_cleansed_data_node(raw2, "fp123", 950, 50, "50 NaN rows")
        prep2 = build_preprocessed_data_node(
            cleansed2, "schema_fp", {"temp": {"mean": 22}},
            {"voc_resistance": 100000}, "scaler_hash_abc", 10, 10, 900,
        )
        split2 = build_split_data_node(prep2, 0.2, 720, 180)
        root2 = build_trained_model_node(
            split2, "weights_hash_xyz",
            {"mae": 5.0, "rmse": 8.0, "r2": 0.9},
            {"epochs": 100, "learning_rate": 0.001, "batch_size": 32},
            "mlp", "mlp-1.0.0",
        )
        root2.compute_hash()

        assert root1.content_hash != root2.content_hash


# ── diff_merkle_trees ──────────────────────────────────────────────────────


class TestDiffMerkleTrees:
    def _make_tree_dict(self, sensor_id="s1", weights_hash="w1"):
        sensor = build_sensor_node(sensor_id, "fw1", "bme680")
        raw = build_raw_data_node(sensor, {"source_type": "synthetic"}, 100)
        cleansed = build_cleansed_data_node(raw, "fp", 90, 10, "ok")
        prep = build_preprocessed_data_node(cleansed, "sfp", {}, {}, "sh", 10, 10, 80)
        split = build_split_data_node(prep, 0.2, 64, 16)
        root = build_trained_model_node(
            split, weights_hash, {"mae": 5.0}, {"epochs": 10}, "mlp", "1.0.0",
        )
        root.compute_hash()
        return root.to_dict()

    def test_identical_trees(self):
        tree = self._make_tree_dict()
        result = diff_merkle_trees(tree, tree)
        assert len(result["changed_levels"]) == 0
        assert len(result["unchanged_levels"]) > 0

    def test_changed_root(self):
        old = self._make_tree_dict(weights_hash="w1")
        new = self._make_tree_dict(weights_hash="w2")
        result = diff_merkle_trees(old, new)
        assert "trained_model" in result["changed_levels"]

    def test_changed_leaf(self):
        old = self._make_tree_dict(sensor_id="s1")
        new = self._make_tree_dict(sensor_id="s2")
        result = diff_merkle_trees(old, new)
        assert "sensor" in result["changed_levels"]
        # All ancestors should also be changed
        assert "trained_model" in result["changed_levels"]

    def test_different_node_types(self):
        old = {"node_type": "a", "content_hash": "h1", "content_inputs": {}, "children": []}
        new = {"node_type": "b", "content_hash": "h1", "content_inputs": {}, "children": []}
        result = diff_merkle_trees(old, new)
        assert len(result["changed_levels"]) > 0


# ── compute_scaler_hash ───────────────────────────────────────────────────


class TestComputeScalerHash:
    def test_deterministic(self):
        fs = StandardScaler()
        ts = MinMaxScaler()
        fs.fit(np.array([[1, 2], [3, 4], [5, 6]]))
        ts.fit(np.array([[10], [20], [30]]))
        h1 = compute_scaler_hash(fs, ts)
        h2 = compute_scaler_hash(fs, ts)
        assert h1 == h2

    def test_different_scalers(self):
        fs1 = StandardScaler()
        ts1 = MinMaxScaler()
        fs1.fit(np.array([[1, 2], [3, 4]]))
        ts1.fit(np.array([[10], [20]]))

        fs2 = StandardScaler()
        ts2 = MinMaxScaler()
        fs2.fit(np.array([[10, 20], [30, 40]]))
        ts2.fit(np.array([[100], [200]]))

        assert compute_scaler_hash(fs1, ts1) != compute_scaler_hash(fs2, ts2)


# ── _hash_dict ─────────────────────────────────────────────────────────────


class TestHashDict:
    def test_deterministic(self):
        d = {"a": 1, "b": 2}
        assert _hash_dict(d) == _hash_dict(d)

    def test_different_input(self):
        assert _hash_dict({"a": 1}) != _hash_dict({"a": 2})

    def test_returns_16_chars(self):
        h = _hash_dict({"test": "value"})
        assert len(h) == 16
