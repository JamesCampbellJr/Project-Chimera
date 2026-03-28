# /tests/test_memory.py

"""Tests for the persistent memory system."""

import json
import os
import shutil
import sys
import tempfile
import unittest

# Ensure the parent package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.memory import MemoryEntry, MemoryStore


class TestMemoryEntry(unittest.TestCase):
    """Tests for the MemoryEntry data class."""

    def test_create_entry_defaults(self):
        entry = MemoryEntry(content="Test content", category="general")
        self.assertEqual(entry.content, "Test content")
        self.assertEqual(entry.category, "general")
        self.assertEqual(entry.tags, [])
        self.assertEqual(entry.metadata, {})
        self.assertEqual(entry.importance, 0.5)
        self.assertEqual(entry.access_count, 0)
        self.assertIn("general_", entry.id)

    def test_create_entry_custom(self):
        entry = MemoryEntry(
            content="Custom entry",
            category="skills",
            tags=["python", "web"],
            metadata={"key": "value"},
            entry_id="custom_id",
            importance=0.9,
        )
        self.assertEqual(entry.id, "custom_id")
        self.assertEqual(entry.tags, ["python", "web"])
        self.assertEqual(entry.importance, 0.9)

    def test_importance_clamped(self):
        entry = MemoryEntry(content="x", category="general", importance=1.5)
        self.assertEqual(entry.importance, 1.0)

        entry2 = MemoryEntry(content="x", category="general", importance=-0.5)
        self.assertEqual(entry2.importance, 0.0)

    def test_to_dict_roundtrip(self):
        entry = MemoryEntry(
            content="Roundtrip test",
            category="feedback",
            tags=["test"],
            metadata={"score": 0.8},
            importance=0.7,
        )
        d = entry.to_dict()
        restored = MemoryEntry.from_dict(d)
        self.assertEqual(restored.content, entry.content)
        self.assertEqual(restored.category, entry.category)
        self.assertEqual(restored.tags, entry.tags)
        self.assertEqual(restored.importance, entry.importance)
        self.assertEqual(restored.metadata, entry.metadata)


class TestMemoryStore(unittest.TestCase):
    """Tests for the MemoryStore persistence and search."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="chimera_memory_test_")
        self.store = MemoryStore(memory_dir=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_store_and_retrieve(self):
        entry = self.store.store("Hello world", category="general", tags=["greeting"])
        self.assertIsNotNone(entry)
        self.assertEqual(entry.content, "Hello world")

        retrieved = self.store.get_by_id(entry.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.content, "Hello world")

    def test_persistence(self):
        self.store.store("Persistent data", category="skills", tags=["test"])

        # Create a new store pointing at the same directory
        new_store = MemoryStore(memory_dir=self.test_dir)
        entries = new_store.get_by_category("skills")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].content, "Persistent data")

    def test_search_by_keyword(self):
        self.store.store("Python web framework Flask", category="skills", tags=["python"])
        self.store.store("Rust CLI application", category="skills", tags=["rust"])
        self.store.store("JavaScript React frontend", category="skills", tags=["javascript"])

        results = self.store.search("python")
        self.assertEqual(len(results), 1)
        self.assertIn("Python", results[0].content)

    def test_search_by_tag(self):
        self.store.store("Entry with tag A", category="general", tags=["tagA"])
        self.store.store("Entry with tag B", category="general", tags=["tagB"])

        results = self.store.search("entry", tags=["tagA"])
        self.assertEqual(len(results), 1)
        self.assertIn("tag A", results[0].content)

    def test_search_with_category_filter(self):
        self.store.store("Skill entry", category="skills")
        self.store.store("Feedback entry", category="feedback")

        results = self.store.search("entry", category="skills")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].category, "skills")

    def test_search_limit(self):
        for i in range(10):
            self.store.store(f"Repeated word item {i}", category="general")

        results = self.store.search("repeated", limit=3)
        self.assertEqual(len(results), 3)

    def test_delete(self):
        entry = self.store.store("To be deleted", category="general")
        self.assertTrue(self.store.delete(entry.id))
        self.assertIsNone(self.store.get_by_id(entry.id))

    def test_delete_nonexistent(self):
        self.assertFalse(self.store.delete("nonexistent_id"))

    def test_invalid_category(self):
        with self.assertRaises(ValueError):
            self.store.store("Bad category", category="invalid_category")

    def test_get_stats(self):
        self.store.store("Skill 1", category="skills")
        self.store.store("Skill 2", category="skills")
        self.store.store("Feedback 1", category="feedback")

        stats = self.store.get_stats()
        self.assertEqual(stats["skills"], 2)
        self.assertEqual(stats["feedback"], 1)
        self.assertEqual(stats["total"], 3)

    def test_clear_category(self):
        self.store.store("Entry 1", category="general")
        self.store.store("Entry 2", category="general")
        self.assertEqual(len(self.store.get_by_category("general")), 2)

        self.store.clear_category("general")
        self.assertEqual(len(self.store.get_by_category("general")), 0)

    def test_get_by_category_ordered_by_recency(self):
        self.store.store("First", category="general")
        self.store.store("Second", category="general")
        self.store.store("Third", category="general")

        entries = self.store.get_by_category("general")
        # Most recent first
        self.assertEqual(entries[0].content, "Third")

    def test_search_updates_access_count(self):
        entry = self.store.store("Searchable content", category="general")
        self.assertEqual(entry.access_count, 0)

        results = self.store.search("searchable")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].access_count, 1)

    def test_file_created_on_disk(self):
        self.store.store("Check file", category="skills")
        path = os.path.join(self.test_dir, "skills.json")
        self.assertTrue(os.path.exists(path))

        with open(path, "r") as fh:
            data = json.load(fh)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["content"], "Check file")


if __name__ == "__main__":
    unittest.main()
