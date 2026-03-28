# /tests/test_self_improvement.py

"""Tests for the self-improvement module."""

import os
import shutil
import sys
import tempfile
import unittest

# Ensure the parent package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.self_improvement import SelfImprovement
from core.memory import MemoryStore


class TestSelfImprovement(unittest.TestCase):
    """Tests for the SelfImprovement module (non-LLM parts)."""

    def setUp(self):
        self.memory_dir = tempfile.mkdtemp(prefix="chimera_si_test_")
        self.memory = MemoryStore(memory_dir=self.memory_dir)
        self.si = SelfImprovement(memory_store=self.memory)

    def tearDown(self):
        shutil.rmtree(self.memory_dir, ignore_errors=True)

    def test_performance_summary_empty(self):
        summary = self.si.get_performance_summary()
        self.assertEqual(summary["total_evaluations"], 0)
        self.assertEqual(summary["average_score"], 0.0)

    def test_performance_summary_with_data(self):
        # Simulate some evaluation history
        self.si.evaluation_history = [
            {"evaluation": {"overall_score": 0.8}, "task": "task1", "output_type": "code"},
            {"evaluation": {"overall_score": 0.6}, "task": "task2", "output_type": "code"},
            {"evaluation": {"overall_score": 0.9}, "task": "task3", "output_type": "code"},
        ]
        summary = self.si.get_performance_summary()
        self.assertEqual(summary["total_evaluations"], 3)
        self.assertAlmostEqual(summary["average_score"], 0.767, places=2)
        self.assertEqual(len(summary["recent_scores"]), 3)

    def test_performance_trend_improving(self):
        self.si.evaluation_history = [
            {"evaluation": {"overall_score": 0.4}, "task": "t1", "output_type": "code"},
            {"evaluation": {"overall_score": 0.6}, "task": "t2", "output_type": "code"},
            {"evaluation": {"overall_score": 0.8}, "task": "t3", "output_type": "code"},
        ]
        summary = self.si.get_performance_summary()
        self.assertEqual(summary["trend"], "improving")

    def test_performance_trend_stable(self):
        self.si.evaluation_history = [
            {"evaluation": {"overall_score": 0.7}, "task": "t1", "output_type": "code"},
            {"evaluation": {"overall_score": 0.7}, "task": "t2", "output_type": "code"},
            {"evaluation": {"overall_score": 0.7}, "task": "t3", "output_type": "code"},
        ]
        summary = self.si.get_performance_summary()
        self.assertEqual(summary["trend"], "stable")

    def test_no_memory_still_works(self):
        si_no_mem = SelfImprovement(memory_store=None)
        summary = si_no_mem.get_performance_summary()
        self.assertEqual(summary["total_evaluations"], 0)


if __name__ == "__main__":
    unittest.main()
