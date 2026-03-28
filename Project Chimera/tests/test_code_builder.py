# /tests/test_code_builder.py

"""Tests for the code builder and project manager modules."""

import json
import os
import shutil
import sys
import tempfile
import unittest

# Ensure the parent package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.code_builder import PROJECT_TEMPLATES, SUPPORTED_LANGUAGES
from core.project_manager import ProjectManager, ProjectState
from core.memory import MemoryStore
from core.business_manager import BusinessManager, ClientProject


class TestProjectTemplates(unittest.TestCase):
    """Tests for project template definitions."""

    def test_templates_have_required_keys(self):
        for name, tmpl in PROJECT_TEMPLATES.items():
            self.assertIn("description", tmpl, f"Template '{name}' missing description")
            self.assertIn("files", tmpl, f"Template '{name}' missing files")
            self.assertIn("language", tmpl, f"Template '{name}' missing language")
            self.assertIsInstance(tmpl["files"], list)
            self.assertTrue(len(tmpl["files"]) > 0, f"Template '{name}' has empty files list")

    def test_template_languages_supported(self):
        for name, tmpl in PROJECT_TEMPLATES.items():
            self.assertIn(
                tmpl["language"],
                SUPPORTED_LANGUAGES,
                f"Template '{name}' uses unsupported language '{tmpl['language']}'",
            )


class TestProjectState(unittest.TestCase):
    """Tests for the ProjectState data class."""

    def test_create_project_state(self):
        proj = ProjectState("test_1", "Build a website")
        self.assertEqual(proj.id, "test_1")
        self.assertEqual(proj.description, "Build a website")
        self.assertEqual(proj.status, "planning")
        self.assertEqual(proj.files_created, [])
        self.assertEqual(proj.tasks_completed, [])
        self.assertEqual(proj.tasks_pending, [])

    def test_update_status(self):
        proj = ProjectState("test_2", "Test project")
        proj.update_status("in_progress")
        self.assertEqual(proj.status, "in_progress")

    def test_invalid_status(self):
        proj = ProjectState("test_3", "Test project")
        with self.assertRaises(ValueError):
            proj.update_status("invalid_status")

    def test_to_dict_roundtrip(self):
        proj = ProjectState("test_4", "Roundtrip test")
        proj.update_status("in_progress")
        proj.files_created = ["main.py", "README.md"]
        proj.tasks_completed = ["Setup"]
        proj.tasks_pending = ["Build", "Test"]

        d = proj.to_dict()
        restored = ProjectState.from_dict(d)
        self.assertEqual(restored.id, proj.id)
        self.assertEqual(restored.description, proj.description)
        self.assertEqual(restored.status, proj.status)
        self.assertEqual(restored.files_created, proj.files_created)
        self.assertEqual(restored.tasks_completed, proj.tasks_completed)
        self.assertEqual(restored.tasks_pending, proj.tasks_pending)


class TestProjectManager(unittest.TestCase):
    """Tests for the ProjectManager persistence."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="chimera_projects_test_")
        self.memory_dir = tempfile.mkdtemp(prefix="chimera_memory_test_")
        self.memory = MemoryStore(memory_dir=self.memory_dir)
        self.pm = ProjectManager(memory_store=self.memory, projects_dir=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
        shutil.rmtree(self.memory_dir, ignore_errors=True)

    def test_list_projects_empty(self):
        self.assertEqual(self.pm.list_projects(), [])

    def test_complete_task(self):
        # Manually create a project state for testing (no LLM needed)
        proj = ProjectState("manual_1", "Manual test")
        proj.tasks_pending = ["Task A", "Task B"]
        self.pm.active_projects["manual_1"] = proj
        self.pm._save_projects()

        self.pm.complete_task("manual_1", "Task A")
        self.assertIn("Task A", proj.tasks_completed)
        self.assertNotIn("Task A", proj.tasks_pending)
        self.assertEqual(proj.status, "in_progress")

        self.pm.complete_task("manual_1", "Task B")
        self.assertEqual(proj.status, "completed")

    def test_add_error(self):
        proj = ProjectState("err_1", "Error test")
        self.pm.active_projects["err_1"] = proj

        self.pm.add_error("err_1", "Something went wrong")
        self.assertEqual(proj.errors, ["Something went wrong"])

    def test_persistence(self):
        proj = ProjectState("persist_1", "Persistence test")
        proj.tasks_pending = ["Build"]
        self.pm.active_projects["persist_1"] = proj
        self.pm._save_projects()

        # Load in a new manager
        pm2 = ProjectManager(projects_dir=self.test_dir)
        loaded = pm2.get_project("persist_1")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.description, "Persistence test")


class TestClientProject(unittest.TestCase):
    """Tests for the ClientProject data class."""

    def test_create_client_project(self):
        proj = ClientProject("Acme Inc", "Build an API", 5000.0)
        self.assertEqual(proj.client_name, "Acme Inc")
        self.assertEqual(proj.description, "Build an API")
        self.assertEqual(proj.price, 5000.0)
        self.assertEqual(proj.status, "proposal")

    def test_to_dict_roundtrip(self):
        proj = ClientProject("Test Corp", "Website", 3000.0, project_id="cp_1")
        proj.status = "accepted"

        d = proj.to_dict()
        restored = ClientProject.from_dict(d)
        self.assertEqual(restored.id, "cp_1")
        self.assertEqual(restored.client_name, "Test Corp")
        self.assertEqual(restored.price, 3000.0)
        self.assertEqual(restored.status, "accepted")


class TestBusinessManager(unittest.TestCase):
    """Tests for the BusinessManager."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="chimera_biz_test_")
        self.bm = BusinessManager(data_dir=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_record_payment(self):
        proj = ClientProject("Client A", "App", 2000.0, project_id="pay_1")
        self.bm.client_projects["pay_1"] = proj

        self.bm.record_payment("pay_1", 2000.0)
        self.assertEqual(proj.status, "paid")
        self.assertEqual(len(self.bm.revenue_log), 1)
        self.assertEqual(self.bm.revenue_log[0]["amount"], 2000.0)

    def test_revenue_summary(self):
        summary = self.bm.get_revenue_summary()
        self.assertEqual(summary["total_revenue"], 0)
        self.assertEqual(summary["total_projects"], 0)

    def test_update_project_status(self):
        proj = ClientProject("Client B", "Service", 1500.0, project_id="upd_1")
        self.bm.client_projects["upd_1"] = proj

        self.bm.update_project_status("upd_1", "delivered")
        self.assertEqual(proj.status, "delivered")
        self.assertIsNotNone(proj.delivered_at)

    def test_persistence(self):
        proj = ClientProject("Persist Client", "DB", 4000.0, project_id="persist_biz_1")
        self.bm.client_projects["persist_biz_1"] = proj
        self.bm.record_payment("persist_biz_1", 4000.0)

        # Load in a new manager
        bm2 = BusinessManager(data_dir=self.test_dir)
        self.assertIn("persist_biz_1", bm2.client_projects)
        self.assertEqual(len(bm2.revenue_log), 1)


if __name__ == "__main__":
    unittest.main()
