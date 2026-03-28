# /core/memory.py

"""
Persistent Memory System for Project Chimera.

Provides long-term memory storage, retrieval, and semantic search capabilities
so that agents can learn from past experiences and continuously improve.
Memory is organized into categories (skills, feedback, projects, business, general)
and persisted as JSON files on disk.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


MEMORY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "memory_store")


class MemoryEntry:
    """A single memory entry with metadata."""

    def __init__(
        self,
        content: str,
        category: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        entry_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        importance: float = 0.5,
    ):
        self.id = entry_id or f"{category}_{int(time.time() * 1000)}"
        self.content = content
        self.category = category
        self.tags = tags or []
        self.metadata = metadata or {}
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.importance = max(0.0, min(1.0, importance))
        self.access_count = 0
        self.last_accessed = self.timestamp

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category,
            "tags": self.tags,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        entry = cls(
            content=data["content"],
            category=data["category"],
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            entry_id=data.get("id"),
            timestamp=data.get("timestamp"),
            importance=data.get("importance", 0.5),
        )
        entry.access_count = data.get("access_count", 0)
        entry.last_accessed = data.get("last_accessed", entry.timestamp)
        return entry


class MemoryStore:
    """
    Persistent memory store backed by JSON files.

    Memories are organized by category, each stored in its own file.
    Supports keyword-based search across all categories.
    """

    VALID_CATEGORIES = ("skills", "feedback", "projects", "business", "general", "errors", "improvements")

    def __init__(self, memory_dir: Optional[str] = None):
        self.memory_dir = memory_dir or MEMORY_DIR
        os.makedirs(self.memory_dir, exist_ok=True)
        self._cache: Dict[str, List[MemoryEntry]] = {}
        self._load_all()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _category_path(self, category: str) -> str:
        return os.path.join(self.memory_dir, f"{category}.json")

    def _load_category(self, category: str) -> List[MemoryEntry]:
        path = self._category_path(category)
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r") as fh:
                raw = json.load(fh)
            return [MemoryEntry.from_dict(item) for item in raw]
        except (json.JSONDecodeError, KeyError) as exc:
            logging.error("Failed to load memory category %s: %s", category, exc)
            return []

    def _load_all(self):
        for cat in self.VALID_CATEGORIES:
            self._cache[cat] = self._load_category(cat)
        logging.info(
            "Memory store loaded: %s",
            {cat: len(entries) for cat, entries in self._cache.items() if entries},
        )

    def _save_category(self, category: str):
        path = self._category_path(category)
        entries = self._cache.get(category, [])
        with open(path, "w") as fh:
            json.dump([e.to_dict() for e in entries], fh, indent=2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(
        self,
        content: str,
        category: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
    ) -> MemoryEntry:
        """Store a new memory entry and persist it to disk."""
        if category not in self.VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{category}'. Must be one of {self.VALID_CATEGORIES}"
            )
        entry = MemoryEntry(
            content=content,
            category=category,
            tags=tags,
            metadata=metadata,
            importance=importance,
        )
        self._cache.setdefault(category, []).append(entry)
        self._save_category(category)
        logging.info("Memory stored [%s]: %s", category, entry.id)
        return entry

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """
        Search memories by keyword matching on content and tags.

        Returns results sorted by relevance (keyword hits + importance).
        """
        query_lower = query.lower()
        query_words = query_lower.split()

        categories = [category] if category else list(self.VALID_CATEGORIES)
        results: List[tuple] = []  # (score, entry)

        for cat in categories:
            for entry in self._cache.get(cat, []):
                # Tag filter
                if tags and not any(t in entry.tags for t in tags):
                    continue

                content_lower = entry.content.lower()
                tag_text = " ".join(entry.tags).lower()

                # Simple relevance scoring
                score = 0.0
                for word in query_words:
                    if word in content_lower:
                        score += 1.0
                    if word in tag_text:
                        score += 0.5

                if score > 0:
                    score += entry.importance * 0.5
                    results.append((score, entry))

        results.sort(key=lambda x: x[0], reverse=True)
        found = [entry for _, entry in results[:limit]]

        # Update access counts
        now = datetime.now(timezone.utc).isoformat()
        for entry in found:
            entry.access_count += 1
            entry.last_accessed = now

        return found

    def get_by_category(self, category: str, limit: int = 50) -> List[MemoryEntry]:
        """Return the most recent entries in a category."""
        entries = self._cache.get(category, [])
        return sorted(entries, key=lambda e: e.timestamp, reverse=True)[:limit]

    def get_by_id(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory entry by its ID."""
        for entries in self._cache.values():
            for entry in entries:
                if entry.id == entry_id:
                    return entry
        return None

    def delete(self, entry_id: str) -> bool:
        """Delete a memory entry by its ID."""
        for cat, entries in self._cache.items():
            for i, entry in enumerate(entries):
                if entry.id == entry_id:
                    entries.pop(i)
                    self._save_category(cat)
                    logging.info("Memory deleted: %s", entry_id)
                    return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics about the memory store."""
        stats: Dict[str, Any] = {}
        total = 0
        for cat in self.VALID_CATEGORIES:
            count = len(self._cache.get(cat, []))
            if count:
                stats[cat] = count
            total += count
        stats["total"] = total
        return stats

    def clear_category(self, category: str):
        """Remove all entries from a category."""
        self._cache[category] = []
        self._save_category(category)
        logging.info("Memory category '%s' cleared.", category)
