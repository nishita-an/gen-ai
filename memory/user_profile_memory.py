"""
Persistent User Profile Memory (Immutable Memory)
───────────────────────────────────────────────────
Stores long-term, durable facts about the user that survive across sessions.
Backed by a JSON file — simple, human-readable, easy to inspect / migrate.

Examples of profile data:
  • primary learning goals
  • preferred programming languages / tech stack
  • project descriptions
  • constraints ("I'm allergic to Java")
  • communication style preferences
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

_DEFAULT_PROFILE: Dict[str, Any] = {
    "user_id": "default",
    "name": "",
    "goals": [],
    "preferences": {},
    "projects": [],
    "technical_interests": [],
    "constraints": [],
    "custom_facts": {},
    "created_at": "",
    "updated_at": "",
}


class UserProfileMemory:
    """
    JSON-backed persistent user profile.

    Design principle: this memory is *never automatically discarded*.
    All writes append or merge; existing data is never deleted unless the
    caller explicitly calls remove_fact().
    """

    def __init__(self, profile_path: str | None = None) -> None:
        self._path = Path(profile_path or settings.memory.profile_path)
        self._profile: Dict[str, Any] = self._load()
        logger.info("UserProfileMemory loaded from %s", self._path)

    # ── Persistence ─────────────────────────────────────────────────────────

    def _load(self) -> Dict[str, Any]:
        if self._path.exists():
            with open(self._path, "r") as f:
                data = json.load(f)
            # Fill missing keys from default template
            for k, v in _DEFAULT_PROFILE.items():
                data.setdefault(k, copy.deepcopy(v))
            return data
        profile = copy.deepcopy(_DEFAULT_PROFILE)
        profile["created_at"] = datetime.now(timezone.utc).isoformat()
        profile["updated_at"] = profile["created_at"]
        self._save(profile)
        return profile

    def _save(self, profile: Optional[Dict[str, Any]] = None) -> None:
        data = profile or self._profile
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug("UserProfile saved to %s", self._path)

    # ── Read operations ─────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        return self._profile.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._profile)

    def format_for_prompt(self) -> str:
        """Return a concise text block suitable for injection into the LLM prompt."""
        lines: List[str] = []

        if name := self._profile.get("name"):
            lines.append(f"User name: {name}")

        goals = self._profile.get("goals", [])
        if goals:
            lines.append("Goals:\n" + "\n".join(f"  • {g}" for g in goals))

        prefs = self._profile.get("preferences", {})
        if prefs:
            lines.append("Preferences:\n" + "\n".join(f"  • {k}: {v}" for k, v in prefs.items()))

        projects = self._profile.get("projects", [])
        if projects:
            lines.append("Projects:\n" + "\n".join(f"  • {p}" for p in projects))

        interests = self._profile.get("technical_interests", [])
        if interests:
            lines.append("Technical interests: " + ", ".join(interests))

        constraints = self._profile.get("constraints", [])
        if constraints:
            lines.append("Constraints:\n" + "\n".join(f"  • {c}" for c in constraints))

        custom = self._profile.get("custom_facts", {})
        if custom:
            lines.append("Other facts:\n" + "\n".join(f"  • {k}: {v}" for k, v in custom.items()))

        return "\n".join(lines) if lines else "No user profile data yet."

    # ── Write operations ────────────────────────────────────────────────────

    def set_name(self, name: str) -> None:
        self._profile["name"] = name
        self._save()

    def add_goal(self, goal: str) -> None:
        if goal not in self._profile["goals"]:
            self._profile["goals"].append(goal)
            self._save()
            logger.info("UserProfile: added goal '%s'", goal)

    def add_project(self, project: str) -> None:
        if project not in self._profile["projects"]:
            self._profile["projects"].append(project)
            self._save()

    def add_interest(self, interest: str) -> None:
        if interest not in self._profile["technical_interests"]:
            self._profile["technical_interests"].append(interest)
            self._save()

    def add_constraint(self, constraint: str) -> None:
        if constraint not in self._profile["constraints"]:
            self._profile["constraints"].append(constraint)
            self._save()

    def set_preference(self, key: str, value: Any) -> None:
        self._profile["preferences"][key] = value
        self._save()

    def set_custom_fact(self, key: str, value: Any) -> None:
        self._profile["custom_facts"][key] = value
        self._save()

    def remove_fact(self, key: str) -> None:
        """Explicit removal — should be called only on user request."""
        removed = False
        for field in ("goals", "projects", "technical_interests", "constraints"):
            if isinstance(self._profile.get(field), list):
                before = len(self._profile[field])
                self._profile[field] = [x for x in self._profile[field] if x != key]
                if len(self._profile[field]) < before:
                    removed = True
        if key in self._profile.get("custom_facts", {}):
            del self._profile["custom_facts"][key]
            removed = True
        if removed:
            self._save()
            logger.info("UserProfile: removed fact '%s'", key)

    def merge_from_llm_extraction(self, extracted: Dict[str, Any]) -> None:
        """
        Merge structured facts extracted by the LLM.
        Expected keys: goals, projects, technical_interests, constraints, preferences, custom_facts
        """
        for goal in extracted.get("goals", []):
            self.add_goal(goal)
        for project in extracted.get("projects", []):
            self.add_project(project)
        for interest in extracted.get("technical_interests", []):
            self.add_interest(interest)
        for constraint in extracted.get("constraints", []):
            self.add_constraint(constraint)
        for k, v in extracted.get("preferences", {}).items():
            self.set_preference(k, v)
        for k, v in extracted.get("custom_facts", {}).items():
            self.set_custom_fact(k, v)
        logger.info("UserProfile: merged LLM extraction")

    def __repr__(self) -> str:
        return (
            f"UserProfileMemory("
            f"goals={len(self._profile.get('goals', []))}, "
            f"projects={len(self._profile.get('projects', []))})"
        )
