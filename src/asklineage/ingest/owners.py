"""Path-based ownership rules for projects that don't record owners in dbt."""

from collections.abc import Iterable
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

import yaml


@dataclass(frozen=True)
class OwnerRule:
    pattern: str
    owner: str


def load_owner_rules(path: Path) -> tuple[OwnerRule, ...]:
    data = yaml.safe_load(path.read_text()) or {}
    return tuple(
        OwnerRule(pattern=rule["path"], owner=rule["owner"]) for rule in data.get("rules", [])
    )


def match_owner(rules: Iterable[OwnerRule], file_path: str) -> str | None:
    # fnmatch's "*" also matches "/", so "models/*/payments*" covers every layer and subfolder.
    return next((rule.owner for rule in rules if fnmatchcase(file_path, rule.pattern)), None)
