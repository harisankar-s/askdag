# 3. Path-based owner rules when dbt has no owners

**Status:** accepted, 2026-11-21

## Context
"Who owns this model?" is a core question. In the Cal-ITP sample, 0 of 628 models record an owner in dbt (`meta.owner` or `group`). In practice, ownership often lives in CODEOWNERS files, Slack or people's heads.

## Decision
Resolve each node's owner in this order: dbt `meta.owner`, then the owner of its dbt `group`, then the first matching rule in an `owners.yml` file of path patterns. Store where the owner came from (`owner_source`) so answers can say how reliable it is. The Cal-ITP rules are synthetic and labelled as such; some folders are left unowned on purpose.

## Consequences
- Works today for projects with no ownership metadata, and dbt's own fields win as soon as a team adds them.
- Path rules go stale if folders move; `owner_source` makes that visible.
- Answers must distinguish "owned per owners.yml" from "owned per dbt", and must say when no owner is recorded.

## Alternatives considered
- **CODEOWNERS:** the same idea in GitHub's format; can be added as another rule source later.
- **Git history (top committer):** needs full history, and "who edited last" isn't "who owns it".
