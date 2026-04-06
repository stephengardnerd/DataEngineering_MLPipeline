# Contributing — Stephen Gardner's Agent Ecosystem

This repository is part of a multi-agent ecosystem managed by Stephen D. Gardner. Four AI agents collaborate on these repositories under unified conventions.

## Agent Identities

| Agent | Framework | Git Name | Git Email |
|-------|-----------|----------|-----------|
| Claude | Anthropic | `Claude (Anthropic)` | `claude@brainhq.local` |
| Aquarius | OpenClaw | `Aquarius (OpenClaw)` | `aquarius@brainhq.local` |
| Cancer | Antigravity | `Cancer (Antigravity)` | `cancer@brainhq.local` |
| Virgo | Hermes | `Virgo (Hermes)` | `virgo@brainhq.local` |
| Stephen | — | `stephengardnerd` | `flight0234@gmail.com` |

All agent commits use `@brainhq.local` emails. These are attribution identifiers and will show as "unverified" on GitHub — this is expected.

## Branch Naming

```
agent/<name>/<feature-or-task-id>    Working branches
collab/<agent1>-<agent2>/<feature>   Cross-agent collaboration
main / master                        Protected — no direct push
```

Examples:
```
agent/claude/T012-cancer-bridge-setup
agent/aquarius/trading-cron-fix
agent/cancer/codebase-audit-april
collab/claude-cancer/gemini-integration
```

## Commit Message Format

```
<type>(<scope>): <description>

[optional body]

Agent: <Name> (<Framework>)
Task: <T-ID or n/a>
```

### Types
- `feat` — new feature or capability
- `fix` — bug fix
- `chore` — maintenance, dependency updates
- `docs` — documentation only
- `refactor` — code restructuring (no behavior change)
- `audit` — code quality or security audit output

### Example
```
feat(etl): add retry logic to disaster message ingestion

Handles transient HTTP failures when fetching message CSVs.
Exponential backoff with 3 retries.

Agent: Claude (Anthropic)
Task: T015
```

## Push Rules

1. **Always push to both `main` and `master`** after merge.
2. **Never push directly to `main` or `master`** — work on agent branches, merge via fast-forward or PR.
3. **Cancer (AI Studio exports)** push to the `aistudio-exports` repo, then route through the bridge.

## Cross-Agent Task Routing

- **Cancer involved** → `brainhq-cancer-bridge` repo (always)
- **Aquarius / Claude only** → `_shared/tasks/TASK_QUEUE.md` in BrainHQ vault (always)
- No overlap. No exceptions.

## Task Schema

All bridge task files include `schema_version: 1`. Unknown fields are ignored (forward-compatible). Breaking changes bump the version.

## Code Review

All significant changes go through review:
- Aquarius submits plans to `_shared/reviews/` in BrainHQ
- Claude reviews and responds in the same directory
- Doc update reminders (README, TECHNICAL, DISASTER_RECOVERY) are included in every review response

## Author Attribution

All documents and products created in this ecosystem list **Stephen D. Gardner** as the author.

---

*Maintained by the BrainHQ agent ecosystem. Last updated: 2026-04-06.*
