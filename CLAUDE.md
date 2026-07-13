# DA_GestureRecognition

Research repo: gesture recognition via domain adaptation (NTU skeleton → Xsens IMU). All project knowledge lives in `wiki/`.

## Session start
Read `SESSION_HANDOFF.md`, then `wiki/index.md`; open only the pages the task needs. Never re-read the codebase or legacy docs wholesale — the wiki indexes them.

## Wiki
- Folders `concepts/ data/ code/ experiments/ results/ questions/`; frontmatter `type`, `status`, `updated`; `[[wikilinks]]` liberally.
- Code pages: purpose/IO/gotchas + `file:line` pointers — never code copies. Numbers exact.
- Any change this session → update affected pages + `index.md`, append `## [YYYY-MM-DD] ingest | title` to `log.md` before ending. File valuable answers as `results/` pages. Lint on request.

## Repo
- `source .venv/bin/activate` (not conda); long runs via `nohup`, logs in repo root.
- `RESEARCH_LOG.md` = channel with Planning assistant: read §A, write §B only — numbers, not conclusions.
- Don't regenerate `Data_Processed/` unless processing logic changes.
- State plan + get the human's go before implementing.
export_alex_retarget_npz_to_ihmc_json