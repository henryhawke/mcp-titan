# Documentation Dependency Map (Phase 0 Audit)

Date: October 10, 2025  
Owner: Phase 0 Documentation Task Force  
Scope: README.md, DETAILED_IMPLEMENTATION_GUIDE.md, IMPLEMENTATION_PROGRESS.md, COMPLETE_IMPLEMENTATION_PACKAGE.md, ROADMAP_ANALYSIS.md, AUDIT_FINDINGS.md, AUDIT_IMPLEMENTATION_SUMMARY.md, GUIDE_UPDATE_SUMMARY.md, mcp-titan-system-audit.md, docs/api/README.md, docs/llm-system-prompt.md, PRODUCTION_READINESS_ANALYSIS.md, IMPLEMENTATION_COMPLETE.md

---

## 1. Canonical Sources & Dependencies

| Document | Primary Focus | Depends On / References | Overlaps With | Current Drift / Contradictions |
| --- | --- | --- | --- | --- |
| README.md | Public quick start, feature list, tool summary | docs/api/README.md, src/index.ts | ROADMAP_ANALYSIS.md, IMPLEMENTATION_COMPLETE.md | Updated to 17-tool registry (includes `get_token_flow_metrics`); monitor for parity with help output. |
| docs/api/README.md | API schema reference | src/index.ts, src/learner.ts, src/types.ts | README.md, docs/llm-system-prompt.md | Table synchronized to 17 tools with roadmap callouts. |
| docs/llm-system-prompt.md | Client-facing tool instructions | docs/api/README.md | README.md, MCP client prompts | Rewritten with complete tool coverage and roadmap caveats. |
| ROADMAP_ANALYSIS.md | Strategic backlog | README.md, docs/api/README.md | IMPLEMENTATION_PROGRESS.md, DETAILED_IMPLEMENTATION_GUIDE.md | Now reflects 17-tool registry; still highlights missing `manifold_step`. |
| IMPLEMENTATION_COMPLETE.md | Delivery scope snapshot | README.md, docs/api/README.md | IMPLEMENTATION_PROGRESS.md, PRODUCTION_READINESS_ANALYSIS.md | Updated to 17 tools; keeps roadmap dependency notices. |
| PRODUCTION_READINESS_ANALYSIS.md | Deployment readiness gaps | src/index.ts, src/model.ts | IMPLEMENTATION_COMPLETE.md | Consistent with current status; references “10+ tools” loosely but doesn’t flag exact count drift. |
| DETAILED_IMPLEMENTATION_GUIDE.md | Deep implementation plan | research_paper_source.md, src/model.ts | COMPLETE_IMPLEMENTATION_PACKAGE.md, GUIDE_UPDATE_SUMMARY.md | Contains redundant sections (Momentum/Token Flow repeated in appendices, duplicated checklists). Action matrix still assumes legacy audit docs. |
| COMPLETE_IMPLEMENTATION_PACKAGE.md | Bundle description | DETAILED_IMPLEMENTATION_GUIDE.md, IMPLEMENTATION_PROGRESS.md | GUIDE_UPDATE_SUMMARY.md | Claims audit to-dos 31% complete (based on Jan 2025 data), references files slated for consolidation. |
| GUIDE_UPDATE_SUMMARY.md | Delta log for guide | DETAILED_IMPLEMENTATION_GUIDE.md | COMPLETE_IMPLEMENTATION_PACKAGE.md | Redundant with package doc; still points to 2,768-line guide structure pre-refactor. |
| AUDIT_FINDINGS.md | Raw audit results | src/index.ts, src/model.ts | AUDIT_IMPLEMENTATION_SUMMARY.md, mcp-titan-system-audit.md | Documents only 11 tools, asserts Week 1 immediate tasks completed that no longer match code. |
| AUDIT_IMPLEMENTATION_SUMMARY.md | Summary of audit fixes | AUDIT_FINDINGS.md, src/index.ts | mcp-titan-system-audit.md | Claims integration tests exist in `src/__tests__/integration.test.ts` (file absent). References `manifold_step` tool. |
| mcp-titan-system-audit.md | Combined audit & roadmap | AUDIT_FINDINGS.md | ROADMAP_ANALYSIS.md | Repeats same tool inaccuracies (“process_input”, 10-tool list). |
| IMPLEMENTATION_PROGRESS.md | Status tracker | ROADMAP_ANALYSIS.md, README.md | IMPLEMENTATION_COMPLETE.md | Labeled Phase 0–1 complete with no checkboxes/matrix; metrics stale, no canonical source mapping, no test coverage stats. |

**Actual Registered Tools (src/index.ts):** `help`, `bootstrap_memory`, `init_model`, `memory_stats`, `forward_pass`, `train_step`, `get_memory_state`, `get_token_flow_metrics`, `reset_gradients`, `prune_memory`, `save_checkpoint`, `load_checkpoint`, `init_learner`, `pause_learner`, `resume_learner`, `get_learner_stats`, `add_training_sample`. `manifold_step`, `encode_text`, `get_surprise_metrics`, `analyze_memory`, and `predict_next` do **not** exist today.

---

## 2. Specific Issues (Requested Focus)

1. **docs/api/README.md references non-existent `process_input`:** ✅ Removed in current doc; needs explicit note confirming text now references `processInput` helper indirectly. → Update narrative to avoid regression.
2. **Tool count parity:** ✅ Resolved — help output, README, API reference, and system prompt now reflect the 17-tool registry. Keep parity checks in the release checklist.
3. **`IMPLEMENTATION_COMPLETE.md` vs `PRODUCTION_READINESS_ANALYSIS.md`:** Current wording aligned, but legacy copies claimed production readiness. → Make new consolidated SYSTEM_AUDIT.md explicitly set status.
4. **CLI name drift:** README now correct (`titan-memory`). → Note in audit so consolidated doc can preserve resolution.
5. **docs/llm-system-prompt.md schema drift:** ✅ Rewritten with current schemas. Monitor future updates.
6. **Help text references `manifold_step`:** ✅ Updated to current registry; keep roadmap note for planned tools.

---

## 3. Proposed Canonical Ownership (post-consolidation)

| Topic | Source of Truth | Supporting Reference |
| --- | --- | --- |
| Tool schemas & parameters | docs/api/README.md (derive from src/index.ts) | README.md, docs/llm-system-prompt.md |
| System status & audits | SYSTEM_AUDIT.md (new consolidated doc) | IMPLEMENTATION_PROGRESS.md |
| Implementation roadmap | ROADMAP_ANALYSIS.md | DETAILED_IMPLEMENTATION_GUIDE.md |
| Delivery bundle overview | IMPLEMENTATION_PACKAGE.md (new consolidated doc) | README.md |
| Phase progress & metrics | IMPLEMENTATION_PROGRESS.md | SYSTEM_AUDIT.md |
| Architecture | docs/architecture-overview.md | README.md |

---

## 4. Immediate Action Items

- Merge AUDIT_* docs + `mcp-titan-system-audit.md` into **SYSTEM_AUDIT.md** with current status tables.
- Merge COMPLETE_IMPLEMENTATION_PACKAGE.md + GUIDE_UPDATE_SUMMARY.md into **IMPLEMENTATION_PACKAGE.md**; archive original docs.
- Update docs/api/README.md, README.md, docs/llm-system-prompt.md, and in-code help text to reflect the 17-tool reality (or explicitly mark planned tools).
- Add code/feature matrix & metrics table to IMPLEMENTATION_PROGRESS.md (Phase checkboxes, status matrix, telemetry placeholders).
- Expand docs/architecture-overview.md with `WorkflowUtils`, additional class docs, and workflow intentions vs reality.

---

## 5. Archive & Versioning Notes

- Obsolete inputs (AUDIT_FINDINGS.md, AUDIT_IMPLEMENTATION_SUMMARY.md, mcp-titan-system-audit.md, COMPLETE_IMPLEMENTATION_PACKAGE.md, GUIDE_UPDATE_SUMMARY.md) should be moved to `docs/archive/` after consolidation, preserving git history.
- New consolidated docs must carry forward dates/versioning to prevent future drift.

---

Prepared by: Codex (Phase 0 audit)  
Next Review: After Phase 0 consolidation tasks are complete.
