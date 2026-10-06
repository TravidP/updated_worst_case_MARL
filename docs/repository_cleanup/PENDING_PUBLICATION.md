# Publication review / 发布前待处理

- No push or Release was performed. The ZIP remains local; obtain author approval and publish it separately, then add the actual URL to RESTORE.md.
- `paper/` remains local and ignored pending publication/copyright clearance. The already tracked `Journal_Paper.pdf` is unchanged; confirm its sharing rights before publishing the repository.
- `.codex` was an empty local marker and is now untracked, not deleted.
- `docs/site/.openai/hosting.json` contains deployment metadata; review whether the existing private-site identifier should be public. It was not changed.
- Machine-specific paths are listed by filename and line only in public_information_review.json. They are retained where they describe historical provenance, not presented as portable commands. No high-confidence credential pattern was detected; this is not a guarantee of absence.
- `docs/history/` links in older documentation refer to local archives, intentionally not available in a fresh clone. Obtain these separately only if auditing historical development.
- Preserved gate JSON files are historical evidence, not fresh gates for the now-modified source tree. New experiments must follow their current gate contract; this cleanup does not rerun verification or evaluation.
- Final models and frozen demand remain local at paths referenced by selections. The display ZIP does not include them. Exact paired demand/SUMO seeds and hashes are in paired_demand_provenance.csv; model mappings remain in runs_eval/revised/selections/.
- Original network inputs and legacy experimental datasets were retained rather than guessing that they are disposable.

未推送、未发布 Release；论文发布权、部署元数据与本机路径仍需公开前人工复核。
历史 gate 只用于溯源，不能视为对本次新源码重新验证。训练数据未删除。
