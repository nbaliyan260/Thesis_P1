# Stage-3 execution record

16 September 2026. This log is separate from immutable earlier-stage evidence.

## Setup and pre-measurement work

- Read existing contract, publication assessment and stage-2 source/results. Independent review selected protection-cost measurement, exact-semantics seal batching, a larger model/longer contexts, broader K/V faults and replayable snapshots as the completion gate.
- Reconnected using the already authorized HPC account; authentication is not stored in any project file. Read-only checks showed no existing jobs, sufficient idle capacity and a 3 TB user quota with about 5.9 GB used before the new model download.
- Resolved public ungated Qwen2.5-7B revision `d149729398750b98c0af14eb82c78cfe92750796` through the official Hugging Face model API before download/measurement. The existing two model revisions are unchanged. Model card: https://huggingface.co/Qwen/Qwen2.5-7B (Apache-2.0, 7.61B parameters). No weight redistribution is planned.
- CPU-only fetch job **195906** failed before downloading because the remote checkout lacked the locally packaged `experiments/fetch_pinned_models.py`. Its logs are preserved. Added a task-owned stage-3 pinned fetcher and resubmitted as **195910**; no measurement code/outcome was changed by this setup repair.
- Implemented an independent stage-3 controller with batched transfer scheduling and exact canonical SHA-256 equivalence. Local final combined suite: **269 passed, 17 CUDA-only skipped**, including all 107 original/stage-2 tests. GPU behavior is not claimed from these local tests.
- Runner's separate tiny-HF CPU integration passed its complete 39-session debug matrix, 111 commits, three expected checkpoint refusals, seven independently reloadable snapshots and separate profile checks. This is synthetic validation, not pretrained evidence.

## Frozen allocated execution

- Pinned model-fetch job **195910** completed successfully at 16:44:59 Dubai time. The 7B model was downloaded on its CPU allocation; existing pinned smaller models were reused.
- The 23-file `protocol_freeze.json` passed both local and remote verification before any pretrained stage-3 measurements. Debug array **195926** and dependency-gated main array **195927** were submitted. Main requires successful completion of every model's debug task.
- The scheduler initially ran two debug tasks while the third waited on the account's `QOSMaxJobsPerUserLimit`; no attempt was made to bypass that limit. Each task requests one GPU, four CPUs, 48 GB host memory and a two-hour limit.
- All three debug tasks completed successfully. Each allocated GPU ran the combined **286-test** suite with no skipped CUDA cases. Each pretrained debug run then completed 39 sessions, 111 commits and three specified checkpoint refusals; these tiny runs are excluded from main-study results.
- Main array **195927** started only after the entire debug array passed. Qwen-7B waits for a scheduler slot while the two smaller models execute; this is the lab's account limit, not insufficient GPU memory.
- Retrieved all debug records, snapshots and setup/debug logs. Separate local CPU audits matched HPC audit outcomes: Smol 24,606 checks; Qwen-0.5B 23,649; Qwen-7B 32,461. Across the 21 debug archives, 2,338 actual/reference tensor pairs passed numeric and stricter byte comparisons. These are reload audits, not local model re-execution.
- Independent raw-result recomputation found one **reporting-only defect**: the frozen analyzer displayed refused-window seal counters as zero because their raw metadata is nested under `rejection`. No timing, fault, correctness or integrity-gate record was affected. A separate post-measurement tool writes `summary_corrected.json` and hash-bound `analysis_corrections.json`, preserving original summaries and raw data. Thirteen persisted offline reporting regression tests passed. No frozen source was changed and no outcome was retuned.
- Qwen eager-attention warnings are retained. Saved configurations explicitly set `use_sliding_window=false`; the engine rejects active sliding-window mode and native/custom parity is still required.

## Completed allocated outcomes

- All three main array tasks completed. Smol and Qwen-0.5B HPC audits passed 312,645 and 313,628 checks respectively. Qwen-7B completed at 17:44:16 Dubai with 392,720 passed audit checks; live `scontrol` reported `COMPLETED`, exit `0:0`, and 32:53 elapsed for that task.
- Each main model finished 786 measured sessions, 2,322 committed windows, 18 specified checkpoint refusals, 228 native/custom parity positions, and seven predeclared tensor archives. Warmups and synchronized profiles remain separate. The 23-file freeze passed before and after allocated execution.
- A post-hoc **validation-only** replay was declared to archive Smol's already-observed code/prefix-128/W8 multi-prefix fallback, seed 7431, repetition 0. Before job **195986**, a separate four-file freeze bound the amendment, reduced config, verifier and submission script. The job log records those bindings before model execution. It completed at 17:37:52 Dubai: 13 sessions, 33 commits, three specified refusals and three archives. Local/HPC audits each passed 8,124 checks, and the selected-case verifier reproduced the exact original applied faults, references and trajectories. Its 366 tensor pairs also passed byte comparisons. Nothing from this selected replay enters main counts, timings or independent-fault claims.
- The frozen amendment's phrase "index 7" is clarified in a separate note: `first_divergence=7` is one-based; predicted token seven changes and step eight starts at layer zero. The pre-run amendment, traces and acceptance code remain unchanged.
- The final local combined core/controller/reporting suite passed **296 tests, 17 CUDA-only skipped**. The earlier allocated suite passed all **286 core/controller tests** on each model's GPU task. Twenty-seven additional reporting tests are CPU-only. A known local LibreSSL warning is preserved.
- Final `sacct` accounting retrieval failed because the lab accounting database refused its connection. No missing accounting totals were invented. The error, completed 7B live-job status, per-job terminal logs and final empty queue are retained.
- At **17:45:26 Dubai**, remote source/amendment verification passed again and the account queue was empty. The task-owned SSH master was then closed. No user terminal or unrelated connection was closed. No HPC jobs remain from this stage.

Earlier-stage source/config/data are not edited. Final local re-audit, independently recomputed numerical report and delivery checks are recorded in their hash-bound outputs and the following completion section.

## Final result and report gates

- All three main local CPU audits match the HPC audits apart from timestamps. They passed **1,018,993 checks** and reloaded **2,338 tensor pairs** across 21 main archives; every pair passed numeric and byte comparisons. Native/custom parity and two-step continuation checks remain precisely scoped as described in the report.
- The independent raw calculator agreed on **81,309 quantities** against hash-bound corrected summaries. Its final combined report is `notes/stage3_main_recomputed.json`. Original summaries and raw evidence are retained. All model/shape rows and all eight fault strata, including unfavorable early-layer results, are reported.
- Final main totals are 2,358 sessions, 7,020 attempted windows, 6,966 commits and 54 specified refusals. These totals include native/bare clean reference runs; repeated policies/timings are not independent faults. Only the Smol main study had natural token divergence: two configured cases, with four partial-to-full executions from one configured case. The selected archive replay remains excluded.
- Final manuscript claims and table/provenance bindings passed a separate internal read-only review. The 12-page PDF was rendered and visually checked page by page. Layout iteration removed orphan overflow pages and fixed literal mathematical `p*` rendering without changing manuscript numbers or any experiment. Earlier PDFs are untouched. The shared PDF formatter's delimiter bug was fixed; this is not a measurement-source change.
- The delivery packager verifies the source freeze, all included raw/snapshot hashes, selected-case amendment binding, PDF/source/builder/layout bindings and all-page visual-review gate. It then checks every archived member's SHA256 and ZIP CRC. `package_summary.json` and the post-packaging extracted-artifact `delivery_smoke.json` are external receipts rather than self-referential archive members.

Completion is the expanded bounded implementation, experiment, audits, report and reproducible package. It is not a completed thesis or A/A* readiness claim. No paper submission, GitHub push, production-cluster change or physical fault injection was performed.
