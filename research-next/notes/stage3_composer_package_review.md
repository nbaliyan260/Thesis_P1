# Read-only composer and package review

Reviewed 2026-09-16 before the combined main report existed. No composer, package, template or frozen-source file was changed, and no manuscript/PDF authoring or renderer was run by this reviewer.

## Confirmed

- Current template has 14 unique placeholders matching all composer blocks, is ASCII, and retains nine explicit page markers. Composition preserves those markers in the manuscript source; only the separate RESULTS.md derivative removes them.
- Actual local JUnit reports 313 tests, 17 skipped, zero failures/errors: 296 passed. All three allocated debug logs contain the 286-pass GPU-suite gate. The reporting tests account for 27 of the local passes.
- The eight fault strata transpose correctly from six model/initial-prefix rows into two eight-row, three-model tables. List-comprehension model indices do not overwrite the later model-name loop.
- Main commit/refusal counts come from measured totals; tensor pairs and audit checks use the local audit once, not the HPC repetition as extra observations.
- The frozen dependency selection includes the stage-2 controller/tests and original engine/runner/tests. The explicit run_optimized_replay.py addition closes the independent-test import chain. The portable PDF layout has built-in font fallbacks and no hidden earlier-project import.
- Selected roots do not include model caches or environments; ordinary known credential-token/private-key patterns are screened. This is not exhaustive secret detection.

## Findings sent to root

1. **Package preflight ordering:** main writes delivery_manifest.json before checking whether the ZIP already exists. A repeated invocation can overwrite the local manifest and then refuse the unchanged archive. Check all output targets first and use exclusive creation.
2. **Archive-hash claim:** all_archived_member_hashes_verified is set true although SHA256 verification loops over records that exclude the manifest. The manifest receives CRC checking only. Verify its archived SHA256 separately or narrow the claim.
3. **Portable guide dependency:** DELIVERY.md references experiments/fetch_pinned_models.py, but the initial package selection omits that helper. Include it or remove that documented route.
4. **Test reproduction command:** DELIVERY.md names test_reporting.py but omits test_report_tables.py. Use the reporting directory or list both files to reproduce the advertised 27 reporting tests.
5. **Debug provenance:** composer reads debug divergence counts after checking only local-audit status. Bind that audit to current debug metadata/raw hashes and require tensor reload before using its counts.
6. **Availability wording:** native/bare do not actually stream/release externally each step; the harness buffers outputs locally and records per-step completion as an availability proxy. Replace “native/bare release each step” with that precise distinction.
7. **Composer preflight:** also check manuscript_provenance.json before writing either manuscript output, to avoid a late exclusive-write failure leaving partially completed outputs.
8. **Hardcoded environment:** verify the stated Python/Torch/GPU and allocation facts against all three main metadata/log sets before final composition. A source freeze alone does not establish the allocated GPU model.

Optional additional gate: bind the final PDF to its source through the renderer's build metadata before packaging. ZIP byte verification establishes archival consistency, not that a particular PDF was rendered from the supplied manuscript.

These findings describe the versions inspected and may subsequently be resolved by root. No final numeric interpretation was reviewed because the combined main evidence was not yet available.
