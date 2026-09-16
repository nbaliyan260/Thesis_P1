# Literature coverage and source-integrity audit

Audit date: 2026-09-16. This is a bounded audit of the existing review ledgers, syntheses, download manifest and file inventory. It does not repeat the 100 reviews, authenticate every paper against a publisher, or claim that every page was read.

## Exact coverage

| Ledger | Assigned IDs | Records | Focused deeper reviews | Other selected-section screens |
| --- | --- | ---: | ---: | ---: |
| [Systems](../literature/systems_001_034.json) | 001-034 | 34 | 7 | 27 |
| [Safety](../literature/safety_035_067.json) | 035-067 | 33 | 3 | 30 |
| [Reliability](../literature/reliability_068_100.json) | 068-100 | 33 | 8 | 25 |
| Total | 001-100 | 100 | 18 | 82 |

The 18 deeper-review IDs are **015, 017, 023, 024, 027, 028, 030, 052, 058, 065, 068, 069, 073, 075, 078, 080, 082, 085**. “Deeper” means the ledger explicitly marks focused method/evaluation/main-text reading; it does not mean a uniform cover-to-cover standard. For example, 082 is a focused methods/evaluation review, whereas several systems records specify full main-text reading but exclude appendices. None of these labels should be converted into “18 papers fully verified.”

Automated local checks found:

- Exactly 100 distinct ledger IDs, covering every integer from 001 through 100; no duplicate or missing ID.
- Exactly one corresponding nonempty extracted-text file for each record and all 100 expected PDF files present.
- All 100 manifest rows say downloaded. Current PDF sizes match the manifest byte counts, and all files begin with the PDF signature.
- 1,767 sequential physical-page extraction markers in total; no marker gaps or entirely empty extracted pages.
- All recorded page references lie within their corresponding extraction's marker range.
- Every record has a title, contribution, limitation/evidence-limit field, connection, publication-status field and review-depth/page references.
- The 67 records that include source_url agree with the manifest URL. Records 035-067 omit that field; their source URLs remain recoverable by ID from the manifest.
- Ninety-nine titles match the manifest exactly. Record 056 differs only in capitalization: ledger “DeepServe,” manifest “DEEPSERVE.” The supplied proceedings cover itself uses “DeepServe,” so this is not an unresolved paper-identity mismatch.

These are completeness and consistency checks, not proof of PDF authenticity or flawless extraction. File magic and matching sizes do not establish cryptographic identity to a publisher copy. Nonempty page text can still omit a figure, equation, table, footnote or column reading order. The extraction contains about 7.38 MB of UTF-8 text; that size says nothing about how much was read.

Manifest source hosts are arxiv.org for 77 PDFs, ACL Anthology for 9, USENIX for 12, NDSS for 1 and HotCarbon for 1. These are source-location counts, **not** counts of peer-reviewed versus unreviewed papers: some arXiv copies report accepted venues, and those claims have different verification depths.

## Metadata and publication-status flags

### Highest-priority identity/history verification

| ID | Flag in the existing evidence | Required treatment |
| --- | --- | --- |
| 036 | Supplied first page says arXiv:2607.16215v1 and 28 May 2026. The July-coded identifier and displayed May date are inconsistent. | Verify the canonical arXiv title, authors, version history and source PDF before using its measurements as firm evidence or finalizing its bibliography. Do not invent an explanation. |
| 039 | Supplied first page says arXiv:2607.19353v1 and 20 May 2026. Same identifier/date inconsistency. | Same verification requirement. Label provisional until resolved. |

This audit reconfirmed those strings in the existing extracted first pages, but did not resolve them by a new publisher lookup. The flags do not by themselves prove that the papers are inauthentic; they make unqualified citation unsafe.

**Later root recheck, 16 September 2026:** the canonical pages for [036 / RAIL Guard](https://arxiv.org/abs/2607.16215) and [039 / Confidential GPU Inference](https://arxiv.org/abs/2607.19353) returned matching titles and the same May submission dates in their version histories, alongside July-coded identifiers. Thus the discrepancy was not resolved by looking up the canonical records; no cause is invented. These two records remain excluded from the central novelty argument. This recheck verifies the observed title/record association, not authenticity of every supplied PDF byte or independent correctness of the reported results.

### Venue, version and provenance checks before a final bibliography

| IDs | Issue | Safe current statement |
| --- | --- | --- |
| 003, 006 | ICML 2026 declaration in an arXiv-hosted supplied PDF, not independently confirmed in proceedings | Author/PDF-declared venue; independently unverified. |
| 021 | ICS 2026 declaration in supplied PDF | Same distinction. |
| 031, 041, 086 | ICLR 2026/2025 headers in arXiv-hosted copies | Verify the corresponding official acceptance/proceedings record if venue status matters. |
| 032 | ICML 2026 Hypothesis Testing workshop header | A workshop claim, not ICML main-conference acceptance. |
| 050, 057, 076 | FORGE 2026, ASPLOS 2025 and SC 2025/author-accepted claims with DOI information inside supplied copies | DOI-bearing author metadata is useful but a final bibliography should resolve the DOI/publisher record. |
| 068, 069, 082 | MLSys 2025, NeurIPS 2025 and ICML 2025/PMLR declarations in supplied arXiv copies | Check official proceedings before treating venue verification as complete. |
| 004, 093, 099 | Placeholder conference/ACM metadata; 099 additionally indicates a MIND 2025 workshop author version | Placeholder metadata does not establish final acceptance or final bibliographic details. |
| 095 | Copyright notice does not establish a specific venue | Keep venue unresolved. |
| 053 | Ledger did not establish a venue from selected text, although the manifest points to a HotCarbon 2025 domain/path | Do not silently promote a URL path to verified acceptance; inspect the official program record when needed. |
| 042 | Submitted to IEEE TDSC | Submission is not acceptance. |
| 019, 065, 086 | Manifest/filename year differs from the initial arXiv year or later supplied version year | Preserve separate initial-submission, accessed-version and publication years. These differences can be legitimate; they are not automatically contradictions. |

Other records labeled “preprint,” “venue not verified,” or “later venue not verified” remain in that state; this audit does not upgrade them. Particularly relevant examples include 075 (KernelBench), 078 (training SDC), and 098 (AIOpsLab). An exhaustive publisher recheck was outside this bounded task.

Official-source PDF URLs strengthen the provenance of 002, 013, 035, 038, 047, 052, 054-056, 058-063, 074, 077, 084-085, 089-090 and 094. Even these were not all independently re-fetched and authenticated in this audit.

### Cross-ledger and extraction limitations

- **Schema gap:** the safety ledger's 33 records lack per-record source_url and filename fields. The manifest resolves the omission, but that JSON should not be distributed alone without the manifest. No source record was edited during this audit.
- **Schema differences:** systems uses evidence_limits/possible_connection/page_references strings; the other ledgers use limitations/connection/page_refs arrays. A combined table must normalize these fields rather than drop one group's limitations.
- **Page numbering:** safety and reliability explicitly use physical PDF pages, including proceedings covers. Systems uses extraction markers. Printed article page numbers may differ; retain that convention in quotations and tables.
- **Known extraction weakness:** 014 explicitly reports an uninterpretable extracted figure. It supports no visually verified figure claim. More generally, no corpus-wide rendering/OCR/figure audit was performed.
- **Review granularity:** all three ledgers explicitly limit their reading claims. Listing page numbers does not assert that every line on those pages was read.
- **Measured results:** paper results are author-reported unless an independent replication is explicitly documented elsewhere. The later RECUT pilot is not a replication of those 100 papers.
- **Scope of source identity:** title-to-manifest agreement confirms bookkeeping against the supplied collection, not agreement with an independently verified publisher record for every PDF.

The external prior-art syntheses also contain status differences that should be reconciled before final references. HybridServe is called a preprint in the reliability synthesis, while the systems/safety reviews record an ICCD 2025 journal reference on the author arXiv page; use the more explicit evidence and keep author-record versus publisher verification distinct. GhostServe's MLSys 2026 status is attributed to its primary author record; the shared-KV paper's SECRYPT claim is similarly not an independently checked proceedings entry. KVBoost (2608.21362) has a reported identifier/date inconsistency in the safety synthesis. Belayer (2608.14635) is described there as having a July submission/August revision; its canonical version chronology should be checked rather than copied uncritically. These are additional sources, not extra papers in the 100-record coverage count.

## Requested website mapping

The review package covers **30 requested web resources plus SPAR and MATS as two additional resources**. This is not 32 distinct organizations: the Anthropic fellowship and Alignment Science index are separate requested resources on overlapping institutional infrastructure.

Sources:

- **P:** [program_agendas_01_11.json](../literature/program_agendas_01_11.json), 11 requested resources.
- **S:** research-site audit section of [systems_synthesis.md](../literature/systems_synthesis.md), 10 requested resources.
- **R:** [website_audit_reliability.md](../literature/website_audit_reliability.md), 9 requested resources plus 2 extras.

All access dates are recorded as 2026-09-16. The following condenses the existing access evidence; this audit did not revisit every URL.

| Resource | Ledger | Recorded access and important limit |
| --- | --- | --- |
| 01 Apart Research fellowship/research | P | Pages accessible; concrete project identified. Associated NeurIPS paper title/abstract only, not a full review. |
| 02 Pivotal Research Q3 program | P | Bare-domain open failed; www Q3 page mostly navigation. Specific Q3 roster unverified; official library project pages read instead. |
| 03 ERA Fellowship | P | Program and project listing read. One Oxford hardware-assurance page read; another linked evaluation paper failed to fetch. |
| 04 LASR Labs | P | Bare-domain failed; www homepage/past projects read. One arXiv abstract read; another linked workshop paper failed. |
| 05 AI Safety Camp | P | Redirected incubator page accessible; concrete project catalog not established. |
| 06 Principles of Intelligence / PIBBSS | P | Fellowship and PIRAMID pages accessible; conceptual project agenda read. |
| 07 Anthropic Fellows Program | P | Official program and circuit-tracing project pages read. |
| 08 OpenAI Safety Fellowship | P | Official announcement read; no concrete project roster or hardware-specific agenda verified. |
| 09 Iliad Fellowship | P | Bare-domain failed; www page read; no specific hardware/KV project catalog verified. |
| 10 GovAI | P | Bare-domain failed; www opportunities and technical-governance overview read; full linked paper not read. |
| 11 IAPS Fellowship | P | Bare-domain failed; www fellowship, research listing and specific monitorability page read. |
| 12 FAR.AI research | S | Redirect/canonical research and publications pages read; specific activation-probe project read. |
| 13 Berkeley CHAI research | S | Research index read; largely 2023-and-earlier material, not a verified complete 2026 agenda. Selected causal-model source accessed. |
| 14 Center for AI Safety | S | Initial research URL failed; canonical research page and MASK project page succeeded. |
| 15 ARC | S | Initial /research URL failed; homepage and specific mechanistic-estimation article succeeded. |
| 16 Redwood Research | S | Research page and selected high-stakes reliability source accessed. |
| 17 METR | S | Research page and automated-kernel-engineering article read. |
| 18 Apollo Research | S | Science page and specific monitorability article read. |
| 19 UK AISI | S | Page displayed a JavaScript notice but substantial text was returned; selected monitoring-deployment page read. |
| 20 Anthropic Alignment Science | S | Index and CHIVE counterfactual-evaluation article read. |
| 21 Transformer Circuits | S | Index read; one attention-tracing link failed; attribution-graphs methods page succeeded. |
| 22 Resolution | R | Homepage, launch statement and research portfolio read; no numerical KV-recovery project located. |
| 23 Cooperative AI | R | Bare-domain failed; www research guidance read. A specific call was found, but the recorded deadline had passed; not current opportunity verification. |
| 24 Epoch AI | R | Homepage and long-context latency report read. |
| 25 AI Safety training directory | R | Directory text read; a discovery resource, not a research agenda or proof opportunities remain open. |
| 26 Alignment Forum | R | Redirected site returned posts with limited dynamic text; a specific primary Redwood proposal was opened. |
| 27 AI Safety Ideas | R | Homepage, list and one attention-only-model idea read; listed idea is not evidence of novelty or endorsement. |
| 28 80,000 Hours job board | R | Page/filter shell only; individual job listings were not returned. No role, salary, deadline or hiring claim verified. |
| 29 ARENA curriculum | R | Initial bare-domain failed; www curriculum, chapter and self-study pages succeeded. |
| 30 Neuronpedia | R | Bare-domain failed; www homepage and a specific model page read; no remote inference was run. |
| Extra: SPAR | R | Homepage read; direct fall-project index exceeded retrieval size limit. Search-indexed descriptions and specific project pages were available; no complete project-catalog audit. |
| Extra: MATS | R | Redirected homepage, research and mentor pages read; no matching Mahmoud/KV-repair mentor was verified. |

Thus all 30 requested resources have an access record, but not all requested pages or project catalogs were successfully read. The most material incomplete cases are Pivotal's Q3 roster, AI Safety Camp's project catalog, the job board's individual listings, the SPAR full index, two failed linked papers at ERA/LASR, and the failed Transformer Circuits subpage. Several initial failures were resolved by canonical/www alternatives and should not be described as wholly inaccessible sites.

Site fit statements are reviewer inferences, not endorsements, admissions predictions or confirmations of open applications. No accounts, applications or outreach messages were created as part of these site checks.

## Suggested methods wording for the final paper

“We conducted a structured screen of the 100 supplied papers using their extracted full text, with selected-section reading for all papers and focused method/evaluation review for 18 closer neighbors. The accompanying ledgers record physical-page references, reading depth, limitations and publication-status uncertainty. We did not conduct a uniform cover-to-cover review, audit every proof/figure, or reproduce the papers' experiments. We additionally inspected 30 requested research/program resources and two related resources, recording incomplete access and distinguishing organizational agendas from technical evidence. This process informed a provisional novelty assessment, not a proof of priority or an exhaustive systematic review.”

Before final bibliography submission, resolve the flagged identifier/date inconsistencies and verify any venue claim central to the argument against its official record. The 100-file completeness result should never be substituted for that source-integrity work.
