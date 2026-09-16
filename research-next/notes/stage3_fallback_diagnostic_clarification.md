# Diagnostic amendment terminology clarification

16 September 2026. The frozen amendment uses the phrase "divergence at index 7". The stored controller field `first_divergence=7` is a **one-based predicted-token ordinal**, not a zero-based array index. The first unequal speculative/corrected token is array position 6 (284 versus 298). The seventh replay step still uses the selected cut; the eighth starts at layer zero.

The declared eight-step replay starts, acceptance criteria, raw traces and verifier are correct and unchanged. This is a wording clarification only. The pre-job amendment and four-file diagnostic freeze are preserved byte-for-byte; the final manuscript explicitly explains the ordinal.
