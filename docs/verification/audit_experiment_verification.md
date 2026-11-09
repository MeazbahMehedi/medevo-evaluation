# Auditor experiment verification

Development verification passed 129 tests in 11.198 seconds, including fourteen new auditor-experiment checks. The current complete log is [test_results.txt](test_results.txt). These are software checks, not live medical results.

The new checks cover the single-field request difference, twelve-call bound, unchanged images/schema/sampling, preserved original records, reference-free requests, invalid repair indexing, conservative hint rejection, unsupported non-repair auxiliary text, malformed replies, transport failure, locks, source/configuration integrity, repeat-worker protection and CLI routing.

A supervised child process used the real local HTTP request implementation against a scripted loopback server. It sent twelve image-capable requests and recorded twelve fixture responses without loading the primary model. Fixture decisions are explicitly not clinical judgements. Positive schema/hint checks never produced continuations or training examples. A controlled API key was absent from saved JSON evidence.

An offline check of the actual uploaded Windows archive verified that all twelve trial requests change only the system instruction. It reproduced the current baseline assessment: eight skips, two unresolved decisions, one rejected hint and one invalid audit. Original records were unchanged. No live model or auditor calls were made during that archive check.

The real local Qwen trial remains the next target-PC checkpoint. The development results do not establish improved auditor coherence, clinical validity, immutable Ollama weight identity or training feasibility. Existing primary-model, auditor, collection, readout, scoring and configuration files remain unchanged; the update adds a separate experiment module and command route.
