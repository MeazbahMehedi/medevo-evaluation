# Readout experiment verification

Development verification on 8 November 2026 passed **115 tests** in **8.394 seconds**, including thirteen new readout checks. The full log is [test_results.txt](test_results.txt).

The new tests cover a single text-only request, fixed 32-token budget, exact twelve-case workload, unchanged original evidence, unchanged lexical verification, reference-free model requests, truncation, partial failure, time limits, source/data/role guards, locks, repeat-worker rejection and CLI routing. A supervised child process completed twelve scripted readouts through the real command entry point. Model responses in these integration checks are fixtures; their positive matches are not medical results.

An additional compatibility check used the saved Windows collection. Its twelve saved cases, source controls and request construction passed validation. The saved collection does not contain the prepared data bundle; bundle reads and its metadata digest were stubbed for this compatibility check only. It did not independently repeat dataset validation or run model inference. The actual PC command performs the data checks against its existing prepared bundle.

The development environment has no cached copy of the full pretrained checkpoint. Real text-only checkpoint inference remains the next target-PC checkpoint. The installed SmolVLM processor implementation accepts text-only input, but the new tests do not establish actual checkpoint answer quality or latency.

Primary-model loading, existing reasoning/readout/continuation code, auditor requests, data preparation, collection, scoring, configuration and dependencies remain unchanged. The update adds the separate experiment module and CLI route, tests and documentation. It does not replace the original training readout method or admit examples for training.
