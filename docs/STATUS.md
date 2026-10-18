# Research status

Both historical studies are complete. No model or specification is currently being selected from these outcomes.

| Study | Classification | Targets | Result |
|---|---|---:|---|
| GARCH(1,1) versus EWMA | Fixed retrospective comparison | 1,508 | Lower observed GARCH loss; registered 95% interval includes zero. |
| Additional GARCH orders | Exploratory extension after the baseline was known | 1,508 per candidate | Neither added order improved observed QLIKE or met the Holm-adjusted improvement rule. |

## Preserved identities

- Original protocol SHA-256: `99a7bbc83f8fed2b26182e4ebb78da30e17351025592a21c3cf8c1ca7df06116`.
- Extension protocol SHA-256: `6c908954ca7cb88033e560cf87f5e14600bba0b376293daeaf43f8623a202f96`.
- Accepted snapshot: 5,283 complete XNYS adjusted-close observations, 2005–2025.
- Both studies completed without model fallbacks or numerical variance floors.
- The extension reused the exact original GARCH(1,1) forecasts without historical baseline refitting.
- Both saved-result audits pass, including code/input/output identities, training windows, losses and statistical calculations.

## Validation and public boundary

The complete package passed 59 offline tests, Ruff and strict mypy before the extension execution. The supplied GitHub Actions workflow performs offline checks and synthetic numerical demonstrations. CI execution status is reported by GitHub Actions, separately from the completed historical research receipts.

The repository includes source, protocols, aggregate summaries, figures and audit/execution receipts. Vendor data, detailed return ledgers, local environments, caches and local operational records are excluded. Exact historical reconstruction requires the matching accepted snapshot.

Read the [original report](results.md), [extension report](higher-order-extension.md), [design](research-design.md), and [contribution disclosure](../CONTRIBUTIONS.md). Neither study evaluates portfolio performance. The other research programme and its prospective records were not used.
