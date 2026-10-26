# Security / Threat-Model Review — `ReplenishmentLedger.sol`

Scope: `blockchain/contracts/ReplenishmentLedger.sol` (v1 as found, kept verbatim at
`blockchain/legacy/ReplenishmentLedger_v1.sol` + `.json`) and the off-chain bridge
`blockchain/scripts/interact.py`. Tooling: solc 0.8.19 (legacy codegen, optimizer on,
runs = 200 — the settings that reproduce the committed v1 bytecode exactly, see
`blockchain/scripts/compile.py`), Ganache v7.9.2, web3.py 8.0.0.
Evidence: `tests/test_contract.py` (23 tests, all passing — see `test_report.md`) and
`publication_package/gas_measurements.csv`.

## Summary of findings

| # | Finding (v1) | Severity | Status in v2 | Test |
|---|---|---|---|---|
| F1 | `logDecision()` and `autoExecutePO()` callable by **any** address | High | Fixed: `onlyAuthorizedCaller` (owner-managed allow-list; deployer authorised in constructor) | `test_log_decision_rejects_unauthorized_caller`, `test_auto_execute_po_rejects_unauthorized_caller`, `test_owner_can_grant_and_revoke_callers`, `test_non_owner_cannot_administer` |
| F2 | `autoExecutePO()` can be called **repeatedly for the same decision**, minting duplicate POs | High | Fixed: `poForDecision[decisionId]` must be 0; set before the event | `test_auto_execute_po_rejects_double_execution` |
| F3 | A record can be logged as `AUTONOMOUS_EXECUTION` with a confidence **below its own threshold**, and the threshold itself is caller-supplied (could be 0) | Medium | Mitigated: for AUTONOMOUS status, `confidence ≥ threshold` and `threshold ≥ minAutonomousThreshold` (owner-set floor, default 0.85e18); `confidence ≤ 1e18` always | `test_autonomous_requires_confidence_at_or_above_threshold`, `test_autonomous_threshold_cannot_undercut_onchain_floor`, `test_confidence_above_one_rejected`, `test_threshold_floor_admin` |
| F4 | Reentrancy in `autoExecutePO()` | Informational | Not exploitable in v1 or v2: the function makes **no external calls** (no ETH/token transfer, no callbacks), so checks–effects–interactions holds trivially. A `nonReentrant` lock was added as defence-in-depth for future versions that might call a supplier/payment contract | `test_reentrancy_lock_is_released_after_each_call` (the guard cannot be *triggered* from a test because there is no external call to re-enter through — stated plainly) |
| F5 | The on-chain `shapHash` field does **not** contain a hash of the SHAP explanation | Medium (integrity / claim accuracy) | **Not changed** (off-chain behaviour, out of scope for an additive change) — documented here and in RESULTS_SUMMARY.md | n/a (code inspection) |
| F6 | Confidence-score oracle: the contract trusts whatever confidence the backend sends | High (architectural) | **Not fixed — inherent limitation of the design**; see section below | `test_v1_contract_had_no_access_control_or_double_spend_protection` shows the v1 exposure |

Before/after evidence: `test_v1_contract_had_no_access_control_or_double_spend_protection`
deploys the original v1 bytecode and shows a never-authorised account logging an
AUTONOMOUS record with confidence 0.10 and threshold 0.0 and then executing **two** POs
for it. Every step of that sequence reverts on v2.

Non-regression: `test_interact_bridge_works_unchanged_against_v2` drives the unchanged
`interact.log_decision()` / `interact.execute_po()` against v2. The ABI of
`logDecision`/`autoExecutePO`, both structs, and all three original events are unchanged,
and `deploy.py` still calls the zero-argument constructor from `accounts[0]` (which
becomes owner and authorised caller), so `pipeline.py`, the FastAPI routers, the
dashboard and `ledger_integrity_check.py` work without modification.

## (a) Access control

**v1:** no modifier on either state-changing function. Anyone with RPC access to the
chain could (i) append arbitrary decision records — polluting the audit trail that
the project presents as its trust anchor — and (ii) trigger PO execution for any
AUTONOMOUS record, any number of times.

**v2:** `owner` (set to `msg.sender` at deployment), `authorizedCallers` mapping,
`onlyOwner` admin functions (`setAuthorizedCaller`, `transferOwnership`,
`setMinAutonomousThreshold`), and `onlyAuthorizedCaller` on `logDecision` and
`autoExecutePO`. Admin actions emit `OwnershipTransferred`,
`CallerAuthorizationChanged` and `MinAutonomousThresholdChanged` events, so changes
to who may write are themselves part of the audit trail.

Residual risk: single-key ownership. A production deployment should put `owner`
behind a multisig or timelock; the backend's signing key should live in an HSM/KMS
rather than in Ganache's unlocked `accounts[0]` (which `interact.py` uses today).

## (b) Reentrancy

`autoExecutePO()` performs `require` checks, then writes storage
(`_nextPOId`, `poForDecision`, `purchaseOrders`, `poIds`), then emits an event.
There is no `call`, `transfer`, `send`, `delegatecall` or external contract
invocation anywhere in the contract, so there is no re-entry point. In v2 the
state write that closes the double-execution hole (`poForDecision`) happens before
anything else, and a `nonReentrant` lock wraps the function anyway. Measured gas
overhead of all v2 checks together: +3,939 gas per `logDecision` (+1.3 %) and
+26,842 gas per `autoExecutePO` (+11.7 %, dominated by the new
`poForDecision` SSTORE); see `gas_cost_analysis.md`.

## (c) The confidence-score oracle risk

The confidence score that decides whether a PO is executed autonomously is computed
**off-chain** (Python, `confidence/confidence_estimator.py`) and passed to the
contract as a plain `uint256` argument. The contract cannot tell a confidence of
0.91 that came from `exp(-alpha * (q90 - q10)/|yhat|)` from a value typed in by an
attacker.

**Attack:** an adversary who compromises the backend host, its signing key, or
any component between the model and `interact.log_decision()` (the YAML config,
the saved model files, the Python process) can submit
`logDecision(status = AUTONOMOUS_EXECUTION, confidenceScore = 0.99e18,
confidenceThreshold = 0.85e18, orderQuantity = <anything>)` and then
`autoExecutePO(id)`. Every on-chain check passes. The PO is executed and the ledger
records — immutably and convincingly — a high-confidence, policy-compliant decision
that never happened. Immutability makes the forgery *harder to repair*, not
easier to detect.

**What v2 does mitigate**
- Only allow-listed addresses can write (so an attacker needs the backend key, not
  just RPC access).
- At most one PO per logged decision (no replay amplification of a single forged
  record).
- The autonomy policy cannot be weakened from the backend: an AUTONOMOUS record must
  satisfy `confidence ≥ threshold ≥ minAutonomousThreshold`, and only the owner can
  lower the floor (an on-chain, evented action). Lowering `CONFIDENCE_THRESHOLD` in
  `threshold_config.yaml` below 0.85 will now make AUTONOMOUS logs revert until the
  owner changes the floor — a deliberate behaviour change, documented here.

**What this project does NOT implement (state explicitly in the paper)**
- **No on-chain or cryptographic verification of the ML computation.** The contract
  does not, and cannot, check that the confidence value was produced by the stated
  model on the stated inputs. There is no zkML proof, no TEE attestation, no
  committee/multi-oracle agreement, and no commitment to model weights or input
  features.
- **No binding of the record to its inputs.** Neither the feature vector, the
  model version/hash, nor the prediction interval bounds are stored or hashed
  on-chain, so a decision cannot be independently recomputed from chain data alone.
- **`shapHash` is not a SHAP hash (finding F5).** `interact._shap_hash()` hashes
  `record.confidence_metadata`, which for XGBoost is the constant
  `{"quantiles": [0.1, 0.9]}` — every XGBoost decision therefore carries the *same*
  32-byte value. `explainability/shap_explainer.hash_explanation()` exists but is
  not called anywhere on the decision path. The claim in
  `docs/novelty_and_contribution.md` §3.4 that "SHAP explanations … are hashed and
  stored on-chain with every decision record" is not true of the current code.
- **No order-quantity bound.** `orderQuantity` is accepted as given; a per-SKU cap
  or rate limit is not enforced on-chain.
- **Human-approval path is off-chain only.** `POST /approvals/{id}/approve` returns
  a JSON note; there is no on-chain approval transaction or approver signature.

**Mitigations a deployment would need (future work, not implemented)**
1. Hash-commit the full decision context (feature vector, model artefact hash,
   interval bounds, the real SHAP payload via `hash_explanation()`) in
   `logDecision`, so any record can be recomputed and audited off-chain.
2. Signed model outputs from an attested enclave (TEE) or k-of-n independent
   inference oracles, verified in the contract before `autoExecutePO`.
3. zkML proofs of inference for the (small) XGBoost quantile models — feasible
   in principle for tree ensembles, at substantial proving cost.
4. On-chain per-SKU quantity caps / daily spend limits and a timelock between
   `logDecision` and `autoExecutePO` during which a guardian can veto.
5. Multisig owner; HSM-held backend key; a real `approveAndExecute()` function
   requiring an approver signature for HUMAN_APPROVAL records.

## Files changed

- `blockchain/contracts/ReplenishmentLedger.sol` — v2 (hardened). Original kept at
  `blockchain/legacy/ReplenishmentLedger_v1.sol`.
- `blockchain/abi/ReplenishmentLedger.json` — recompiled v2 artefact. Original kept
  at `blockchain/legacy/ReplenishmentLedger_v1.json`.
- `blockchain/scripts/compile.py` — new, pinned compiler settings.
- `tests/test_contract.py`, `tests/conftest.py` — new.

Note: the repository is not a git repository, so these changes are saved on disk but
not committed. Any already-running Ganache still has the v1 contract deployed at the
address in `deployed_address.json`; re-run `python blockchain/scripts/deploy.py` to
deploy v2.
