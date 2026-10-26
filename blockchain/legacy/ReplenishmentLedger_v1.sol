// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title ReplenishmentLedger
 * @notice On-chain audit trail for the Confidence-Gated Supply Chain
 *         Replenishment pipeline.
 *
 * Every decision made by the ML → Decision Gate pipeline is recorded here
 * regardless of outcome.  When the confidence gate passes (decision == 1
 * AUTONOMOUS_EXECUTION), a separate function mints a Purchase Order.
 * When confidence is insufficient (decision == 2 HUMAN_APPROVAL), only
 * the ledger entry is written and an event is emitted for off-chain
 * monitoring.
 *
 * Fixed-point encoding: all float values from the Python pipeline are
 * encoded as uint256 with 18 decimal places (1e18 = 1.0) to preserve
 * precision without floating-point on-chain.
 */
contract ReplenishmentLedger {

    // -----------------------------------------------------------------------
    //  Enums
    // -----------------------------------------------------------------------

    enum ExecutionStatus {
        NO_ACTION,              // 0 – stock sufficient, no reorder needed
        AUTONOMOUS_EXECUTION,   // 1 – reorder triggered, confidence ≥ threshold
        HUMAN_APPROVAL          // 2 – reorder flagged, confidence < threshold
    }

    // -----------------------------------------------------------------------
    //  Structs
    // -----------------------------------------------------------------------

    struct DecisionRecord {
        uint256 id;                   // auto-incrementing decision ID
        string  sku;                  // product identifier (e.g. "SKU_A")
        string  region;               // e.g. "North"
        uint256 orderQuantity;        // suggested replenishment qty (fixed-point 1e18)
        string  modelType;            // "xgboost" | "prophet" | "lstm"
        uint256 forecastValue;        // point forecast (fixed-point 1e18)
        uint256 confidenceScore;      // model confidence (fixed-point 1e18, range 0–1e18)
        uint256 confidenceThreshold;  // gate threshold (fixed-point 1e18)
        bytes32 shapHash;             // SHA-256 of the SHAP explanation payload
        ExecutionStatus status;
        uint256 timestamp;            // block.timestamp at time of logging
    }

    struct PurchaseOrder {
        uint256 id;
        uint256 decisionId;           // links back to the originating DecisionRecord
        string  sku;
        string  region;
        uint256 quantity;             // fixed-point 1e18
        uint256 timestamp;
        ExecutionStatus status;
    }

    // -----------------------------------------------------------------------
    //  State
    // -----------------------------------------------------------------------

    uint256 private _nextDecisionId = 1;
    uint256 private _nextPOId       = 1;

    mapping(uint256 => DecisionRecord) public decisions;
    mapping(uint256 => PurchaseOrder)  public purchaseOrders;

    // All decision IDs in insertion order (for off-chain enumeration)
    uint256[] public decisionIds;

    // All PO IDs in insertion order
    uint256[] public poIds;

    // -----------------------------------------------------------------------
    //  Events
    // -----------------------------------------------------------------------

    event DecisionLogged(
        uint256 indexed decisionId,
        string  sku,
        string  region,
        string  modelType,
        uint256 confidenceScore,
        ExecutionStatus status,
        bytes32 shapHash,
        uint256 timestamp
    );

    event OrderExecuted(
        uint256 indexed poId,
        uint256 indexed decisionId,
        string  sku,
        string  region,
        uint256 quantity,
        uint256 timestamp
    );

    event HumanApprovalRequired(
        uint256 indexed decisionId,
        string  sku,
        string  region,
        uint256 confidenceScore,
        uint256 confidenceThreshold,
        uint256 timestamp
    );

    // -----------------------------------------------------------------------
    //  Core functions
    // -----------------------------------------------------------------------

    /**
     * @notice Log a decision from the ML -> Decision Gate pipeline.
     *         Called for EVERY decision regardless of outcome (audit trail).
     * @param _sku               Product identifier
     * @param _region            Geographic region
     * @param _orderQuantity     Suggested replenishment qty (fixed-point 1e18)
     * @param _modelType         Forecasting model that was used
     * @param _forecastValue     Point forecast value (fixed-point 1e18)
     * @param _confidenceScore   Model confidence score (fixed-point 1e18)
     * @param _confidenceThreshold  Gate threshold (fixed-point 1e18)
     * @param _shapHash          SHA-256 hash of the SHAP explanation payload
     * @param _status            Execution status (NO_ACTION, AUTONOMOUS, or HUMAN)
     * @return decisionId       The ID of the newly created DecisionRecord
     */
    function logDecision(
        string  calldata _sku,
        string  calldata _region,
        uint256 _orderQuantity,
        string  calldata _modelType,
        uint256 _forecastValue,
        uint256 _confidenceScore,
        uint256 _confidenceThreshold,
        bytes32 _shapHash,
        ExecutionStatus _status
    )
        external
        returns (uint256 decisionId)
    {
        decisionId = _nextDecisionId++;

        decisions[decisionId] = DecisionRecord({
            id:                  decisionId,
            sku:                 _sku,
            region:              _region,
            orderQuantity:       _orderQuantity,
            modelType:           _modelType,
            forecastValue:       _forecastValue,
            confidenceScore:     _confidenceScore,
            confidenceThreshold: _confidenceThreshold,
            shapHash:            _shapHash,
            status:              _status,
            timestamp:           block.timestamp
        });

        decisionIds.push(decisionId);

        emit DecisionLogged(
            decisionId,
            _sku,
            _region,
            _modelType,
            _confidenceScore,
            _status,
            _shapHash,
            block.timestamp
        );

        // Emit secondary event for downstream systems to act on
        if (_status == ExecutionStatus.HUMAN_APPROVAL) {
            emit HumanApprovalRequired(
                decisionId,
                _sku,
                _region,
                _confidenceScore,
                _confidenceThreshold,
                block.timestamp
            );
        }
    }

    /**
     * @notice Auto-execute a Purchase Order on-chain.  Only callable when
     *         the decision status is AUTONOMOUS_EXECUTION.
     * @param decisionId  ID of the DecisionRecord that triggered this PO
     */
    function autoExecutePO(uint256 decisionId)
        external
        returns (uint256 poId)
    {
        DecisionRecord storage rec = decisions[decisionId];
        require(rec.id != 0, "Decision does not exist");
        require(
            rec.status == ExecutionStatus.AUTONOMOUS_EXECUTION,
            "Decision was not marked AUTONOMOUS_EXECUTION"
        );
        require(rec.orderQuantity > 0, "Order quantity is zero");

        poId = _nextPOId++;

        purchaseOrders[poId] = PurchaseOrder({
            id:          poId,
            decisionId:  decisionId,
            sku:         rec.sku,
            region:      rec.region,
            quantity:    rec.orderQuantity,
            timestamp:   block.timestamp,
            status:      ExecutionStatus.AUTONOMOUS_EXECUTION
        });

        poIds.push(poId);

        emit OrderExecuted(
            poId,
            decisionId,
            rec.sku,
            rec.region,
            rec.orderQuantity,
            block.timestamp
        );
    }

    // -----------------------------------------------------------------------
    //  Read helpers
    // -----------------------------------------------------------------------

    function getDecisionCount() external view returns (uint256) {
        return decisionIds.length;
    }

    function getPOCount() external view returns (uint256) {
        return poIds.length;
    }

    function getDecision(uint256 decisionId)
        external
        view
        returns (DecisionRecord memory)
    {
        require(decisions[decisionId].id != 0, "Decision does not exist");
        return decisions[decisionId];
    }

    function getPO(uint256 poId)
        external
        view
        returns (PurchaseOrder memory)
    {
        require(purchaseOrders[poId].id != 0, "Purchase order does not exist");
        return purchaseOrders[poId];
    }

    /**
     * @notice Return all decision IDs stored on-chain.
     */
    function getAllDecisionIds() external view returns (uint256[] memory) {
        return decisionIds;
    }

    /**
     * @notice Return all PO IDs stored on-chain.
     */
    function getAllPOIds() external view returns (uint256[] memory) {
        return poIds;
    }
}
