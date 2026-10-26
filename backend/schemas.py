"""
schemas.py

Pydantic models for FastAPI request/response schemas.
Mirrors the data shapes from gate.py DecisionRecord and interact.py chain results.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
#  Enums
# ---------------------------------------------------------------------------

class GateDecisionEnum(str, Enum):
    NO_ACTION = "NO_ACTION"
    AUTONOMOUS_EXECUTION = "AUTONOMOUS_EXECUTION"
    HUMAN_APPROVAL = "HUMAN_APPROVAL"


class ModelTypeEnum(str, Enum):
    XGBOOST = "xgboost"
    PROPHET = "prophet"
    LSTM = "lstm"


# ---------------------------------------------------------------------------
#  Decision Gate
# ---------------------------------------------------------------------------

class DecisionRecordResponse(BaseModel):
    product_id: str
    region: str
    model_type: str
    timestamp: str
    forecast: float
    confidence: float
    confidence_threshold: float
    reorder_required: bool
    safety_stock: float
    reorder_point: float
    current_inventory: float
    suggested_order_quantity: float
    decision: GateDecisionEnum
    reasoning: str
    confidence_metadata: dict = Field(default_factory=dict)


class ChainResult(BaseModel):
    decision_id: Optional[int] = None
    tx_hash: Optional[str] = None
    gas_used: Optional[int] = None
    status: str


class theresult(BaseModel):
    po_id: Optional[int] = None
    tx_hash: Optional[str] = None
    gas_used: Optional[int] = None


class PipelineResult(BaseModel):
    decision_record: DecisionRecordResponse
    chain_result: Optional[ChainResult] = None
    po_result: Optional[theresult] = None


# ---------------------------------------------------------------------------
#  Pipeline run request
# ---------------------------------------------------------------------------

class PipelineRunRequest(BaseModel):
    product_id: str = Field(..., examples=["SKU_A"])
    region: str = Field(..., examples=["North"])
    model_type: ModelTypeEnum = ModelTypeEnum.XGBOOST
    inventory_scenario: str = Field("below_rop", examples=["below_rop", "above_rop", "random"])
    current_inventory: Optional[float] = None
    log_to_chain: bool = True


# ---------------------------------------------------------------------------
#  Ledger
# ---------------------------------------------------------------------------

class LedgerDecision(BaseModel):
    id: int
    sku: str
    region: str
    order_quantity: float
    model_type: str
    forecast_value: float
    confidence_score: float
    confidence_threshold: float
    shap_hash: str
    status: int  # 0=NO_ACTION, 1=AUTO, 2=HUMAN
    timestamp: int


class LedgerPO(BaseModel):
    id: int
    decision_id: int
    sku: str
    region: str
    quantity: float
    timestamp: int
    status: int


class LedgerSummary(BaseModel):
    total_decisions: int
    total_pos: int
    decisions: list[LedgerDecision]
    purchase_orders: list[LedgerPO]


# ---------------------------------------------------------------------------
#  Approvals queue
# ---------------------------------------------------------------------------

class ApprovalAction(BaseModel):
    action: str = Field(..., examples=["approve", "reject"])
    notes: Optional[str] = None


class PendingApproval(BaseModel):
    decision_id: int
    product_id: str
    region: str
    model_type: str
    forecast: float
    confidence: float
    suggested_order_quantity: float
    reasoning: str
    timestamp: str


# ---------------------------------------------------------------------------
#  Scan / accuracy
# ---------------------------------------------------------------------------

class ScanRequest(BaseModel):
    scenario: str = "below_rop"
    model_types: list[ModelTypeEnum] = [ModelTypeEnum.XGBOOST, ModelTypeEnum.PROPHET, ModelTypeEnum.LSTM]
    log_to_chain: bool = True


class ScanResult(BaseModel):
    product_id: str
    region: str
    model_type: str
    confidence: float
    decision: str
    chain_decision_id: Optional[int] = None
    chain_tx_hash: Optional[str] = None


class ScanSummary(BaseModel):
    total: int
    no_action: int
    autonomous: int
    human_approval: int
    results: list[ScanResult]
