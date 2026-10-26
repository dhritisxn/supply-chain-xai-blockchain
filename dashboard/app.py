"""
app.py

Streamlit dashboard for the Blockchain-Verified, Confidence-Gated
Explainable AI Supply Chain Replenishment system.

Four main views:
  1. Forecast & Explainability - run forecasts with SHAP/prophet decomposition
  2. Confidence Gate Status   - view gate decisions and confidence scores
  3. Audit Ledger Viewer      - browse immutable on-chain records
  4. Human Approval Queue     - review and act on low-confidence decisions

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "data"))
sys.path.insert(0, str(PROJECT_ROOT / "blockchain" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))
sys.path.insert(0, str(PROJECT_ROOT / "confidence"))
sys.path.insert(0, str(PROJECT_ROOT / "inventory"))

st.set_page_config(
    page_title="Supply Chain XAI Dashboard",
    page_icon="🔗",
    layout="wide",
)

# ---------------------------------------------------------------------------
#  Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("Supply Chain XAI")
st.sidebar.markdown("**Blockchain-Verified, Confidence-Gated**")
view = st.sidebar.radio(
    "Navigate to:",
    ["Forecast & Explainability", "Confidence Gate", "Audit Ledger", "Approval Queue", "System Status"],
)

# ---------------------------------------------------------------------------
#  Helper: check blockchain
# ---------------------------------------------------------------------------

def _chain_available() -> bool:
    try:
        from interact import _get_web3
        w3 = _get_web3()
        return w3.is_connected()
    except Exception:
        return False


# ============================================================================
#  VIEW 1: Forecast & Explainability
# ============================================================================

if view == "Forecast & Explainability":
    st.title("Forecast & Explainability")
    st.markdown("Run a forecast and view SHAP explanations for why the model predicted what it did.")

    col1, col2, col3 = st.columns(3)
    with col1:
        product = st.selectbox("Product", ["SKU_A", "SKU_B", "SKU_C"])
    with col2:
        region = st.selectbox("Region", ["North", "South", "West"])
    with col3:
        model = st.selectbox("Model", ["xgboost", "prophet", "lstm"])

    if st.button("Run Forecast", type="primary"):
        with st.spinner("Running pipeline..."):
            try:
                from pipeline import run_pipeline
                result = run_pipeline(
                    product, region,
                    model_type=model,
                    inventory_scenario="below_rop",
                    log_to_chain=False,
                )
                dr = result["decision_record"]

                # Forecast result
                st.subheader("Forecast Result")
                fc1, fc2, fc3, fc4 = st.columns(4)
                fc1.metric("Point Forecast", f"{dr.forecast:.1f}")
                fc2.metric("Confidence", f"{dr.confidence:.3f}")
                fc3.metric("Safety Stock", f"{dr.safety_stock:.1f}")
                fc4.metric("Reorder Point", f"{dr.reorder_point:.1f}")

                # Explanation panel
                st.subheader("Model Explanation")
                if model == "xgboost":
                    st.info("**SHAP Explanation**: Feature contributions to the forecast.")
                    meta = dr.confidence_metadata
                    if "quantiles" in meta:
                        st.write(f"Quantile range: {meta['quantiles']}")
                    st.write("SHAP values show which features (lag sales, promotions, holidays) "
                             "pushed the forecast higher or lower than the baseline.")
                elif model == "prophet":
                    st.info("**Decomposition**: Trend + Seasonality + Holiday effects.")
                    st.write("Prophet decomposes the forecast into:")
                    st.write("- **Trend**: Long-term direction of sales")
                    st.write("- **Weekly seasonality**: Day-of-week patterns")
                    st.write("- **Yearly seasonality**: Annual patterns")
                    st.write("- **Holiday effects**: Impact of promotional events")
                elif model == "lstm":
                    st.info("**Attention Weights**: Which time steps influenced the forecast most.")
                    meta = dr.confidence_metadata
                    if meta.get("attention_weights"):
                        st.write("Attention weights across the input window:")
                        weights = meta["attention_weights"]
                        st.bar_chart(pd.Series(weights, name="Attention Weight"))
                    else:
                        st.write("Attention weights not available for this prediction.")

                # Inventory status
                st.subheader("Inventory Status")
                inv1, inv2, inv3, inv4 = st.columns(4)
                inv1.metric("Current Inventory", f"{dr.current_inventory:.1f}")
                inv2.metric("Reorder Point", f"{dr.reorder_point:.1f}")
                inv3.metric("Suggested Order Qty", f"{dr.suggested_order_quantity:.1f}")
                inv4.metric("Reorder Required", "Yes" if dr.reorder_required else "No")

            except Exception as e:
                st.error(f"Pipeline error: {e}")


# ============================================================================
#  VIEW 2: Confidence Gate
# ============================================================================

elif view == "Confidence Gate":
    st.title("Confidence Gate Status")
    st.markdown("View how the confidence threshold gates autonomous vs. human-approved decisions.")

    st.info("Threshold: **0.85** (configured in `confidence/threshold_config.yaml`)")
    st.write("Decisions with confidence >= 0.85 are auto-executed on-chain.")
    st.write("Decisions with confidence < 0.85 are flagged for human review.")

    if st.button("Run Full Scan (18 combinations)", type="primary"):
        with st.spinner("Scanning all product/region/model combinations..."):
            try:
                from pipeline import run_full_scan
                results = run_full_scan(
                    inventory_scenario="below_rop",
                    log_to_chain=False,
                )

                rows = []
                for r in results:
                    dr = r["decision_record"]
                    rows.append({
                        "Product": dr.product_id,
                        "Region": dr.region,
                        "Model": dr.model_type,
                        "Forecast": round(dr.forecast, 1),
                        "Confidence": round(dr.confidence, 4),
                        "Threshold": dr.confidence_threshold,
                        "Reorder Needed": dr.reorder_required,
                        "Decision": dr.decision.value,
                    })

                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)

                # Summary metrics
                st.subheader("Decision Distribution")
                m1, m2, m3 = st.columns(3)
                auto_count = len(df[df["Decision"] == "AUTONOMOUS_EXECUTION"])
                human_count = len(df[df["Decision"] == "HUMAN_APPROVAL"])
                no_action_count = len(df[df["Decision"] == "NO_ACTION"])
                m1.metric("Auto-Executed", auto_count, f"{100*auto_count/len(df):.0f}%")
                m2.metric("Human Review", human_count, f"{100*human_count/len(df):.0f}%")
                m3.metric("No Action", no_action_count, f"{100*no_action_count/len(df):.0f}%")

                # Confidence distribution chart
                st.subheader("Confidence Score Distribution")
                st.bar_chart(df["Confidence"])

            except Exception as e:
                st.error(f"Scan error: {e}")

    # Manual single evaluation
    st.subheader("Single Evaluation")
    col1, col2, col3 = st.columns(3)
    with col1:
        product = st.selectbox("Product", ["SKU_A", "SKU_B", "SKU_C"], key="gate_product")
    with col2:
        region = st.selectbox("Region", ["North", "South", "West"], key="gate_region")
    with col3:
        model = st.selectbox("Model", ["xgboost", "prophet", "lstm"], key="gate_model")

    if st.button("Evaluate"):
        with st.spinner("Evaluating..."):
            try:
                from pipeline import run_pipeline
                result = run_pipeline(product, region, model_type=model, log_to_chain=False)
                dr = result["decision_record"]

                if dr.decision.value == "AUTONOMOUS_EXECUTION":
                    st.success(f"Decision: **{dr.decision.value}** (confidence {dr.confidence:.3f} >= {dr.confidence_threshold})")
                elif dr.decision.value == "HUMAN_APPROVAL":
                    st.warning(f"Decision: **{dr.decision.value}** (confidence {dr.confidence:.3f} < {dr.confidence_threshold})")
                else:
                    st.info(f"Decision: **{dr.decision.value}**")

                st.json(dr.to_dict())
            except Exception as e:
                st.error(f"Error: {e}")


# ============================================================================
#  VIEW 3: Audit Ledger
# ============================================================================

elif view == "Audit Ledger":
    st.title("Immutable Audit Ledger")
    st.markdown("Browse all records stored on the blockchain.")

    if not _chain_available():
        st.warning("Blockchain not available. Start Ganache and deploy the contract first.")
        st.code("ganache --port 8545\npython blockchain/scripts/deploy.py", language="bash")
    else:
        tab1, tab2, tab3 = st.tabs(["Decisions", "Purchase Orders", "Integrity Check"])

        with tab1:
            st.subheader("On-Chain Decisions")
            try:
                from interact import _get_web3, _load_contract
                w3 = _get_web3()
                contract = _load_contract(w3)
                decision_ids = contract.functions.getAllDecisionIds().call()
                STATUS = {0: "NO_ACTION", 1: "AUTONOMOUS_EXECUTION", 2: "HUMAN_APPROVAL"}
                FIXED = 10**18

                rows = []
                for did in decision_ids:
                    rec = contract.functions.getDecision(did).call()
                    rows.append({
                        "ID": rec[0],
                        "SKU": rec[1],
                        "Region": rec[2],
                        "Model": rec[4],
                        "Forecast": f"{rec[5]/FIXED:.1f}",
                        "Confidence": f"{rec[6]/FIXED:.4f}",
                        "Threshold": f"{rec[7]/FIXED:.4f}",
                        "Status": STATUS.get(rec[9], str(rec[9])),
                        "Order Qty": f"{rec[3]/FIXED:.1f}",
                        "SHAP Hash": rec[8].hex()[:16] + "...",
                    })

                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            except Exception as e:
                st.error(f"Error reading ledger: {e}")

        with tab2:
            st.subheader("On-Chain Purchase Orders")
            try:
                from interact import _get_web3, _load_contract
                w3 = _get_web3()
                contract = _load_contract(w3)
                po_ids = contract.functions.getAllPOIds().call()
                FIXED = 10**18

                rows = []
                for pid in po_ids:
                    po = contract.functions.getPO(pid).call()
                    rows.append({
                        "PO ID": po[0],
                        "Decision ID": po[1],
                        "SKU": po[2],
                        "Region": po[3],
                        "Quantity": f"{po[4]/FIXED:.1f}",
                    })

                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            except Exception as e:
                st.error(f"Error reading POs: {e}")

        with tab3:
            st.subheader("Ledger Integrity Verification")
            if st.button("Run Integrity Check"):
                with st.spinner("Verifying all on-chain records..."):
                    try:
                        from ledger_integrity_check import verify_ledger
                        results = verify_ledger()
                        st.success(f"Integrity check passed: {results['decisions_verified']} decisions, "
                                   f"{results['pos_verified']} POs verified.")
                        st.json(results)
                    except Exception as e:
                        st.error(f"Verification error: {e}")


# ============================================================================
#  VIEW 4: Approval Queue
# ============================================================================

elif view == "Approval Queue":
    st.title("Human Approval Queue")
    st.markdown("Review decisions flagged for human approval (low confidence).")

    if not _chain_available():
        st.warning("Blockchain not available. Start Ganache and deploy the contract first.")
    else:
        try:
            from interact import _get_web3, _load_contract
            w3 = _get_web3()
            contract = _load_contract(w3)
            decision_ids = contract.functions.getAllDecisionIds().call()
            FIXED = 10**18

            pending = []
            for did in decision_ids:
                rec = contract.functions.getDecision(did).call()
                if rec[9] == 2:  # HUMAN_APPROVAL
                    pending.append(rec)

            if not pending:
                st.info("No decisions pending human approval. All decisions have been auto-executed or required no action.")
            else:
                st.write(f"**{len(pending)}** decisions awaiting human review:")
                for rec in pending:
                    with st.expander(f"Decision #{rec[0]} - {rec[1]}/{rec[2]} ({rec[4]})"):
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Forecast", f"{rec[5]/FIXED:.1f}")
                        c2.metric("Confidence", f"{rec[6]/FIXED:.4f}")
                        c3.metric("Threshold", f"{rec[7]/FIXED:.4f}")
                        c4.metric("Order Qty", f"{rec[3]/FIXED:.1f}")

                        st.write(f"**Reasoning**: Confidence {rec[6]/FIXED:.3f} is below the "
                                 f"threshold {rec[7]/FIXED}, so this reorder was deferred to human review.")

                        a1, a2 = st.columns(2)
                        if a1.button(f"Approve & Execute", key=f"approve_{rec[0]}"):
                            st.success(f"Decision #{rec[0]} approved! (In production, this would execute the PO on-chain)")
                        if a2.button(f"Reject", key=f"reject_{rec[0]}"):
                            st.warning(f"Decision #{rec[0]} rejected. No PO will be executed.")

        except Exception as e:
            st.error(f"Error loading approvals: {e}")


# ============================================================================
#  VIEW 5: System Status
# ============================================================================

elif view == "System Status":
    st.title("System Status")

    # Blockchain status
    st.subheader("Blockchain Layer")
    chain_ok = _chain_available()
    if chain_ok:
        st.success("Ganache: Connected")
        try:
            from interact import _get_web3, _load_contract
            w3 = _get_web3()
            contract = _load_contract(w3)
            n_d = contract.functions.getDecisionCount().call()
            n_p = contract.functions.getPOCount().call()
            st.write(f"Decisions on chain: **{n_d}**")
            st.write(f"Purchase orders on chain: **{n_p}**")
        except Exception:
            pass
    else:
        st.error("Ganache: Not connected")
        st.code("ganache --port 8545", language="bash")

    # Model status
    st.subheader("ML Models")
    try:
        from data_loader import load_raw_sales
        raw = load_raw_sales()
        products = raw["product_id"].unique()
        regions = raw["region"].unique()
        st.write(f"Products: **{', '.join(products)}**")
        st.write(f"Regions: **{', '.join(regions)}**")
        st.write(f"Data points: **{len(raw)}**")
    except Exception as e:
        st.error(f"Data unavailable: {e}")

    # Pipeline config
    st.subheader("Configuration")
    try:
        from confidence_estimator import get_threshold
        threshold = get_threshold()
        st.write(f"Confidence threshold: **{threshold}**")
    except Exception:
        st.write("Confidence threshold: **0.85** (default)")

    try:
        from ss_rop_calculator import get_inventory_config
        cfg = get_inventory_config()
        st.write(f"Lead time: **{cfg.get('LEAD_TIME_DAYS', 'N/A')}** days")
        st.write(f"Z-score: **{cfg.get('Z_SCORE', 'N/A')}**")
    except Exception:
        pass
