import streamlit as st
import os
import json
import joblib
import numpy as np
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

st.set_page_config(page_title="Insurance Claim Agent", page_icon="🛡️")
st.title("🛡️ Insurance Claim Processing Agent (LangGraph)")

# Load ML Model and Scaler (.pkl)
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model.pkl")
scaler_path = os.path.join(current_dir, "scaler.pkl")

model = joblib.load(model_path) if os.path.exists(model_path) else None
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

# Sidebar Configuration
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input(
    "API Key",
    value="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",
    type="password"
)
base_url = st.sidebar.text_input("Base URL", value="https://openrouter.ai/api/v1")
model_name = st.sidebar.text_input("Model Name", value="openrouter/free")

# Preset Scenarios
st.subheader("1. Claim Details")
preset = st.selectbox(
    "Choose Test Scenario",
    [
        "Complete Claim ($1,200) -> Auto Approve",
        "Missing Documents ($3,500) -> Reject",
        "Expired Policy ($2,500) -> Reject",
        "High Value Claim ($15,000) -> Human Review",
        "Custom Claim"
    ]
)

if "Complete Claim" in preset:
    c_id, c_name, c_amt = "CLM-001", "Sarah Miller", 1200.0
    c_docs = ["National_ID", "Medical_Invoice", "Incident_Report"]
    c_active, c_exp, c_inc = True, "2026-12-31", "2026-06-10"
    c_notes = "Routine emergency clinic visit for ankle sprain."
elif "Missing Documents" in preset:
    c_id, c_name, c_amt = "CLM-002", "David Brown", 3500.0
    c_docs = ["National_ID"]  # Missing invoice and incident report
    c_active, c_exp, c_inc = True, "2026-12-31", "2026-06-10"
    c_notes = "Car repair claim but did not attach garage invoice."
elif "Expired Policy" in preset:
    c_id, c_name, c_amt = "CLM-003", "Marcus Vance", 2500.0
    c_docs = ["National_ID", "Medical_Invoice", "Incident_Report"]
    c_active, c_exp, c_inc = False, "2025-12-31", "2026-06-10"
    c_notes = "Claim submitted after policy expired."
elif "High Value" in preset:
    c_id, c_name, c_amt = "CLM-004", "Elena Rostova", 15000.0
    c_docs = ["National_ID", "Medical_Invoice", "Incident_Report"]
    c_active, c_exp, c_inc = True, "2026-12-31", "2026-06-10"
    c_notes = "Major surgery and hospitalization with verified medical records."
else:
    c_id, c_name, c_amt = "CLM-100", "Alex Smith", 2000.0
    c_docs = ["National_ID", "Medical_Invoice", "Incident_Report"]
    c_active, c_exp, c_inc = True, "2026-12-31", "2026-06-10"
    c_notes = "Standard outpatient claim."

col1, col2 = st.columns(2)
with col1:
    claim_id = st.text_input("Claim ID", value=c_id)
    claimant_name = st.text_input("Claimant Name", value=c_name)
    claim_amount = st.number_input("Claim Amount ($)", value=float(c_amt), step=100.0)
with col2:
    docs_submitted = st.multiselect(
        "Submitted Documents",
        ["National_ID", "Medical_Invoice", "Incident_Report"],
        default=c_docs
    )
    policy_active = st.checkbox("Policy is Active", value=c_active)

claim_notes = st.text_area("Incident Description", value=c_notes, height=80)

# LangGraph State Schema
class ClaimState(TypedDict):
    claim_id: str
    claim_amount: float
    documents_submitted: List[str]
    policy_active: bool
    policy_expiry_date: str
    incident_date: str
    claim_notes: str
    doc_result: dict
    eligibility_result: dict
    fraud_result: dict
    claim_summary: str
    decision: str
    human_notes: str

# Execute Workflow
if st.button("Process Claim with LangGraph", type="primary"):
    with st.spinner("Executing 5-node LangGraph multi-agent workflow..."):
        try:
            # 1. Document Verification Agent
            def doc_agent(state: ClaimState):
                req = ["National_ID", "Medical_Invoice", "Incident_Report"]
                sub = set(state.get("documents_submitted", []))
                missing = [d for d in req if d not in sub]
                return {
                    "doc_result": {
                        "status": "PASSED" if not missing else "FAILED",
                        "missing": missing
                    }
                }

            # 2. Eligibility Agent
            def elig_agent(state: ClaimState):
                active = state.get("policy_active", False)
                inc = state.get("incident_date", "")
                exp = state.get("policy_expiry_date", "")
                return {
                    "eligibility_result": {
                        "status": "ELIGIBLE" if (active and inc <= exp) else "INELIGIBLE"
                    }
                }

            # 3. Fraud Detection Agent
            def fraud_agent(state: ClaimState):
                amt = state.get("claim_amount", 0)
                notes = state.get("claim_notes", "")
                if amt > 20000 or "urgent" in notes.lower():
                    risk = "HIGH"
                elif amt > 10000:
                    risk = "MEDIUM"
                else:
                    risk = "LOW"
                return {"fraud_result": {"risk": risk}}

            # 4. Summary Agent & Routing Logic
            def summary_agent(state: ClaimState):
                d = state["doc_result"]["status"]
                e = state["eligibility_result"]["status"]
                f = state["fraud_result"]["risk"]
                amt = state["claim_amount"]

                if d == "FAILED" or e == "INELIGIBLE":
                    dec = "REJECT"
                elif f in ["MEDIUM", "HIGH"] or amt > 10000:
                    dec = "HUMAN_APPROVAL"
                else:
                    dec = "AUTO_APPROVE"

                summary = f"Claim {state['claim_id']}: Docs={d}, Eligibility={e}, Fraud Risk={f}"
                return {"claim_summary": summary, "decision": dec}

            # 5. Human Approval Agent
            def human_agent(state: ClaimState):
                amt = state["claim_amount"]
                risk = state["fraud_result"]["risk"]
                if risk == "HIGH":
                    note = "Adjuster rejected claim due to high risk flags."
                    status = "REJECTED_BY_ADJUSTER"
                else:
                    note = f"Adjuster reviewed and approved high value claim (${amt:,.2f})."
                    status = "APPROVED_BY_ADJUSTER"
                return {"human_notes": note, "decision": status}

            # Graph Assembly
            g = StateGraph(ClaimState)
            g.add_node("doc_verification", doc_agent)
            g.add_node("eligibility_check", elig_agent)
            g.add_node("fraud_detection", fraud_agent)
            g.add_node("claim_summary", summary_agent)
            g.add_node("human_approval", human_agent)

            # Parallel Fan-Out
            g.add_edge(START, "doc_verification")
            g.add_edge(START, "eligibility_check")
            g.add_edge(START, "fraud_detection")

            # Fan-In
            g.add_edge("doc_verification", "claim_summary")
            g.add_edge("eligibility_check", "claim_summary")
            g.add_edge("fraud_detection", "claim_summary")

            # Routing
            g.add_conditional_edges(
                "claim_summary",
                lambda s: s["decision"],
                {
                    "AUTO_APPROVE": END,
                    "REJECT": END,
                    "HUMAN_APPROVAL": "human_approval"
                }
            )
            g.add_edge("human_approval", END)
            app = g.compile()

            res = app.invoke({
                "claim_id": claim_id,
                "claim_amount": float(claim_amount),
                "documents_submitted": docs_submitted,
                "policy_active": policy_active,
                "policy_expiry_date": c_exp,
                "incident_date": c_inc,
                "claim_notes": claim_notes
            })

            # Show Results
            st.success("Workflow Execution Completed!")
            st.metric("Final Claim Decision", res["decision"])

            c1, c2, c3 = st.columns(3)
            c1.write(f"**Document Check:** {res['doc_result']['status']}")
            c2.write(f"**Eligibility Check:** {res['eligibility_result']['status']}")
            c3.write(f"**Fraud Risk:** {res['fraud_result']['risk']}")

            if res.get("human_notes"):
                st.info(f"**Human-in-the-Loop Review:** {res['human_notes']}")

            # ML Model Prediction (.pkl)
            if model and scaler:
                feats = scaler.transform([[float(claim_amount), len(docs_submitted), 5, 0]])
                pred = model.predict(feats)[0]
                st.write("---")
                st.caption(f"🤖 Machine Learning Model (.pkl) Score: {'Flagged for Review (Class 1)' if pred == 1 else 'Low Risk Routine Claim (Class 0)'}")

        except Exception as e:
            st.error(f"Processing error: {e}")

if __name__ == "__main__":
    pass
