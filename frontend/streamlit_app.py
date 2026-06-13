"""Streamlit frontend: chat, document upload, and observability dashboard."""
import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Enterprise Agentic RAG", layout="wide", page_icon="🔎")
st.title("🔎 Enterprise Agentic RAG Platform")

tab_chat, tab_ingest, tab_obs = st.tabs(["💬 Chat", "📥 Ingest", "📊 Observability"])

with tab_ingest:
    st.subheader("Upload documents")
    files = st.file_uploader("PDF / TXT / MD", accept_multiple_files=True,
                             type=["pdf", "txt", "md"])
    if st.button("Ingest", disabled=not files):
        for f in files:
            r = requests.post(f"{API}/ingest", files={"file": (f.name, f.getvalue())})
            if r.ok:
                d = r.json()
                st.success(f"{f.name}: {d['chunks']} chunks indexed")
            else:
                st.error(f"{f.name}: {r.text}")

with tab_chat:
    st.subheader("Ask across your documents")
    col1, col2 = st.columns([3, 1])
    with col2:
        roles = st.multiselect("Roles (RBAC)", ["employee", "manager", "finance", "admin"],
                               default=["employee"])
        use_agent = st.toggle("Agent mode (tools)", value=True)
    with col1:
        q = st.text_input("Your question", placeholder="e.g. What's the status of TKT-1003?")
    if st.button("Ask", type="primary", disabled=not q):
        with st.spinner("Thinking..."):
            r = requests.post(f"{API}/query", json={
                "query": q, "roles": roles, "use_agent": use_agent})
        if r.ok:
            data = r.json()
            if data["blocked"]:
                st.error(f"🛡️ Blocked: {data['block_reason']}")
            else:
                st.markdown("### Answer")
                st.write(data["answer"])
                if data["steps"]:
                    with st.expander(f"🧠 Reasoning trace ({len(data['steps'])} steps)"):
                        for i, s in enumerate(data["steps"], 1):
                            st.markdown(f"**Step {i}:** {s['thought']}")
                            if s.get("tool_call"):
                                st.code(f"{s['tool_call']['name']}({s['tool_call']['arguments']})")
                            if s.get("observation"):
                                st.text(s["observation"][:500])
                if data["citations"]:
                    with st.expander(f"📚 Citations ({len(data['citations'])})"):
                        for c in data["citations"]:
                            st.markdown(f"- **{c['source']}** (p.{c['page']}): {c['snippet']}")
                st.caption(f"⏱️ {data['latency_ms']:.0f}ms · 🔒 {data['redactions']} redactions")
        else:
            st.error(r.text)

with tab_obs:
    st.subheader("Platform metrics")
    if st.button("Refresh"):
        st.rerun()
    r = requests.get(f"{API}/metrics")
    if r.ok:
        m = r.json()
        c = st.columns(5)
        c[0].metric("Total queries", m["total_queries"])
        c[1].metric("Avg latency (ms)", m["avg_latency_ms"])
        c[2].metric("Blocked", m["blocked_queries"])
        c[3].metric("PII redactions", m["total_redactions"])
        c[4].metric("Tool calls", m["total_tool_calls"])
        st.markdown("#### Recent queries")
        st.dataframe(m["recent"], use_container_width=True)
    else:
        st.info("No metrics yet — run some queries first.")
