"""
Streamlit UI — MedVision RAG (Structured Output Edition)
Updated to display structured fields from new API response schema.
"""

import os
import requests
from PIL import Image
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="MedVision RAG", page_icon="🏥", layout="wide")

st.markdown("""
<style>
.header { background: linear-gradient(135deg,#1a73e8,#0d47a1);
          padding:1.5rem 2rem; border-radius:12px; color:white; margin-bottom:1.5rem; }
.card   { background:#f8f9fa; border-left:4px solid #1a73e8;
          padding:1rem 1.5rem; border-radius:0 8px 8px 0; margin-bottom:1rem; }
.warn   { background:#fff3e0; border:1px solid #ff9800;
          padding:.75rem 1rem; border-radius:8px; margin-bottom:1rem; }
.danger { background:#fce4ec; border:1px solid #e53935;
          padding:.75rem 1rem; border-radius:8px; margin-bottom:.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
  <h1 style="margin:0">🏥 MedVision RAG</h1>
  <p style="margin:.25rem 0 0 0;opacity:.85">
    Multimodal Medical AI — MedGemma 4B · FAISS RAG · LangGraph
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="warn">⚠️ <b>Disclaimer:</b> Educational demo only. '
            'Not a substitute for professional medical advice.</div>',
            unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("📤 Input")
    uploaded = st.file_uploader("Upload Medical Image (optional)", type=["jpg","jpeg","png"])
    if uploaded:
        st.image(Image.open(uploaded), caption=uploaded.name, use_column_width=True)
    else:
        st.info("💡 No image? You can still ask general medical questions.")

    question = st.text_area("Your Question", height=120,
        placeholder="e.g. What abnormalities are visible in this X-ray?")
    go = st.button("🔍 Analyze", type="primary", use_container_width=True,
                   disabled=not bool(question.strip()))

with right:
    st.subheader("📊 Analysis Results")

    if go and question.strip():
        with st.spinner("🧠 Running MedGemma pipeline..."):
            try:
                form = {"question": question}
                files = {}
                if uploaded:
                    uploaded.seek(0)
                    files = {"file": (uploaded.name, uploaded.read(), uploaded.type)}

                resp = requests.post(f"{API_URL}/analyze", data=form,
                                     files=files or None, timeout=300)
                resp.raise_for_status()
                r = resp.json()

                st.success("✅ Analysis complete!")

                # ── Image Findings ──────────────────────────────────────────
                with st.expander("🔬 Image Findings (VLM)", expanded=True):
                    fi = r["image_findings"]
                    col1, col2 = st.columns(2)
                    col1.metric("Study Type", fi["study_type"])
                    col2.metric("Anatomy", fi["anatomy"])
                    st.markdown(f"**Findings:** {fi['findings']}")
                    st.markdown(f"**Impression:** {fi['impression']}")

                # ── RAG Context ─────────────────────────────────────────────
                with st.expander("📚 Retrieved Medical Context (RAG)"):
                    st.markdown(r["context"])

                # ── LLM Answer ──────────────────────────────────────────────
                with st.expander("💡 AI Answer", expanded=True):
                    st.markdown(f"**Summary:** {r['summary']}")
                    st.markdown(f"**Answer:** {r['answer']}")
                    st.markdown(f"**Context used:** {r['context_used']}")

                # ── Red Flags ───────────────────────────────────────────────
                if r.get("red_flags"):
                    st.markdown("**⚠️ Red Flags:**")
                    for flag in r["red_flags"]:
                        st.markdown(
                            f'<div class="danger">🚨 {flag}</div>',
                            unsafe_allow_html=True)

                # ── Recommendation ──────────────────────────────────────────
                st.info(f"📋 **Recommendation:** {r['recommendation']}")

            except requests.exceptions.ConnectionError:
                st.error(f"❌ Cannot reach API at `{API_URL}`. Is FastAPI running?")
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. MedGemma inference can take 1-3 min on CPU.")
            except requests.exceptions.HTTPError as e:
                st.error(f"API Error {e.response.status_code}: {e.response.text}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
    else:
        st.markdown("""
**Pipeline:**
1. 🔬 **VLM** — MedGemma reads image → structured findings JSON
2. 📚 **RAG** — FAISS retrieves relevant medical knowledge
3. 💡 **LLM** — MedGemma synthesizes → structured answer JSON

*All inference runs on MedGemma 4B IT — locally or on HF Spaces T4 GPU.*
        """)

st.divider()
st.caption("MedVision RAG OSS | MedGemma 4B | Educational use only")
