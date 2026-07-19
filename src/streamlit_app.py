import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from generate import generate_answer

st.set_page_config(page_title="Urban Parking Policy Assistant", page_icon="🅿️")

st.title("🅿️ Urban Parking Policy Assistant")
st.write(
    "Ask a question about parking policy, zoning requirements, or planning "
    "strategy. Answers are grounded in the loaded documents and cite their sources."
)

city = st.selectbox(
    "Scope to a city (optional)",
    options=["All", "Columbus", "Dubai", "Seattle", "Methodology"],
    index=0,
)

query = st.text_input("Your question", placeholder="e.g. What are the minimum off-street parking requirements for retail uses?")

if st.button("Ask") and query.strip():
    with st.spinner("Retrieving and generating answer..."):
        selected_city = None if city == "All" else city.lower()
        answer, sources = generate_answer(query, city=selected_city)

    st.markdown("### Answer")
    st.write(answer)

    st.markdown("### Sources")
    for doc in sources:
        doc_city = doc.metadata.get("city", "unknown")
        source_name = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", "?")
        st.write(f"- **{doc_city}** | {source_name}, p.{page}")