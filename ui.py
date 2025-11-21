import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Simple Doc Search", page_icon="📄", layout="wide")

st.title("📄 Document Search")

with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Results to show", 1, 20, 5)
    threshold = st.slider("Minimum Cosine Similarity", 0.0, 1.0, 0.0, 0.05, help="Filter out results below this similarity score.")
    
    st.divider()
    st.header("Query Expansion")
    
    query_status = st.empty()
    
query = st.text_input("Search Query", placeholder="Type here...")
search_button = st.button("Search")

if search_button and query:
    with st.spinner("Searching..."):
        try:
            response = requests.post(f"{API_URL}/search", json={"query": query, "top_k": top_k, "threshold": threshold})
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                expanded_query = data.get("expanded_query", "")
                
                with query_status.container():
                    st.info(f"Expanded: {expanded_query}")
                
                if not results:
                    st.info("No results found.")
                else:
                    for res in results:
                        with st.expander(f"📄 {res['doc_id']} (Score: {res['score']:.2f})", expanded=False):
                            st.markdown(f"**Cleaned Content Preview:**")
                            st.info(res['preview'])
                            
                            st.markdown("---")
                            st.subheader("📊 Similarity Metrics")
                            m_col1, m_col2, m_col3 = st.columns(3)
                            
                            metrics = res.get('metrics', {})
                            
                            with m_col1:
                                st.metric("Cosine Similarity", metrics.get('Cosine Similarity', 'N/A'), help="Measures semantic similarity (1.0 is identical). Higher is better.")
                            with m_col2:
                                st.metric("Euclidean Dist.", metrics.get('Euclidean Distance', 'N/A'), help="Straight-line distance between vectors. Lower is better.")
                            with m_col3:
                                st.metric("Manhattan Dist.", metrics.get('Manhattan Distance', 'N/A'), help="Sum of absolute differences. Robust to outliers. Lower is better.")

                            st.markdown("---")
                            st.caption("Explanation")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Why:** {res['explanation'].get('why', '-')}")
                                st.write(f"**Overlap:** {res['explanation'].get('overlap_ratio', '-')}")
                            with col2:
                                keywords = ", ".join(res['explanation'].get('keywords_overlapped', []))
                                st.write(f"**Keywords:** {keywords if keywords else 'None'}")
            else:
                st.error(f"API Error: {response.status_code}")
        except Exception as e:
            st.error(f"Connection Error: {e}")
elif search_button:
    st.warning("Please enter a query.")

