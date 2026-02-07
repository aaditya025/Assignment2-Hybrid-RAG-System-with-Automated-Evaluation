"""
Streamlit Web Application for Hybrid RAG System
"""

import streamlit as st
import time
import os
from retrieval import DenseRetriever, SparseRetriever, HybridRetriever
from generator import ResponseGenerator
import config


# Page configuration
st.set_page_config(
    page_title="Hybrid RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .chunk-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def load_system():
    """
    Load the retrieval and generation system
    """
    # Use session state to cache models
    if 'system_loaded' not in st.session_state:
        with st.spinner("Loading RAG system... This may take a minute."):
            # Load retrievers
            dense_retriever = DenseRetriever(model_name=config.EMBEDDING_MODEL)
            dense_retriever.load_index(config.INDEX_DIR)

            sparse_retriever = SparseRetriever()
            sparse_retriever.load_index(config.INDEX_DIR)

            # Create hybrid retriever
            hybrid_retriever = HybridRetriever(
                dense_retriever=dense_retriever,
                sparse_retriever=sparse_retriever,
                rrf_k=config.RRF_K
            )

            # Load generator
            generator = ResponseGenerator(model_name=config.LLM_MODEL)

            st.session_state.hybrid_retriever = hybrid_retriever
            st.session_state.generator = generator
            st.session_state.system_loaded = True

    return st.session_state.hybrid_retriever, st.session_state.generator


def rag_interface():
    """Main RAG Q&A interface"""
    # Header
    st.markdown('<p class="main-header">🔍 Hybrid RAG System</p>', unsafe_allow_html=True)
    st.markdown("**Combining Dense Vector Retrieval, BM25 Sparse Retrieval, and Reciprocal Rank Fusion**")

    # Check if indexes exist
    if not os.path.exists(config.INDEX_DIR) or not os.path.exists(os.path.join(config.INDEX_DIR, 'faiss_index.bin')):
        st.error("⚠️ Indexes not found! Please run `python build_index.py` first to build the system.")
        st.info("This will fetch Wikipedia articles, create chunks, and build the retrieval indexes.")
        return

    # Load system
    try:
        hybrid_retriever, generator = load_system()
        st.success("✅ System loaded successfully!")
    except Exception as e:
        st.error(f"Error loading system: {e}")
        return

    # Configuration
    st.sidebar.header("⚙️ Configuration")
    top_k = st.sidebar.slider("Top-K (Dense/Sparse)", min_value=5, max_value=20, value=10,
                               help="Number of chunks to retrieve from each method")
    top_n = st.sidebar.slider("Top-N (Final Context)", min_value=3, max_value=10, value=5,
                               help="Number of chunks to use for final answer generation")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This system combines:\n"
        "- **Dense Retrieval**: Semantic similarity using sentence embeddings\n"
        "- **Sparse Retrieval**: Keyword matching using BM25\n"
        "- **RRF**: Reciprocal Rank Fusion to combine results"
    )

    # Main query interface
    st.markdown("---")
    query = st.text_input(
        "🔎 Enter your question:",
        placeholder="E.g., What is quantum computing?",
        help="Ask any question based on the Wikipedia articles in the system"
    )

    if st.button("🚀 Generate Answer", type="primary"):
        if not query:
            st.warning("Please enter a question!")
            return

        # Create tabs for results
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Answer", "🎯 Retrieved Chunks", "📊 Retrieval Details", "⏱️ Metrics"])

        with st.spinner("Generating answer..."):
            start_time = time.time()

            # Retrieve
            retrieval_results = hybrid_retriever.retrieve(query, top_k=top_k, top_n=top_n)

            # Generate
            response = generator.generate_with_metadata(query, retrieval_results)

            end_time = time.time()
            response_time = end_time - start_time

        # Tab 1: Answer
        with tab1:
            st.markdown("### 💡 Generated Answer")
            st.markdown(f"**Question:** {query}")
            st.success(response['answer'])

        # Tab 2: Retrieved Chunks (RRF Results)
        with tab2:
            st.markdown("### 🎯 Top Retrieved Chunks (After RRF)")
            st.caption(f"Showing top {len(response['retrieved_chunks'])} chunks used for answer generation")

            for i, chunk in enumerate(response['retrieved_chunks'], 1):
                with st.expander(f"**Chunk {i}** - {chunk['title']} (RRF Score: {chunk['rrf_score']})"):
                    st.markdown(f"**Source:** [{chunk['title']}]({chunk['url']})")
                    st.markdown(f"**Text:** {chunk['text']}")
                    st.markdown("**Scores:**")
                    cols = st.columns(3)
                    cols[0].metric("RRF Score", chunk['rrf_score'])
                    cols[1].metric("Dense Score", chunk['dense_score'])
                    cols[2].metric("Sparse Score", chunk['sparse_score'])

        # Tab 3: Retrieval Details
        with tab3:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🔵 Dense Retrieval Results")
                for i, result in enumerate(response['dense_results'], 1):
                    st.markdown(f"**{i}. {result['title']}** (Score: {result['score']})")
                    st.caption(result['text'])
                    st.markdown(f"[View Source]({result['url']})")
                    st.markdown("---")

            with col2:
                st.markdown("#### 🟢 Sparse (BM25) Retrieval Results")
                for i, result in enumerate(response['sparse_results'], 1):
                    st.markdown(f"**{i}. {result['title']}** (Score: {result['score']})")
                    st.caption(result['text'])
                    st.markdown(f"[View Source]({result['url']})")
                    st.markdown("---")

        # Tab 4: Metrics
        with tab4:
            st.markdown("### ⏱️ Performance Metrics")
            metric_cols = st.columns(4)
            metric_cols[0].metric("Response Time", f"{response_time:.2f}s")
            metric_cols[1].metric("Chunks Retrieved", len(response['retrieved_chunks']))
            metric_cols[2].metric("Top-K", top_k)
            metric_cols[3].metric("Top-N", top_n)

            st.markdown("### 🔧 System Configuration")
            config_data = {
                "Embedding Model": config.EMBEDDING_MODEL,
                "LLM Model": config.LLM_MODEL,
                "RRF K": config.RRF_K,
                "Chunk Size": f"{config.MIN_CHUNK_SIZE}-{config.MAX_CHUNK_SIZE} tokens",
                "Chunk Overlap": f"{config.CHUNK_OVERLAP} tokens"
            }
            for key, value in config_data.items():
                st.text(f"{key}: {value}")


def main():
    """Main app with page navigation"""
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🔍 RAG Q&A", "📊 Evaluation Dashboard"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")

    if page == "🔍 RAG Q&A":
        rag_interface()
    else:
        from evaluation_dashboard import show_evaluation_dashboard
        show_evaluation_dashboard()


if __name__ == "__main__":
    main()
