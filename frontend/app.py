"""ChipMind Streamlit dashboard."""

import json
from pathlib import Path

import streamlit as st

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@st.cache_resource
def load_chipmind_graph():
    """Load ChipMindGraph once. Returns None if indexes missing."""
    try:
        from chipmind.agents.graph import ChipMindGraph

        index_dir = PROJECT_ROOT / "data" / "processed" / "indexes"
        if not index_dir.exists():
            return None
        return ChipMindGraph(index_dir=str(index_dir))
    except Exception:
        return None


def get_kb_stats():
    """Get knowledge base stats."""
    try:
        chunks_path = PROJECT_ROOT / "data" / "processed" / "all_chunks.jsonl"
        if not chunks_path.exists():
            return 0, 0
        modules = docs = 0
        with open(chunks_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    if d.get("chunk_type") == "verilog_code":
                        modules += 1
                    elif d.get("chunk_type") == "eda_doc":
                        docs += 1
                except Exception:
                    pass
        return modules, docs
    except Exception:
        return 0, 0


def main():
    st.set_page_config(
        page_title="ChipMind",
        page_icon="🧠",
        layout="wide",
    )

    # Sidebar
    with st.sidebar:
        st.title("🧠 ChipMind")
        st.caption("v0.1")
        st.subheader("Project Info")
        st.markdown("[GitHub](https://github.com/your-org/chipmind)")
        st.divider()

        from chipmind.config import settings

        st.subheader("Model")
        st.markdown(f"**{settings.LLM_MODEL}**")
        st.divider()

        modules, docs = get_kb_stats()
        st.subheader("Knowledge Base")
        st.metric("Verilog modules", f"{modules:,}" if modules else "N/A")
        st.metric("EDA docs", f"{docs:,}" if docs else "N/A")

    # Tab 1: Generate Verilog
    tab1, tab2, tab3 = st.tabs(["Generate Verilog", "Knowledge Base Explorer", "Benchmark Results"])

    with tab1:
        st.title("🧠 ChipMind")
        st.markdown("Describe your design in natural language. ChipMind will generate Verilog and validate it.")

        query = st.text_area(
            "Design description",
            placeholder="Design a 4-bit ALU with add, subtract, AND, OR operations",
            height=120,
        )
        max_iterations = st.slider("Max debug iterations", 1, 10, 5)

        if st.button("Generate", type="primary"):
            if not query.strip():
                st.warning("Please enter a design description.")
            else:
                graph = load_chipmind_graph()
                if graph is None:
                    st.error(
                        "ChipMind index not found. Run `make build-index` first. "
                        "See README for setup instructions."
                    )
                else:
                    with st.spinner("Analyzing specification..."):
                        try:
                            result = graph.run(query=query, max_iterations=max_iterations)
                            st.session_state["generate_result"] = result
                        except Exception as e:
                            st.error(f"Generation failed: {e}")
                            st.exception(e)

        if "generate_result" in st.session_state:
            result = st.session_state["generate_result"]

            # Section 1: Specification
            with st.expander("📋 Specification", expanded=False):
                spec = result.get("spec", {})
                if spec:
                    st.json(spec)
                else:
                    st.info("No spec parsed.")

            # Section 2: Retrieved Examples
            with st.expander("📚 Retrieved Examples", expanded=False):
                retrieved = result.get("retrieved_modules", [])
                if retrieved:
                    for i, m in enumerate(retrieved[:3]):
                        name = m.get("module_name", f"Example {i+1}")
                        score = m.get("rrf_score")
                        code = m.get("code", "")
                        st.markdown(f"**{name}**" + (f" (score: {score:.4f})" if score else ""))
                        st.code(code[:500] + ("..." if len(code) > 500 else ""), language="verilog")
                else:
                    st.info("No examples retrieved.")

            # Section 3: Generated Verilog
            st.subheader("Generated Verilog")
            code = result.get("final_code", result.get("generated_code", ""))
            st.code(code, language="verilog")

            status = result.get("final_status", "unknown")
            if status == "success":
                st.success("✅ Compiled and passed simulation!")
            elif status == "compile_success_sim_fail":
                st.warning("⚠️ Compiled but simulation failed.")
            else:
                st.error("❌ Compilation failed.")

            # Section 4: Compilation Result
            st.subheader("Compilation Result")
            hist = result.get("iteration_history", [])
            if hist:
                last = hist[-1]
                if last.get("compiled"):
                    st.success("✅ Compiled successfully")
                    if last.get("simulated"):
                        st.info("Simulation output:")
                        st.text(last.get("sim_output", ""))
                else:
                    st.error("❌ Compilation failed")
                    for e in last.get("errors", []):
                        st.error(f"Line {e.get('line', '?')}: {e.get('message', '')}")
            else:
                st.info("No compilation history.")

            # Section 5: Debug History
            if len(hist) > 1:
                st.subheader("Debug History")
                for i, entry in enumerate(hist):
                    with st.expander(f"Iteration {i}"):
                        st.json(entry)

            # Section 6: Metrics
            st.subheader("Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total time", f"{result.get('total_time_seconds', 0):.2f}s")
            with col2:
                st.metric("Iterations", len(hist))
            with col3:
                st.metric("Tokens used", result.get("total_tokens_used", 0))

    with tab2:
        st.title("Knowledge Base Explorer")
        search_query = st.text_input("Search query", placeholder="4-bit counter")
        search_type = st.selectbox("Search type", ["Code", "Docs"])
        k = st.slider("Number of results", 1, 20, 5)

        if st.button("Search"):
            graph = load_chipmind_graph()
            if graph is None:
                st.error("Index not loaded. Run `make build-index` first.")
            elif not search_query.strip():
                st.warning("Enter a search query.")
            else:
                try:
                    if search_type == "Docs":
                        results = graph.retriever.search_docs(search_query, k=k)
                    else:
                        results = graph.retriever.search_code(search_query, k=k)
                    st.session_state["search_results"] = results
                except Exception as e:
                    st.error(str(e))
                    st.session_state["search_results"] = []

        if "search_results" in st.session_state:
            for r in st.session_state["search_results"]:
                name = r.get("module_name") or r.get("section_title") or "Unknown"
                score = r.get("rrf_score")
                tags = r.get("tags", [])
                tag_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
                st.markdown(f"**{name}**" + (f" (score: {score:.4f})" if score else ""))
                if tag_str:
                    st.caption(f"Tags: {tag_str}")
                code = r.get("code", r.get("text", ""))
                st.code(code[:500] + ("..." if len(code) > 500 else ""), language="verilog")

    with tab3:
        st.title("Benchmark Results")
        metrics_path = PROJECT_ROOT / "chipmind" / "evaluation" / "results" / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            def fmt_iter(x):
                if isinstance(x, (int, float)):
                    return f"{x:.1f}"
                return "N/A"

            st.dataframe(
                [
                    {
                        "Mode": {"baseline": "Baseline", "rag_only": "RAG Only", "chipmind_agentic": "Agentic"}.get(k, k),
                        "Pass@1": f"{v.get('pass_at_1', 0)*100:.1f}%",
                        "Syntax Rate": f"{v.get('syntax_rate', 0)*100:.1f}%",
                        "Avg Iterations": fmt_iter(v.get("avg_iterations")),
                    }
                    for k, v in metrics.items()
                ],
                use_container_width=True,
            )
            col1, col2, col3 = st.columns(3)
            base = metrics.get("baseline", {}).get("pass_at_1", 0) or 0.001
            rag = metrics.get("rag_only", {}).get("pass_at_1", 0)
            agentic = metrics.get("chipmind_agentic", {}).get("pass_at_1", 0)
            with col1:
                st.metric("Baseline Pass@1", f"{base*100:.1f}%")
            with col2:
                imp = ((rag - base) / base * 100) if base else 0
                st.metric("RAG Pass@1", f"{rag*100:.1f}%", delta=f"+{imp:.1f}%" if imp > 0 else None)
            with col3:
                imp = ((agentic - base) / base * 100) if base else 0
                st.metric("Agentic Pass@1", f"{agentic*100:.1f}%", delta=f"+{imp:.1f}%" if imp > 0 else None)
        else:
            st.info("Run `make eval-quick` to generate benchmark results.")


if __name__ == "__main__":
    main()
