"""
Evaluation Dashboard for Streamlit
Interactive visualization of evaluation results
"""

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


def load_evaluation_data():
    """Load evaluation results and summaries"""
    data = {}

    files = {
        'results': 'data/comprehensive_results.json',
        'summary': 'data/comprehensive_summary.json',
        'ablation': 'data/ablation_results.json',
        'error': 'data/error_analysis.json'
    }

    for key, filepath in files.items():
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data[key] = json.load(f)
        else:
            data[key] = None

    return data


def show_evaluation_dashboard():
    """Main evaluation dashboard"""

    st.title("📊 Evaluation Dashboard")
    st.markdown("---")

    # Load data
    data = load_evaluation_data()

    if data['results'] is None:
        st.warning("⚠️ No evaluation results found. Please run the evaluation first:")
        st.code("python run_evaluation.py")
        return

    results = data['results']
    summary = data['summary']
    ablation = data['ablation']
    error_analysis = data['error']

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview",
        "🔬 Ablation Study",
        "📊 Detailed Results",
        "❌ Error Analysis",
        "📋 Raw Data"
    ])

    # ===== TAB 1: Overview =====
    with tab1:
        st.header("Overall Performance Summary")

        if summary:
            # Metric cards
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Mean Reciprocal Rank",
                    f"{summary['standard_metrics']['mrr']['mean']:.4f}",
                    f"±{summary['standard_metrics']['mrr']['std']:.4f}"
                )

            with col2:
                st.metric(
                    "F1 Score",
                    f"{summary['standard_metrics']['f1_score']['mean']:.4f}",
                    f"±{summary['standard_metrics']['f1_score']['std']:.4f}"
                )

            with col3:
                st.metric(
                    "NDCG@10",
                    f"{summary['standard_metrics']['ndcg@10']['mean']:.4f}",
                    f"±{summary['standard_metrics']['ndcg@10']['std']:.4f}"
                )

            with col4:
                st.metric(
                    "Avg Response Time",
                    f"{summary['performance']['avg_response_time']:.2f}s"
                )

            st.markdown("---")

            # Score distributions
            st.subheader("Score Distributions")

            df = pd.DataFrame(results)

            col1, col2 = st.columns(2)

            with col1:
                fig_mrr = px.histogram(
                    df,
                    x='mrr',
                    nbins=20,
                    title='MRR Distribution',
                    labels={'mrr': 'MRR Score'},
                    color_discrete_sequence=['#3498db']
                )
                fig_mrr.add_vline(
                    x=summary['standard_metrics']['mrr']['mean'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Mean: {summary['standard_metrics']['mrr']['mean']:.3f}"
                )
                st.plotly_chart(fig_mrr, use_container_width=True)

            with col2:
                fig_f1 = px.histogram(
                    df,
                    x='f1_score',
                    nbins=20,
                    title='F1 Score Distribution',
                    labels={'f1_score': 'F1 Score'},
                    color_discrete_sequence=['#2ecc71']
                )
                fig_f1.add_vline(
                    x=summary['standard_metrics']['f1_score']['mean'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Mean: {summary['standard_metrics']['f1_score']['mean']:.3f}"
                )
                st.plotly_chart(fig_f1, use_container_width=True)

            # Performance by question type
            st.subheader("Performance by Question Type")

            type_stats = df.groupby('question_type').agg({
                'mrr': 'mean',
                'f1_score': 'mean',
                'ndcg@10': 'mean'
            }).reset_index()

            fig_type = go.Figure(data=[
                go.Bar(name='MRR', x=type_stats['question_type'], y=type_stats['mrr'], marker_color='#3498db'),
                go.Bar(name='F1 Score', x=type_stats['question_type'], y=type_stats['f1_score'], marker_color='#2ecc71'),
                go.Bar(name='NDCG@10', x=type_stats['question_type'], y=type_stats['ndcg@10'], marker_color='#e74c3c')
            ])
            fig_type.update_layout(
                title='Average Scores by Question Type',
                xaxis_title='Question Type',
                yaxis_title='Score',
                barmode='group'
            )
            st.plotly_chart(fig_type, use_container_width=True)

    # ===== TAB 2: Ablation Study =====
    with tab2:
        st.header("Ablation Study Results")

        if ablation:
            # Retrieval mode comparison
            st.subheader("1. Retrieval Mode Comparison")

            mode_data = []
            for mode in ['mode_dense', 'mode_sparse', 'mode_hybrid']:
                if mode in ablation:
                    mode_name = mode.replace('mode_', '').upper()
                    metrics = ablation[mode]['metrics']
                    mode_data.append({
                        'Mode': mode_name,
                        'MRR': metrics['mrr']['mean'],
                        'F1 Score': metrics['f1_score']['mean'],
                        'NDCG@10': metrics['ndcg@10']['mean']
                    })

            if mode_data:
                df_modes = pd.DataFrame(mode_data)

                fig_modes = go.Figure(data=[
                    go.Bar(name='MRR', x=df_modes['Mode'], y=df_modes['MRR'], marker_color='#3498db'),
                    go.Bar(name='F1 Score', x=df_modes['Mode'], y=df_modes['F1 Score'], marker_color='#2ecc71'),
                    go.Bar(name='NDCG@10', x=df_modes['Mode'], y=df_modes['NDCG@10'], marker_color='#e74c3c')
                ])
                fig_modes.update_layout(
                    title='Performance Comparison: Dense vs Sparse vs Hybrid',
                    xaxis_title='Retrieval Mode',
                    yaxis_title='Score',
                    barmode='group'
                )
                st.plotly_chart(fig_modes, use_container_width=True)

                st.dataframe(df_modes, use_container_width=True)

            # Parameter tuning
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("2. Top-K Parameter Tuning")
                k_data = []
                for key in sorted([k for k in ablation.keys() if k.startswith('top_k_')]):
                    k = int(key.split('_')[-1])
                    mrr = ablation[key]['metrics']['mrr']['mean']
                    k_data.append({'K': k, 'MRR': mrr})

                if k_data:
                    df_k = pd.DataFrame(k_data)
                    fig_k = px.line(
                        df_k,
                        x='K',
                        y='MRR',
                        markers=True,
                        title='Effect of Top-K on MRR'
                    )
                    st.plotly_chart(fig_k, use_container_width=True)

            with col2:
                st.subheader("3. Top-N Parameter Tuning")
                n_data = []
                for key in sorted([k for k in ablation.keys() if k.startswith('top_n_')]):
                    n = int(key.split('_')[-1])
                    f1 = ablation[key]['metrics']['f1_score']['mean']
                    n_data.append({'N': n, 'F1': f1})

                if n_data:
                    df_n = pd.DataFrame(n_data)
                    fig_n = px.line(
                        df_n,
                        x='N',
                        y='F1',
                        markers=True,
                        title='Effect of Top-N on F1 Score'
                    )
                    st.plotly_chart(fig_n, use_container_width=True)

    # ===== TAB 3: Detailed Results =====
    with tab3:
        st.header("Detailed Question Results")

        df = pd.DataFrame(results)

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            question_types = ['All'] + list(df['question_type'].unique())
            selected_type = st.selectbox("Question Type", question_types)

        with col2:
            min_mrr = st.slider("Minimum MRR", 0.0, 1.0, 0.0)

        with col3:
            min_f1 = st.slider("Minimum F1", 0.0, 1.0, 0.0)

        # Apply filters
        filtered_df = df.copy()
        if selected_type != 'All':
            filtered_df = filtered_df[filtered_df['question_type'] == selected_type]
        filtered_df = filtered_df[filtered_df['mrr'] >= min_mrr]
        filtered_df = filtered_df[filtered_df['f1_score'] >= min_f1]

        st.write(f"Showing {len(filtered_df)} / {len(df)} questions")

        # Display results
        for _, row in filtered_df.iterrows():
            with st.expander(f"**{row['question_id']}** - {row['question']} (Type: {row['question_type']})"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Generated Answer:**")
                    st.write(row['generated_answer'])

                with col2:
                    st.markdown("**Ground Truth:**")
                    st.write(row['ground_truth_answer'][:200] + '...')

                # Metrics
                metric_cols = st.columns(5)
                metric_cols[0].metric("MRR", f"{row['mrr']:.4f}")
                metric_cols[1].metric("F1", f"{row['f1_score']:.4f}")
                metric_cols[2].metric("NDCG@10", f"{row['ndcg@10']:.4f}")
                metric_cols[3].metric("P@5", f"{row['precision@5']:.4f}")
                metric_cols[4].metric("Time", f"{row['response_time']:.2f}s")

    # ===== TAB 4: Error Analysis =====
    with tab4:
        st.header("Error Analysis")

        if error_analysis:
            col1, col2, col3 = st.columns(3)

            col1.metric("Retrieval Failures (MRR=0)", error_analysis['retrieval_failures_count'])
            col2.metric("Generation Failures (F1<0.3)", error_analysis['generation_failures_count'])
            col3.metric("Ranking Issues (0<MRR<0.5)", error_analysis['ranking_issues_count'])

            st.markdown("---")
            st.subheader("Performance by Question Type")

            error_df = pd.DataFrame([
                {
                    'Question Type': qtype.capitalize(),
                    'Total': data['total'],
                    'Avg MRR': data['avg_mrr'],
                    'Avg F1': data['avg_f1']
                }
                for qtype, data in error_analysis['by_question_type'].items()
            ])

            st.dataframe(error_df, use_container_width=True)

            # Visualize
            fig_error = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Avg MRR by Type', 'Avg F1 by Type')
            )

            fig_error.add_trace(
                go.Bar(x=error_df['Question Type'], y=error_df['Avg MRR'], name='MRR', marker_color='#3498db'),
                row=1, col=1
            )

            fig_error.add_trace(
                go.Bar(x=error_df['Question Type'], y=error_df['Avg F1'], name='F1', marker_color='#2ecc71'),
                row=1, col=2
            )

            fig_error.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_error, use_container_width=True)

    # ===== TAB 5: Raw Data =====
    with tab5:
        st.header("Raw Data Export")

        df = pd.DataFrame(results)

        # Display summary stats
        st.subheader("Summary Statistics")
        st.dataframe(df[['mrr', 'f1_score', 'ndcg@10', 'precision@5', 'recall@5', 'response_time']].describe())

        st.markdown("---")

        # Display full data
        st.subheader("Full Results Table")
        st.dataframe(
            df[['question_id', 'question', 'question_type', 'mrr', 'f1_score', 'ndcg@10', 'response_time']],
            use_container_width=True
        )

        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="evaluation_results.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    show_evaluation_dashboard()
