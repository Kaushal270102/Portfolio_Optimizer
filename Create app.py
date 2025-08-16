import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
import os

# Import helper functions from utils.py
from utils import RISK_PROFILE_MAPPING, convert_weights_to_shares

# --- Configuration and Data Loading ---
# Centralized paths to backend outputs
DATA_PATHS = {
    'section6_weights': 'consolidated_data/models/section6_clean/clean_final_portfolio_weights.csv',
    'section6_summary': 'consolidated_data/ready_for_section6/clean_section6_complete.json',
    'section7_complete_results': 'consolidated_data/models/section7_portfolio_management/portfolio_management_complete.json',
    'section3_clusters': 'consolidated_data/processed_data/pattern_matching_clusters.csv',
    'section2_fundamentals': 'consolidated_data/processed_data/enhanced_master_nse_fundamental_data.csv',
    'section4_integrated_external_analysis': 'consolidated_data/external_factors/integrated_external_analysis.json',
    'section2_sector_mapping': 'consolidated_data/processed_data/corrected_nse_sector_mapping.json', # For pie charts
    'nse_returns_correlation_matrix': 'consolidated_data/processed_data/nse_returns_correlation_matrix.csv', # For correlation heatmap
}

# Cache data loading for performance
@st.cache_data
def load_all_backend_data():
    data = {}
    for key, path in DATA_PATHS.items():
        if os.path.exists(path):
            try:
                if path.endswith('.csv'):
                    # Handle index_col for specific CSVs if needed, or adjust load_data() in utils.py
                    if key in ['section3_clusters', 'section2_fundamentals', 'nse_returns_correlation_matrix']:
                        data[key] = pd.read_csv(path, index_col=0) # Assume first column is index
                    else:
                        data[key] = pd.read_csv(path)
                elif path.endswith('.json'):
                    with open(path, 'r') as f:
                        data[key] = json.load(f)
            except Exception as e:
                st.error(f"Error loading {key} from {path}: {e}")
                data[key] = None
        else:
            st.warning(f"Required data file not found: {path}. Please ensure backend script has run successfully.")
            data[key] = None
    return data

ALL_BACKEND_DATA = load_all_backend_data()


# --- Streamlit App Layout ---
st.set_page_config(
    page_title="NSE 500 Portfolio Optimization & Management",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ Institutional-Grade Portfolio Manager")
st.markdown("---")

# Sidebar Navigation
st.sidebar.header("Navigation")
page_selection = st.sidebar.radio(
    "Go to",
    ("🚀 Overview Dashboard", "🎯 Build/Analyze Portfolio", "📈 Performance & Risk Deep Dive", "🔍 Analytical Insights")
)

# --- 1. Overview Dashboard Section ---
if page_selection == "🚀 Overview Dashboard":
    st.header("🚀 Market & System Overview")
    
    if ALL_BACKEND_DATA['section4_integrated_external_analysis']:
        st.subheader("Current Market Regime")
        regime_assessment = ALL_BACKEND_DATA['section4_integrated_external_analysis'].get('market_regime_assessment', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Regime", regime_assessment.get('overall_assessment', 'N/A').replace('_', ' ').title())
        with col2:
            st.metric("Market Trend", regime_assessment.get('market_trend', 'N/A').replace('_', ' ').title())
        with col3:
            st.metric("Volatility Env.", regime_assessment.get('volatility_environment', 'N/A').replace('_', ' ').title())
        
        st.subheader("System Readiness")
        st.info(f"Data last processed: {ALL_BACKEND_DATA['section7_complete_results'].get('metadata',{}).get('creation_date', 'N/A')}")
        st.info(f"High-quality stocks processed: {len(ALL_BACKEND_DATA['section2_fundamentals']) if ALL_BACKEND_DATA['section2_fundamentals'] is not None else 'N/A'}")
    else:
        st.warning("Market overview data not fully available. Please ensure backend has run successfully.")

# --- 2. Build/Analyze Portfolio Section ---
elif page_selection == "🎯 Build/Analyze Portfolio":
    st.header("🎯 Define & Generate Your Portfolio")

    if ALL_BACKEND_DATA['section6_weights'] is None or ALL_BACKEND_DATA['section6_summary'] is None:
        st.error("Required backend data for portfolio building is missing. Please ensure backend has run successfully.")
    else:
        # User Inputs
        with st.expander("Your Investment Preferences", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                investment_amount_input = st.number_input(
                    "Total Investment Amount (₹)",
                    min_value=10000.0,
                    max_value=100000000.0,
                    value=100000.0,
                    step=10000.0,
                    format="%.2f"
                )
                st.info("Actual invested amount may vary due to whole-share purchases.")

            with col2:
                # This input is for display/information, backend run determines actual lookback
                investment_horizon_years = st.slider(
                    "Intended Investment Horizon (Years)",
                    min_value=1,
                    max_value=10,
                    value=5,
                    step=1
                )
                st.info("Backend optimization uses a fixed/pre-determined lookback period (e.g., 5 years) for calculations. This input is for your information only.")
                
            with col3:
                risk_profile_options = list(RISK_PROFILE_MAPPING.keys())
                risk_profile = st.radio(
                    "Your Risk Profile",
                    risk_profile_options,
                    index=risk_profile_options.index('Moderate') # Default to Moderate
                )
                st.info("Backend optimization parameters are fixed/pre-determined (e.g., based on a 'Moderate' profile). This input is for your information only, matching the backend's assumptions.")

        st.subheader("Generated Portfolio Recommendation")
        
        # This button is only needed if you want to trigger the share conversion
        if st.button("Calculate Shares for this Investment"):
            
            # Retrieve the optimized weights from Section 6 output
            optimized_weights_df = ALL_BACKEND_DATA['section6_weights']
            optimized_weights_series = pd.Series(
                optimized_weights_df['weight'].values, 
                index=optimized_weights_df['stock_symbol']
            )

            with st.spinner("Calculating optimal shares based on live prices..."):
                portfolio_in_shares_df = convert_weights_to_shares(
                    optimized_weights_series,
                    investment_amount_input
                )

                if not portfolio_in_shares_df.empty:
                    st.dataframe(
                        portfolio_in_shares_df.set_index('stock_symbol').style.format({
                            'shares_to_buy': '{:,.0f}',
                            'invested_amount': '₹{:,.2f}',
                            'final_weight_pct': '{:,.2f}%'
                        }),
                        use_container_width=True
                    )
                    
                    st.markdown(f"**Initial Budget:** ₹{investment_amount_input:,.2f}")
                    st.markdown(f"**Actual Invested:** ₹{portfolio_in_shares_df['invested_amount'].sum():,.2f}")
                    st.markdown(f"**Remaining Cash:** ₹{(investment_amount_input - portfolio_in_shares_df['invested_amount'].sum()):,.2f}")
                    st.markdown(f"**Stocks in Portfolio:** {len(portfolio_in_shares_df)}")
                else:
                    st.warning("Could not generate a portfolio in shares. Check data and stock prices.")

        # Display Portfolio Characteristics & Sector Allocation (Always display, not just on button click)
        st.subheader("Portfolio Characteristics & Diversification (from Backend Output)")
        section6_sum = ALL_BACKEND_DATA['section6_summary'].get('portfolio_results', {}).get('validation_results', {}).get('basic_checks', {})
        section6_div = ALL_BACKEND_DATA['section6_summary'].get('portfolio_results', {}).get('validation_results', {}).get('diversification', {})

        col_char1, col_char2 = st.columns(2)
        with col_char1:
            st.metric("Total Positions", section6_sum.get('total_positions', 'N/A'))
            st.metric("Max Position", f"{section6_sum.get('max_position', 0)*100:.2f}%")
            st.metric("Sum of Weights", f"{section6_sum.get('sum_of_weights', 0)*100:.2f}%")
        with col_char2:
            st.metric("Effective Positions", f"{section6_div.get('effective_positions', 0):.1f}")
            st.metric("Top 10 Concentration", f"{section6_div.get('top_10_concentration', 0)*100:.2f}%")
            st.metric("Problematic Stocks Included?", str(section6_sum.get('no_problematic_stocks', 'N/A')))

        # Sector Allocation Pie Chart
        if ALL_BACKEND_DATA['section3_clusters'] is not None and ALL_BACKEND_DATA['section2_sector_mapping'] is not None and ALL_BACKEND_DATA['section6_weights'] is not None:
            st.subheader("Portfolio Sector Allocation")
            portfolio_symbols = ALL_BACKEND_DATA['section6_weights']['stock_symbol'].tolist()
            
            sector_map = ALL_BACKEND_DATA['section2_sector_mapping']
            stock_to_sector_series = pd.Series(sector_map)
            
            # Ensure only symbols present in the sector_map are used
            valid_portfolio_symbols = [s for s in portfolio_symbols if s in stock_to_sector_series.index]
            final_portfolio_sectors = stock_to_sector_series.loc[valid_portfolio_symbols]

            # Get the weights for these valid symbols
            valid_weights_df = ALL_BACKEND_DATA['section6_weights'][ALL_BACKEND_DATA['section6_weights']['stock_symbol'].isin(valid_portfolio_symbols)]
            valid_optimized_weights_series = pd.Series(valid_weights_df['weight'].values, index=valid_weights_df['stock_symbol'])

            portfolio_sector_weights = pd.DataFrame({
                'Weight': valid_optimized_weights_series,
                'Sector': final_portfolio_sectors
            }).groupby('Sector')['Weight'].sum().reset_index()

            fig_sector_pie = px.pie(
                portfolio_sector_weights,
                values='Weight',
                names='Sector',
                title='Portfolio Allocation by Sector',
                hole=0.3
            )
            st.plotly_chart(fig_sector_pie, use_container_width=True)
        else:
            st.info("Sector allocation data not fully available. Please ensure relevant backend files are present.")


# --- 3. Performance & Risk Deep Dive Section ---
elif page_selection == "📈 Performance & Risk Deep Dive":
    st.header("📈 Portfolio Performance & Risk Analysis")

    if ALL_BACKEND_DATA['section7_complete_results']:
        performance = ALL_BACKEND_DATA['section7_complete_results'].get('performance_tracking', {})
        risk_monitoring = ALL_BACKEND_DATA['section7_complete_results'].get('risk_monitoring', {})
        stress_testing = ALL_BACKEND_DATA['section7_complete_results'].get('stress_testing', {})
        rebalancing = ALL_BACKEND_DATA['section7_complete_results'].get('rebalancing_framework', {}).get('last_assessment', {})

        st.subheader("Performance Summary")
        col_perf1, col_perf2, col_perf3 = st.columns(3)
        with col_perf1:
            st.metric("Annualized Return", f"{performance.get('annualized_return', 0)*100:.2f}%")
            st.metric("Sharpe Ratio", f"{performance.get('sharpe_ratio', 0):.2f}")
        with col_perf2:
            st.metric("Volatility", f"{performance.get('volatility', 0)*100:.2f}%")
            st.metric("Max Drawdown", f"{performance.get('max_drawdown', 0)*100:.2f}%")
        with col_perf3:
            st.metric("Active Return (vs. Benchmark)", f"{performance.get('active_return', 0)*100:.2f}%")
            st.metric("Information Ratio", f"{performance.get('information_ratio', 0):.2f}")
        
        st.subheader("Risk Monitoring & Controls")
        col_risk1, col_risk2, col_risk3 = st.columns(3)
        with col_risk1:
            st.metric("Portfolio Volatility (Recent)", f"{risk_monitoring.get('portfolio_risks',{}).get('current_volatility',0)*100:.2f}%")
        with col_risk2:
            st.metric("Concentration Risk (HHI)", f"{risk_monitoring.get('portfolio_risks',{}).get('concentration_risk',0):.3f}")
        with col_risk3:
            st.metric("Effective Positions", f"{risk_monitoring.get('portfolio_risks',{}).get('effective_positions',0):.1f}")
        
        if risk_monitoring.get('limit_breaches'):
            st.warning("⚠️ **RISK LIMIT BREACHES DETECTED!**")
            st.dataframe(pd.DataFrame(risk_monitoring['limit_breaches']))
        
        st.subheader("Rebalancing Needs")
        st.metric("Needs Rebalancing?", "✅ YES" if rebalancing.get('needs_rebalancing', False) else "❌ NO")
        if rebalancing.get('needs_rebalancing'):
            st.info(f"Max Weight Drift: {rebalancing.get('max_drift',0)*100:.2f}%")
            st.info(f"Significant Drifts: {rebalancing.get('significant_drifts', 0)} positions")

        st.subheader("Stress Testing Results")
        if stress_testing:
            worst_scenario_loss = min([s.get('portfolio_loss',0) for s in stress_testing.values()])
            st.metric("Worst Case Loss", f"{worst_scenario_loss*100:.2f}%")
            if worst_scenario_loss < -0.25:
                st.error("🔴 **PORTFOLIO RESILIENCE: REVIEW NEEDED** - Significant loss in stress scenarios.")
            else:
                st.success("🟢 **PORTFOLIO RESILIENCE: GOOD**")
            
            st.write("---")
            st.markdown("**Scenario Details:**")
            for scenario_name, scenario_data in stress_testing.items():
                st.markdown(f"**{scenario_name.replace('_',' ').title()}:**")
                st.write(f"  - Loss: {scenario_data.get('portfolio_loss',0)*100:.2f}%")
                st.write(f"  - Max Drawdown: {scenario_data.get('max_drawdown',0)*100:.2f}%")
        else:
            st.info("Stress testing results not available.")


# --- 4. Analytical Insights Section ---
elif page_selection == "🔍 Analytical Insights":
    st.header("🔍 Deeper Analytical Insights")

    # Stock Clustering Visualization
    if ALL_BACKEND_DATA['section3_clusters'] is not None and ALL_BACKEND_DATA['section4_integrated_external_analysis'] is not None:
        st.subheader("Stock Clustering: Risk-Return Patterns")
        cluster_df = ALL_BACKEND_DATA['section3_clusters'].copy()
        
        cluster_favorability_map = {}
        if ALL_BACKEND_DATA['section4_integrated_external_analysis'].get('cluster_analysis'):
            for cluster_id_str, data in ALL_BACKEND_DATA['section4_integrated_external_analysis']['cluster_analysis'].items():
                cluster_favorability_map[int(cluster_id_str)] = data.get('regime_impact',{}).get('overall_favorability', 'neutral')

        if not cluster_favorability_map:
            st.warning("Cluster favorability data not available in external analysis.")
            cluster_df['Favorability'] = 'N/A'
        else:
            cluster_df['Favorability'] = cluster_df['Cluster'].map(cluster_favorability_map)

        # Ensure that the columns used for plotting (Volatility_1Y, Momentum_1Y) exist
        if 'Volatility_1Y' in cluster_df.columns and 'Momentum_1Y' in cluster_df.columns:
            fig_clusters = px.scatter(
                cluster_df.reset_index(), # Reset index to make 'Symbol' a column
                x='Volatility_1Y',
                y='Momentum_1Y',
                color='Cluster',
                hover_name='Symbol',
                hover_data=['Sector', 'Favorability'],
                title='Stock Clusters by Volatility vs. Momentum',
                labels={'Volatility_1Y': 'Annualized Volatility (1Y)', 'Momentum_1Y': '1-Year Momentum'}
            )
            st.plotly_chart(fig_clusters, use_container_width=True)
        else:
            st.info("Additional feature data (Volatility_1Y, Momentum_1Y) needed for plotting stock clusters. Showing raw data instead.")
            st.dataframe(cluster_df) # Show raw cluster data if plotting features are missing

    # Fundamental Scores Leaderboard
    if ALL_BACKEND_DATA['section2_fundamentals'] is not None:
        st.subheader("Fundamental Quality Leaderboard")
        fund_df = ALL_BACKEND_DATA['section2_fundamentals'].copy()
        if 'Overall_Score' in fund_df.columns:
            top_bottom_fund = fund_df[['Overall_Score', 'Sector', 'Trailing_PE', 'Return_on_Equity']].sort_values(by='Overall_Score', ascending=False)
            st.dataframe(top_bottom_fund.head(10).style.format({'Overall_Score': '{:.2f}', 'Trailing_PE': '{:.1f}', 'Return_on_Equity': '{:.2%}'}))
            st.write("---")
            st.dataframe(top_bottom_fund.tail(10).style.format({'Overall_Score': '{:.2f}', 'Trailing_PE': '{:.1f}', 'Return_on_Equity': '{:.2%}'}))
        else:
            st.info("Overall_Score not found in fundamental data. Please ensure it's calculated in backend.")
            st.dataframe(fund_df.head())

    # Correlation Heatmap (using nse_returns_correlation_matrix.csv)
    if ALL_BACKEND_DATA['nse_returns_correlation_matrix'] is not None:
        st.subheader("Stock Returns Correlation Heatmap (Partial View)")
        corr_matrix = ALL_BACKEND_DATA['nse_returns_correlation_matrix']
        
        # Select a smaller subset for display to avoid performance issues
        if len(corr_matrix.columns) > 50:
            sample_tickers = corr_matrix.columns[:50] # Or randomly select
            corr_matrix_display = corr_matrix.loc[sample_tickers, sample_tickers]
        else:
            corr_matrix_display = corr_matrix

        fig_corr_heatmap = px.imshow(
            corr_matrix_display,
            labels=dict(x="Stock", y="Stock", color="Correlation"),
            x=corr_matrix_display.columns,
            y=corr_matrix_display.columns,
            color_continuous_scale=px.colors.sequential.RdBu,
            title="Correlation Matrix of Stock Returns"
        )
        st.plotly_chart(fig_corr_heatmap, use_container_width=True)
    else:
        st.info("Correlation matrix data not available. Please ensure backend has generated 'nse_returns_correlation_matrix.csv'.")
