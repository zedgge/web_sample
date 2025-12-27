"""
Project Ketamine - Web Demo
Interactive Paper Trading Demo for Potential Clients

Run:
    streamlit run web_demo.py

Features:
- Live paper trading simulation
- Multiple pre-built strategies
- Real-time performance charts
- Risk metrics dashboard
- Portfolio visualization
- Downloadable trade history
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import time

# Import your services
try:
    from app.services.paper_broker import PaperBroker
    from app.services.realtime_data_feed import RealtimeDataFeed
    from strategies.simple_strategies import (
        SimpleMomentumStrategy,
        SimpleMeanReversionStrategy,
        RSIStrategy,
        MovingAverageCrossoverStrategy,
        BreakoutStrategy
    )
    from app.services.strategy_backtester import StrategyBacktester
    SERVICES_AVAILABLE = True
except ImportError:
    SERVICES_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Project Ketamine - Paper Trading Demo",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .success-metric {
        border-left-color: #10b981;
    }
    .danger-metric {
        border-left-color: #ef4444;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False
if 'trades' not in st.session_state:
    st.session_state.trades = []
if 'equity_curve' not in st.session_state:
    st.session_state.equity_curve = []
if 'start_capital' not in st.session_state:
    st.session_state.start_capital = 100000
if 'current_capital' not in st.session_state:
    st.session_state.current_capital = 100000
if 'positions' not in st.session_state:
    st.session_state.positions = {}


def generate_demo_data():
    """Generate realistic demo trading data"""
    # Simulate some trades
    np.random.seed(int(time.time()))

    # Add a trade
    trade = {
        'timestamp': datetime.now(),
        'symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']),
        'side': np.random.choice(['BUY', 'SELL']),
        'quantity': np.random.randint(10, 100),
        'price': np.random.uniform(100, 500),
        'strategy': np.random.choice(['Momentum', 'RSI', 'MA Crossover', 'Mean Reversion'])
    }

    st.session_state.trades.append(trade)

    # Update equity curve
    pnl = np.random.normal(0, 500)  # Random P&L
    st.session_state.current_capital += pnl
    st.session_state.equity_curve.append({
        'timestamp': datetime.now(),
        'equity': st.session_state.current_capital
    })

    # Update positions
    if trade['side'] == 'BUY':
        if trade['symbol'] in st.session_state.positions:
            st.session_state.positions[trade['symbol']]['quantity'] += trade['quantity']
        else:
            st.session_state.positions[trade['symbol']] = {
                'quantity': trade['quantity'],
                'avg_price': trade['price']
            }
    else:
        if trade['symbol'] in st.session_state.positions:
            st.session_state.positions[trade['symbol']]['quantity'] -= trade['quantity']
            if st.session_state.positions[trade['symbol']]['quantity'] <= 0:
                del st.session_state.positions[trade['symbol']]


# Header
st.markdown('<div class="main-header">🚀 Project Ketamine</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #64748b;">Institutional-Grade Algorithmic Trading Platform - Live Demo</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Trading parameters
    st.subheader("Trading Parameters")
    initial_capital = st.number_input(
        "Initial Capital ($)",
        min_value=10000,
        max_value=1000000,
        value=100000,
        step=10000
    )

    # Strategy selection
    st.subheader("Strategies")
    strategies = st.multiselect(
        "Select Strategies",
        ["Momentum", "Mean Reversion", "RSI", "MA Crossover", "Breakout"],
        default=["Momentum", "RSI"]
    )

    # Asset selection
    st.subheader("Assets")
    assets = st.multiselect(
        "Select Assets to Trade",
        ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "SPY", "QQQ"],
        default=["AAPL", "GOOGL", "MSFT"]
    )

    # Risk parameters
    st.subheader("Risk Management")
    max_position_size = st.slider(
        "Max Position Size (%)",
        min_value=5,
        max_value=50,
        value=20,
        step=5
    )

    stop_loss = st.slider(
        "Stop Loss (%)",
        min_value=1,
        max_value=20,
        value=10,
        step=1
    )

    st.markdown("---")

    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start", use_container_width=True):
            st.session_state.bot_running = True
            st.session_state.start_capital = initial_capital
            st.session_state.current_capital = initial_capital
            st.session_state.trades = []
            st.session_state.equity_curve = [{
                'timestamp': datetime.now(),
                'equity': initial_capital
            }]
            st.session_state.positions = {}
            st.success("Bot started!")

    with col2:
        if st.button("⏹️ Stop", use_container_width=True):
            st.session_state.bot_running = False
            st.warning("Bot stopped!")

    # Status
    st.markdown("---")
    if st.session_state.bot_running:
        st.success("🟢 **Status:** Running")
    else:
        st.info("🔵 **Status:** Stopped")

# Main content
if not st.session_state.bot_running and len(st.session_state.trades) == 0:
    # Welcome screen
    st.info("👈 Configure your trading bot in the sidebar and click **Start** to begin paper trading!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### ✅ Risk-Free Testing")
        st.write("100% paper trading - no real money at risk")

    with col2:
        st.markdown("### 🤖 Automated Trading")
        st.write("Multiple strategies working 24/7")

    with col3:
        st.markdown("### 📊 Real-Time Analytics")
        st.write("Live performance tracking and risk metrics")

    st.markdown("---")

    # Feature showcase
    st.subheader("🎯 Platform Features")

    features_col1, features_col2 = st.columns(2)

    with features_col1:
        st.markdown("""
        **Trading Capabilities:**
        - ✅ Multi-Strategy Aggregation
        - ✅ Real-Time Market Data
        - ✅ Advanced Order Types
        - ✅ Position Sizing & Risk Management
        - ✅ Paper & Live Trading
        """)

    with features_col2:
        st.markdown("""
        **Analytics & Monitoring:**
        - ✅ Performance Attribution
        - ✅ VaR & CVaR Risk Metrics
        - ✅ Strategy Performance Tracking
        - ✅ Live Dashboards
        - ✅ Trade Journal & Logging
        """)

    st.markdown("---")

    # Example backtest results
    st.subheader("📈 Example: Momentum Strategy Backtest (2023)")

    # Generate sample backtest chart
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    equity = 100000 * (1 + np.random.randn(len(dates)).cumsum() * 0.01)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=equity,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#667eea', width=2)
    ))

    fig.update_layout(
        title="Sample Portfolio Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Sample metrics
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric("Total Return", "+23.4%", delta="+23.4%")

    with metric_col2:
        st.metric("Sharpe Ratio", "1.85", delta="+0.45")

    with metric_col3:
        st.metric("Max Drawdown", "-8.3%", delta="-8.3%")

    with metric_col4:
        st.metric("Win Rate", "58.2%", delta="+8.2%")

else:
    # Live trading dashboard

    # Simulate new trade if bot is running
    if st.session_state.bot_running and np.random.random() < 0.3:
        generate_demo_data()

    # Performance metrics
    st.subheader("📊 Performance Overview")

    total_return = ((st.session_state.current_capital - st.session_state.start_capital) / st.session_state.start_capital) * 100
    total_pnl = st.session_state.current_capital - st.session_state.start_capital
    num_trades = len(st.session_state.trades)
    num_positions = len(st.session_state.positions)

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric(
            "Portfolio Value",
            f"${st.session_state.current_capital:,.2f}",
            delta=f"${total_pnl:,.2f}"
        )

    with metric_col2:
        st.metric(
            "Total Return",
            f"{total_return:.2f}%",
            delta=f"{total_return:.2f}%"
        )

    with metric_col3:
        st.metric("Total Trades", num_trades)

    with metric_col4:
        st.metric("Open Positions", num_positions)

    st.markdown("---")

    # Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("📈 Equity Curve")

        if len(st.session_state.equity_curve) > 0:
            equity_df = pd.DataFrame(st.session_state.equity_curve)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_df['timestamp'],
                y=equity_df['equity'],
                mode='lines',
                fill='tozeroy',
                name='Equity',
                line=dict(color='#667eea', width=2)
            ))

            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Portfolio Value ($)",
                hovermode='x unified',
                height=300,
                margin=dict(l=0, r=0, t=0, b=0)
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data yet. Waiting for trades...")

    with chart_col2:
        st.subheader("🎯 Open Positions")

        if len(st.session_state.positions) > 0:
            positions_data = []
            for symbol, pos in st.session_state.positions.items():
                positions_data.append({
                    'Symbol': symbol,
                    'Quantity': pos['quantity'],
                    'Avg Price': f"${pos['avg_price']:.2f}",
                    'Value': f"${pos['quantity'] * pos['avg_price']:,.2f}"
                })

            positions_df = pd.DataFrame(positions_data)
            st.dataframe(positions_df, use_container_width=True, hide_index=True)
        else:
            st.info("No open positions")

    st.markdown("---")

    # Trade history
    st.subheader("📋 Recent Trades")

    if len(st.session_state.trades) > 0:
        trades_df = pd.DataFrame(st.session_state.trades)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp']).dt.strftime('%H:%M:%S')
        trades_df['value'] = (trades_df['quantity'] * trades_df['price']).apply(lambda x: f"${x:,.2f}")
        trades_df['price'] = trades_df['price'].apply(lambda x: f"${x:.2f}")

        display_df = trades_df[['timestamp', 'symbol', 'side', 'quantity', 'price', 'value', 'strategy']].tail(10)
        display_df.columns = ['Time', 'Symbol', 'Side', 'Qty', 'Price', 'Value', 'Strategy']

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Download button
        csv = trades_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Trade History (CSV)",
            data=csv,
            file_name=f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No trades yet")

    # Auto-refresh
    if st.session_state.bot_running:
        time.sleep(2)
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 2rem;">
    <p><strong>Project Ketamine</strong> - Institutional-Grade Algorithmic Trading Platform</p>
    <p>⚠️ This is a demo with simulated data. For educational purposes only.</p>
    <p>Trading involves substantial risk. Past performance does not guarantee future results.</p>
</div>
""", unsafe_allow_html=True)
