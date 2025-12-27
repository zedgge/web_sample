"""
QuantEdge Pro - Professional Trading Platform Demo
Interactive Paper Trading Demonstration

Features:
- Professional trading terminal UI
- Real-time data updates (smooth, no page refresh)
- Advanced performance metrics & risk analytics
- Portfolio heatmap visualization
- Strategy comparison & leaderboard
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import time

# Page config
st.set_page_config(
    page_title="QuantEdge Pro - Trading Platform",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Bloomberg-style CSS
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}

    /* Professional color palette - Bloomberg inspired */
    :root {
        --primary: #FF6600;
        --secondary: #1E3A8A;
        --success: #16A34A;
        --danger: #DC2626;
        --warning: #CA8A04;
        --dark: #0F172A;
        --light: #F8FAFC;
        --border: #E2E8F0;
        --text-primary: #1E293B;
        --text-secondary: #64748B;
    }

    /* Clean transitions */
    * {
        transition: all 0.2s ease;
    }

    /* Main header - professional, no emojis */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary);
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 1rem;
        color: var(--text-secondary);
        margin-bottom: 2rem;
        font-weight: 500;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }

    /* Live indicator - minimal, professional */
    .live-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: var(--success);
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 8px var(--success);
    }

    /* Status badge - Bloomberg style */
    .status-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        border: 2px solid;
        font-family: 'Roboto Mono', monospace;
    }

    .status-running {
        background: var(--success);
        color: white;
        border-color: var(--success);
    }

    .status-stopped {
        background: transparent;
        color: var(--text-secondary);
        border-color: var(--border);
    }

    /* Feature pill - professional */
    .feature-pill {
        display: inline-block;
        background: var(--light);
        color: var(--text-primary);
        padding: 8px 16px;
        border-radius: 4px;
        font-size: 0.85rem;
        margin: 4px;
        font-weight: 600;
        border: 1px solid var(--border);
    }

    /* Button styling - Bloomberg inspired */
    .stButton > button {
        background: linear-gradient(180deg, #FFFFFF 0%, #F1F5F9 100%);
        border: 1px solid var(--border);
        border-radius: 4px;
        color: var(--text-primary);
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        padding: 10px 24px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }

    .stButton > button:hover {
        background: linear-gradient(180deg, #F8FAFC 0%, #E2E8F0 100%);
        border-color: var(--primary);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(180deg, var(--success) 0%, #15803D 100%);
        border-color: var(--success);
        color: white;
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(180deg, #15803D 0%, #166534 100%);
        box-shadow: 0 4px 8px rgba(22, 163, 74, 0.3);
    }

    /* Data tables - terminal style */
    .dataframe {
        font-family: 'Roboto Mono', monospace;
        font-size: 0.85rem;
    }

    /* Section headers */
    h3 {
        color: var(--text-primary);
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 1rem;
        letter-spacing: -0.3px;
        border-bottom: 2px solid var(--primary);
        padding-bottom: 8px;
    }

    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-family: 'Roboto Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: var(--text-secondary);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%);
        border-right: 1px solid var(--border);
    }

    /* Remove emoji-like elements */
    .element-container {
        animation: none !important;
    }

    /* Professional metric cards */
    .metric-container {
        background: white;
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    .metric-container:hover {
        border-color: var(--primary);
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
    }

    /* Chart containers */
    .chart-container {
        background: white;
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 16px;
        margin: 12px 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    /* Professional table styling */
    table {
        border-collapse: collapse;
        border: 1px solid var(--border);
    }

    th {
        background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%);
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.5px;
        padding: 12px;
        border-bottom: 2px solid var(--border);
    }

    td {
        padding: 10px 12px;
        border-bottom: 1px solid var(--border);
    }

    /* Remove all gradients and flashy colors */
    .stApp {
        background: #FFFFFF;
    }

    /* Professional expander */
    .streamlit-expanderHeader {
        background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%);
        border: 1px solid var(--border);
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.9rem;
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
if 'strategy_performance' not in st.session_state:
    st.session_state.strategy_performance = {
        'Momentum': {'trades': 0, 'wins': 0, 'pnl': 0, 'returns': []},
        'RSI': {'trades': 0, 'wins': 0, 'pnl': 0, 'returns': []},
        'MA Crossover': {'trades': 0, 'wins': 0, 'pnl': 0, 'returns': []},
        'Mean Reversion': {'trades': 0, 'wins': 0, 'pnl': 0, 'returns': []},
    }
if 'daily_returns' not in st.session_state:
    st.session_state.daily_returns = []
if 'max_capital' not in st.session_state:
    st.session_state.max_capital = 100000
if 'trade_count' not in st.session_state:
    st.session_state.trade_count = 0
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()


def generate_realistic_trade():
    """Generate realistic trade with better data"""
    np.random.seed(int(time.time() * 1000) % 2**32)

    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX', 'ORCL', 'AMD']
    strategies = list(st.session_state.strategy_performance.keys())

    symbol = np.random.choice(symbols)
    strategy = np.random.choice(strategies)
    side = np.random.choice(['BUY', 'SELL'], p=[0.55, 0.45])
    quantity = np.random.randint(5, 150)
    price = np.random.uniform(80, 900)

    # More realistic P&L based on strategy
    strategy_multipliers = {
        'Momentum': 1.2,
        'RSI': 1.0,
        'MA Crossover': 0.9,
        'Mean Reversion': 1.1
    }

    base_pnl = np.random.normal(150, 400)
    pnl = base_pnl * strategy_multipliers.get(strategy, 1.0)
    is_win = pnl > 0

    trade = {
        'timestamp': datetime.now(),
        'symbol': symbol,
        'side': side,
        'quantity': quantity,
        'price': price,
        'value': quantity * price,
        'strategy': strategy,
        'pnl': pnl,
        'is_win': is_win
    }

    st.session_state.trades.append(trade)
    st.session_state.trade_count += 1

    # Update strategy performance
    perf = st.session_state.strategy_performance[strategy]
    perf['trades'] += 1
    perf['pnl'] += pnl
    perf['returns'].append(pnl / (quantity * price))
    if is_win:
        perf['wins'] += 1

    # Update capital
    st.session_state.current_capital += pnl
    st.session_state.max_capital = max(st.session_state.max_capital, st.session_state.current_capital)

    # Update equity curve
    st.session_state.equity_curve.append({
        'timestamp': datetime.now(),
        'equity': st.session_state.current_capital
    })

    # Update positions
    if side == 'BUY':
        if symbol in st.session_state.positions:
            current_qty = st.session_state.positions[symbol]['quantity']
            current_avg = st.session_state.positions[symbol]['avg_price']
            new_qty = current_qty + quantity
            new_avg = (current_qty * current_avg + quantity * price) / new_qty
            st.session_state.positions[symbol]['quantity'] = new_qty
            st.session_state.positions[symbol]['avg_price'] = new_avg
        else:
            st.session_state.positions[symbol] = {
                'quantity': quantity,
                'avg_price': price,
                'current_price': price * (1 + np.random.normal(0, 0.015))
            }
    else:
        if symbol in st.session_state.positions:
            st.session_state.positions[symbol]['quantity'] -= quantity
            if st.session_state.positions[symbol]['quantity'] <= 0:
                del st.session_state.positions[symbol]

    # Update position prices (simulate market movement)
    for sym in st.session_state.positions:
        price_change = np.random.normal(0, 0.008)
        st.session_state.positions[sym]['current_price'] *= (1 + price_change)

    # Update daily returns
    daily_return = (st.session_state.current_capital / st.session_state.start_capital - 1) * 100
    st.session_state.daily_returns.append(daily_return)

    st.session_state.last_update = datetime.now()


def calculate_advanced_metrics():
    """Calculate comprehensive performance metrics"""
    if len(st.session_state.trades) == 0:
        return {
            'total_return': 0,
            'total_return_abs': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'avg_trade': 0,
            'best_trade': 0,
            'worst_trade': 0,
            'consecutive_wins': 0,
            'consecutive_losses': 0
        }

    trades_df = pd.DataFrame(st.session_state.trades)

    # Basic metrics
    total_return_abs = st.session_state.current_capital - st.session_state.start_capital
    total_return = (total_return_abs / st.session_state.start_capital) * 100

    # Sharpe ratio
    if len(st.session_state.daily_returns) > 1:
        returns_array = np.array(st.session_state.daily_returns)
        returns_mean = np.mean(returns_array)
        returns_std = np.std(returns_array)
        sharpe = (returns_mean / returns_std * np.sqrt(252)) if returns_std > 0 else 0

        # Sortino ratio (only downside deviation)
        negative_returns = returns_array[returns_array < 0]
        downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0.0001
        sortino = (returns_mean / downside_std * np.sqrt(252)) if downside_std > 0 else 0
    else:
        sharpe = 0
        sortino = 0

    # Max drawdown
    if len(st.session_state.equity_curve) > 1:
        equity_series = pd.Series([e['equity'] for e in st.session_state.equity_curve])
        running_max = equity_series.expanding().max()
        drawdown = ((equity_series - running_max) / running_max * 100)
        max_dd = drawdown.min()
    else:
        max_dd = 0

    # Win rate
    wins = trades_df['is_win'].sum()
    win_rate = (wins / len(trades_df)) * 100 if len(trades_df) > 0 else 0

    # Profit factor
    total_wins = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    total_losses = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

    # Average metrics
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
    avg_trade = trades_df['pnl'].mean()

    best_trade = trades_df['pnl'].max()
    worst_trade = trades_df['pnl'].min()

    # Consecutive wins/losses
    consecutive_wins = 0
    consecutive_losses = 0
    current_streak_wins = 0
    current_streak_losses = 0

    for win in trades_df['is_win']:
        if win:
            current_streak_wins += 1
            current_streak_losses = 0
            consecutive_wins = max(consecutive_wins, current_streak_wins)
        else:
            current_streak_losses += 1
            current_streak_wins = 0
            consecutive_losses = max(consecutive_losses, current_streak_losses)

    return {
        'total_return': total_return,
        'total_return_abs': total_return_abs,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_trade': avg_trade,
        'best_trade': best_trade,
        'worst_trade': worst_trade,
        'consecutive_wins': consecutive_wins,
        'consecutive_losses': consecutive_losses
    }


# Header
st.markdown('<div class="main-header">QUANTEDGE PRO</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Algorithmic Trading Platform</div>', unsafe_allow_html=True)

# Live status indicator
if st.session_state.bot_running:
    st.markdown('''
    <div style="text-align: center; margin-bottom: 2rem;">
        <span class="live-indicator"></span>
        <span style="font-weight: 600; color: #16A34A; font-size: 0.9rem; letter-spacing: 1px; font-family: 'Roboto Mono', monospace;">LIVE TRADING</span>
    </div>
    ''', unsafe_allow_html=True)

st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Configuration")

    st.subheader("Capital")
    initial_capital = st.number_input(
        "Initial Capital ($)",
        min_value=10000,
        max_value=10000000,
        value=100000,
        step=10000
    )

    st.subheader("Strategies")
    selected_strategies = st.multiselect(
        "Active Strategies",
        ["Momentum", "RSI", "MA Crossover", "Mean Reversion"],
        default=["Momentum", "RSI", "MA Crossover"]
    )

    st.subheader("Assets")
    assets = st.multiselect(
        "Trading Universe",
        ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "NFLX", "ORCL", "AMD", "SPY", "QQQ"],
        default=["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"]
    )

    st.subheader("Risk Management")
    max_position = st.slider("Max Position Size (%)", 5, 50, 20, 5)
    stop_loss = st.slider("Stop Loss (%)", 1, 25, 10, 1)
    take_profit = st.slider("Take Profit (%)", 5, 100, 25, 5)

    st.subheader("Trading Speed")
    speed = st.select_slider(
        "Update Frequency",
        options=["Slow (5s)", "Normal (3s)", "Fast (1s)", "Maximum (0.5s)"],
        value="Normal (3s)"
    )

    speed_map = {
        "Slow (5s)": 5,
        "Normal (3s)": 3,
        "Fast (1s)": 1,
        "Maximum (0.5s)": 0.5
    }
    trade_interval = speed_map[speed]

    st.markdown("---")

    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("START", use_container_width=True, type="primary"):
            st.session_state.bot_running = True
            st.session_state.start_capital = initial_capital
            st.session_state.current_capital = initial_capital
            st.session_state.max_capital = initial_capital
            st.session_state.trades = []
            st.session_state.equity_curve = [{
                'timestamp': datetime.now(),
                'equity': initial_capital
            }]
            st.session_state.positions = {}
            st.session_state.trade_count = 0
            st.session_state.daily_returns = [0]
            for strategy in st.session_state.strategy_performance:
                st.session_state.strategy_performance[strategy] = {
                    'trades': 0, 'wins': 0, 'pnl': 0, 'returns': []
                }

    with col2:
        if st.button("STOP", use_container_width=True):
            st.session_state.bot_running = False

    st.markdown("---")

    # Status display
    if st.session_state.bot_running:
        st.markdown('<div class="status-badge status-running">RUNNING</div>', unsafe_allow_html=True)
        st.caption(f"Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")
    else:
        st.markdown('<div class="status-badge status-stopped">STOPPED</div>', unsafe_allow_html=True)

    # Quick stats in sidebar
    if len(st.session_state.trades) > 0:
        st.markdown("---")
        st.markdown("### Quick Stats")
        st.metric("Total Trades", st.session_state.trade_count)
        st.metric("Open Positions", len(st.session_state.positions))

# Main content
if not st.session_state.bot_running and len(st.session_state.trades) == 0:
    # Welcome screen
    st.info("Configure your trading bot in the sidebar and click START to begin")

    # Feature showcase
    col1, col2, col3, col4 = st.columns(4)

    features = [
        ("Automated Trading", "Multi-strategy execution\nReal-time decisions\n24/7 operation"),
        ("Advanced Analytics", "Sharpe & Sortino ratios\nDrawdown monitoring\nWin rate tracking"),
        ("Risk Management", "Position sizing\nStop-loss protection\nDiversification"),
        ("Multiple Strategies", "Momentum\nMean reversion\nTechnical indicators\nML-based")
    ]

    for col, (title, desc) in zip([col1, col2, col3, col4], features):
        with col:
            st.markdown(f"### {title}")
            for line in desc.split('\n'):
                st.markdown(f"- {line}")

    st.markdown("---")

    # Example chart
    st.subheader("Multi-Strategy Performance Comparison")

    dates = pd.date_range(start='2024-01-01', end='2025-12-27', freq='D')

    fig = go.Figure()

    strategies_colors = {
        'Momentum': '#FF6600',
        'RSI': '#16A34A',
        'MA Crossover': '#CA8A04',
        'Mean Reversion': '#DC2626'
    }

    for strategy, color in strategies_colors.items():
        returns = np.random.randn(len(dates)).cumsum() * 0.015 + 0.12
        equity = 100000 * (1 + returns)

        fig.add_trace(go.Scatter(
            x=dates,
            y=equity,
            mode='lines',
            name=strategy,
            line=dict(width=2, color=color),
            hovertemplate='%{y:$,.0f}<extra></extra>'
        ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font=dict(family='Inter, sans-serif', size=12, color='#1E293B')
    )

    st.plotly_chart(fig, use_container_width=True)

    # Sample metrics
    st.subheader("Performance Metrics")

    m1, m2, m3, m4, m5, m6 = st.columns(6)

    with m1:
        st.metric("Return", "+27.3%", "+27.3%")
    with m2:
        st.metric("Sharpe", "2.18", "+0.68")
    with m3:
        st.metric("Sortino", "3.24", "+1.24")
    with m4:
        st.metric("Max DD", "-6.2%", "-6.2%")
    with m5:
        st.metric("Win Rate", "63.7%", "+13.7%")
    with m6:
        st.metric("Profit Factor", "2.89", "+1.89")

else:
    # LIVE DASHBOARD

    # Generate trades
    if st.session_state.bot_running and np.random.random() < 0.35:
        generate_realistic_trade()

    # Calculate metrics
    metrics = calculate_advanced_metrics()

    # Performance metrics
    st.subheader("Performance Dashboard")

    p1, p2, p3, p4, p5, p6 = st.columns(6)

    with p1:
        delta_color = "normal" if metrics['total_return'] >= 0 else "inverse"
        st.metric(
            "Portfolio Value",
            f"${st.session_state.current_capital:,.0f}",
            delta=f"{metrics['total_return']:+.2f}%",
            delta_color=delta_color
        )

    with p2:
        st.metric(
            "Total Return",
            f"{metrics['total_return']:+.2f}%",
            delta=f"${metrics['total_return_abs']:+,.0f}",
            delta_color=delta_color
        )

    with p3:
        st.metric(
            "Sharpe Ratio",
            f"{metrics['sharpe_ratio']:.2f}",
            delta=f"{metrics['sharpe_ratio']:.2f}"
        )

    with p4:
        st.metric(
            "Sortino Ratio",
            f"{metrics['sortino_ratio']:.2f}",
            delta=f"{metrics['sortino_ratio']:.2f}"
        )

    with p5:
        st.metric(
            "Max Drawdown",
            f"{metrics['max_drawdown']:.2f}%",
            delta=f"{metrics['max_drawdown']:.2f}%",
            delta_color="inverse"
        )

    with p6:
        st.metric(
            "Win Rate",
            f"{metrics['win_rate']:.1f}%",
            delta=f"{metrics['win_rate'] - 50:+.1f}%"
        )

    st.markdown("---")

    # Charts
    chart_col1, chart_col2 = st.columns([2, 1])

    with chart_col1:
        st.subheader("Equity Curve")

        if len(st.session_state.equity_curve) > 0:
            equity_df = pd.DataFrame(st.session_state.equity_curve)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=equity_df['timestamp'],
                y=equity_df['equity'],
                mode='lines',
                fill='tozeroy',
                name='Portfolio Value',
                line=dict(color='#FF6600', width=2),
                fillcolor='rgba(255, 102, 0, 0.1)',
                hovertemplate='%{y:$,.0f}<extra></extra>'
            ))

            fig.add_hline(
                y=st.session_state.start_capital,
                line_dash="dash",
                line_color="#64748B",
                line_width=1,
                annotation_text="Start",
                annotation_position="right"
            )

            fig.add_hline(
                y=st.session_state.max_capital,
                line_dash="dot",
                line_color="#16A34A",
                line_width=1,
                annotation_text="ATH",
                annotation_position="right"
            )

            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Value ($)",
                hovermode='x unified',
                height=450,
                showlegend=False,
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FFFFFF',
                font=dict(family='Roboto Mono, monospace', size=11),
                margin=dict(l=0, r=0, t=10, b=0)
            )

            st.plotly_chart(fig, use_container_width=True, key="equity_chart")

    with chart_col2:
        st.subheader("Strategy Leaderboard")

        strategy_data = []
        for strategy, perf in st.session_state.strategy_performance.items():
            if perf['trades'] > 0:
                win_rate = (perf['wins'] / perf['trades']) * 100
                avg_return = np.mean(perf['returns']) * 100 if perf['returns'] else 0

                strategy_data.append({
                    'Strategy': strategy,
                    'P&L': perf['pnl'],
                    'Trades': perf['trades'],
                    'Win %': win_rate,
                    'Avg Return': avg_return
                })

        if strategy_data:
            strategy_df = pd.DataFrame(strategy_data).sort_values('P&L', ascending=False)

            fig = go.Figure()

            colors = ['#16A34A' if x > 0 else '#DC2626' for x in strategy_df['P&L']]

            fig.add_trace(go.Bar(
                x=strategy_df['Strategy'],
                y=strategy_df['P&L'],
                marker_color=colors,
                text=strategy_df['P&L'].apply(lambda x: f"${x:,.0f}"),
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>P&L: %{y:$,.0f}<extra></extra>'
            ))

            fig.update_layout(
                xaxis_title="",
                yaxis_title="P&L ($)",
                height=450,
                showlegend=False,
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FFFFFF',
                font=dict(family='Roboto Mono, monospace', size=11),
                margin=dict(l=0, r=0, t=10, b=0)
            )

            st.plotly_chart(fig, use_container_width=True, key="strategy_chart")

            st.dataframe(
                strategy_df[['Strategy', 'Trades', 'Win %', 'P&L']].style.format({
                    'Win %': '{:.1f}%',
                    'P&L': '${:,.0f}'
                }),
                use_container_width=True,
                hide_index=True
            )

    st.markdown("---")

    # Portfolio heatmap
    with st.expander("Portfolio Heatmap", expanded=False):
        if len(st.session_state.positions) > 0:
            heatmap_data = []
            for symbol, pos in st.session_state.positions.items():
                value = pos['quantity'] * pos['current_price']
                cost = pos['quantity'] * pos['avg_price']
                pnl_pct = ((value - cost) / cost) * 100 if cost > 0 else 0

                heatmap_data.append({
                    'Symbol': symbol,
                    'Value': value,
                    'P&L %': pnl_pct
                })

            heatmap_df = pd.DataFrame(heatmap_data)

            fig = px.treemap(
                heatmap_df,
                path=['Symbol'],
                values='Value',
                color='P&L %',
                color_continuous_scale=['#DC2626', '#CA8A04', '#16A34A'],
                color_continuous_midpoint=0,
                hover_data={'Value': ':$,.0f', 'P&L %': ':.2f'}
            )

            fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))

            st.plotly_chart(fig, use_container_width=True, key="heatmap")
        else:
            st.info("No positions to display")

    # Data tables
    table_col1, table_col2 = st.columns(2)

    with table_col1:
        st.subheader("Open Positions")

        if len(st.session_state.positions) > 0:
            pos_data = []
            for symbol, pos in st.session_state.positions.items():
                current_value = pos['quantity'] * pos['current_price']
                cost_basis = pos['quantity'] * pos['avg_price']
                unrealized_pnl = current_value - cost_basis
                unrealized_pnl_pct = (unrealized_pnl / cost_basis) * 100 if cost_basis > 0 else 0

                pos_data.append({
                    'Symbol': symbol,
                    'Qty': pos['quantity'],
                    'Avg': f"${pos['avg_price']:.2f}",
                    'Current': f"${pos['current_price']:.2f}",
                    'Value': f"${current_value:,.0f}",
                    'P&L': unrealized_pnl,
                    'P&L %': f"{unrealized_pnl_pct:+.2f}%"
                })

            pos_df = pd.DataFrame(pos_data)

            def highlight_pnl(row):
                pnl = row['P&L']
                color = '#D1FAE5' if pnl > 0 else '#FEE2E2' if pnl < 0 else ''
                return ['background-color: ' + color] * len(row)

            styled_pos = pos_df.style.apply(highlight_pnl, axis=1).format({
                'P&L': '${:,.0f}'
            })

            st.dataframe(styled_pos, use_container_width=True, hide_index=True, key="positions_table")
        else:
            st.info("No open positions")

    with table_col2:
        st.subheader("Recent Trades")

        if len(st.session_state.trades) > 0:
            recent = pd.DataFrame(st.session_state.trades).tail(10)
            recent['Time'] = pd.to_datetime(recent['timestamp']).dt.strftime('%H:%M:%S')

            trade_display = recent[['Time', 'symbol', 'side', 'quantity', 'price', 'strategy', 'pnl']].copy()
            trade_display.columns = ['Time', 'Symbol', 'Side', 'Qty', 'Price', 'Strategy', 'P&L']
            trade_display['Price'] = trade_display['Price'].apply(lambda x: f"${x:.2f}")
            trade_display['P&L'] = trade_display['P&L'].apply(lambda x: f"${x:,.0f}")

            def color_side(row):
                colors = [''] * len(row)
                if row['Side'] == 'BUY':
                    colors[2] = 'color: #16A34A; font-weight: bold'
                else:
                    colors[2] = 'color: #DC2626; font-weight: bold'
                return colors

            styled_trades = trade_display.style.apply(color_side, axis=1)

            st.dataframe(styled_trades, use_container_width=True, hide_index=True, key="trades_table")

            csv = pd.DataFrame(st.session_state.trades).to_csv(index=False)
            st.download_button(
                "Download Full History",
                csv,
                f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.info("No trades yet")

    # Advanced metrics
    with st.expander("Advanced Metrics", expanded=False):
        adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)

        with adv_col1:
            st.metric("Avg Win", f"${metrics['avg_win']:,.0f}")
            st.metric("Best Trade", f"${metrics['best_trade']:,.0f}")

        with adv_col2:
            st.metric("Avg Loss", f"${metrics['avg_loss']:,.0f}")
            st.metric("Worst Trade", f"${metrics['worst_trade']:,.0f}")

        with adv_col3:
            st.metric("Avg Trade", f"${metrics['avg_trade']:,.0f}")
            st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")

        with adv_col4:
            st.metric("Win Streak", metrics['consecutive_wins'])
            st.metric("Loss Streak", metrics['consecutive_losses'])

    # Auto-refresh
    if st.session_state.bot_running:
        time.sleep(trade_interval)
        st.rerun()

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 2.5rem; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; margin-top: 2rem;">
    <p style="font-size: 1.2rem; font-weight: 700; margin-bottom: 0.8rem; color: #FF6600;">
        QUANTEDGE PRO
    </p>
    <p style="margin-bottom: 1.5rem; font-size: 0.95rem; color: #64748B;">
        Algorithmic Trading Platform
    </p>
    <div style="margin: 1.5rem 0; display: flex; justify-content: center; flex-wrap: wrap; gap: 8px;">
        <span class="feature-pill">100% Paper Trading</span>
        <span class="feature-pill">Risk-Free Testing</span>
        <span class="feature-pill">Real-Time Analytics</span>
        <span class="feature-pill">Multi-Strategy</span>
        <span class="feature-pill">Advanced Metrics</span>
    </div>
    <p style="font-size: 0.85rem; margin-top: 1.5rem; color: #64748B;">
        Demo uses simulated data. For educational purposes only. Trading involves substantial risk.
    </p>
    <p style="font-size: 0.75rem; margin-top: 0.8rem; color: #94A3B8;">
        Powered by Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
