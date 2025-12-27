"""
Project Ketamine - ULTRA Web Demo
INSANE Interactive Paper Trading Demo with Smooth Data Updates

Features:
- Professional trading terminal UI with animations
- Real-time data updates (NO page refresh flicker!)
- Advanced performance metrics & Sharpe ratio
- Dark mode toggle
- Portfolio heatmap
- Strategy comparison
- Performance leaderboard
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
    page_title="Project Ketamine - Professional Trading Demo",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with animations
st.markdown("""
<style>
    /* Hide Streamlit elements for cleaner UI */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}

    /* Professional color palette */
    :root {
        --primary: #667eea;
        --secondary: #764ba2;
        --success: #10b981;
        --danger: #ef4444;
        --warning: #f59e0b;
        --dark: #1e293b;
        --light: #f8fafc;
    }

    /* Smooth transitions for all elements */
    * {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Main header with gradient animation */
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        animation: gradientShift 3s ease infinite, fadeIn 1s ease-in;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #64748b;
        margin-bottom: 2rem;
        animation: fadeIn 1.5s ease-in;
    }

    /* Metric cards with hover effects */
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border: 2px solid rgba(102, 126, 234, 0.2);
        margin: 0.5rem 0;
        animation: slideInUp 0.5s ease-out;
    }

    .metric-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.25);
        border-color: #667eea;
    }

    /* Live indicator with pulse */
    .live-indicator {
        display: inline-block;
        width: 14px;
        height: 14px;
        background: radial-gradient(circle, #10b981 0%, #059669 100%);
        border-radius: 50%;
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        margin-right: 10px;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    }

    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        }
        50% {
            transform: scale(1.15);
            box-shadow: 0 0 0 10px rgba(16, 185, 129, 0);
        }
    }

    /* Fade in animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Slide in up */
    @keyframes slideInUp {
        from {
            transform: translateY(30px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }

    /* Status badge with glow */
    .status-badge {
        display: inline-block;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin: 0.25rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    .status-running {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.4);
        animation: glow 2s ease-in-out infinite;
    }

    @keyframes glow {
        0%, 100% { box-shadow: 0 4px 20px rgba(16, 185, 129, 0.4); }
        50% { box-shadow: 0 4px 30px rgba(16, 185, 129, 0.7); }
    }

    .status-stopped {
        background: linear-gradient(135deg, #64748b 0%, #475569 100%);
        color: white;
    }

    /* Feature pill */
    .feature-pill {
        display: inline-block;
        background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
        color: #4338ca;
        padding: 0.6rem 1.2rem;
        border-radius: 25px;
        font-size: 0.9rem;
        margin: 0.3rem;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(67, 56, 202, 0.15);
    }

    .feature-pill:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(67, 56, 202, 0.25);
    }

    /* Chart container with smooth entry */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
        animation: fadeIn 0.6s ease-out;
    }

    /* Trade notification toast */
    .trade-toast {
        position: fixed;
        top: 80px;
        right: 20px;
        background: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 5px solid var(--success);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        animation: slideInRight 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        z-index: 9999;
        min-width: 300px;
    }

    @keyframes slideInRight {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    /* Performance indicators */
    .perf-positive {
        color: var(--success);
        font-weight: 800;
        text-shadow: 0 0 10px rgba(16, 185, 129, 0.3);
    }

    .perf-negative {
        color: var(--danger);
        font-weight: 800;
        text-shadow: 0 0 10px rgba(239, 68, 68, 0.3);
    }

    /* Smooth data update (no flicker) */
    .element-container {
        animation: none !important;
    }

    /* Dark mode support */
    .dark-mode {
        background: #0f172a;
        color: #f8fafc;
    }

    /* Loading skeleton */
    .skeleton {
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 200% 100%;
        animation: loading 1.5s ease-in-out infinite;
    }

    @keyframes loading {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }

    /* Leaderboard item */
    .leaderboard-item {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.6rem 0;
        border-left: 5px solid var(--primary);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }

    .leaderboard-item:hover {
        transform: translateX(8px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.2);
    }

    /* Heatmap cell */
    .heatmap-cell {
        padding: 1rem;
        text-align: center;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .heatmap-cell:hover {
        transform: scale(1.1);
        z-index: 10;
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
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False


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
st.markdown('<div class="main-header">🚀 PROJECT KETAMINE</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Institutional-Grade Algorithmic Trading Platform</div>', unsafe_allow_html=True)

# Live status indicator
if st.session_state.bot_running:
    st.markdown('''
    <div style="text-align: center; margin-bottom: 2rem;">
        <span class="live-indicator"></span>
        <span style="font-weight: 700; color: #10b981; font-size: 1.1rem; letter-spacing: 1px;">LIVE TRADING</span>
    </div>
    ''', unsafe_allow_html=True)

st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    # Dark mode toggle
    dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    st.session_state.dark_mode = dark_mode

    st.subheader("💰 Capital")
    initial_capital = st.number_input(
        "Initial Capital ($)",
        min_value=10000,
        max_value=10000000,
        value=100000,
        step=10000
    )

    st.subheader("🎯 Strategies")
    selected_strategies = st.multiselect(
        "Active Strategies",
        ["Momentum", "RSI", "MA Crossover", "Mean Reversion"],
        default=["Momentum", "RSI", "MA Crossover"]
    )

    st.subheader("📈 Assets")
    assets = st.multiselect(
        "Trading Universe",
        ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "NFLX", "ORCL", "AMD", "SPY", "QQQ"],
        default=["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"]
    )

    st.subheader("🛡️ Risk Management")
    max_position = st.slider("Max Position Size (%)", 5, 50, 20, 5)
    stop_loss = st.slider("Stop Loss (%)", 1, 25, 10, 1)
    take_profit = st.slider("Take Profit (%)", 5, 100, 25, 5)

    st.subheader("⚡ Trading Speed")
    speed = st.select_slider(
        "Update Frequency",
        options=["Slow (5s)", "Normal (3s)", "Fast (1s)", "Ludicrous (0.5s)"],
        value="Normal (3s)"
    )

    speed_map = {
        "Slow (5s)": 5,
        "Normal (3s)": 3,
        "Fast (1s)": 1,
        "Ludicrous (0.5s)": 0.5
    }
    trade_interval = speed_map[speed]

    st.markdown("---")

    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ START", use_container_width=True, type="primary"):
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
            st.balloons()

    with col2:
        if st.button("⏹️ STOP", use_container_width=True):
            st.session_state.bot_running = False

    st.markdown("---")

    # Status display
    if st.session_state.bot_running:
        st.markdown('<div class="status-badge status-running">🟢 RUNNING</div>', unsafe_allow_html=True)
        st.caption(f"Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")
    else:
        st.markdown('<div class="status-badge status-stopped">⚫ STOPPED</div>', unsafe_allow_html=True)

    # Quick stats in sidebar
    if len(st.session_state.trades) > 0:
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        st.metric("Total Trades", st.session_state.trade_count)
        st.metric("Open Positions", len(st.session_state.positions))

# Main content
if not st.session_state.bot_running and len(st.session_state.trades) == 0:
    # Welcome screen
    st.info("👈 **Configure your bot** and click **START** to begin trading!")

    # Feature showcase
    col1, col2, col3, col4 = st.columns(4)

    features = [
        ("⚡ Automated Trading", "Multi-strategy execution\nReal-time decisions\n24/7 operation"),
        ("📊 Advanced Analytics", "Sharpe & Sortino ratios\nDrawdown monitoring\nWin rate tracking"),
        ("🛡️ Risk Management", "Position sizing\nStop-loss protection\nDiversification"),
        ("🎯 Multiple Strategies", "Momentum\nMean reversion\nTechnical indicators\nML-based")
    ]

    for col, (title, desc) in zip([col1, col2, col3, col4], features):
        with col:
            st.markdown(f"### {title}")
            for line in desc.split('\n'):
                st.markdown(f"- {line}")

    st.markdown("---")

    # Example chart
    st.subheader("📈 Example: Multi-Strategy Performance")

    dates = pd.date_range(start='2024-01-01', end='2024-12-27', freq='D')

    fig = go.Figure()

    strategies_colors = {
        'Momentum': '#667eea',
        'RSI': '#10b981',
        'MA Crossover': '#f59e0b',
        'Mean Reversion': '#ef4444'
    }

    for strategy, color in strategies_colors.items():
        # Generate realistic equity curve
        returns = np.random.randn(len(dates)).cumsum() * 0.015 + 0.12
        equity = 100000 * (1 + returns)

        fig.add_trace(go.Scatter(
            x=dates,
            y=equity,
            mode='lines',
            name=strategy,
            line=dict(width=3, color=color),
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
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Sample metrics
    st.subheader("📊 Example Performance Metrics")

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

    # Generate trades (smooth, no full page refresh)
    if st.session_state.bot_running and np.random.random() < 0.35:
        generate_realistic_trade()

    # Calculate metrics
    metrics = calculate_advanced_metrics()

    # Create placeholder containers for smooth updates
    perf_container = st.container()
    charts_container = st.container()
    data_container = st.container()

    # Performance metrics (updates smoothly)
    with perf_container:
        st.subheader("📊 Performance Dashboard")

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
    with charts_container:
        chart_col1, chart_col2 = st.columns([2, 1])

        with chart_col1:
            st.subheader("📈 Equity Curve")

            if len(st.session_state.equity_curve) > 0:
                equity_df = pd.DataFrame(st.session_state.equity_curve)

                fig = go.Figure()

                # Main equity line
                fig.add_trace(go.Scatter(
                    x=equity_df['timestamp'],
                    y=equity_df['equity'],
                    mode='lines',
                    fill='tozeroy',
                    name='Portfolio Value',
                    line=dict(color='#667eea', width=3),
                    fillcolor='rgba(102, 126, 234, 0.15)',
                    hovertemplate='%{y:$,.0f}<extra></extra>'
                ))

                # Starting capital
                fig.add_hline(
                    y=st.session_state.start_capital,
                    line_dash="dash",
                    line_color="#94a3b8",
                    line_width=2,
                    annotation_text="Start",
                    annotation_position="right"
                )

                # All-time high
                fig.add_hline(
                    y=st.session_state.max_capital,
                    line_dash="dot",
                    line_color="#10b981",
                    line_width=2,
                    annotation_text="ATH",
                    annotation_position="right"
                )

                fig.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Value ($)",
                    hovermode='x unified',
                    height=450,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=10, b=0)
                )

                st.plotly_chart(fig, use_container_width=True, key="equity_chart")

        with chart_col2:
            st.subheader("🏆 Strategy Leaderboard")

            # Strategy ranking
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

                # Create bar chart
                fig = go.Figure()

                colors = ['#10b981' if x > 0 else '#ef4444' for x in strategy_df['P&L']]

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
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=10, b=0)
                )

                st.plotly_chart(fig, use_container_width=True, key="strategy_chart")

                # Strategy table
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
    with st.expander("🗺️ Portfolio Heatmap", expanded=False):
        if len(st.session_state.positions) > 0:
            # Create heatmap data
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

            # Create treemap
            fig = px.treemap(
                heatmap_df,
                path=['Symbol'],
                values='Value',
                color='P&L %',
                color_continuous_scale=['#ef4444', '#f59e0b', '#10b981'],
                color_continuous_midpoint=0,
                hover_data={'Value': ':$,.0f', 'P&L %': ':.2f'}
            )

            fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))

            st.plotly_chart(fig, use_container_width=True, key="heatmap")
        else:
            st.info("No positions to display")

    # Data tables
    with data_container:
        table_col1, table_col2 = st.columns(2)

        with table_col1:
            st.subheader("📋 Open Positions")

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

                # Style the table
                def highlight_pnl(row):
                    pnl = row['P&L']
                    color = '#d1fae5' if pnl > 0 else '#fee2e2' if pnl < 0 else ''
                    return ['background-color: ' + color] * len(row)

                styled_pos = pos_df.style.apply(highlight_pnl, axis=1).format({
                    'P&L': '${:,.0f}'
                })

                st.dataframe(styled_pos, use_container_width=True, hide_index=True, key="positions_table")
            else:
                st.info("No open positions")

        with table_col2:
            st.subheader("💹 Recent Trades")

            if len(st.session_state.trades) > 0:
                recent = pd.DataFrame(st.session_state.trades).tail(10)
                recent['Time'] = pd.to_datetime(recent['timestamp']).dt.strftime('%H:%M:%S')

                trade_display = recent[['Time', 'symbol', 'side', 'quantity', 'price', 'strategy', 'pnl']].copy()
                trade_display.columns = ['Time', 'Symbol', 'Side', 'Qty', 'Price', 'Strategy', 'P&L']
                trade_display['Price'] = trade_display['Price'].apply(lambda x: f"${x:.2f}")
                trade_display['P&L'] = trade_display['P&L'].apply(lambda x: f"${x:,.0f}")

                # Style trades
                def color_side(row):
                    colors = [''] * len(row)
                    if row['Side'] == 'BUY':
                        colors[2] = 'color: #10b981; font-weight: bold'
                    else:
                        colors[2] = 'color: #ef4444; font-weight: bold'
                    return colors

                styled_trades = trade_display.style.apply(color_side, axis=1)

                st.dataframe(styled_trades, use_container_width=True, hide_index=True, key="trades_table")

                # Download button
                csv = pd.DataFrame(st.session_state.trades).to_csv(index=False)
                st.download_button(
                    "📥 Download Full History",
                    csv,
                    f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
            else:
                st.info("No trades yet")

    # Advanced metrics
    with st.expander("📈 Advanced Metrics", expanded=False):
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

    # Auto-refresh with smooth transition
    if st.session_state.bot_running:
        time.sleep(trade_interval)
        st.rerun()

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 2.5rem; background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%); border-radius: 16px; margin-top: 2rem;">
    <p style="font-size: 1.3rem; font-weight: 700; margin-bottom: 0.8rem;">
        <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            PROJECT KETAMINE
        </span>
    </p>
    <p style="margin-bottom: 1.5rem; font-size: 1.05rem; color: #64748b;">
        Institutional-Grade Algorithmic Trading Platform
    </p>
    <div style="margin: 1.5rem 0; display: flex; justify-content: center; flex-wrap: wrap; gap: 0.5rem;">
        <span class="feature-pill">✅ 100% Paper Trading</span>
        <span class="feature-pill">✅ Risk-Free Testing</span>
        <span class="feature-pill">✅ Real-Time Analytics</span>
        <span class="feature-pill">✅ Multi-Strategy</span>
        <span class="feature-pill">✅ Advanced Metrics</span>
    </div>
    <p style="font-size: 0.9rem; margin-top: 1.5rem; color: #64748b;">
        ⚠️ Demo uses simulated data • Educational purposes only • Trading involves risk
    </p>
    <p style="font-size: 0.8rem; margin-top: 0.8rem; opacity: 0.6;">
        Powered by Streamlit • Built with 💜
    </p>
</div>
""", unsafe_allow_html=True)
