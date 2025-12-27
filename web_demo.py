"""
QuantEdge Pro - Professional Trading Platform
Institutional-Grade Paper Trading Demonstration

Real-time algorithmic trading simulation with advanced analytics
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
    page_title="QuantEdge Pro",
    page_icon="■",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Terminal CSS - NO EMOJIS, NO CARTOON STUFF
st.markdown("""
<style>
    /* Hide all Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    header {visibility: hidden;}

    /* Professional terminal colors */
    :root {
        --terminal-green: #00FF41;
        --terminal-red: #FF3B30;
        --terminal-orange: #FF9500;
        --terminal-blue: #007AFF;
        --bg-dark: #0A0E27;
        --bg-darker: #050814;
        --text-primary: #E8EAED;
        --text-secondary: #9AA0A6;
        --border-color: #202940;
        --success: #00C853;
        --danger: #FF1744;
    }

    /* Main app background */
    .stApp {
        background: var(--bg-dark);
        color: var(--text-primary);
    }

    /* Remove ALL animations */
    * {
        transition: none !important;
        animation: none !important;
    }

    /* Header - clean, professional */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--terminal-orange);
        text-align: left;
        margin-bottom: 0.3rem;
        letter-spacing: 2px;
        font-family: 'Courier New', monospace;
        text-transform: uppercase;
        border-left: 4px solid var(--terminal-orange);
        padding-left: 16px;
    }

    .subtitle {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-bottom: 1.5rem;
        font-family: 'Courier New', monospace;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding-left: 20px;
    }

    /* Live indicator - simple dot */
    .live-status {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: var(--terminal-green);
        margin-right: 8px;
    }

    /* Status badges - terminal style */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        background: transparent;
        border: 1px solid var(--border-color);
        font-family: 'Courier New', monospace;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .status-running {
        border-color: var(--terminal-green);
        color: var(--terminal-green);
    }

    .status-stopped {
        border-color: var(--text-secondary);
        color: var(--text-secondary);
    }

    /* Buttons - terminal style, NO GRADIENTS */
    .stButton > button {
        background: var(--bg-darker);
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        font-family: 'Courier New', monospace;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        padding: 8px 20px;
    }

    .stButton > button:hover {
        background: var(--border-color);
        border-color: var(--terminal-orange);
    }

    .stButton > button[kind="primary"] {
        background: var(--terminal-green);
        border-color: var(--terminal-green);
        color: var(--bg-dark);
    }

    .stButton > button[kind="primary"]:hover {
        background: #00FF41;
        border-color: #00FF41;
    }

    /* Metrics - terminal display */
    [data-testid="stMetricValue"] {
        font-family: 'Courier New', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    [data-testid="stMetricLabel"] {
        font-family: 'Courier New', monospace;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--text-secondary);
    }

    /* Section headers */
    h3 {
        color: var(--terminal-orange);
        font-family: 'Courier New', monospace;
        font-weight: 700;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 8px;
        margin-top: 1rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--bg-darker);
        border-right: 1px solid var(--border-color);
    }

    [data-testid="stSidebar"] h2 {
        color: var(--terminal-orange);
        font-family: 'Courier New', monospace;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    [data-testid="stSidebar"] h3 {
        color: var(--text-primary);
        font-size: 0.85rem;
        border: none;
    }

    /* Input fields */
    input, select, .stSelectbox, .stMultiSelect {
        background: var(--bg-darker) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        font-family: 'Courier New', monospace !important;
    }

    /* Data tables - terminal style */
    .dataframe {
        background: var(--bg-darker);
        border: 1px solid var(--border-color);
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
        color: var(--text-primary);
    }

    table {
        background: var(--bg-darker);
        border: 1px solid var(--border-color);
    }

    th {
        background: var(--bg-dark);
        color: var(--terminal-orange);
        font-family: 'Courier New', monospace;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.7rem;
        letter-spacing: 1px;
        padding: 10px;
        border-bottom: 2px solid var(--border-color);
    }

    td {
        padding: 8px 10px;
        border-bottom: 1px solid var(--border-color);
        color: var(--text-primary);
    }

    /* Stock performance card */
    .stock-card {
        background: var(--bg-darker);
        border: 1px solid var(--border-color);
        border-left: 3px solid var(--terminal-green);
        padding: 10px;
        margin: 6px 0;
        font-family: 'Courier New', monospace;
    }

    .stock-card.negative {
        border-left-color: var(--terminal-red);
    }

    .stock-ticker {
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: 1px;
    }

    .stock-pnl {
        font-size: 1.1rem;
        font-weight: 700;
        margin-top: 4px;
    }

    .stock-pnl.positive {
        color: var(--terminal-green);
    }

    .stock-pnl.negative {
        color: var(--terminal-red);
    }

    /* Info boxes */
    .stAlert {
        background: var(--bg-darker);
        border: 1px solid var(--border-color);
        color: var(--text-primary);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: var(--bg-darker);
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        font-family: 'Courier New', monospace;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.8rem;
    }

    /* Remove all rounded corners */
    * {
        border-radius: 0 !important;
    }

    /* Horizontal rule */
    hr {
        border: none;
        border-top: 1px solid var(--border-color);
        margin: 1.5rem 0;
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
if 'stock_performance' not in st.session_state:
    st.session_state.stock_performance = {}


def generate_realistic_trade():
    """Generate realistic trade"""
    np.random.seed(int(time.time() * 1000) % 2**32)

    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX', 'ORCL', 'AMD']
    strategies = list(st.session_state.strategy_performance.keys())

    symbol = np.random.choice(symbols)
    strategy = np.random.choice(strategies)
    side = np.random.choice(['BUY', 'SELL'], p=[0.55, 0.45])
    quantity = np.random.randint(5, 150)
    price = np.random.uniform(80, 900)

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

    # Update stock performance tracking
    if symbol not in st.session_state.stock_performance:
        st.session_state.stock_performance[symbol] = {
            'total_pnl': 0,
            'trade_count': 0,
            'last_pnl': 0,
            'position': 0
        }

    st.session_state.stock_performance[symbol]['total_pnl'] += pnl
    st.session_state.stock_performance[symbol]['trade_count'] += 1
    st.session_state.stock_performance[symbol]['last_pnl'] = pnl

    if side == 'BUY':
        st.session_state.stock_performance[symbol]['position'] += quantity
    else:
        st.session_state.stock_performance[symbol]['position'] -= quantity

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
            st.session_state.positions[symbol]['last_update'] = datetime.now()
        else:
            st.session_state.positions[symbol] = {
                'quantity': quantity,
                'avg_price': price,
                'current_price': price * (1 + np.random.normal(0, 0.015)),
                'last_update': datetime.now()
            }
    else:
        if symbol in st.session_state.positions:
            st.session_state.positions[symbol]['quantity'] -= quantity
            st.session_state.positions[symbol]['last_update'] = datetime.now()
            if st.session_state.positions[symbol]['quantity'] <= 0:
                del st.session_state.positions[symbol]

    # Update position prices
    for sym in st.session_state.positions:
        price_change = np.random.normal(0, 0.008)
        st.session_state.positions[sym]['current_price'] *= (1 + price_change)

    # Update daily returns
    daily_return = (st.session_state.current_capital / st.session_state.start_capital - 1) * 100
    st.session_state.daily_returns.append(daily_return)

    st.session_state.last_update = datetime.now()


def calculate_advanced_metrics():
    """Calculate performance metrics"""
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
        }

    trades_df = pd.DataFrame(st.session_state.trades)

    total_return_abs = st.session_state.current_capital - st.session_state.start_capital
    total_return = (total_return_abs / st.session_state.start_capital) * 100

    # Sharpe & Sortino
    if len(st.session_state.daily_returns) > 1:
        returns_array = np.array(st.session_state.daily_returns)
        returns_mean = np.mean(returns_array)
        returns_std = np.std(returns_array)
        sharpe = (returns_mean / returns_std * np.sqrt(252)) if returns_std > 0 else 0

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

    wins = trades_df['is_win'].sum()
    win_rate = (wins / len(trades_df)) * 100 if len(trades_df) > 0 else 0

    total_wins = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    total_losses = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
    avg_trade = trades_df['pnl'].mean()

    best_trade = trades_df['pnl'].max()
    worst_trade = trades_df['pnl'].min()

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
    }


# Header
st.markdown('<div class="main-header">QUANTEDGE PRO</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">ALGORITHMIC TRADING PLATFORM</div>', unsafe_allow_html=True)

# Live status
if st.session_state.bot_running:
    st.markdown('''
    <div style="margin-bottom: 1.5rem; font-family: 'Courier New', monospace; font-size: 0.85rem;">
        <span class="live-status"></span>
        <span style="color: #00FF41; font-weight: 700; letter-spacing: 1px;">LIVE TRADING</span>
        <span style="color: #9AA0A6; margin-left: 20px;">LAST UPDATE: {}</span>
    </div>
    '''.format(st.session_state.last_update.strftime('%H:%M:%S')), unsafe_allow_html=True)

st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("CONFIGURATION")

    st.subheader("CAPITAL")
    initial_capital = st.number_input(
        "Initial Capital ($)",
        min_value=10000,
        max_value=10000000,
        value=100000,
        step=10000
    )

    st.subheader("STRATEGIES")
    selected_strategies = st.multiselect(
        "Active Strategies",
        ["Momentum", "RSI", "MA Crossover", "Mean Reversion"],
        default=["Momentum", "RSI", "MA Crossover"]
    )

    st.subheader("ASSETS")
    assets = st.multiselect(
        "Trading Universe",
        ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "NFLX", "ORCL", "AMD", "SPY", "QQQ"],
        default=["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"]
    )

    st.subheader("RISK PARAMETERS")
    max_position = st.slider("Max Position Size (%)", 5, 50, 20, 5)
    stop_loss = st.slider("Stop Loss (%)", 1, 25, 10, 1)

    st.subheader("EXECUTION SPEED")
    speed = st.select_slider(
        "Update Frequency",
        options=["SLOW (5s)", "NORMAL (3s)", "FAST (1s)", "MAXIMUM (0.5s)"],
        value="NORMAL (3s)"
    )

    speed_map = {
        "SLOW (5s)": 5,
        "NORMAL (3s)": 3,
        "FAST (1s)": 1,
        "MAXIMUM (0.5s)": 0.5
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
            st.session_state.equity_curve = [{'timestamp': datetime.now(), 'equity': initial_capital}]
            st.session_state.positions = {}
            st.session_state.trade_count = 0
            st.session_state.daily_returns = [0]
            st.session_state.stock_performance = {}
            for strategy in st.session_state.strategy_performance:
                st.session_state.strategy_performance[strategy] = {
                    'trades': 0, 'wins': 0, 'pnl': 0, 'returns': []
                }

    with col2:
        if st.button("STOP", use_container_width=True):
            st.session_state.bot_running = False

    st.markdown("---")

    # Status
    if st.session_state.bot_running:
        st.markdown('<div class="status-badge status-running">RUNNING</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge status-stopped">STOPPED</div>', unsafe_allow_html=True)

    if len(st.session_state.trades) > 0:
        st.markdown("---")
        st.markdown("### STATISTICS")
        st.metric("TOTAL TRADES", st.session_state.trade_count)
        st.metric("OPEN POSITIONS", len(st.session_state.positions))

# Main content
if not st.session_state.bot_running and len(st.session_state.trades) == 0:
    # Welcome screen
    st.info("Configure trading parameters in sidebar and click START to begin simulation")

    # Features
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### AUTOMATED EXECUTION")
        st.markdown("- Multi-strategy aggregation")
        st.markdown("- Real-time decision engine")
        st.markdown("- 24/7 operation capability")

    with col2:
        st.markdown("### RISK ANALYTICS")
        st.markdown("- Sharpe & Sortino ratios")
        st.markdown("- Drawdown monitoring")
        st.markdown("- Position-level risk tracking")

    with col3:
        st.markdown("### PERFORMANCE TRACKING")
        st.markdown("- Stock-level P&L")
        st.markdown("- Strategy comparison")
        st.markdown("- Real-time equity curve")

    st.markdown("---")

    # Example chart
    st.subheader("STRATEGY PERFORMANCE COMPARISON")

    dates = pd.date_range(start='2024-01-01', end='2025-12-27', freq='D')
    fig = go.Figure()

    strategies_colors = {
        'Momentum': '#FF9500',
        'RSI': '#00FF41',
        'MA Crossover': '#007AFF',
        'Mean Reversion': '#FF3B30'
    }

    for strategy, color in strategies_colors.items():
        returns = np.random.randn(len(dates)).cumsum() * 0.015 + 0.12
        equity = 100000 * (1 + returns)

        fig.add_trace(go.Scatter(
            x=dates,
            y=equity,
            mode='lines',
            name=strategy,
            line=dict(width=1.5, color=color),
            hovertemplate='%{y:$,.0f}<extra></extra>'
        ))

    fig.update_layout(
        xaxis_title="DATE",
        yaxis_title="PORTFOLIO VALUE ($)",
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='#0A0E27',
        paper_bgcolor='#0A0E27',
        font=dict(family='Courier New, monospace', size=10, color='#E8EAED')
    )

    st.plotly_chart(fig, use_container_width=True)

    # Sample metrics
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("RETURN", "+27.3%")
    with m2:
        st.metric("SHARPE", "2.18")
    with m3:
        st.metric("SORTINO", "3.24")
    with m4:
        st.metric("MAX DD", "-6.2%")
    with m5:
        st.metric("WIN RATE", "63.7%")
    with m6:
        st.metric("PROFIT FACTOR", "2.89")

else:
    # LIVE DASHBOARD

    # Generate trades
    if st.session_state.bot_running and np.random.random() < 0.35:
        generate_realistic_trade()

    # Calculate metrics
    metrics = calculate_advanced_metrics()

    # Performance metrics
    st.subheader("PERFORMANCE DASHBOARD")

    p1, p2, p3, p4, p5, p6 = st.columns(6)

    with p1:
        delta_color = "normal" if metrics['total_return'] >= 0 else "inverse"
        st.metric(
            "PORTFOLIO VALUE",
            f"${st.session_state.current_capital:,.0f}",
            delta=f"{metrics['total_return']:+.2f}%",
            delta_color=delta_color
        )

    with p2:
        st.metric(
            "TOTAL RETURN",
            f"{metrics['total_return']:+.2f}%",
            delta=f"${metrics['total_return_abs']:+,.0f}",
            delta_color=delta_color
        )

    with p3:
        st.metric("SHARPE RATIO", f"{metrics['sharpe_ratio']:.2f}")

    with p4:
        st.metric("SORTINO RATIO", f"{metrics['sortino_ratio']:.2f}")

    with p5:
        st.metric("MAX DRAWDOWN", f"{metrics['max_drawdown']:.2f}%", delta_color="inverse")

    with p6:
        st.metric("WIN RATE", f"{metrics['win_rate']:.1f}%")

    st.markdown("---")

    # Main content area - 3 columns
    main_col1, main_col2, main_col3 = st.columns([3, 2, 2])

    with main_col1:
        st.subheader("EQUITY CURVE")

        if len(st.session_state.equity_curve) > 0:
            equity_df = pd.DataFrame(st.session_state.equity_curve)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=equity_df['timestamp'],
                y=equity_df['equity'],
                mode='lines',
                fill='tozeroy',
                name='Portfolio Value',
                line=dict(color='#FF9500', width=2),
                fillcolor='rgba(255, 149, 0, 0.1)',
                hovertemplate='%{y:$,.0f}<extra></extra>'
            ))

            fig.add_hline(
                y=st.session_state.start_capital,
                line_dash="dash",
                line_color="#9AA0A6",
                line_width=1,
                annotation_text="START",
                annotation_position="right"
            )

            fig.add_hline(
                y=st.session_state.max_capital,
                line_dash="dot",
                line_color="#00FF41",
                line_width=1,
                annotation_text="ATH",
                annotation_position="right"
            )

            fig.update_layout(
                xaxis_title="TIME",
                yaxis_title="VALUE ($)",
                hovermode='x unified',
                height=400,
                showlegend=False,
                plot_bgcolor='#0A0E27',
                paper_bgcolor='#0A0E27',
                font=dict(family='Courier New, monospace', size=9, color='#E8EAED'),
                margin=dict(l=0, r=0, t=10, b=0)
            )

            st.plotly_chart(fig, use_container_width=True, key="equity_chart")

    with main_col2:
        st.subheader("STRATEGY P&L")

        strategy_data = []
        for strategy, perf in st.session_state.strategy_performance.items():
            if perf['trades'] > 0:
                strategy_data.append({
                    'Strategy': strategy,
                    'P&L': perf['pnl'],
                    'Trades': perf['trades'],
                })

        if strategy_data:
            strategy_df = pd.DataFrame(strategy_data).sort_values('P&L', ascending=False)

            fig = go.Figure()

            colors = ['#00FF41' if x > 0 else '#FF3B30' for x in strategy_df['P&L']]

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
                height=400,
                showlegend=False,
                plot_bgcolor='#0A0E27',
                paper_bgcolor='#0A0E27',
                font=dict(family='Courier New, monospace', size=9, color='#E8EAED'),
                margin=dict(l=0, r=0, t=10, b=0)
            )

            st.plotly_chart(fig, use_container_width=True, key="strategy_chart")

    with main_col3:
        st.subheader("STOCK PERFORMANCE")

        # Sort stocks by total P&L
        sorted_stocks = sorted(
            st.session_state.stock_performance.items(),
            key=lambda x: x[1]['total_pnl'],
            reverse=True
        )

        if sorted_stocks:
            for symbol, perf in sorted_stocks[:10]:  # Top 10
                pnl = perf['total_pnl']
                trades = perf['trade_count']
                position = perf['position']

                pnl_class = "positive" if pnl > 0 else "negative"
                card_class = "stock-card" if pnl > 0 else "stock-card negative"

                st.markdown(f'''
                <div class="{card_class}">
                    <div class="stock-ticker">{symbol}</div>
                    <div class="stock-pnl {pnl_class}">${pnl:+,.2f}</div>
                    <div style="font-size: 0.75rem; color: #9AA0A6; margin-top: 4px;">
                        TRADES: {trades} | POS: {position}
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.info("No stock data yet")

    st.markdown("---")

    # Data tables
    table_col1, table_col2 = st.columns(2)

    with table_col1:
        st.subheader("OPEN POSITIONS")

        if len(st.session_state.positions) > 0:
            pos_data = []
            for symbol, pos in st.session_state.positions.items():
                current_value = pos['quantity'] * pos['current_price']
                cost_basis = pos['quantity'] * pos['avg_price']
                unrealized_pnl = current_value - cost_basis
                unrealized_pnl_pct = (unrealized_pnl / cost_basis) * 100 if cost_basis > 0 else 0

                pos_data.append({
                    'SYMBOL': symbol,
                    'QTY': pos['quantity'],
                    'AVG': f"${pos['avg_price']:.2f}",
                    'CURRENT': f"${pos['current_price']:.2f}",
                    'VALUE': f"${current_value:,.0f}",
                    'P&L': f"${unrealized_pnl:+,.0f}",
                    'P&L %': f"{unrealized_pnl_pct:+.2f}%"
                })

            pos_df = pd.DataFrame(pos_data)
            st.dataframe(pos_df, use_container_width=True, hide_index=True, key="positions_table")
        else:
            st.info("No open positions")

    with table_col2:
        st.subheader("RECENT TRADES")

        if len(st.session_state.trades) > 0:
            recent = pd.DataFrame(st.session_state.trades).tail(10)
            recent['TIME'] = pd.to_datetime(recent['timestamp']).dt.strftime('%H:%M:%S')

            trade_display = recent[['TIME', 'symbol', 'side', 'quantity', 'price', 'pnl']].copy()
            trade_display.columns = ['TIME', 'SYMBOL', 'SIDE', 'QTY', 'PRICE', 'P&L']
            trade_display['PRICE'] = trade_display['PRICE'].apply(lambda x: f"${x:.2f}")
            trade_display['P&L'] = trade_display['P&L'].apply(lambda x: f"${x:+,.0f}")

            st.dataframe(trade_display, use_container_width=True, hide_index=True, key="trades_table")

            csv = pd.DataFrame(st.session_state.trades).to_csv(index=False)
            st.download_button(
                "DOWNLOAD HISTORY",
                csv,
                f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.info("No trades yet")

    # Auto-refresh
    if st.session_state.bot_running:
        time.sleep(trade_interval)
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; font-family: 'Courier New', monospace; color: #9AA0A6;">
    <div style="font-size: 1.1rem; font-weight: 700; color: #FF9500; letter-spacing: 2px; margin-bottom: 0.5rem;">
        QUANTEDGE PRO
    </div>
    <div style="font-size: 0.85rem; margin-bottom: 1rem; letter-spacing: 1px;">
        ALGORITHMIC TRADING PLATFORM
    </div>
    <div style="font-size: 0.75rem; line-height: 1.6;">
        SIMULATED DATA | EDUCATIONAL USE ONLY | SUBSTANTIAL RISK WARNING
    </div>
</div>
""", unsafe_allow_html=True)
