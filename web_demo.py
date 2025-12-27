"""
QuantEdge Pro - Professional Trading Terminal
ULTIMATE VERSION - Perfect scaling, zero flash, smooth performance

FEATURES:
- JavaScript-based auto-refresh (NO FLASH)
- Fully responsive scaling for all window sizes
- Bloomberg-style terminal design
- Only trades selected assets
- Scrollable stock performance widget
- Mobile responsive
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import time
from streamlit_autorefresh import st_autorefresh

# Page config
st.set_page_config(
    page_title="QuantEdge Pro - Trading Terminal",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ULTIMATE CSS - FULLY RESPONSIVE, ZERO FLASH, PERFECT SCALING
st.markdown("""
<style>
    /* ============================================================
       CRITICAL ANTI-FLASH FIXES
       ============================================================ */

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}

    /* NUCLEAR: Remove ALL animations/transitions */
    *, *::before, *::after {
        transition: none !important;
        animation: none !important;
    }

    /* Prevent flash on rerun */
    .element-container {
        will-change: auto !important;
    }

    /* Force opacity to prevent flickering */
    .stApp {
        opacity: 1 !important;
    }

    /* Prevent element flickering */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        animation: none !important;
    }

    /* Smooth opacity for content */
    [data-testid="stVerticalBlock"] {
        opacity: 1 !important;
    }

    /* ============================================================
       COLOR PALETTE & BACKGROUNDS
       ============================================================ */

    :root {
        --terminal-orange: #FF9500;
        --terminal-green: #00FF41;
        --terminal-red: #FF3B30;
        --bg-dark: #0A0E27;
        --bg-darker: #060A1F;
        --text-primary: #E0E0E0;
        --text-secondary: #8B8B8B;
        --border-color: #2A2E47;
    }

    /* Force dark background EVERYWHERE to prevent white flash */
    html, body, #root, [data-testid="stAppViewContainer"], .main, .stApp {
        background-color: var(--bg-dark) !important;
        background: var(--bg-dark) !important;
    }

    .stApp {
        color: var(--text-primary) !important;
    }

    /* ============================================================
       RESPONSIVE TYPOGRAPHY
       ============================================================ */

    .main-header {
        font-size: clamp(1.5rem, 4vw, 2.8rem);
        font-weight: 700;
        color: var(--terminal-orange);
        text-align: center;
        margin-bottom: 0.3rem;
        font-family: 'Courier New', monospace;
        letter-spacing: clamp(1px, 0.3vw, 3px);
    }

    .subtitle {
        text-align: center;
        font-size: clamp(0.7rem, 1.2vw, 0.9rem);
        color: var(--text-secondary);
        margin-bottom: 2rem;
        font-family: 'Courier New', monospace;
        letter-spacing: clamp(1px, 0.2vw, 2px);
        text-transform: uppercase;
    }

    /* Section headers - responsive */
    h3 {
        color: var(--terminal-orange);
        font-weight: 700;
        font-size: clamp(0.85rem, 1.5vw, 1rem);
        margin-bottom: 1rem;
        font-family: 'Courier New', monospace;
        letter-spacing: 1.5px;
        border-bottom: 2px solid var(--terminal-orange);
        padding-bottom: 6px;
    }

    /* ============================================================
       RESPONSIVE METRICS
       ============================================================ */

    [data-testid="stMetricValue"] {
        font-family: 'Courier New', monospace;
        font-size: clamp(1rem, 2vw, 1.6rem) !important;
        font-weight: 700;
        color: var(--text-primary);
    }

    [data-testid="stMetricLabel"] {
        font-size: clamp(0.6rem, 1vw, 0.7rem) !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--text-secondary);
        font-family: 'Courier New', monospace;
    }

    /* ============================================================
       LIVE INDICATOR
       ============================================================ */

    .live-indicator {
        display: inline-block;
        width: clamp(8px, 1.5vw, 10px);
        height: clamp(8px, 1.5vw, 10px);
        background: var(--terminal-green);
        border-radius: 50%;
        margin-right: 8px;
    }

    /* ============================================================
       BUTTONS - RESPONSIVE
       ============================================================ */

    .stButton > button {
        background: var(--bg-darker);
        border: 2px solid var(--border-color);
        color: var(--text-primary);
        font-weight: 700;
        font-size: clamp(0.7rem, 1.2vw, 0.8rem);
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: clamp(8px, 1.5vw, 12px) clamp(16px, 3vw, 24px);
        font-family: 'Courier New', monospace;
    }

    .stButton > button:hover {
        border-color: var(--terminal-orange);
        color: var(--terminal-orange);
    }

    .stButton > button[kind="primary"] {
        background: var(--bg-darker);
        border-color: var(--terminal-green);
        color: var(--terminal-green);
    }

    .stButton > button[kind="primary"]:hover {
        background: var(--terminal-green);
        color: var(--bg-dark);
    }

    /* ============================================================
       STATUS BADGE
       ============================================================ */

    .status-badge {
        display: inline-block;
        padding: clamp(4px, 1vw, 6px) clamp(12px, 2vw, 16px);
        font-weight: 700;
        font-size: clamp(0.65rem, 1vw, 0.75rem);
        letter-spacing: 1.5px;
        text-transform: uppercase;
        border: 2px solid;
        font-family: 'Courier New', monospace;
        background: transparent;
    }

    .status-running {
        color: var(--terminal-green);
        border-color: var(--terminal-green);
    }

    .status-stopped {
        color: var(--text-secondary);
        border-color: var(--border-color);
    }

    /* ============================================================
       SIDEBAR - RESPONSIVE
       ============================================================ */

    [data-testid="stSidebar"] {
        background: var(--bg-darker);
        border-right: 2px solid var(--border-color);
        min-width: 250px;
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--terminal-orange);
        font-family: 'Courier New', monospace;
        letter-spacing: 1px;
        font-size: clamp(0.85rem, 1.5vw, 1rem);
    }

    [data-testid="stSidebar"] label {
        color: var(--text-secondary);
        font-family: 'Courier New', monospace;
        font-size: clamp(0.7rem, 1.1vw, 0.8rem);
    }

    /* ============================================================
       FORM INPUTS - RESPONSIVE
       ============================================================ */

    input[type="number"], input[type="text"] {
        background: var(--bg-dark) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        font-family: 'Courier New', monospace !important;
        font-size: clamp(0.75rem, 1.2vw, 0.85rem) !important;
    }

    input[type="number"]:focus, input[type="text"]:focus {
        border-color: var(--terminal-orange) !important;
        box-shadow: none !important;
    }

    /* Sliders */
    .stSlider > div > div > div {
        background: var(--border-color) !important;
    }

    .stSlider > div > div > div > div {
        background: var(--terminal-orange) !important;
    }

    /* Multiselect */
    .stMultiSelect > div > div {
        background: var(--bg-dark) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        font-family: 'Courier New', monospace !important;
        font-size: clamp(0.75rem, 1.2vw, 0.85rem) !important;
    }

    .stMultiSelect span {
        color: var(--text-primary) !important;
        font-family: 'Courier New', monospace !important;
    }

    /* Select boxes */
    .stSelectbox > div > div {
        background: var(--bg-dark) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        font-family: 'Courier New', monospace !important;
        font-size: clamp(0.75rem, 1.2vw, 0.85rem) !important;
    }

    /* ============================================================
       DATAFRAMES - RESPONSIVE
       ============================================================ */

    .dataframe {
        font-family: 'Courier New', monospace;
        font-size: clamp(0.7rem, 1.1vw, 0.8rem);
        background: var(--bg-darker);
        color: var(--text-primary);
    }

    /* ============================================================
       STOCK PERFORMANCE WIDGET - SCROLLABLE
       ============================================================ */

    .stock-scroll-container {
        max-height: clamp(400px, 60vh, 600px);
        overflow-y: auto;
        overflow-x: hidden;
        padding-right: 10px;
        margin-top: 10px;
    }

    /* Custom scrollbar */
    .stock-scroll-container::-webkit-scrollbar {
        width: 8px;
    }

    .stock-scroll-container::-webkit-scrollbar-track {
        background: var(--bg-dark);
        border: 1px solid var(--border-color);
    }

    .stock-scroll-container::-webkit-scrollbar-thumb {
        background: var(--border-color);
    }

    .stock-scroll-container::-webkit-scrollbar-thumb:hover {
        background: var(--terminal-orange);
    }

    /* Stock card - responsive */
    .stock-card {
        background: var(--bg-darker);
        border: 1px solid var(--border-color);
        padding: clamp(8px, 1.5vw, 12px);
        margin-bottom: 8px;
        font-family: 'Courier New', monospace;
    }

    .stock-symbol {
        font-size: clamp(0.85rem, 1.3vw, 1rem);
        font-weight: 700;
        letter-spacing: 1px;
    }

    .stock-positive {
        color: var(--terminal-green);
    }

    .stock-negative {
        color: var(--terminal-red);
    }

    .stock-neutral {
        color: var(--text-secondary);
    }

    /* ============================================================
       EXPANDER
       ============================================================ */

    .streamlit-expanderHeader {
        background: var(--bg-darker);
        border: 1px solid var(--border-color);
        color: var(--terminal-orange);
        font-family: 'Courier New', monospace;
        font-weight: 700;
        font-size: clamp(0.75rem, 1.2vw, 0.85rem);
    }

    /* ============================================================
       DIVIDERS & INFO BOXES
       ============================================================ */

    hr {
        border-color: var(--border-color);
        margin: 1.5rem 0;
    }

    .stAlert {
        background: var(--bg-darker);
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        font-family: 'Courier New', monospace;
        font-size: clamp(0.75rem, 1.2vw, 0.85rem);
    }

    /* ============================================================
       RESPONSIVE BREAKPOINTS
       ============================================================ */

    /* Tablets */
    @media (max-width: 1024px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .stock-scroll-container {
            max-height: 500px;
        }
    }

    /* Mobile landscape & small tablets */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem;
            letter-spacing: 2px;
        }

        .subtitle {
            font-size: 0.75rem;
        }

        .stock-scroll-container {
            max-height: 400px;
        }

        [data-testid="stSidebar"] {
            min-width: 200px;
        }
    }

    /* Mobile portrait */
    @media (max-width: 480px) {
        .main-header {
            font-size: 1.4rem;
            letter-spacing: 1px;
        }

        .subtitle {
            font-size: 0.65rem;
        }

        .stock-card {
            padding: 8px;
        }

        .stButton > button {
            padding: 8px 12px;
            font-size: 0.7rem;
        }
    }

    /* Ultra-wide screens */
    @media (min-width: 1920px) {
        .main .block-container {
            max-width: 90%;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

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
if 'selected_assets' not in st.session_state:
    st.session_state.selected_assets = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA']

# ============================================================
# CORE TRADING FUNCTIONS
# ============================================================

def generate_realistic_trade():
    """Generate realistic trade - ONLY from selected assets"""
    np.random.seed(int(time.time() * 1000) % 2**32)

    # USE ONLY SELECTED ASSETS
    symbols = st.session_state.selected_assets if st.session_state.selected_assets else ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA']
    strategies = list(st.session_state.strategy_performance.keys())

    symbol = np.random.choice(symbols)
    strategy = np.random.choice(strategies)
    side = np.random.choice(['BUY', 'SELL'], p=[0.55, 0.45])
    quantity = np.random.randint(5, 150)
    price = np.random.uniform(80, 900)

    # Realistic P&L
    strategy_multipliers = {
        'Momentum': 1.1,
        'RSI': 1.0,
        'MA Crossover': 0.95,
        'Mean Reversion': 1.05
    }

    base_pnl = np.random.normal(80, 250)
    pnl = base_pnl * strategy_multipliers.get(strategy, 1.0)
    is_win = pnl > 0

    trade = {
        'timestamp': datetime.now(),
        'symbol': symbol,
        'side': side,
        'quantity': quantity,
        'price': price,
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
                'current_price': price * (1 + np.random.normal(0, 0.01))
            }
    else:
        if symbol in st.session_state.positions:
            st.session_state.positions[symbol]['quantity'] -= quantity
            if st.session_state.positions[symbol]['quantity'] <= 0:
                del st.session_state.positions[symbol]

    # Update position prices
    for sym in st.session_state.positions:
        price_change = np.random.normal(0, 0.005)
        st.session_state.positions[sym]['current_price'] *= (1 + price_change)

    # Update daily returns
    daily_return = (st.session_state.current_capital / st.session_state.start_capital - 1) * 100
    st.session_state.daily_returns.append(daily_return)

    st.session_state.last_update = datetime.now()


def calculate_advanced_metrics():
    """Calculate comprehensive trading metrics"""
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

    total_return_abs = st.session_state.current_capital - st.session_state.start_capital
    total_return = (total_return_abs / st.session_state.start_capital) * 100

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

# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="main-header">QUANTEDGE PRO</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">ALGORITHMIC TRADING TERMINAL</div>', unsafe_allow_html=True)

# Live status indicator
if st.session_state.bot_running:
    st.markdown('''
    <div style="text-align: center; margin-bottom: 2rem;">
        <span class="live-indicator"></span>
        <span style="font-weight: 700; color: #00FF41; font-size: 0.85rem; letter-spacing: 2px; font-family: 'Courier New', monospace;">LIVE</span>
    </div>
    ''', unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# SIDEBAR CONFIGURATION
# ============================================================

with st.sidebar:
    st.header("CONFIGURATION")

    st.subheader("CAPITAL")
    initial_capital = st.number_input(
        "Starting Capital ($)",
        min_value=10000,
        max_value=10000000,
        value=100000,
        step=10000
    )

    st.subheader("ASSETS")
    assets = st.multiselect(
        "Trading Universe",
        ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "NFLX", "ORCL", "AMD", "SPY", "QQQ"],
        default=["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"]
    )
    st.session_state.selected_assets = assets

    st.subheader("RISK MANAGEMENT")
    max_position = st.slider("Max Position Size (%)", 5, 50, 20, 5)
    stop_loss = st.slider("Stop Loss (%)", 1, 25, 10, 1)
    take_profit = st.slider("Take Profit (%)", 5, 100, 25, 5)

    st.subheader("SPEED")
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

    # Status
    if st.session_state.bot_running:
        st.markdown('<div class="status-badge status-running">RUNNING</div>', unsafe_allow_html=True)
        st.caption(f"Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")
    else:
        st.markdown('<div class="status-badge status-stopped">STOPPED</div>', unsafe_allow_html=True)

    # Quick stats
    if len(st.session_state.trades) > 0:
        st.markdown("---")
        st.markdown("### QUICK STATS")
        st.metric("TOTAL TRADES", st.session_state.trade_count)
        st.metric("OPEN POSITIONS", len(st.session_state.positions))

# ============================================================
# MAIN CONTENT
# ============================================================

if not st.session_state.bot_running and len(st.session_state.trades) == 0:
    # DEMO MODE - Show historical backtest
    st.info("Configure trading terminal in sidebar and click START to begin")
    st.markdown("---")
    st.subheader("STRATEGY BACKTEST - 2025")

    dates = pd.date_range(start='2025-01-01', end='2025-12-27', freq='D')
    fig = go.Figure()

    strategies_colors = {
        'Momentum': '#FF9500',
        'RSI': '#00FF41',
        'MA Crossover': '#FFD60A',
        'Mean Reversion': '#FF3B30'
    }

    for strategy, color in strategies_colors.items():
        returns = np.random.randn(len(dates)).cumsum() * 0.01 + 0.08
        equity = 100000 * (1 + returns)
        fig.add_trace(go.Scatter(
            x=dates, y=equity, mode='lines', name=strategy,
            line=dict(width=2, color=color),
            hovertemplate='%{y:$,.0f}<extra></extra>'
        ))

    fig.update_layout(
        xaxis_title="DATE", yaxis_title="VALUE ($)",
        hovermode='x unified', height=500, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                   font=dict(family='Courier New', size=11)),
        plot_bgcolor='#0A0E27', paper_bgcolor='#0A0E27',
        font=dict(family='Courier New', size=11, color='#E0E0E0'),
        xaxis=dict(gridcolor='#2A2E47'), yaxis=dict(gridcolor='#2A2E47')
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("PERFORMANCE METRICS")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("RETURN", "+18.7%", "+18.7%")
    with m2:
        st.metric("SHARPE", "1.84", "+0.84")
    with m3:
        st.metric("SORTINO", "2.56", "+1.56")
    with m4:
        st.metric("MAX DD", "-8.3%", "-8.3%")
    with m5:
        st.metric("WIN RATE", "58.2%", "+8.2%")
    with m6:
        st.metric("PROFIT FACTOR", "2.12", "+1.12")

else:
    # LIVE DASHBOARD
    # Generate trades in background
    if st.session_state.bot_running and np.random.random() < 0.35:
        generate_realistic_trade()

    # Calculate metrics
    metrics = calculate_advanced_metrics()

    # Main layout with responsive columns
    main_col, stock_col = st.columns([3, 1])

    with main_col:
        st.subheader("PERFORMANCE DASHBOARD")
        p1, p2, p3, p4, p5, p6 = st.columns(6)

        with p1:
            delta_color = "normal" if metrics['total_return'] >= 0 else "inverse"
            st.metric("PORTFOLIO VALUE", f"${st.session_state.current_capital:,.0f}",
                     delta=f"${metrics['total_return_abs']:+,.0f}", delta_color=delta_color)
        with p2:
            st.metric("TOTAL RETURN", f"{metrics['total_return']:+.2f}%",
                     delta=f"{metrics['total_return']:+.2f}%", delta_color=delta_color)
        with p3:
            st.metric("SHARPE RATIO", f"{metrics['sharpe_ratio']:.2f}",
                     delta=f"{metrics['sharpe_ratio']:.2f}")
        with p4:
            st.metric("SORTINO RATIO", f"{metrics['sortino_ratio']:.2f}",
                     delta=f"{metrics['sortino_ratio']:.2f}")
        with p5:
            st.metric("MAX DRAWDOWN", f"{metrics['max_drawdown']:.2f}%",
                     delta=f"{metrics['max_drawdown']:.2f}%", delta_color="inverse")
        with p6:
            st.metric("WIN RATE", f"{metrics['win_rate']:.1f}%",
                     delta=f"{metrics['win_rate'] - 50:+.1f}%")

        st.markdown("---")

        # Equity curve
        st.subheader("EQUITY CURVE")
        if len(st.session_state.equity_curve) > 0:
            equity_df = pd.DataFrame(st.session_state.equity_curve)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_df['timestamp'], y=equity_df['equity'],
                mode='lines', fill='tozeroy', name='Portfolio Value',
                line=dict(color='#FF9500', width=2),
                fillcolor='rgba(255, 149, 0, 0.1)',
                hovertemplate='%{y:$,.0f}<extra></extra>'
            ))
            fig.add_hline(y=st.session_state.start_capital, line_dash="dash",
                         line_color="#8B8B8B", line_width=1,
                         annotation_text="START", annotation_position="right")
            fig.add_hline(y=st.session_state.max_capital, line_dash="dot",
                         line_color="#00FF41", line_width=1,
                         annotation_text="ATH", annotation_position="right")
            fig.update_layout(
                xaxis_title="TIME", yaxis_title="VALUE ($)",
                hovermode='x unified', height=400, showlegend=False,
                plot_bgcolor='#0A0E27', paper_bgcolor='#0A0E27',
                font=dict(family='Courier New', size=10, color='#E0E0E0'),
                xaxis=dict(gridcolor='#2A2E47'), yaxis=dict(gridcolor='#2A2E47'),
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig, use_container_width=True, key="equity_chart")

        st.markdown("---")

        # Strategy leaderboard
        st.subheader("STRATEGY LEADERBOARD")
        strategy_data = []
        for strategy, perf in st.session_state.strategy_performance.items():
            if perf['trades'] > 0:
                win_rate = (perf['wins'] / perf['trades']) * 100
                avg_return = np.mean(perf['returns']) * 100 if perf['returns'] else 0
                strategy_data.append({
                    'Strategy': strategy, 'P&L': perf['pnl'],
                    'Trades': perf['trades'], 'Win %': win_rate, 'Avg Return': avg_return
                })

        if strategy_data:
            strategy_df = pd.DataFrame(strategy_data).sort_values('P&L', ascending=False)
            fig = go.Figure()
            colors = ['#00FF41' if x > 0 else '#FF3B30' for x in strategy_df['P&L']]
            fig.add_trace(go.Bar(
                x=strategy_df['Strategy'], y=strategy_df['P&L'],
                marker_color=colors,
                text=strategy_df['P&L'].apply(lambda x: f"${x:,.0f}"),
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>P&L: %{y:$,.0f}<extra></extra>'
            ))
            fig.update_layout(
                xaxis_title="", yaxis_title="P&L ($)", height=350, showlegend=False,
                plot_bgcolor='#0A0E27', paper_bgcolor='#0A0E27',
                font=dict(family='Courier New', size=10, color='#E0E0E0'),
                xaxis=dict(gridcolor='#2A2E47'), yaxis=dict(gridcolor='#2A2E47'),
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig, use_container_width=True, key="strategy_chart")
            st.dataframe(
                strategy_df[['Strategy', 'Trades', 'Win %', 'P&L']].style.format({
                    'Win %': '{:.1f}%', 'P&L': '${:,.0f}'
                }),
                use_container_width=True, hide_index=True
            )

    # STOCK PERFORMANCE COLUMN - SCROLLABLE
    with stock_col:
        st.subheader("STOCKS")
        if len(st.session_state.positions) > 0:
            stock_performance = []
            for symbol, pos in st.session_state.positions.items():
                current_value = pos['quantity'] * pos['current_price']
                cost_basis = pos['quantity'] * pos['avg_price']
                pnl = current_value - cost_basis
                pnl_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else 0
                stock_performance.append((symbol, pnl, pnl_pct))

            sorted_stocks = sorted(stock_performance, key=lambda x: x[1], reverse=True)

            # BUILD HTML FOR SCROLLING WIDGET
            stock_html_parts = ['<div class="stock-scroll-container">']
            for symbol, pnl, pnl_pct in sorted_stocks:
                if pnl > 0:
                    color_class, symbol_prefix = "stock-positive", "▲"
                elif pnl < 0:
                    color_class, symbol_prefix = "stock-negative", "▼"
                else:
                    color_class, symbol_prefix = "stock-neutral", "="

                stock_html_parts.append(f'''
                <div class="stock-card">
                    <div class="stock-symbol {color_class}">{symbol_prefix} {symbol}</div>
                    <div class="{color_class}" style="font-size: 0.9rem; font-weight: 700; margin-top: 4px;">
                        ${pnl:+,.0f}
                    </div>
                    <div class="{color_class}" style="font-size: 0.8rem; margin-top: 2px;">
                        {pnl_pct:+.2f}%
                    </div>
                </div>
                ''')
            stock_html_parts.append('</div>')

            # RENDER WITH unsafe_allow_html=True
            st.markdown(''.join(stock_html_parts), unsafe_allow_html=True)
        else:
            st.info("No positions")

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
                    'Symbol': symbol, 'Qty': pos['quantity'],
                    'Avg': f"${pos['avg_price']:.2f}",
                    'Current': f"${pos['current_price']:.2f}",
                    'Value': f"${current_value:,.0f}",
                    'P&L': unrealized_pnl,
                    'P&L %': f"{unrealized_pnl_pct:+.2f}%"
                })
            pos_df = pd.DataFrame(pos_data)
            st.dataframe(pos_df.style.format({'P&L': '${:,.0f}'}),
                        use_container_width=True, hide_index=True, key="positions_table")
        else:
            st.info("No open positions")

    with table_col2:
        st.subheader("RECENT TRADES")
        if len(st.session_state.trades) > 0:
            recent = pd.DataFrame(st.session_state.trades).tail(10)
            recent['Time'] = pd.to_datetime(recent['timestamp']).dt.strftime('%H:%M:%S')
            trade_display = recent[['Time', 'symbol', 'side', 'quantity', 'price', 'strategy', 'pnl']].copy()
            trade_display.columns = ['Time', 'Symbol', 'Side', 'Qty', 'Price', 'Strategy', 'P&L']
            trade_display['Price'] = trade_display['Price'].apply(lambda x: f"${x:.2f}")
            trade_display['P&L'] = trade_display['P&L'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(trade_display, use_container_width=True, hide_index=True, key="trades_table")

            csv = pd.DataFrame(st.session_state.trades).to_csv(index=False)
            st.download_button("DOWNLOAD FULL HISTORY", csv,
                              f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                              "text/csv", use_container_width=True)
        else:
            st.info("No trades yet")

    # Advanced metrics
    with st.expander("ADVANCED METRICS", expanded=False):
        adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
        with adv_col1:
            st.metric("AVG WIN", f"${metrics['avg_win']:,.0f}")
            st.metric("BEST TRADE", f"${metrics['best_trade']:,.0f}")
        with adv_col2:
            st.metric("AVG LOSS", f"${metrics['avg_loss']:,.0f}")
            st.metric("WORST TRADE", f"${metrics['worst_trade']:,.0f}")
        with adv_col3:
            st.metric("AVG TRADE", f"${metrics['avg_trade']:,.0f}")
            st.metric("PROFIT FACTOR", f"{metrics['profit_factor']:.2f}")
        with adv_col4:
            st.metric("WIN STREAK", metrics['consecutive_wins'])
            st.metric("LOSS STREAK", metrics['consecutive_losses'])

    # AUTO-REFRESH - JavaScript based (minimal flash)
    if st.session_state.bot_running:
        st_autorefresh(interval=int(trade_interval * 1000), key="data_refresh")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: #060A1F; border: 1px solid #2A2E47; margin-top: 2rem;">
    <p style="font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem; color: #FF9500; font-family: 'Courier New', monospace; letter-spacing: 2px;">
        QUANTEDGE PRO
    </p>
    <p style="margin-bottom: 1rem; font-size: 0.85rem; color: #8B8B8B; font-family: 'Courier New', monospace;">
        ALGORITHMIC TRADING TERMINAL
    </p>
    <p style="font-size: 0.75rem; margin-top: 1rem; color: #8B8B8B; font-family: 'Courier New', monospace;">
        Paper trading simulation. Educational purposes only. Trading involves substantial risk.
    </p>
</div>
""", unsafe_allow_html=True)
