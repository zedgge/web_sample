"""
QuantEdge Pro - Professional Trading Terminal
ULTIMATE VERSION - Perfect UI/UX, professional design, zero compromises

FEATURES:
- Professional Bloomberg-style design
- Intuitive, easy-to-use interface
- Perfect responsive scaling
- Smooth animations (minimal flash)
- Clean, modern aesthetic
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
    page_title="QuantEdge Pro - Trading Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal CSS - Let config.toml handle theming
st.markdown("""
<style>
    /* Hide ALL Streamlit branding and GitHub link */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    header {visibility: hidden;}

    /* Hide GitHub link in top right */
    [data-testid="stToolbar"] {display: none;}
    .viewerBadge_link__qRIco {display: none;}
    .viewerBadge_container__r5tak {display: none;}
    iframe[src*="github"] {display: none;}
    a[href*="github.com"] {display: none !important;}

    /* Hide fullscreen button completely */
    [data-testid="StyledFullScreenButton"] {display: none !important;}
    button[title="View fullscreen"] {display: none !important;}

    /* Hide GitHub link if fullscreen is triggered */
    [data-testid="StyledFullScreenButton"] ~ div {display: none;}

    /* Nuclear option - hide ALL fullscreen elements */
    iframe[title*="streamlit"] {pointer-events: none;}
    .fullScreenFrame {display: none !important;}

    /* Clean header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        color: #3B82F6;
    }

    .subtitle {
        text-align: center;
        font-size: 0.9rem;
        color: #9CA3AF;
        margin-bottom: 2rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
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
# CORE FUNCTIONS
# ============================================================

def generate_realistic_trade():
    """Generate realistic trade - ONLY from selected assets"""
    np.random.seed(int(time.time() * 1000) % 2**32)

    symbols = st.session_state.selected_assets if st.session_state.selected_assets else ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA']
    strategies = list(st.session_state.strategy_performance.keys())

    symbol = np.random.choice(symbols)
    strategy = np.random.choice(strategies)
    side = np.random.choice(['BUY', 'SELL'], p=[0.55, 0.45])
    quantity = np.random.randint(5, 150)
    price = np.random.uniform(80, 900)

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

    perf = st.session_state.strategy_performance[strategy]
    perf['trades'] += 1
    perf['pnl'] += pnl
    perf['returns'].append(pnl / (quantity * price))
    if is_win:
        perf['wins'] += 1

    st.session_state.current_capital += pnl
    st.session_state.max_capital = max(st.session_state.max_capital, st.session_state.current_capital)

    st.session_state.equity_curve.append({
        'timestamp': datetime.now(),
        'equity': st.session_state.current_capital
    })

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

    for sym in st.session_state.positions:
        price_change = np.random.normal(0, 0.005)
        st.session_state.positions[sym]['current_price'] *= (1 + price_change)

    daily_return = (st.session_state.current_capital / st.session_state.start_capital - 1) * 100
    st.session_state.daily_returns.append(daily_return)

    st.session_state.last_update = datetime.now()


def calculate_advanced_metrics():
    """Calculate comprehensive trading metrics"""
    if len(st.session_state.trades) == 0:
        return {
            'total_return': 0, 'total_return_abs': 0, 'sharpe_ratio': 0,
            'sortino_ratio': 0, 'max_drawdown': 0, 'win_rate': 0,
            'profit_factor': 0, 'avg_win': 0, 'avg_loss': 0, 'avg_trade': 0,
            'best_trade': 0, 'worst_trade': 0, 'consecutive_wins': 0,
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
        sharpe = sortino = 0

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

    consecutive_wins = consecutive_losses = 0
    current_streak_wins = current_streak_losses = 0
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
        'total_return': total_return, 'total_return_abs': total_return_abs,
        'sharpe_ratio': sharpe, 'sortino_ratio': sortino, 'max_drawdown': max_dd,
        'win_rate': win_rate, 'profit_factor': profit_factor, 'avg_win': avg_win,
        'avg_loss': avg_loss, 'avg_trade': avg_trade, 'best_trade': best_trade,
        'worst_trade': worst_trade, 'consecutive_wins': consecutive_wins,
        'consecutive_losses': consecutive_losses
    }

# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="main-header">QuantEdge Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Algorithmic Trading Terminal - Free Demo</div>', unsafe_allow_html=True)

# Status indicator
if st.session_state.bot_running:
    st.markdown('''
    <div style="text-align: center; margin-bottom: 2rem;">
        <span class="status-badge status-running">
            <span class="live-pulse"></span>
            Live Trading
        </span>
    </div>
    ''', unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### Configuration")

    st.markdown("**Capital**")
    initial_capital = st.number_input(
        "Starting Capital ($)",
        min_value=10000,
        max_value=500000000,
        value=100000,
        step=10000,
        format="%d"
    )

    st.markdown("**Trading Universe**")
    assets = st.multiselect(
        "Select Assets",
        ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "NFLX", "ORCL", "AMD", "SPY", "QQQ"],
        default=["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"]
    )
    st.session_state.selected_assets = assets

    st.markdown("**Strategy**")
    strategy_select = st.selectbox(
        "Trading Strategy",
        ["Momentum", "RSI Reversal", "MA Crossover", "Mean Reversion"],
        index=0
    )

    st.markdown("**Risk Management**")
    max_position = st.slider(
        "Max Position (%)",
        min_value=5,
        max_value=50,
        value=20,
        step=5
    )

    stop_loss = st.slider(
        "Stop Loss (%)",
        min_value=1,
        max_value=25,
        value=10,
        step=1
    )

    take_profit = st.slider(
        "Take Profit (%)",
        min_value=5,
        max_value=100,
        value=25,
        step=5
    )

    st.markdown("**Update Speed**")
    speed = st.select_slider(
        "Refresh Rate",
        options=["Slow (5s)", "Normal (3s)", "Fast (1s)", "Maximum (0.5s)"],
        value="Slow (5s)"
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
    st.markdown("**Controls**")
    col1, col2 = st.columns(2)

    with col1:
        start_clicked = st.button(
            "START",
            use_container_width=True,
            type="primary"
        )

    with col2:
        stop_clicked = st.button(
            "STOP",
            use_container_width=True
        )

    if start_clicked:
        st.session_state.bot_running = True
        st.session_state.start_capital = initial_capital
        st.session_state.current_capital = initial_capital
        st.session_state.max_capital = initial_capital
        st.session_state.trades = []
        st.session_state.equity_curve = [{'timestamp': datetime.now(), 'equity': initial_capital}]
        st.session_state.positions = {}
        st.session_state.trade_count = 0
        st.session_state.daily_returns = [0]
        for strategy in st.session_state.strategy_performance:
            st.session_state.strategy_performance[strategy] = {'trades': 0, 'wins': 0, 'pnl': 0, 'returns': []}
        st.rerun()

    if stop_clicked:
        st.session_state.bot_running = False
        st.rerun()

    st.markdown("---")

    # Status
    st.markdown("**Status**")
    if st.session_state.bot_running:
        st.success("RUNNING")
        st.caption(f"Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")
    else:
        st.warning("STOPPED")

    # Quick stats
    if len(st.session_state.trades) > 0:
        st.markdown("**Quick Stats**")
        st.metric("Trades", st.session_state.trade_count)
        st.metric("Positions", len(st.session_state.positions))

# ============================================================
# MAIN CONTENT
# ============================================================

if not st.session_state.bot_running and len(st.session_state.trades) == 0:
    # DEMO MODE
    st.info("Configure your trading terminal in the sidebar and click **START** to begin paper trading")

    st.markdown("### Historical Backtest Performance")

    dates = pd.date_range(start='2025-01-01', end='2025-12-27', freq='D')
    fig = go.Figure()

    strategies_colors = {
        'Momentum': '#3B82F6',
        'RSI': '#10B981',
        'MA Crossover': '#F59E0B',
        'Mean Reversion': '#EF4444'
    }

    for strategy, color in strategies_colors.items():
        returns = np.random.randn(len(dates)).cumsum() * 0.01 + 0.08
        equity = 100000 * (1 + returns)
        fig.add_trace(go.Scatter(
            x=dates, y=equity, mode='lines', name=strategy,
            line=dict(width=3, color=color),
            hovertemplate='%{y:$,.0f}<extra></extra>'
        ))

    fig.update_layout(
        xaxis_title="Date", yaxis_title="Portfolio Value ($)",
        hovermode='x unified', height=500, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        plot_bgcolor='#0B0E1A', paper_bgcolor='#0B0E1A',
        font=dict(family='Inter', size=12, color='#F9FAFB'),
        xaxis=dict(gridcolor='#374151', showgrid=True),
        yaxis=dict(gridcolor='#374151', showgrid=True),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Sample Metrics")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Return", "+18.7%", "18.7%")
    with m2:
        st.metric("Sharpe", "1.84", "0.84")
    with m3:
        st.metric("Sortino", "2.56", "1.56")
    with m4:
        st.metric("Max DD", "-8.3%", "-8.3%", delta_color="inverse")
    with m5:
        st.metric("Win Rate", "58.2%", "8.2%")
    with m6:
        st.metric("Profit Factor", "2.12", "1.12")

else:
    # LIVE DASHBOARD
    if st.session_state.bot_running and np.random.random() < 0.35:
        generate_realistic_trade()

    metrics = calculate_advanced_metrics()

    # Performance Metrics
    st.markdown("### Performance Dashboard")
    p1, p2, p3, p4, p5, p6 = st.columns(6)

    with p1:
        delta_color = "normal" if metrics['total_return'] >= 0 else "inverse"
        st.metric(
            "Portfolio Value",
            f"${st.session_state.current_capital:,.0f}",
            delta=f"${metrics['total_return_abs']:+,.0f}",
            delta_color=delta_color
        )
    with p2:
        st.metric(
            "Total Return",
            f"{metrics['total_return']:+.2f}%",
            delta=f"{metrics['total_return']:+.2f}%",
            delta_color=delta_color
        )
    with p3:
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}", f"{metrics['sharpe_ratio']:.2f}")
    with p4:
        st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}", f"{metrics['sortino_ratio']:.2f}")
    with p5:
        st.metric(
            "Max Drawdown",
            f"{metrics['max_drawdown']:.2f}%",
            f"{metrics['max_drawdown']:.2f}%",
            delta_color="inverse"
        )
    with p6:
        st.metric("Win Rate", f"{metrics['win_rate']:.1f}%", f"{metrics['win_rate'] - 50:+.1f}%")

    st.markdown("---")

    # Layout
    main_col, stock_col = st.columns([3, 1])

    with main_col:
        # Equity Curve
        st.markdown("### Equity Curve")
        if len(st.session_state.equity_curve) > 0:
            equity_df = pd.DataFrame(st.session_state.equity_curve)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_df['timestamp'], y=equity_df['equity'],
                mode='lines', fill='tozeroy', name='Portfolio Value',
                line=dict(color='#3B82F6', width=3),
                fillcolor='rgba(59, 130, 246, 0.1)',
                hovertemplate='%{y:$,.0f}<extra></extra>'
            ))
            fig.add_hline(
                y=st.session_state.start_capital, line_dash="dash",
                line_color="#9CA3AF", line_width=2,
                annotation_text="Start", annotation_position="right"
            )
            fig.add_hline(
                y=st.session_state.max_capital, line_dash="dot",
                line_color="#10B981", line_width=2,
                annotation_text="ATH", annotation_position="right"
            )
            fig.update_layout(
                xaxis_title="Time", yaxis_title="Value ($)",
                hovermode='x unified', height=400, showlegend=False,
                plot_bgcolor='#0B0E1A', paper_bgcolor='#0B0E1A',
                font=dict(family='Inter', size=11, color='#F9FAFB'),
                xaxis=dict(gridcolor='#374151'), yaxis=dict(gridcolor='#374151'),
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Strategy Leaderboard
        st.markdown("### 🏆 Strategy Leaderboard")
        strategy_data = []
        for strategy, perf in st.session_state.strategy_performance.items():
            if perf['trades'] > 0:
                win_rate = (perf['wins'] / perf['trades']) * 100
                avg_return = np.mean(perf['returns']) * 100 if perf['returns'] else 0
                strategy_data.append({
                    'Strategy': strategy, 'P&L': perf['pnl'],
                    'Trades': perf['trades'], 'Win %': win_rate
                })

        if strategy_data:
            strategy_df = pd.DataFrame(strategy_data).sort_values('P&L', ascending=False)
            fig = go.Figure()
            colors = ['#10B981' if x > 0 else '#EF4444' for x in strategy_df['P&L']]
            fig.add_trace(go.Bar(
                x=strategy_df['Strategy'], y=strategy_df['P&L'],
                marker_color=colors,
                text=strategy_df['P&L'].apply(lambda x: f"${x:,.0f}"),
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>P&L: %{y:$,.0f}<extra></extra>'
            ))
            fig.update_layout(
                xaxis_title="", yaxis_title="P&L ($)", height=350, showlegend=False,
                plot_bgcolor='#0B0E1A', paper_bgcolor='#0B0E1A',
                font=dict(family='Inter', size=11, color='#F9FAFB'),
                xaxis=dict(gridcolor='#374151'), yaxis=dict(gridcolor='#374151'),
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                strategy_df[['Strategy', 'Trades', 'Win %', 'P&L']].style.format({
                    'Win %': '{:.1f}%', 'P&L': '${:,.0f}'
                }),
                use_container_width=True, hide_index=True
            )

    # Stock Performance
    with stock_col:
        st.markdown("### Positions P&L")
        if len(st.session_state.positions) > 0:
            stock_performance = []
            for symbol, pos in st.session_state.positions.items():
                current_value = pos['quantity'] * pos['current_price']
                cost_basis = pos['quantity'] * pos['avg_price']
                pnl = current_value - cost_basis
                pnl_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else 0
                stock_performance.append((symbol, pnl, pnl_pct))

            sorted_stocks = sorted(stock_performance, key=lambda x: x[1], reverse=True)

            # Use native Streamlit components instead of HTML
            for symbol, pnl, pnl_pct in sorted_stocks:
                delta_color = "normal" if pnl >= 0 else "inverse"
                st.metric(
                    label=symbol,
                    value=f"${pnl:+,.0f}",
                    delta=f"{pnl_pct:+.2f}%",
                    delta_color=delta_color
                )
        else:
            st.info("No open positions")

    st.markdown("---")

    # Tables
    table_col1, table_col2 = st.columns(2)

    with table_col1:
        st.markdown("### Open Positions")
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
            st.dataframe(
                pos_df.style.format({'P&L': '${:,.0f}'}),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("No open positions")

    with table_col2:
        st.markdown("### Recent Trades")
        if len(st.session_state.trades) > 0:
            recent = pd.DataFrame(st.session_state.trades).tail(10)
            recent['Time'] = pd.to_datetime(recent['timestamp']).dt.strftime('%H:%M:%S')
            trade_display = recent[['Time', 'symbol', 'side', 'quantity', 'price', 'strategy', 'pnl']].copy()
            trade_display.columns = ['Time', 'Symbol', 'Side', 'Qty', 'Price', 'Strategy', 'P&L']
            trade_display['Price'] = trade_display['Price'].apply(lambda x: f"${x:.2f}")
            trade_display['P&L'] = trade_display['P&L'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(trade_display, use_container_width=True, hide_index=True)

            csv = pd.DataFrame(st.session_state.trades).to_csv(index=False)
            st.download_button(
                "Download History",
                csv,
                f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.info("No trades yet")

    # Advanced Metrics
    with st.expander("📊 Advanced Metrics", expanded=False):
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

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("Paper trading simulation for educational purposes only. Not investment advice.")
