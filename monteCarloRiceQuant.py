import numpy as np
import pandas as pd
from rqalpha.api import logger, history_bars, all_instruments, get_factor, update_universe, order_target_percent, scheduler
from rqalpha.portfolio import Portfolio
from typing import List, Dict, Tuple

# ================== 配置参数 ==================
class Config:
    SIM_DAYS = 60
    NUM_SIMULATIONS = 1000
    MIN_HISTORY = 252
    RISK_PER_POSITION = 0.9
    FILTER_CONDITIONS = {
        'profit_growth': ('net_profit_growth_ratio_ttm', lambda x: x > 0),
        'cash_adequacy': (
            ['cash_equivalent_ttm_0', 'short_term_loans_ttm_0'], 
            lambda cash, debt: cash > debt
        ),
        'cash_profit': (
            ['cash_equivalent_ttm_0', 'net_profit_ttm_0'],
            lambda cash, profit: cash >= profit
        )
    }

# ================== 核心函数 ==================
def vectorized_monte_carlo(initial_price: float, returns: np.ndarray, days: int, n_sims: int) -> np.ndarray:
    """向量化蒙特卡洛模拟"""
    mu = np.nanmean(returns)
    sigma = np.nanstd(returns)
    
    dt = 1/252
    rand = np.random.normal(
        loc=(mu - 0.5 * sigma**2) * dt,
        scale=sigma * np.sqrt(dt),
        size=(n_sims, days)
    )
    
    cum_returns = np.exp(rand.cumsum(axis=1))
    return initial_price * cum_returns

# ... 前面的代码保持不变

def get_qualified_stocks(context) -> pd.DataFrame:
    """获取符合财务条件的股票"""
    try:
        # 批量获取因子数据
        factors = [
            'net_profit_ttm_0', 'net_profit_growth_ratio_ttm',
            'cash_equivalent_ttm_0', 'short_term_loans_ttm_0'
        ]
        stocks = all_instruments('CS').order_book_id
        df = get_factor(stocks, factors)
        
        # 复合条件筛选
        cond = True
        for name, (cols, func) in Config.FILTER_CONDITIONS.items():
            if isinstance(cols, list):
                cond &= df[cols].apply(lambda x: func(*x), axis=1)
            else:
                cond &= df[cols].apply(func)
                
        return df[cond].dropna()
    except Exception as e:
        logger.error(f"财务数据获取失败: {str(e)}")
        return pd.DataFrame()

def calculate_upside_probability(context, stock: str) -> float:
    """计算股票上涨概率"""
    try:
        prices = history_bars(stock, Config.MIN_HISTORY, '1d', 'close')
        if len(prices) < 10:  # 数据不足检查
            return 0.0
            
        returns = np.diff(prices) / prices[:-1]
        initial_price = prices[-1]
        
        # 向量化模拟
        simulations = vectorized_monte_carlo(
            initial_price, returns,
            Config.SIM_DAYS, Config.NUM_SIMULATIONS
        )
        
        # 计算上涨概率
        final_prices = simulations[:, -1]
        return np.mean(final_prices > initial_price)
    except Exception as e:
        logger.warning(f"股票{stock}模拟失败: {str(e)}")
        return 0.0

# ================== 策略逻辑 ==================
def rebalance_portfolio(context):
    """执行投资组合再平衡"""
    # 清仓不在候选列表中的股票
    for holding_stock in list(context.portfolio.positions.keys()):
        logger.info(f"holding_stock:{holding_stock}")
        if holding_stock not in context.stocks:
            if context.portfolio.positions[holding_stock].quantity != 0:
                order_target_percent(holding_stock, 0)
                logger.info(f"Sold all positions of {holding_stock}.")

    # 建仓/调仓目标股票
    position_size = Config.RISK_PER_POSITION / len(context.stocks)
    for stock in context.stocks:
        if stock not in context.portfolio.positions.keys():
            logger.info(f"stock={stock}")
            order_target_percent(stock, position_size)
            logger.debug(f"调仓: {stock} -> {position_size*100:.2f}%")

def query_fundamental(context, bar_dict):
    """核心策略逻辑"""
    try:
        # 获取基本面合格的股票
        fundamental_df = get_qualified_stocks(context)
        logger.info(f"初步筛选合格股票: {len(fundamental_df)}只")
        
        # 并行计算上涨概率
        stock_list = fundamental_df.index.tolist()
        results = [
            (stock, calculate_upside_probability(context, stock[0]))
            for stock in stock_list
        ]
        
        # 筛选Top200
        top_stocks = sorted(results, key=lambda x: x[1], reverse=True)[:200]
        top_codes = [s[0] for s in top_stocks]
        
        # 最终财务筛选
        context.target_stocks = fundamental_df.loc[top_codes]\
            .sort_values('net_profit_growth_ratio_ttm', ascending=False)\
            .head(10).reset_index(1, drop=True)
            
        context.stocks = context.target_stocks.index.values
        logger.info(f"最终入选股票: {context.stocks}")
        update_universe(context.stocks)
        
        # 执行调仓
        rebalance_portfolio(context)
    except Exception as e:
        logger.error(f"策略执行异常: {str(e)}")
        context.target_stocks = []

# ================== 框架函数 ==================
def init(context):
    """初始化"""
    # 每周一开盘前运行策略
    scheduler.run_weekly(query_fundamental, weekday=1)
    context.target_stocks = []

def handle_bar(context, bar_dict):
    """空实现，由定时任务驱动"""
    pass

def after_trading(context):
    """交易后处理"""
    logger.info(f"当日持仓: {context.portfolio.positions}")
