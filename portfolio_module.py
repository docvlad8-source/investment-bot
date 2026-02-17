import requests
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)
MOEX_BASE_URL = "https://iss.moex.com/iss"

BLUE_CHIPS = {
    "SBER": "Сбербанк", "GAZP": "Газпром", "LKOH": "ЛУКОЙЛ",
    "GMKN": "Норникель", "NVTK": "НОВАТЭК", "ROSN": "Роснефть",
    "YNDX": "Яндекс", "MTSS": "МТС", "MGNT": "Магнит",
    "ALRS": "АЛРОСА", "CHMF": "Северсталь", "NLMK": "НЛМК",
    "PLZL": "Полюс", "TATN": "Татнефть", "SNGS": "Сургутнефтегаз",
    "VTBR": "ВТБ", "MOEX": "Мосбиржа", "PHOR": "ФосАгро",
    "RUAL": "РУСАЛ", "AFKS": "АФК Система", "PIKK": "ПИК",
    "OZON": "OZON", "TCSG": "TCS Group", "FIVE": "X5 Group",
}

class MOEXClient:
    def __init__(self):
        self.session = requests.Session()
    
    def get_current_price(self, secid: str) -> Optional[float]:
        url = f"{MOEX_BASE_URL}/engines/stock/markets/shares/boards/TQBR/securities/{secid}.json"
        try:
            response = self.session.get(url, params={"marketdata.columns": "SECID,LAST,PREVPRICE"}, timeout=10)
            data = response.json()
            if "marketdata" in data and data["marketdata"]["data"]:
                row = data["marketdata"]["data"][0]
                return float(row[1] or row[2]) if (row[1] or row[2]) else None
        except Exception as e:
            logger.error(f"Price error {secid}: {e}")
        return None
    
    def get_history(self, secid: str, days: int = 365) -> pd.DataFrame:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        url = f"{MOEX_BASE_URL}/history/engines/stock/markets/shares/boards/TQBR/securities/{secid}.json"
        params = {"from": start_date.strftime("%Y-%m-%d"), "till": end_date.strftime("%Y-%m-%d"),
                  "history.columns": "TRADEDATE,SECID,CLOSE,VOLUME", "start": 0}
        all_data = []
        try:
            while True:
                response = self.session.get(url, params=params, timeout=10)
                data = response.json()
                if "history" in data and data["history"]["data"]:
                    rows = data["history"]["data"]
                    if not rows: break
                    all_data.extend(rows)
                    params["start"] += len(rows)
                else: break
            if all_data:
                df = pd.DataFrame(all_data, columns=["date", "secid", "close", "volume"])
                df["date"] = pd.to_datetime(df["date"])
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                return df.dropna().sort_values("date")
        except Exception as e:
            logger.error(f"History error {secid}: {e}")
        return pd.DataFrame()
    
    def get_multiple_history(self, secids: List[str], days: int = 365) -> Dict[str, pd.DataFrame]:
        result = {}
        for secid in secids:
            df = self.get_history(secid, days)
            if not df.empty:
                result[secid] = df
        return result

class PortfolioAnalyzer:
    def __init__(self, moex_client: MOEXClient):
        self.moex = moex_client
        self.risk_free_rate = 0.16
    
    def get_portfolio_data(self, tickers: List[str], days: int = 365) -> Tuple[pd.DataFrame, pd.DataFrame]:
        history = self.moex.get_multiple_history(tickers, days)
        if not history: return pd.DataFrame(), pd.DataFrame()
        prices = pd.DataFrame()
        for ticker, df in history.items():
            temp = df[["date", "close"]].rename(columns={"close": ticker})
            prices = temp if prices.empty else prices.merge(temp, on="date", how="outer")
        if prices.empty: return pd.DataFrame(), pd.DataFrame()
        prices = prices.set_index("date").sort_index().dropna()
        return prices, prices.pct_change().dropna()
    
    def portfolio_performance(self, weights, mean_returns, cov_matrix):
        port_return = np.sum(mean_returns * weights) * 252
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        sharpe = (port_return - self.risk_free_rate) / port_std if port_std > 0 else 0
        return port_return, port_std, sharpe
    
    def optimize_portfolio(self, tickers: List[str], days: int = 365) -> Dict:
        prices, returns = self.get_portfolio_data(tickers, days)
        if returns.empty or len(returns.columns) < 2:
            return {"error": "Недостаточно данных"}
        mean_returns, cov_matrix = returns.mean().values, returns.cov().values
        n = len(tickers)
        def neg_sharpe(w):
            ret = np.sum(mean_returns * w) * 252
            std = np.sqrt(np.dot(w.T, np.dot(cov_matrix * 252, w)))
            return -(ret - self.risk_free_rate) / std if std > 0 else 0
        result = minimize(neg_sharpe, np.array([1/n]*n), method="SLSQP",
                         bounds=tuple((0, 1) for _ in range(n)),
                         constraints={"type": "eq", "fun": lambda x: np.sum(x) - 1})
        if not result.success: return {"error": "Оптимизация не сошлась"}
        weights = result.x
        ret, vol, sharpe = self.portfolio_performance(weights, mean_returns, cov_matrix)
        portfolio = {t: {"weight": round(w*100, 2), "name": BLUE_CHIPS.get(t, t)} 
                    for t, w in zip(tickers, weights) if w > 0.001}
        return {"portfolio": portfolio, "expected_return": round(ret*100, 2),
                "volatility": round(vol*100, 2), "sharpe_ratio": round(sharpe, 3), "period_days": days}
    
    def calculate_portfolio_metrics(self, tickers: List[str], weights: List[float],
                                   investment: float = 1000000, days: int = 365) -> Dict:
        prices, returns = self.get_portfolio_data(tickers, days)
        if returns.empty: return {"error": "Недостаточно данных"}
        weights = np.array(weights) / sum(weights)
        portfolio_returns = (returns * weights).sum(axis=1)
        cumulative_return = (1 + portfolio_returns).prod() - 1
        trading_days = len(portfolio_returns)
        annual_return = ((1 + cumulative_return) ** (252 / trading_days) - 1) if trading_days > 0 else 0
        mean_returns, cov_matrix = returns.mean().values, returns.cov().values
        ret, vol, sharpe = self.portfolio_performance(weights, mean_returns, cov_matrix)
        from scipy.stats import norm
        port_std = portfolio_returns.std()
        daily_var = investment * (portfolio_returns.mean() + norm.ppf(0.05) * port_std)
        composition = []
        for i, t in enumerate(tickers):
            if weights[i] > 0.001:
                price = self.moex.get_current_price(t) or 0
                amount = investment * weights[i]
                composition.append({"ticker": t, "name": BLUE_CHIPS.get(t, t), "weight": round(weights[i]*100, 2),
                                   "amount": round(amount, 2), "price": price, "shares": int(amount/price) if price else 0})
        return {"investment": investment, "composition": composition,
                "metrics": {"historical_return": round(cumulative_return*100, 2),
                           "annualized_return": round(annual_return*100, 2),
                           "volatility": round(vol*100, 2), "sharpe_ratio": round(sharpe, 3),
                           "daily_var_95": round(abs(daily_var), 2)},
                "profit_loss": round(investment * cumulative_return, 2),
                "final_value": round(investment * (1 + cumulative_return), 2),
                "trading_days": trading_days}

def format_portfolio_result(result: Dict) -> str:
    if "error" in result: return f"❌ {result['error']}"
    text = "📊 **ОПТИМАЛЬНЫЙ ПОРТФЕЛЬ**\n━━━━━━━━━━━━━━━━━━━━\n\n**Состав:**\n"
    for t, d in result.get("portfolio", {}).items():
        text += f"• `{t}` ({d['name']}): {d['weight']}%\n"
    text += f"\n📈 Ожидаемая доходность: **{result['expected_return']}%** годовых\n"
    text += f"📉 Волатильность: **{result['volatility']}%**\n"
    text += f"⚖️ Sharpe Ratio: **{result['sharpe_ratio']}**\n"
    text += f"\n_Период анализа: {result['period_days']} дней_"
    return text

def format_metrics_result(result: Dict) -> str:
    if "error" in result: return f"❌ {result['error']}"
    m = result.get("metrics", {})
    text = f"💼 **АНАЛИЗ ПОРТФЕЛЯ**\n━━━━━━━━━━━━━━━━━━━━\n\n"
    text += "**Состав:**\n"
    for item in result.get("composition", []):
        text += f"• `{item['ticker']}`: {item['weight']}% ({item['shares']} шт. × {item['price']:.2f}₽)\n"
    profit = result.get('profit_loss', 0)
    sign = "+" if profit >= 0 else ""
    emoji = "📈" if profit >= 0 else "📉"
    text += f"\n**Доходность:**\n"
    text += f"💰 Вложено: **{result['investment']:,.0f}₽**\n"
    text += f"{emoji} Доходность: **{sign}{m.get('historical_return', 0)}%**\n"
    text += f"💵 Прибыль: **{sign}{profit:,.0f}₽**\n"
    text += f"🏦 Итого: **{result.get('final_value', 0):,.0f}₽**\n"
    text += f"\n**Риски:**\n"
    text += f"📉 Волатильность: {m.get('volatility', 0)}%\n"
    text += f"⚖️ Sharpe: {m.get('sharpe_ratio', 0)}\n"
    text += f"🎯 VaR (95%): {m.get('daily_var_95', 0):,.0f}₽/день\n"
    text += f"\n_Период: {result.get('trading_days', 0)} дней_"
    return text
