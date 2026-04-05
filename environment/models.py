from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class ActionType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class Action(BaseModel):
    action_type: ActionType
    stock_index: int = Field(ge=0, le=4)
    amount: float = Field(ge=0.0, le=1.0)


class StockInfo(BaseModel):
    name: str
    current_value: float
    target_allocation: float
    current_allocation: float
    drift: float


class Observation(BaseModel):
    stocks: List[StockInfo]
    total_portfolio_value: float
    step_number: int
    max_steps: int
    transaction_cost_rate: float
    total_transaction_costs: float
    balance_score: float


class PortfolioState(BaseModel):
    task_id: int
    holdings: List[float]
    prices: List[float]
    target_allocations: List[float]
    stock_names: List[str]
    step_number: int
    max_steps: int
    transaction_cost_rate: float
    total_transaction_costs: float
    done: bool
    reward: float
    observation: Optional[Observation] = None


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict


class ResetResult(BaseModel):
    observation: Observation
    info: dict


class TaskConfig(BaseModel):
    task_id: int
    name: str
    description: str
    num_stocks: int
    stock_names: List[str]
    initial_holdings: List[float]
    initial_prices: List[float]
    target_allocations: List[float]
    max_steps: int
    transaction_cost_rate: float
    volatility: float
    max_trade_fraction: float