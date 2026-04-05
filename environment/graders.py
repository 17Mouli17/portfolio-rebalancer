from typing import List
from environment.reward import compute_allocations, compute_balance_score


def grade_task_1(
    holdings: List[float],
    prices: List[float],
    target_allocations: List[float],
    step_number: int,
    max_steps: int,
) -> float:
    current_allocations = compute_allocations(holdings, prices)
    balance_score = compute_balance_score(current_allocations, target_allocations)
    efficiency_bonus = 0.1 * (1 - step_number / max_steps) if balance_score >= 0.95 else 0.0
    return round(min(1.0, balance_score + efficiency_bonus), 4)


def grade_task_2(
    holdings: List[float],
    prices: List[float],
    target_allocations: List[float],
    total_transaction_costs: float,
    total_portfolio_value: float,
    step_number: int,
    max_steps: int,
) -> float:
    current_allocations = compute_allocations(holdings, prices)
    balance_score = compute_balance_score(current_allocations, target_allocations)
    cost_ratio = total_transaction_costs / total_portfolio_value if total_portfolio_value > 0 else 0.0
    cost_penalty = min(cost_ratio * 20, 0.2)
    efficiency_bonus = 0.1 * (1 - step_number / max_steps) if balance_score >= 0.90 else 0.0
    return round(max(0.0, min(1.0, balance_score - cost_penalty + efficiency_bonus)), 4)


def grade_task_3(
    holdings: List[float],
    prices: List[float],
    target_allocations: List[float],
    total_transaction_costs: float,
    total_portfolio_value: float,
    step_number: int,
    max_steps: int,
) -> float:
    current_allocations = compute_allocations(holdings, prices)
    balance_score = compute_balance_score(current_allocations, target_allocations)
    cost_ratio = total_transaction_costs / total_portfolio_value if total_portfolio_value > 0 else 0.0
    cost_penalty = min(cost_ratio * 15, 0.25)
    time_pressure_bonus = 0.1 * (1 - step_number / max_steps) if balance_score >= 0.85 else 0.0
    return round(max(0.0, min(1.0, balance_score - cost_penalty + time_pressure_bonus)), 4)


GRADERS = {
    1: grade_task_1,
    2: grade_task_2,
    3: grade_task_3,
}


def grade(task_id: int, **kwargs) -> float:
    if task_id not in GRADERS:
        raise ValueError(f"No grader found for task_id={task_id}")
    return GRADERS[task_id](**kwargs)