from typing import List


def compute_allocations(holdings: List[float], prices: List[float]) -> List[float]:
    values = [h * p for h, p in zip(holdings, prices)]
    total = sum(values)
    if total == 0:
        return [0.0] * len(holdings)
    return [v / total for v in values]


def compute_balance_score(current_allocations: List[float], target_allocations: List[float]) -> float:
    total_drift = sum(abs(c - t) for c, t in zip(current_allocations, target_allocations))
    return round(max(0.0, 1.0 - total_drift), 4)


def compute_drift(current_allocations: List[float], target_allocations: List[float]) -> List[float]:
    return [round(c - t, 4) for c, t in zip(current_allocations, target_allocations)]


def compute_step_reward(
    current_allocations: List[float],
    target_allocations: List[float],
    transaction_cost: float,
    transaction_cost_rate: float,
    step_number: int,
    max_steps: int,
) -> float:
    balance_score = compute_balance_score(current_allocations, target_allocations)
    cost_penalty = min(transaction_cost * 10, 0.3)
    time_bonus = 0.05 * (1 - step_number / max_steps) if balance_score > 0.95 else 0.0
    reward = balance_score - cost_penalty + time_bonus
    return round(max(0.0, min(1.0, reward)), 4)