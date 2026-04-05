import random
from typing import List
from environment.models import (
    Action, ActionType, Observation, PortfolioState,
    StepResult, ResetResult, StockInfo, TaskConfig
)
from environment.tasks import get_task
from environment.reward import compute_allocations, compute_balance_score, compute_drift, compute_step_reward
from environment.graders import grade


class PortfolioEnv:
    def __init__(self, seed: int = None):
        self.state: PortfolioState = None
        if seed is not None:
            random.seed(seed)

    def reset(self, task_id: int = 1) -> ResetResult:
        task: TaskConfig = get_task(task_id)
        self.state = PortfolioState(
            task_id=task_id,
            holdings=list(task.initial_holdings),
            prices=list(task.initial_prices),
            target_allocations=list(task.target_allocations),
            stock_names=list(task.stock_names),
            step_number=0,
            max_steps=task.max_steps,
            transaction_cost_rate=task.transaction_cost_rate,
            total_transaction_costs=0.0,
            done=False,
            reward=0.0,
        )
        obs = self._build_observation()
        self.state.observation = obs

        return ResetResult(
            observation=obs,
            info={
                "task_id": task_id,
                "task_name": task.name,
                "description": task.description,
                "goal": "Minimize allocation drift while reducing transaction costs"
            }
        )

    def step(self, action: Action) -> StepResult:
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call /reset first.")
        if self.state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        task: TaskConfig = get_task(self.state.task_id)
        transaction_cost = 0.0

        # --- FIXED INDEX HANDLING ---
        if action.action_type != ActionType.HOLD:
            idx = action.stock_index

            # FIX: correct validation using number of stocks
            if idx >= len(self.state.holdings):
                idx = 0  # fallback instead of crash

            total_value = sum(h * p for h, p in zip(self.state.holdings, self.state.prices))
            trade_amount = min(action.amount, task.max_trade_fraction) * total_value

            if action.action_type == ActionType.BUY:
                shares_to_buy = trade_amount / self.state.prices[idx]
                transaction_cost = trade_amount * self.state.transaction_cost_rate
                self.state.holdings[idx] += shares_to_buy

            elif action.action_type == ActionType.SELL:
                current_value = self.state.holdings[idx] * self.state.prices[idx]
                sell_value = min(trade_amount, current_value * 0.99)
                shares_to_sell = sell_value / self.state.prices[idx]
                transaction_cost = sell_value * self.state.transaction_cost_rate
                self.state.holdings[idx] = max(0.0, self.state.holdings[idx] - shares_to_sell)

            self.state.total_transaction_costs += transaction_cost

        # --- PRICE VOLATILITY ---
        if task.volatility > 0.0:
            self.state.prices = [
                max(1.0, p * (1 + random.uniform(-task.volatility, task.volatility)))
                for p in self.state.prices
            ]

        self.state.step_number += 1

        # --- REWARD ---
        current_allocations = compute_allocations(self.state.holdings, self.state.prices)
        reward = compute_step_reward(
            current_allocations=current_allocations,
            target_allocations=self.state.target_allocations,
            transaction_cost=transaction_cost,
            transaction_cost_rate=self.state.transaction_cost_rate,
            step_number=self.state.step_number,
            max_steps=self.state.max_steps,
        )
        self.state.reward = reward

        # --- DONE CONDITION ---
        balance_score = compute_balance_score(current_allocations, self.state.target_allocations)

        done = self.state.step_number >= self.state.max_steps

        # OPTIONAL improvement: early stopping
        if balance_score > 0.98:
            done = True

        self.state.done = done

        obs = self._build_observation()
        self.state.observation = obs

        total_value = sum(h * p for h, p in zip(self.state.holdings, self.state.prices))

        # --- FINAL SCORING ---
        final_score = None
        if done:
            if self.state.task_id == 1:
                final_score = grade(
                    task_id=1,
                    holdings=self.state.holdings,
                    prices=self.state.prices,
                    target_allocations=self.state.target_allocations,
                    step_number=self.state.step_number,
                    max_steps=self.state.max_steps,
                )
            elif self.state.task_id == 2:
                final_score = grade(
                    task_id=2,
                    holdings=self.state.holdings,
                    prices=self.state.prices,
                    target_allocations=self.state.target_allocations,
                    total_transaction_costs=self.state.total_transaction_costs,
                    total_portfolio_value=total_value,
                    step_number=self.state.step_number,
                    max_steps=self.state.max_steps,
                )
            elif self.state.task_id == 3:
                final_score = grade(
                    task_id=3,
                    holdings=self.state.holdings,
                    prices=self.state.prices,
                    target_allocations=self.state.target_allocations,
                    total_transaction_costs=self.state.total_transaction_costs,
                    total_portfolio_value=total_value,
                    step_number=self.state.step_number,
                    max_steps=self.state.max_steps,
                )

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "step": self.state.step_number,
                "transaction_cost": round(transaction_cost, 4),
                "total_transaction_costs": round(self.state.total_transaction_costs, 4),
                "final_score": final_score,
            }
        )

    def state_info(self) -> PortfolioState:
        if self.state is None:
            raise RuntimeError("Call reset() before state()")
        return self.state

    def _build_observation(self) -> Observation:
        current_allocations = compute_allocations(self.state.holdings, self.state.prices)
        drifts = compute_drift(current_allocations, self.state.target_allocations)
        balance_score = compute_balance_score(current_allocations, self.state.target_allocations)
        total_value = sum(h * p for h, p in zip(self.state.holdings, self.state.prices))

        stocks = [
            StockInfo(
                name=self.state.stock_names[i],
                current_value=round(self.state.holdings[i] * self.state.prices[i], 2),
                target_allocation=round(self.state.target_allocations[i], 4),
                current_allocation=round(current_allocations[i], 4),
                drift=drifts[i],
            )
            for i in range(len(self.state.holdings))
        ]

        return Observation(
            stocks=stocks,
            total_portfolio_value=round(total_value, 2),
            step_number=self.state.step_number,
            max_steps=self.state.max_steps,
            transaction_cost_rate=self.state.transaction_cost_rate,
            total_transaction_costs=round(self.state.total_transaction_costs, 4),
            balance_score=balance_score,
        )