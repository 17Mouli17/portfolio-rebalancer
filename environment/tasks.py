from environment.models import TaskConfig

TASK_1 = TaskConfig(
    task_id=1,
    name="Two-Stock Basic Rebalancer",
    description="Rebalance a 2-stock portfolio to a 50/50 target allocation. No transaction costs, no price volatility. Learn the basic buy/sell mechanics.",
    num_stocks=2,
    stock_names=["Reliance", "TCS"],
    initial_holdings=[6000.0, 4000.0],
    initial_prices=[100.0, 100.0],
    target_allocations=[0.50, 0.50],
    max_steps=10,
    transaction_cost_rate=0.0,
    volatility=0.0,
    max_trade_fraction=1.0,
)

TASK_2 = TaskConfig(
    task_id=2,
    name="Four-Stock Rebalancer with Transaction Costs",
    description="Rebalance a 4-stock portfolio to a 40/30/20/10 target. Each trade incurs a 0.1% transaction cost. Minimize drift while keeping costs low.",
    num_stocks=4,
    stock_names=["Reliance", "TCS", "Gold", "Cash"],
    initial_holdings=[5000.0, 2000.0, 2000.0, 1000.0],
    initial_prices=[100.0, 100.0, 100.0, 100.0],
    target_allocations=[0.40, 0.30, 0.20, 0.10],
    max_steps=15,
    transaction_cost_rate=0.001,
    volatility=0.0,
    max_trade_fraction=1.0,
)

TASK_3 = TaskConfig(
    task_id=3,
    name="Five-Stock Rebalancer under Volatility",
    description="Rebalance a 5-stock portfolio to a 30/25/20/15/10 target under market volatility. Prices shift every step. Transaction costs apply. Max trade per step is 30% of portfolio.",
    num_stocks=5,
    stock_names=["Reliance", "TCS", "Gold", "Infosys", "Cash"],
    initial_holdings=[4000.0, 2000.0, 1500.0, 1500.0, 1000.0],
    initial_prices=[100.0, 100.0, 100.0, 100.0, 100.0],
    target_allocations=[0.30, 0.25, 0.20, 0.15, 0.10],
    max_steps=20,
    transaction_cost_rate=0.002,
    volatility=0.05,
    max_trade_fraction=0.30,
)

ALL_TASKS = {
    1: TASK_1,
    2: TASK_2,
    3: TASK_3,
}


def get_task(task_id: int) -> TaskConfig:
    if task_id not in ALL_TASKS:
        raise ValueError(f"Task {task_id} not found. Available tasks: {list(ALL_TASKS.keys())}")
    return ALL_TASKS[task_id]