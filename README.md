---
title: Portfolio Rebalancer OpenEnv
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Portfolio Rebalancer — OpenEnv Environment

A real-world reinforcement learning environment where an AI agent learns to rebalance a stock portfolio to match a target allocation. Built using the OpenEnv framework by Meta and Hugging Face.

## Objective

The goal of this environment is to train an RL agent to:

- Minimize allocation drift from target portfolio weights
- Minimize transaction costs incurred during trading
- Achieve optimal portfolio balance in the fewest steps

## Problem Overview

A portfolio rebalancer manages a set of assets and must continuously buy or sell to keep allocations close to a predefined target. This mirrors real-world systems used in mutual funds, ETFs, and robo-advisors.

At each step, the agent observes current allocations, drift from target allocations, and transaction costs — then decides whether to buy, sell, or hold.

## Tasks

| Task | Difficulty | Stocks | Max Steps | Transaction Cost | Volatility |
|------|------------|--------|-----------|-----------------|------------|
| 1 | Easy | 2 (Reliance, TCS) | 10 | 0% | None |
| 2 | Medium | 4 (Reliance, TCS, Gold, Cash) | 15 | 0.1% | None |
| 3 | Hard | 5 (Reliance, TCS, Gold, Infosys, Cash) | 20 | 0.2% | 5% per step |

Each task increases complexity by introducing more assets, transaction costs, and market volatility.

## Action Space

```json
{
  "action_type": "buy | sell | hold",
  "stock_index": 0,
  "amount": 0.05
}
```

| Field | Type | Description |
|-------|------|-------------|
| action_type | string | One of: buy, sell, hold |
| stock_index | integer | Index of target stock (0 to num_stocks-1) |
| amount | float | Fraction of total portfolio value to trade (0.0 to 1.0) |

## Observation Space

```json
{
  "stocks": [
    {
      "name": "Reliance",
      "current_value": 6000.0,
      "target_allocation": 0.50,
      "current_allocation": 0.60,
      "drift": 0.10
    }
  ],
  "total_portfolio_value": 10000.0,
  "step_number": 1,
  "max_steps": 10,
  "transaction_cost_rate": 0.0,
  "total_transaction_costs": 0.0,
  "balance_score": 0.80
}
```

## Reward Function

- Base reward = balance score (how close allocations are to target)
- Cost penalty = deducted for transaction costs incurred
- Time bonus = small bonus for achieving balance early in the episode
- Final score = graded at episode end between 0.0 and 1.0

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /reset | Start a new episode. Body: `{"task_id": 1}` |
| POST | /step | Take an action. Body: Action object |
| GET | /state | Get current internal environment state |
| GET | /tasks | List all available tasks |
| GET | /health | Health check |

## Setup and Running Locally

```bash
git clone <your-repo-url>
cd portfolio-rebalancer-env

pip install -r requirements.txt

uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Running with Docker

```bash
docker build -t portfolio-rebalancer .
docker run -p 7860:7860 portfolio-rebalancer
```

## Running Inference

```bash
export API_BASE_URL=http://localhost:7860
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export HF_TOKEN=your_hf_token_here

python inference.py
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| API_BASE_URL | The running environment server URL |
| MODEL_NAME | LLM model identifier for inference |
| HF_TOKEN | Hugging Face API token |

## Project Structure

```
portfolio-rebalancer-env/
├── environment/
│   ├── __init__.py
│   ├── env.py
│   ├── models.py
│   ├── tasks.py
│   ├── reward.py
│   └── graders.py
├── server/
│   ├── __init__.py
│   └── app.py
├── inference.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```
## Scoring

Submissions are evaluated by running `inference.py` against all 3 tasks. Each task returns a score between 0.0 and 1.0. The final evaluation checks runtime correctness, OpenEnv spec compliance, task design quality, and grading logic.