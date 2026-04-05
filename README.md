---
title: Portfolio Rebalancer OpenEnv
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 📈 Portfolio Rebalancer — OpenEnv Environment

A real-world reinforcement learning environment where an AI agent learns to rebalance a stock portfolio to match a target allocation. Built using the OpenEnv framework.

## 🎯 Objective

The goal of this environment is to train an RL agent to:

- Minimize allocation drift from target portfolio weights  
- Minimize transaction costs incurred during trading  
- Achieve optimal portfolio balance in the fewest steps  

## 🧠 Problem Overview

A portfolio rebalancer manages a set of assets and must continuously buy or sell to keep allocations close to a predefined target.

This mirrors real-world systems used in:
- Mutual funds  
- ETFs  
- Robo-advisors  

At each step, the agent observes:
- Current allocations  
- Drift from target allocations  
- Transaction costs  

…and decides whether to:
- **Buy**
- **Sell**
- **Hold**

## 🧩 Tasks

| Task | Difficulty | Stocks | Max Steps | Transaction Cost | Volatility |
|------|------------|--------|-----------|-----------------|------------|
| 1 | Easy | 2 (Reliance, TCS) | 10 | 0% | None |
| 2 | Medium | 4 (Reliance, TCS, Gold, Cash) | 15 | 0.1% | None |
| 3 | Hard | 5 (Reliance, TCS, Gold, Infosys, Cash) | 20 | 0.2% | 5% per step |

Each task increases complexity by introducing:
- More assets  
- Transaction costs  
- Market volatility  

## ⚙️ Action Space

```json
{
  "action_type": "buy | sell | hold",
  "stock_index": 0,
  "amount": 0.05
}