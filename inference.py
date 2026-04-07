import os
import json
import time
import requests

API_BASE_URL = os.environ.get(
    "API_BASE_URL",
    "https://17mouli17-portfolio-rebalancer.hf.space"
)

def reset_env(task_id: int) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/reset",
        json={"task_id": task_id},
        timeout=30
    )
    response.raise_for_status()
    return response.json()

def step_env(action: dict) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/step",
        json=action,
        timeout=30
    )
    response.raise_for_status()
    return response.json()

def rule_based_action(observation: dict) -> dict:
    stocks = observation["stocks"]
    score = observation["balance_score"]

    if score >= 0.95:
        return {"action_type": "hold", "stock_index": 0, "amount": 0.0}

    drifts = [s["drift"] for s in stocks]

    max_idx = max(range(len(drifts)), key=lambda i: drifts[i])
    min_idx = min(range(len(drifts)), key=lambda i: drifts[i])

    max_drift = drifts[max_idx]
    min_drift = drifts[min_idx]

    if max_drift > 0.02:
        return {"action_type": "sell", "stock_index": max_idx, "amount": round(min(max_drift, 0.2), 4)}

    if min_drift < -0.02:
        return {"action_type": "buy", "stock_index": min_idx, "amount": round(min(abs(min_drift), 0.2), 4)}

    return {"action_type": "hold", "stock_index": 0, "amount": 0.0}

def run_task(task_id: int):
    reset_data = reset_env(task_id)
    observation = reset_data["observation"]
    info = reset_data.get("info", {})

    print(f"[START] task={task_id} name={info.get('task_name','')} score={observation['balance_score']}", flush=True)

    done = False
    steps = 0
    final_score = observation["balance_score"]

    while not done:
        action = rule_based_action(observation)
        step_data = step_env(action)

        observation = step_data["observation"]
        reward = step_data["reward"]
        done = step_data["done"]

        steps += 1
        final_score = observation["balance_score"]

        print(f"[STEP] step={steps} reward={round(reward,4)} score={round(final_score,4)}", flush=True)

    print(f"[END] task={task_id} final_score={round(final_score,4)} steps={steps}", flush=True)

    return final_score

def main():
    scores = []

    for task_id in [1, 2, 3]:
        try:
            score = run_task(task_id)
        except Exception:
            score = 0.0
            print(f"[END] task={task_id} final_score=0.0 steps=0", flush=True)
        scores.append(score)
        time.sleep(1)

    avg = round(sum(scores) / len(scores), 4)
    print(f"[SUMMARY] avg_score={avg}", flush=True)

if __name__ == "__main__":
    main()