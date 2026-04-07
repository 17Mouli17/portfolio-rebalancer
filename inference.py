import os
import time
import requests

API_BASE_URL = os.environ.get(
    "API_BASE_URL",
    "https://17mouli17-portfolio-rebalancer.hf.space"
)

def reset_env(task_id):
    r = requests.post(f"{API_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def step_env(action):
    r = requests.post(f"{API_BASE_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()

def get_action(obs):
    stocks = obs["stocks"]
    score = obs["balance_score"]

    if score >= 0.95:
        return {"action_type": "hold", "stock_index": 0, "amount": 0.0}

    drifts = [s["drift"] for s in stocks]

    max_i = max(range(len(drifts)), key=lambda i: drifts[i])
    min_i = min(range(len(drifts)), key=lambda i: drifts[i])

    if drifts[max_i] > 0.02:
        return {"action_type": "sell", "stock_index": max_i, "amount": 0.1}

    if drifts[min_i] < -0.02:
        return {"action_type": "buy", "stock_index": min_i, "amount": 0.1}

    return {"action_type": "hold", "stock_index": 0, "amount": 0.0}

def run_task(task_id):
    try:
        data = reset_env(task_id)
    except Exception as e:
        print(f"[START] task={task_id}", flush=True)
        print(f"[END] task={task_id} final_score=0.0 steps=0", flush=True)
        return 0.0

    obs = data["observation"]
    print(f"[START] task={task_id}", flush=True)

    done = False
    steps = 0
    score = obs["balance_score"]

    while not done:
        try:
            action = get_action(obs)
            step = step_env(action)
        except Exception:
            break

        obs = step["observation"]
        reward = step["reward"]
        done = step["done"]

        steps += 1
        score = obs["balance_score"]

        print(f"[STEP] step={steps} reward={round(reward,4)} score={round(score,4)}", flush=True)

    print(f"[END] task={task_id} final_score={round(score,4)} steps={steps}", flush=True)

    return score

def main():
    scores = []

    for t in [1, 2, 3]:
        s = run_task(t)
        scores.append(s)
        time.sleep(1)

    avg = round(sum(scores) / len(scores), 4)
    print(f"[SUMMARY] avg_score={avg}", flush=True)

if __name__ == "__main__":
    main()