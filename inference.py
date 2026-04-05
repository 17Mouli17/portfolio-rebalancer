import os
import json
import time
import requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

client = None
if HF_TOKEN:
    client = OpenAI(
        base_url="https://api-inference.huggingface.co/v1/",
        api_key=HF_TOKEN,
    )


def reset_env(task_id: int) -> dict:
    response = requests.post(f"{API_BASE_URL}/reset", json={"task_id": task_id})
    response.raise_for_status()
    return response.json()


def step_env(action: dict) -> dict:
    response = requests.post(f"{API_BASE_URL}/step", json=action)
    response.raise_for_status()
    return response.json()


def rule_based_action(observation: dict) -> dict:
    stocks = observation["stocks"]
    balance_score = observation["balance_score"]
    cost = observation.get("transaction_cost_rate", 0.0)

    if balance_score > 0.97:
        return {"action_type": "hold", "stock_index": 0, "amount": 0.0}

    drifts = [s["drift"] for s in stocks]

    max_idx = max(range(len(stocks)), key=lambda i: drifts[i])
    min_idx = min(range(len(stocks)), key=lambda i: drifts[i])

    max_drift = drifts[max_idx]
    min_drift = drifts[min_idx]

    if abs(max_drift) > 0.08 or abs(min_drift) > 0.08:
        amount = 0.12
    elif abs(max_drift) > 0.04 or abs(min_drift) > 0.04:
        amount = 0.08
    else:
        amount = 0.04

    if cost > 0:
        amount *= 0.7

    if max_drift > 0.02:
        return {
            "action_type": "sell",
            "stock_index": max_idx,
            "amount": round(amount, 3)
        }

    if min_drift < -0.02:
        return {
            "action_type": "buy",
            "stock_index": min_idx,
            "amount": round(amount, 3)
        }

    return {"action_type": "hold", "stock_index": 0, "amount": 0.0}


def build_prompt(observation: dict) -> str:
    stocks = observation["stocks"]
    lines = []
    for s in stocks:
        lines.append(
            f"- {s['name']}: current={round(s['current_allocation']*100, 1)}%, "
            f"target={round(s['target_allocation']*100, 1)}%, "
            f"drift={round(s['drift']*100, 2)}%"
        )
    stock_summary = "\n".join(lines)

    return f"""You are a portfolio rebalancing agent.

{stock_summary}

Balance Score: {observation['balance_score']}

Respond ONLY JSON:
{{"action_type": "buy", "stock_index": 1, "amount": 0.1}}"""


def llm_action(observation: dict) -> dict:
    if client is None:
        return None

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": build_prompt(observation)}],
            max_tokens=100,
            temperature=0.1,
        )

        content = response.choices[0].message.content.strip()
        start = content.find("{")
        end = content.rfind("}") + 1

        if start != -1 and end != 0:
            return json.loads(content[start:end])

    except Exception:
        return None

    return None


def get_action(observation: dict) -> dict:
    action = llm_action(observation)
    if action is not None:
        return action
    return rule_based_action(observation)


def run_task(task_id: int):
    reset_data = reset_env(task_id)
    observation = reset_data["observation"]
    info = reset_data.get("info", {})

    print(json.dumps({
        "event": "START",
        "task_id": task_id,
        "task_name": info.get("task_name", ""),
        "initial_balance_score": observation["balance_score"],
    }))

    done = False
    total_reward = 0.0
    final_score = None

    while not done:
        action = get_action(observation)

        step_data = step_env(action)
        observation = step_data["observation"]
        reward = step_data["reward"]
        done = step_data["done"]
        step_info = step_data.get("info", {})

        total_reward += reward

        if step_info.get("final_score") is not None:
            final_score = step_info["final_score"]

        print(json.dumps({
            "step": observation["step_number"],
            "action": action,
            "reward": reward,
            "balance_score": observation["balance_score"],
        }))

    print(json.dumps({
        "event": "END",
        "task_id": task_id,
        "final_score": final_score,
    }))

    return final_score


def main():
    scores = {}
    for task_id in [1, 2, 3]:
        score = run_task(task_id)
        scores[f"task_{task_id}"] = score
        time.sleep(1)

    print(json.dumps({
        "scores": scores,
        "average_score": round(sum(scores.values()) / len(scores), 4),
    }))


if __name__ == "__main__":
    main()