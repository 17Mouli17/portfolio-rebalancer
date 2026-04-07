import os
import json
import time
import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

llm_client = None
try:
    if HF_TOKEN:
        from openai import OpenAI
        llm_client = OpenAI(
            base_url="https://api-inference.huggingface.co/v1/",
            api_key=HF_TOKEN,
        )
except Exception:
    llm_client = None


def reset_env(task_id: int) -> dict:
    try:
        response = requests.post(
            f"{API_BASE_URL}/reset",
            json={"task_id": task_id},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise RuntimeError(f"Failed to reset env: {e}")


def step_env(action: dict) -> dict:
    try:
        response = requests.post(
            f"{API_BASE_URL}/step",
            json=action,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise RuntimeError(f"Failed to step env: {e}")


def rule_based_action(observation: dict) -> dict:
    stocks = observation["stocks"]
    balance_score = observation["balance_score"]

    if balance_score >= 0.95:
        return {"action_type": "hold", "stock_index": 0, "amount": 0.0}

    max_positive_drift = -999
    max_negative_drift = 999
    sell_idx = 0
    buy_idx = 0

    for i, stock in enumerate(stocks):
        if stock["drift"] > max_positive_drift:
            max_positive_drift = stock["drift"]
            sell_idx = i
        if stock["drift"] < max_negative_drift:
            max_negative_drift = stock["drift"]
            buy_idx = i

    if max_positive_drift > 0.02:
        return {"action_type": "sell", "stock_index": sell_idx, "amount": round(min(max_positive_drift, 0.2), 4)}
    elif max_negative_drift < -0.02:
        return {"action_type": "buy", "stock_index": buy_idx, "amount": round(min(abs(max_negative_drift), 0.2), 4)}
    else:
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
    return f"""You are a portfolio rebalancing agent. Bring allocations close to targets.

{stock_summary}

Balance Score: {observation['balance_score']} (1.0 = perfect)
Step: {observation['step_number']} / {observation['max_steps']}

If drift is positive -> sell that stock. If negative -> buy it. If score > 0.95 -> hold.

Respond ONLY with JSON like: {{"action_type": "buy", "stock_index": 1, "amount": 0.05}}"""


def get_action(observation: dict) -> dict:
    if llm_client is not None:
        try:
            response = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": build_prompt(observation)}],
                max_tokens=100,
                temperature=0.1,
            )
            content = response.choices[0].message.content.strip()
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > 0:
                action = json.loads(content[start:end])
                action.setdefault("action_type", "hold")
                action.setdefault("stock_index", 0)
                action.setdefault("amount", 0.0)
                return action
        except Exception:
            pass
    return rule_based_action(observation)


def run_task(task_id: int):
    reset_data = reset_env(task_id)
    observation = reset_data["observation"]
    info = reset_data.get("info", {})

    print(json.dumps({
        "event": "START",
        "task_id": task_id,
        "task_name": info.get("task_name", ""),
        "description": info.get("description", ""),
        "num_stocks": len(observation["stocks"]),
        "max_steps": observation["max_steps"],
        "initial_balance_score": observation["balance_score"],
        "timestamp": time.time(),
    }))

    done = False
    total_reward = 0.0
    final_score = None
    step_count = 0

    while not done:
        action = get_action(observation)
        step_data = step_env(action)
        observation = step_data["observation"]
        reward = step_data["reward"]
        done = step_data["done"]
        step_info = step_data.get("info", {})

        total_reward += reward
        step_count += 1

        if step_info.get("final_score") is not None:
            final_score = step_info["final_score"]

        print(json.dumps({
            "event": "STEP",
            "task_id": task_id,
            "step": observation["step_number"],
            "action": action,
            "reward": reward,
            "balance_score": observation["balance_score"],
            "total_transaction_costs": observation["total_transaction_costs"],
            "done": done,
        }))

    final = final_score if final_score is not None else observation["balance_score"]

    print(json.dumps({
        "event": "END",
        "task_id": task_id,
        "total_steps": step_count,
        "total_reward": round(total_reward, 4),
        "final_score": final,
        "final_balance_score": observation["balance_score"],
        "timestamp": time.time(),
    }))

    return final


def main():
    scores = {}
    for task_id in [1, 2, 3]:
        try:
            score = run_task(task_id)
            scores[f"task_{task_id}"] = score
        except Exception as e:
            print(json.dumps({
                "event": "ERROR",
                "task_id": task_id,
                "error": str(e),
            }))
            scores[f"task_{task_id}"] = 0.0
        time.sleep(1)

    print(json.dumps({
        "event": "SUMMARY",
        "scores": scores,
        "average_score": round(sum(scores.values()) / len(scores), 4),
    }))


if __name__ == "__main__":
    main()