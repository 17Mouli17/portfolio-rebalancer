from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from environment.env import PortfolioEnv
from environment.models import Action, StepResult, ResetResult, PortfolioState

app = FastAPI(title="Portfolio Rebalancer OpenEnv", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = PortfolioEnv()


class ResetRequest(BaseModel):
    task_id: int = 1


@app.get("/")
def root():
    return {"status": "ok", "environment": "Portfolio Rebalancer", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {
        "objective": "Minimize allocation drift while reducing transaction costs",
        "action_space": "buy, sell, hold with stock_index and amount",
        "reward": "balance_score - cost_penalty + time_bonus"
    }


@app.post("/reset", response_model=ResetResult)
def reset(request: ResetRequest):
    try:
        result = env.reset(task_id=request.task_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResult)
def step(action: Action):
    try:
        # Ensure environment is initialized
        if env.state is None:
            raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

        # Validate stock index
        if action.stock_index >= len(env.state.holdings):
            raise HTTPException(status_code=400, detail="Invalid stock index for current task")

        result = env.step(action)
        return result

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=PortfolioState)
def state():
    try:
        return env.state_info()
    except RuntimeError:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")


@app.get("/tasks")
def list_tasks():
    from environment.tasks import ALL_TASKS
    return {
        task_id: {
            "name": task.name,
            "description": task.description,
            "num_stocks": task.num_stocks,
            "max_steps": task.max_steps,
        }
        for task_id, task in ALL_TASKS.items()
    }