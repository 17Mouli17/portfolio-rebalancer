import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
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
    task_id: Optional[int] = 1


@app.get("/")
def root():
    return {"status": "ok", "environment": "Portfolio Rebalancer", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset", response_model=ResetResult)
def reset_post(request: Optional[ResetRequest] = None):
    try:
        task_id = request.task_id if request and request.task_id else 1
        result = env.reset(task_id=task_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reset", response_model=ResetResult)
def reset_get(task_id: int = 1):
    try:
        result = env.reset(task_id=task_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResult)
def step(action: Action):
    try:
        result = env.step(action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=PortfolioState)
def state():
    try:
        return env.state_info()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()