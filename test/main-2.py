import os
import pandas as pd
import requests
from typing import Any, List, Optional
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from langchain.llms.base import LLM
from langchain.agents import Tool, initialize_agent, AgentType

# ===============================
# 自定义 DashScope LLM (不使用 super().__init__)
class DashScope(LLM):
    def __init__(self, api_key: str):
        object.__setattr__(self, "cache", None)
        object.__setattr__(self, "tags", [])
        object.__setattr__(self, "metadata", {})
        object.__setattr__(self, "verbose", False)
        object.__setattr__(self, "callbacks", None)
        # 不调用 BaseModel 的初始化，避免 Pydantic 校验
        object.__setattr__(self, "_api_key", api_key)

    @property
    def _llm_type(self) -> str:
        return "dashscope"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload = {"prompt": prompt, "max_tokens": 512}
        if stop:
            payload["stop"] = stop
        response = requests.post(
            "https://api.dashscope.ai/v1/generate",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        return data.get("text", "")



# ===============================
# 工具函数
def count_period_variable_occurrences(file_path: str, sheet_name: Any, date_column: str,
                                      period: str, column: str, variable: Any) -> int:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df[date_column] = pd.to_datetime(df[date_column])
    period_dt = pd.to_datetime(period)
    mask = df[date_column].dt.to_period("M") == period_dt.to_period("M")
    return int((df.loc[mask, column] == variable).sum())

def compare_period_variable_counts(file_path: str, sheet_name: Any, date_column: str,
                                   column: str, variable: Any, period1: str, period2: str) -> int:
    c1 = count_period_variable_occurrences(file_path, sheet_name, date_column, period1, column, variable)
    c2 = count_period_variable_occurrences(file_path, sheet_name, date_column, period2, column, variable)
    return c2 - c1

def top_k_variables_in_period(file_path: str, sheet_name: Any, date_column: str,
                              period: str, column: str, k: int) -> List[tuple]:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df[date_column] = pd.to_datetime(df[date_column])
    period_dt = pd.to_datetime(period)
    mask = df[date_column].dt.to_period("M") == period_dt.to_period("M")
    counts = df.loc[mask, column].value_counts().head(k)
    return list(counts.items())

def error_tool(args: dict) -> str:
    return f"错误：{args.get('message', '未知错误')}"

# ===============================
# 初始化 LLM 和 Agent
api_key = 'sk-51341d6718514f128743252bc13b5651'  # ← 这里填你的 DashScope API Key
llm = DashScope(api_key=api_key)

tools = [
    Tool(
        name="count_period_variable",
        func=lambda args: count_period_variable_occurrences(
            file_path=args["file_path"],
            sheet_name=args.get("sheet_name", 0),
            date_column=args["date_column"],
            period=args["period"],
            column=args["column"],
            variable=args["variable"],
        ),
        description="统计指定年月下某列中变量出现的次数。"
    ),
    Tool(
        name="compare_variable_counts",
        func=lambda args: compare_period_variable_counts(
            file_path=args["file_path"],
            sheet_name=args.get("sheet_name", 0),
            date_column=args["date_column"],
            column=args["column"],
            variable=args["variable"],
            period1=args["period1"],
            period2=args["period2"],
        ),
        description="比较两个不同年月下同一列同一变量的出现次数差值。"
    ),
    Tool(
        name="top_k_variables",
        func=lambda args: top_k_variables_in_period(
            file_path=args["file_path"],
            sheet_name=args.get("sheet_name", 0),
            date_column=args["date_column"],
            period=args["period"],
            column=args["column"],
            k=args["k"],
        ),
        description="找出某列中出现次数最多的前 K 个变量。"
    ),
    Tool(
        name="error",
        func=lambda args: error_tool(args),
        description="当解析失败或操作不支持时，返回错误提示。"
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ===============================
# FastAPI 后端
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
CURRENT_FILE_PATH = None

@app.post("/api/upload-excel")
async def upload_excel(file: UploadFile = File(...)):
    global CURRENT_FILE_PATH
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)
    CURRENT_FILE_PATH = save_path
    return {"status": "ok", "file_path": save_path}

class Command(BaseModel):
    command: str

@app.post("/api/analyze")
async def analyze(cmd: Command):
    global CURRENT_FILE_PATH
    if not CURRENT_FILE_PATH:
        return {"result": "请先上传 Excel 文件"}
    query = f"文件路径是 {CURRENT_FILE_PATH}，{cmd.command}"
    try:
        result = agent.run(query)
    except Exception as e:
        result = f"分析出错: {e}"
    return {"result": result}

# 启动
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
