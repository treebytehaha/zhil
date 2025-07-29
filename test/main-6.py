import os
import pandas as pd
from typing import Any, List, Optional
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from langchain.llms.base import LLM
from langchain.agents import Tool, initialize_agent, AgentType


# ========================
# 自定义 DashScope LLM（新版接口）
class DashScopeLLM(LLM):
    model: str = "qwen-turbo"
    dashscope_api_key: str

    class Config:
        extra = "allow"  # 允许额外字段，避免 pydantic 报错

    @property
    def _llm_type(self) -> str:
        return "dashscope_custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        import dashscope
        from dashscope import Generation

        dashscope.api_key = self.dashscope_api_key

        messages = [
            {"role": "system", "content": "你是一个专业的excel数据分析助手"},
            {"role": "user", "content": prompt}
        ]

        response = Generation.call(
            model=self.model,
            messages=messages
        )

        return response['output']['text']


# ========================
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


# ========================
# 初始化 LLM 和 Agent
api_key ='sk-51341d6718514f128743252bc13b5651'

llm = DashScopeLLM(dashscope_api_key=api_key)

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
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # 使用结构化聊天Agent
    verbose=True,
    max_iterations=5,  # 限制迭代次数
    early_stopping_method="generate",
    handle_parsing_errors=True,
    return_intermediate_steps=False  # 不返回中间步骤
)


# ========================
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

@app.post("/api/ask")
async def ask_question(q: Question):
    global agent
    if agent is None:
        return {"status": "error", "answer": "请先上传FFR数据文件"}
    
    try:
        # 更清晰的问题提示模板
        prompt_template = """请严格按照以下格式回答：
        Question: {question}
        Thought: 逐步分析问题
        Final Answer: 最终答案（必须是纯文本或数字）
        """
        
        response = agent.run(
            input=prompt_template.format(question=q.question),
            stop=["Final Answer:"]
        )
        
        # 强化输出解析
        if "Final Answer:" in response:
            answer = response.split("Final Answer:")[-1].strip()
        elif "最终答案:" in response:
            answer = response.split("最终答案:")[-1].strip()
        else:
            answer = response.strip()
            
        # 清理答案中的调试信息
        answer = answer.split("For troubleshooting")[0].strip()
        
        return {"status": "success", "answer": answer}
        
    except Exception as e:
        return {"status": "error", "answer": f"分析失败: {str(e)}"}
# 启动服务
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
