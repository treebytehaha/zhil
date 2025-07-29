import os
import uuid
import shutil
from typing import Any, List, Optional, Dict

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from langchain.llms.base import LLM
from langchain.agents import Tool, initialize_agent, AgentType

# ========================
# 自定义 DashScope LLM
class DashScopeLLM(LLM):
    model: str = "qwen-turbo"
    dashscope_api_key: str

    class Config:
        extra = "allow"

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
        resp = Generation.call(model=self.model, messages=messages)
        return resp["output"]["text"]

# ========================
# 读表工具函数
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
# 初始化 LLM & Agent
api_key = os.getenv("DASHSCOPE_API_KEY", "sk-51341d6718514f128743252bc13b5651")
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
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    max_iterations=3,
    early_stopping_method="generate",
    verbose=True
)

# ========================
# FastAPI
app = FastAPI(title="Data Chat Backend", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 全局存储（内存）
CURRENT_FILE_PATH: Optional[str] = None   # 兼容老接口
# sessions[session_id] = {
#   "messages": [ {"role":"user|assistant","content": "..."} ],
#   "docs": { doc_id: {"filename": "...", "path": "...", "uploaded_at": float_timestamp} },
#   "current_doc_id": Optional[str]
# }
sessions: Dict[str, Dict[str, Any]] = {}

# ========================
# Pydantic 模型
class Command(BaseModel):
    command: str

class SessionResponse(BaseModel):
    session_id: str

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class AnalyzeInSession(BaseModel):
    command: str
    doc_id: Optional[str] = None    # 指定分析的文档；为空则自动选
    sheet_name: Optional[Any] = 0   # 可选：Excel 的 sheet
    date_column: Optional[str] = None  # 若需要用到日期列统计
    # 其他可扩展参数...

class SetCurrentDocReq(BaseModel):
    doc_id: str

# ========================
# 旧接口（保留兼容）
@app.post("/api/upload-excel")
async def upload_excel(file: UploadFile = File(...)):
    """
    旧接口：上传一个 Excel，写到 CURRENT_FILE_PATH。
    新前端已使用会话内上传接口；保留以兼容。
    """
    global CURRENT_FILE_PATH
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)
    CURRENT_FILE_PATH = save_path
    return {"status": "ok", "file_path": save_path}

def _pick_latest_excel_global() -> Optional[str]:
    latest_path, latest_ts = None, -1
    for sess in sessions.values():
        for v in sess["docs"].values():
            p = v["path"].lower()
            if p.endswith((".xlsx", ".xls")) and os.path.exists(v["path"]):
                ts = v.get("uploaded_at", 0)
                if ts > latest_ts:
                    latest_ts = ts
                    latest_path = v["path"]
    return latest_path

@app.post("/api/analyze")
async def analyze(cmd: Command):
    global CURRENT_FILE_PATH
    file_path = CURRENT_FILE_PATH
    if not file_path or not os.path.exists(file_path):
        file_path = _pick_latest_excel_global()

    if not file_path:
        return {"result": "请先上传 Excel 文件"}

    query = f"文件路径是 {file_path}，{cmd.command}"
    try:
        result = agent.run(query)
    except Exception as e:
        result = f"分析出错: {e}"
    return {"result": result, "file_path": file_path}


# ========================
# 多会话接口
@app.post("/api/chat/sessions", response_model=SessionResponse)
async def create_session():
    """
    创建会话，返回 session_id
    """
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"messages": [], "docs": {}, "current_doc_id": None}
    return {"session_id": session_id}

@app.get("/api/chat/sessions")
async def list_sessions():
    """
    列出所有会话
    """
    return {"sessions": list(sessions.keys())}

@app.delete("/api/chat/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    删除会话及其文件
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)
    del sessions[session_id]
    return {"status": "deleted"}

# ========================
# 消息接口
@app.post("/api/chat/sessions/{session_id}/messages")
async def post_message(session_id: str, msg: ChatMessage):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    if msg.role not in ("user", "assistant"):
        raise HTTPException(status_code=400, detail="role must be 'user' or 'assistant'")
    entry = msg.dict()
    sessions[session_id]["messages"].append(entry)
    return {"message": entry}

@app.get("/api/chat/sessions/{session_id}/messages")
async def get_messages(session_id: str, offset: int = 0, limit: int = 50):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    msgs = sessions[session_id]["messages"]
    return {"messages": msgs[offset: offset + limit]}

# ========================
# 文档接口
@app.post("/api/chat/sessions/{session_id}/docs")
async def upload_doc(session_id: str, file: UploadFile = File(...)):
    """
    上传文档（任何类型），并存入会话
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    doc_id = str(uuid.uuid4())
    save_path = os.path.join(session_dir, f"{doc_id}_{file.filename}")
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    sessions[session_id]["docs"][doc_id] = {
        "filename": file.filename,
        "path": save_path,
        "uploaded_at": os.path.getmtime(save_path),
    }
    return {"doc_id": doc_id, "filename": file.filename}

@app.get("/api/chat/sessions/{session_id}/docs")
async def list_docs(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    docs = sessions[session_id]["docs"]
    return {
        "docs": [
            {"doc_id": k, "filename": v["filename"], "uploaded_at": v.get("uploaded_at")}
            for k, v in docs.items()
        ]
    }

@app.get("/api/chat/sessions/{session_id}/docs/{doc_id}")
async def get_doc(session_id: str, doc_id: str):
    if session_id not in sessions or doc_id not in sessions[session_id]["docs"]:
        raise HTTPException(status_code=404, detail="Doc not found")
    doc = sessions[session_id]["docs"][doc_id]
    return FileResponse(path=doc["path"], filename=doc["filename"])

# 设置/获取当前分析文档
@app.put("/api/chat/sessions/{session_id}/current-doc")
async def set_current_doc(session_id: str, req: SetCurrentDocReq):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    if req.doc_id not in sessions[session_id]["docs"]:
        raise HTTPException(status_code=404, detail="Doc not found")
    sessions[session_id]["current_doc_id"] = req.doc_id
    return {"status": "ok", "current_doc_id": req.doc_id}

@app.get("/api/chat/sessions/{session_id}/current-doc")
async def get_current_doc(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"current_doc_id": sessions[session_id].get("current_doc_id")}

# ========================
# 会话内分析
def _pick_excel_for_session(session_id: str) -> Optional[str]:
    """
    选择一个可分析的 Excel 文件路径：
    1) 若会话设置了 current_doc_id，且对应文件为 .xlsx / .xls，则使用；
    2) 否则选择该会话中最新上传且扩展名为 .xlsx/.xls 的文件；
    3) 若没有，返回 None。
    """
    sess = sessions.get(session_id)
    if not sess:
        return None

    # 优先 current_doc_id
    cur_id = sess.get("current_doc_id")
    if cur_id:
        doc = sess["docs"].get(cur_id)
        if doc:
            path = doc["path"]
            if path.lower().endswith((".xlsx", ".xls")) and os.path.exists(path):
                return path

    # 其次最新上传的 Excel
    excel_docs = [
        v for v in sess["docs"].values()
        if v["path"].lower().endswith((".xlsx", ".xls")) and os.path.exists(v["path"])
    ]
    if not excel_docs:
        return None
    # 根据上传时间排序
    excel_docs.sort(key=lambda d: d.get("uploaded_at", 0), reverse=True)
    return excel_docs[0]["path"]

@app.post("/api/chat/sessions/{session_id}/analyze")
async def analyze_in_session(session_id: str, req: AnalyzeInSession):
    """
    在指定会话中执行分析。
    - 优先使用 req.doc_id 对应文件；
    - 否则自动选择该会话最新 Excel；
    - 若会话里没有 Excel，回退到旧接口的 CURRENT_FILE_PATH（便于过渡）；
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # 选择文件路径
    file_path: Optional[str] = None
    if req.doc_id:
        doc = sessions[session_id]["docs"].get(req.doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Doc not found")
        file_path = doc["path"]
    else:
        file_path = _pick_excel_for_session(session_id)

    # 仍未找到就回退到旧的 CURRENT_FILE_PATH
    if not file_path:
        if CURRENT_FILE_PATH and os.path.exists(CURRENT_FILE_PATH):
            file_path = CURRENT_FILE_PATH
        else:
            raise HTTPException(status_code=400, detail="未找到可用的 Excel，请先上传。")

    query = f"文件路径是 {file_path}，{req.command}"
    try:
        result = agent.run(query)
    except Exception as e:
        result = f"分析出错: {e}"

    # 保存消息（用户 -> 助手）
    sessions[session_id]["messages"].append({"role": "user", "content": req.command})
    sessions[session_id]["messages"].append({"role": "assistant", "content": result})

    return {"result": result, "file_path": file_path}

# ========================
# 启动
if __name__ == "__main__":
    # 为了避免 dashscope 未设置 API key 导致报错，这里给出提示
    if api_key == "YOUR_API_KEY":
        print("[WARN] DASHSCOPE_API_KEY 未设置，记得在环境变量中配置。")
    uvicorn.run(app, host="0.0.0.0", port=8000)
