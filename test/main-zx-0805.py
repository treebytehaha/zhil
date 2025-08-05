import os
from unittest import result
import uuid
import shutil
from typing import Any, List, Optional, Dict
import json
from typing import Tuple
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from pydantic import BaseModel, Field, ValidationError
from langchain.llms.base import LLM
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.agents.structured_chat.output_parser import (
    StructuredChatOutputParserWithRetries,
)
#记忆模块
from langchain.memory import ConversationBufferMemory           # ⏳ 短期记忆
from langchain.vectorstores import Chroma                       # 🗄️ 向量数据库
from langchain.embeddings import DashScopeEmbeddings            # 向量化模型
from langchain.memory import VectorStoreRetrieverMemory         # 🧠 长期记忆包装

from passwd import app_id, api_key  # 从 passwd.py 中导入 app_id 和 api_key

from langchain.tools import StructuredTool
from tools.tools import (
    count_period_variable_occurrences,
    count_period_variable_wrapped,
    compare_period_variable_counts,
    compare_variable_counts_wrapped,
    top_k_variables_in_period,
    top_k_variables_wrapped,
    count_rows_in_period,
    count_rows_in_period_wrapped,
    PeriodCountArgs,
    CountArgs,
    CompareArgs,
    TopKArgs,
    error_tool,
    list_column_values_wrapped,
    ColumnValuesArgs,
    column_top_k_wrapped,
    ColumnTopKArgs,
    count_column_value_wrapped,
    ColumnValueCountArgs
)
from langchain.callbacks.base import BaseCallbackHandler
import re

# 依赖
from io import BytesIO
from typing import List, Optional, Literal
CURRENT_SESSION_ID = None 

# ==== 新增工具函数 ===============================================
def get_active_session_id(passed_id: Optional[str] = None) -> Optional[str]:
    """
    规则：
    1) 使用显式传入的 session_id
    2) 否则用全局 CURRENT_SESSION_ID，且必须在 sessions 字典里
    3) 否则如果当前只有一个会话，默认就是它
    """
    if passed_id and passed_id in sessions:
        return passed_id
    if CURRENT_SESSION_ID and CURRENT_SESSION_ID in sessions:
        return CURRENT_SESSION_ID
    if len(sessions) == 1:
        return next(iter(sessions))
    return None


# ---------- Pydantic ----------
class PlotChartArgs(BaseModel):
    labels: List[str]          # x 轴或分类
    values: List[float]        # y 轴
    chart_type: Literal["bar", "line", "pie"] = "bar"
    title: Optional[str] = None
    session_id: Optional[str] = None  # 建议传当前会话 id，方便前端访问

# ---------- 核心函数 ----------
def plot_chart(
    labels: List[str],
    values: List[float],
    chart_type: str = "bar",
    title: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict:
    # 自动判断有效 session_id，避免 None
    session_id = get_active_session_id(session_id)
    if len(labels) != len(values):
        raise ValueError("labels 与 values 长度必须一致")

    # 1) 画图
    fig, ax = plt.subplots(figsize=(4, 3))
    if chart_type == "line":
        ax.plot(labels, values, marker="o")
    elif chart_type == "pie":
        ax.pie(values, labels=labels, autopct="%1.1f%%")
    else:  # bar
        ax.bar(labels, values)
    ax.set_title(title or "")
    if chart_type != "pie":
        ax.set_ylabel("Value")
    fig.tight_layout()

    # 2) 输出
    buf = BytesIO()
    fig.savefig(buf, format="jpeg", dpi=150)
    plt.close(fig)
    buf.seek(0)

    image_url = None
    # print("sessions:", sessions)
    # print("session_id:", session_id)
    if session_id and session_id in sessions:
        session_dir = os.path.join(UPLOAD_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)
        doc_id = str(uuid.uuid4())
        filename = f"{doc_id}_chart.jpg"
        save_path = os.path.join(session_dir, filename)
        with open(save_path, "wb") as f:
            f.write(buf.read())
        sessions[session_id]["docs"][doc_id] = {
            "filename": filename,
            "path": save_path,
            "uploaded_at": os.path.getmtime(save_path),
        }
        image_url = f"/api/chat/sessions/{session_id}/docs/{doc_id}"

    return {"image_url": image_url, "labels": labels, "values": values}



class ToolRefRecorder(BaseCallbackHandler):
    def __init__(self):
        self.refs = []
        self._current_tool = None
        self._current_input = None

    def on_tool_start(self, serialized, input_str, **kwargs):
        self._current_tool = None
        if isinstance(serialized, dict):
            self._current_tool = serialized.get("name")
        self._current_input = input_str

    def on_tool_end(self, output, **kwargs):
        s = str(output).strip()
        payload = None
        # 1) 直接 JSON
        try:
            payload = json.loads(s)
        except Exception:
            # 2) 匹配 "Observation: {...}" 里的 JSON
            m = re.search(r'Observation:\s*(\{.*\})', s, re.S)
            if m:
                js = m.group(1)
                try:
                    payload = json.loads(js)
                except Exception:
                    payload = {"raw": s}
            else:
                payload = {"raw": s}

        # 解析输入（尽量 JSON 化）
        try:
            parsed_input = json.loads(self._current_input) if self._current_input else None
        except Exception:
            parsed_input = self._current_input

        img = payload.get("image_url") if isinstance(payload, dict) else None
        self.refs.append({
            "tool": self._current_tool,
            "input": parsed_input,
            "payload": payload,
            "image_url": img,          # ★ 新增
        })

SYSTEM_PROMPT = """
你是一个专业的 Excel 数据分析和知识问答助手。
你可以调用 4 个 Excel 统计工具，也可以直接回答常识/知识库问题。\n
如果用户的问题不需要读取 Excel，请不要调用任何工具，直接回答。\n

当收到用户提问是数据分析问题时，请按以下流程思考并回复（遵循 LangChain Structured Chat 格式）：
1. **问题拆解**  
   • 如果用户的问题比较笼统或复杂，先在 Thought 中把它拆成 1–3 个更具体的子问题（如需要哪几列、哪几个时间段、应使用哪个统计指标等）。  
2. **调用工具**  
    • 你不用一定需要用到工具，如果没有让你统计你可以不使用工具，也可以部分使用工具部分不使用工具，具体由用户的输入命令决定。
    • 对于每个子问题，使用 `Thought` 说明你要做什么。
   • 对每个子问题，选择最合适的工具（count_period_variable、compare_variable_counts、top_k_variables 等）。  
   • 在 Action 块中给出 JSON 参数，等待 Observation。  
   • 按需多轮调用，直到获得完成回答所需的全部数据。  
3. **整合结果**  （***对于简单的问题，例如“2024-03 的Offer中PIX的数目”，“2024-03 的Offer中出现次数最多的前 5 个变量”，“在Offer中，2024-2出现PIX的次数，同比增长多少”，无需遵循下面的复杂内容***）
   • 在 Final Answer 中，用清晰自然的语言 + Markdown 表格 / 列表 **完整呈现：  
     - 得到的统计数字 / 对比结果  
     - 关键行数据摘要（若行数过多可说明已截断）  
     - 对数据的解读、结论、建议**  
   • Final Answer 必须包含所有结论、表格和关键信息；不要把重要内容留在 Thought 或 Observation 里。  
   • 请你尤其**注意**！Final Answer 中不需要包含 Thought 和 Action 的内容。
4. **工具使用说明**  
   • 如果你收到了一段时间和一个明显的统计变量（如“2024-03 的Offer中PIX的数目”），请使用 `count_period_variable` 工具。
   • 如果你收到两个时间段和同一变量（如“在Offer中，2024-2出现PIX的次数，同比增长多少”），请使用 `compare_variable_counts` 工具。
   • 如果你收到一个时间段和一个列名（如“2024-03 的Offer中出现次数最多的前 5 个变量”），请使用 `top_k_variables` 工具。
   • 如果你收到一个时间段，并不清楚统计变量（如“2024年1月，有没有人员损伤，内燃弧，无法分合闸，跳闸，放电的问题，多少例”），请使用 `count_rows_in_period` 工具。
5. **推理说明**
    如果被问到：在xx列中，某某变量出现了多少次？但是没有明确的时间信息，并且意思是该变量全部时间下的内容。请分多次使用 `count_period_variable` 工具，你需要自己读取表格的时间信息，多次传参。
"""
# ========================
# 自定义 DashScope LLM
class DashScopeLLM(LLM):
    model: str = "qwen-max"
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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        resp = Generation.call(model=self.model, messages=messages)
        return resp["output"]["text"]

# ========================
# 读表工具函数
import re
import pandas as pd
from typing import Any




# ========================
# 初始化 LLM & Agen
api_key = os.getenv("DASHSCOPE_API_KEY", api_key)
#加入阿里云的 app_id

llm = DashScopeLLM(dashscope_api_key=api_key, dashscope_app_id=app_id)

tools = [
    StructuredTool.from_function(
        func=count_period_variable_wrapped,
        name="count_period_variable",
        description="统计指定年月下某列中变量出现的次数。如果被问到某某列总共有多少（时间模糊并且意思是全部时间），多次调用改工具并且汇总。仅在需要统计 Excel 列中变量出现次数时使用",
        args_schema=CountArgs,         # 刚才写的 Pydantic
        return_direct=False,            # 结果直接回给 Agent
    ),
    StructuredTool.from_function(
        func=compare_variable_counts_wrapped,
        name="compare_variable_counts",
        description="比较两个不同年月下同一列同一变量出现次数的差值（period2 - period1）。仅在需要统计 Excel 列中不同变量出现次数差值时使用",
        args_schema=CompareArgs,
        return_direct=False,
    ),
    StructuredTool.from_function(
        func=top_k_variables_wrapped,
        name="top_k_variables",
        description="返回指定月份中出现次数最多的前 K 个取值及其计数。仅在需要统计 Excel 列中前K个取值时使用",
        args_schema=TopKArgs,
        return_direct=False,
    ),
    StructuredTool.from_function(
        func=count_rows_in_period_wrapped,
        name="count_rows_in_period",
        description="返回指定年月内的记录总数及所有行数据。仅在需要统计 Excel 列中记录总数及所有行数据使时使用",
        args_schema=PeriodCountArgs,
        return_direct=False,
    ),
    StructuredTool.from_function(
        func=plot_chart,
        name="plot_chart",
        description=(
            "请注意！！这个工具传入的表格标题等参数必须是全英文的，否则在画图的时候会出现乱码！！必须全英文！！"
            "请注意！！只用相对 URL /api/chat/sessions/.../docs/... ——来自你后端 plot_chart() 返回的 image_url，保证同源可取。正文里别再出现其它 ![...](...) 避免解析到外链；有多张图就多行并列写。"
            "请注意！！！使用本工具后，你得到的 image_url 已经是可以直接在前端显示的图片地址了。在回答中请直接使用 **同源相对路径**，不要改动或附加协议头。"
            "根据 labels 与 values 生成图表；"
            "chart_type 可选 bar|line|pie；"
            "session_id 传当前会话可在前端显示图片"
        ),
        args_schema=PlotChartArgs,
        return_direct=False,
    ),
    StructuredTool.from_function(
        func=list_column_values_wrapped,
        name="list_column_values",
        description="获取指定列的所有非空内容，按行顺序返回,例如你需要获取所有年月信息，或者获取所有的Offer信息，则可以调用该工具",
        args_schema=ColumnValuesArgs,
        return_direct=False,
    ),
    StructuredTool.from_function(
        func=column_top_k_wrapped,
        name="column_top_k",
        description="统计某列中出现次数最多的前 k 个取值",
        args_schema=ColumnTopKArgs,
        return_direct=False,
    ),
    StructuredTool.from_function(
        func=count_column_value_wrapped,
        name="count_column_value",
        description="统计某列中特定取值的出现次数",
        args_schema=ColumnValueCountArgs,
        return_direct=False,
    ),
]

short_term_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

## 2️⃣ 长期记忆：向量数据库 + RetrieverMemory
VECTOR_DIR = "vector_store"  # 持久化目录
embeddings = DashScopeEmbeddings(dashscope_api_key=api_key)
vectorstore = Chroma(persist_directory=VECTOR_DIR, embedding_function=embeddings)
retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
long_term_memory = VectorStoreRetrieverMemory(retriever=retriever)

def write_long_term_memory(question: str, answer: str):
    text = f"Q: {question}\nA: {answer}"
    vectorstore.add_texts([text])

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    max_iterations=3,
    handle_parsing_errors=False,  # 开发期如出错直接抛
    early_stopping_method="generate",
    verbose=True,
    memory=short_term_memory,                 # ⏳ 短期
    additional_memories=[long_term_memory],   # 🧠 长期
    output_parser=StructuredChatOutputParserWithRetries(),  # 避免 LLM 格式小错误
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
    name: str

class RenameReq(BaseModel):
    name: str

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

# @app.post("/api/analyze")
# async def analyze(cmd: Command):
#     # 旧接口也占个“全局”会话位，防止后续 plot_chart 没 session_id
#     global CURRENT_SESSION_ID
#     CURRENT_SESSION_ID = "__legacy__"
#     global CURRENT_FILE_PATH
#     file_path = CURRENT_FILE_PATH
#     if not file_path or not os.path.exists(file_path):
#         file_path = _pick_latest_excel_global()

#     if not file_path:
#         return {"result": "请先上传 Excel 文件"}

#     query = f"文件路径是 {file_path}，{cmd.command}"
#     try:
#         recorder = ToolRefRecorder()                           # ★ 新建回调
#         result   = agent.run(query, callbacks=[recorder])      # ★ 挂进去
#     except Exception as e:
#         result = f"分析出错: {e}"

#     print("RECORD-REFS:", json.dumps(recorder.refs, ensure_ascii=False)[:200])
#     return {"result": result, "file_path": file_path, "refs": recorder.refs or None}


# ========================
# 多会话接口
@app.post("/api/chat/sessions", response_model=SessionResponse)
async def create_session():
    sid = str(uuid.uuid4())
    sessions[sid] = {
        "name": "EXCEL表格分析",
        "messages": [],
        "docs": {},
        "current_doc_id": None,
    }
    return {"session_id": sid, "name": sessions[sid]["name"]}

@app.get("/api/chat/sessions")
async def list_sessions():
    return {
        "sessions": [
            {"id": k, "name": v.get("name", k[:8])}
            for k, v in sessions.items()
        ]
    }


import matplotlib
matplotlib.use("Agg")              # 无窗环境
import matplotlib.pyplot as plt
import numpy as np

# ==== 新接口：生成并返回示例图表 ====
# @app.post("/api/chat/sessions/{session_id}/demo-chart")
# async def demo_chart(session_id: str):
#     if session_id not in sessions:
#         raise HTTPException(status_code=404, detail="Session not found")

#     # 1) 会话文件夹
#     session_dir = os.path.join(UPLOAD_DIR, session_id)
#     os.makedirs(session_dir, exist_ok=True)

#     # 2) 画一张简单柱状图
#     x = np.arange(5)
#     y = np.random.randint(10, 100, size=5)
#     fig, ax = plt.subplots()
#     ax.bar(x, y)
#     ax.set_title("随机销量示例")
#     fig.tight_layout()

#     # 3) 保存 PNG
#     doc_id   = str(uuid.uuid4())
#     filename = f"{doc_id}_demo.png"
#     save_path = os.path.join(session_dir, filename)
#     fig.savefig(save_path, dpi=150)
#     plt.close(fig)

#     # 4) 注册到 docs
#     sessions[session_id]["docs"][doc_id] = {
#         "filename": filename,
#         "path": save_path,
#         "uploaded_at": os.path.getmtime(save_path),
#     }

#     # 5) 生成图片 URL
#     img_url = f"/api/chat/sessions/{session_id}/docs/{doc_id}"

#     # 6) 写一条 assistant 消息（Markdown 图片语法）
#     msg = {
#         "role": "assistant",
#         "content": f"这是一个示例图表：\n\n![示例图表]({img_url})"
#     }
#     sessions[session_id]["messages"].append(msg)

#     return {
#         "status": "ok",
#         "doc_id": doc_id,
#         "image_url": img_url,
#         "message": msg,
#     }

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
# === 创建（或追加）消息 ===
@app.post("/api/chat/sessions/{session_id}/messages")
async def post_message(session_id: str, msg: ChatMessage):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    if msg.role not in ("user", "assistant"):
        raise HTTPException(status_code=400, detail="role must be 'user' or 'assistant'")
    entry = msg.dict()
    # sessions[session_id]["messages"].append(entry)
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
    global CURRENT_SESSION_ID
    CURRENT_SESSION_ID = session_id 
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

    try:
        query = f"文件路径是 {file_path}，{req.command}"
        recorder = ToolRefRecorder()                           # ★ 新建回调
        result   = agent.run(query, callbacks=[recorder])      # ★ 挂进去
        print("RECORD-REFS:", json.dumps(recorder.refs,ensure_ascii=False)[:200])
    except Exception as e:
        result = f"分析出错: {e}"

    # 保存消息（用户 -> 助手）
    sessions[session_id]["messages"].append({"role": "user", "content": req.command})
    assistant_msg = {"role": "assistant", "content": result}
    if recorder.refs:
        assistant_msg["refs"] = recorder.refs
    sessions[session_id]["messages"].append(assistant_msg)   # ✅ 仅这一次写库

    return {
        "result": result,
        "refs": recorder.refs or None,     # 前端要用来渲染
        "assistant_msg": assistant_msg     # 前端若想直接拼列表，可用这条
    }

# ---------- rename ----------
@app.put("/api/chat/sessions/{session_id}/name")
async def rename_session(session_id: str, req: RenameReq):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    sessions[session_id]["name"] = req.name.strip() or "未命名会话"
    return {"status": "ok", "name": sessions[session_id]["name"]}


# ========================
# 启动
if __name__ == "__main__":
    # 为了避免 dashscope 未设置 API key 导致报错，这里给出提示
    if api_key == "YOUR_API_KEY":
        print("[WARN] DASHSCOPE_API_KEY 未设置，记得在环境变量中配置。")
    uvicorn.run(app, host="0.0.0.0", port=8000)
