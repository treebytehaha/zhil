import os
import pandas as pd
from typing import Any, List, Optional, Dict
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
            {"role": "system", "content": "你是一个专业的FFR数据分析助手，能够准确回答关于现场故障报告的各种统计和分析问题。"},
            {"role": "user", "content": prompt}
        ]

        response = Generation.call(
            model=self.model,
            messages=messages
        )

        return response['output']['text']

# ========================
# FFR 数据分析工具类
class FFRDataAnalyzer:
    def __init__(self, file_path: str):
        self.df = self._load_and_preprocess_data(file_path)
    
    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """加载并预处理FFR数据"""
        df = pd.read_excel(file_path, sheet_name=0)
        
        # 日期处理
        if '填表日期' in df.columns:
            df['填表日期'] = pd.to_datetime(df['填表日期'])
            df['月份'] = df['填表日期'].dt.to_period('M')
            df['年份'] = df['填表日期'].dt.year
        
        # 统一供应商分类
        if 'RCA2' in df.columns:
            df['RCA2'] = df['RCA2'].str.replace(' ', '').str.upper()
        
        return df

    def get_cases_by_product_month(self, product: str, month: str) -> int:
        """获取指定产品和月份的case数量"""
        period = pd.to_datetime(month).to_period('M')
        mask = (self.df['月份'] == period)
        if product.lower() != 'all':
            mask &= (self.df['Offer'].str.lower() == product.lower())
        return int(mask.sum())

    def get_yoy_comparison(self, product: str, month: str) -> dict:
        """同比分析"""
        current_year = pd.to_datetime(month).year
        last_year = str(current_year - 1) + month[4:]
        
        current = self.get_cases_by_product_month(product, month)
        last = self.get_cases_by_product_month(product, last_year)
        
        return {
            'current': current,
            'last_year': last,
            'difference': current - last,
            'trend': '增长' if current > last else '降低'
        }

    def get_product_distribution(self, month: str) -> dict:
        """产品投诉占比分析"""
        period = pd.to_datetime(month).to_period('M')
        month_data = self.df[self.df['月份'] == period]
        total = len(month_data)
        if total == 0:
            return {}
        dist = month_data['Offer'].value_counts(normalize=True).to_dict()
        return {k: f"{round(v*100, 2)}%" for k, v in dist.items()}

    def get_protection_cases(self, start_month: str, end_month: str = None, compare_start: str = None, compare_end: str = None) -> dict:
        """保护产品失效分析"""
        if end_month is None:
            end_month = start_month
            
        start_dt = pd.to_datetime(start_month).to_period('M')
        end_dt = pd.to_datetime(end_month).to_period('M')
        
        mask = (self.df['月份'] >= start_dt) & (self.df['月份'] <= end_dt)
        mask &= (self.df['RCA1'].str.contains('保护|relay', case=False, na=False))
        
        result = {'current_period': len(self.df[mask])}
        
        if compare_start:
            compare_start_dt = pd.to_datetime(compare_start).to_period('M')
            compare_end_dt = pd.to_datetime(compare_end if compare_end else compare_start).to_period('M')
            
            compare_mask = (self.df['月份'] >= compare_start_dt) & (self.df['月份'] <= compare_end_dt)
            compare_mask &= (self.df['RCA1'].str.contains('保护|relay', case=False, na=False))
            
            result['compare_period'] = len(self.df[compare_mask])
            result['difference'] = result['current_period'] - result['compare_period']
        
        return result

    def get_supplier_cases(self, supplier_type: str, period: str) -> dict:
        """供应商相关case分析"""
        period_dt = pd.to_datetime(period).to_period('M')
        supplier_key = f"{supplier_type.upper()}-SUPPLIER"
        
        mask = (self.df['月份'] == period_dt) & (self.df['RCA2'] == supplier_key)
        data = self.df[mask]
        
        return {
            'count': len(data),
            'top_failure_modes': data['RCA3'].value_counts().head(3).to_dict(),
            'top_components': data['RCA4'].value_counts().head(3).to_dict()
        }

    def get_top_components(self, start_month: str, end_month: str = None, k: int = 3) -> dict:
        """失效零部件TopK分析"""
        if end_month is None:
            end_month = start_month
            
        start_dt = pd.to_datetime(start_month).to_period('M')
        end_dt = pd.to_datetime(end_month).to_period('M')
        
        mask = (self.df['月份'] >= start_dt) & (self.df['月份'] <= end_dt)
        data = self.df[mask]
        
        return data['RCA4'].value_counts().head(k).to_dict()

    def get_customer_complaints(self, month: str, top_n: int = 3) -> dict:
        """客户投诉分析"""
        period_dt = pd.to_datetime(month).to_period('M')
        data = self.df[self.df['月份'] == period_dt]
        
        return {
            'top_customers': data['项目名称'].value_counts().head(top_n).to_dict(),
            'top_regions': data['区域'].value_counts().to_dict()
        }

    def get_case_status(self, month: str) -> dict:
        """Case状态分析"""
        period_dt = pd.to_datetime(month).to_period('M')
        data = self.df[self.df['月份'] == period_dt]
        
        if '项目状态' not in data.columns:
            return {}
        
        status_counts = data['项目状态'].value_counts().to_dict()
        total = sum(status_counts.values())
        
        return {
            'status_distribution': status_counts,
            'closed_rate': f"{round(status_counts.get('关闭', 0) / total * 100, 2)}%" if total > 0 else "0%"
        }

# ========================
# 工具函数
def error_tool(args: dict) -> str:
    return f"错误：{args.get('message', '未知错误')}"

# ========================
# 初始化 LLM 和 Agent
api_key = 'sk-51341d6718514f128743252bc13b5651'  # 请替换为您的实际API密钥
llm = DashScopeLLM(dashscope_api_key=api_key)

# 全局变量
ffr_analyzer = None
tools = []

def initialize_ffr_tools():
    """初始化FFR分析工具"""
    global tools
    
    tools = [
        Tool(
            name="ffr_cases_by_product_month",
            func=lambda args: ffr_analyzer.get_cases_by_product_month(args["product"], args["month"]),
            description="获取指定产品和月份的FFR case数量。输入: product(产品名,如'PIX'), month(年月，如'2024-01')"
        ),
        Tool(
            name="ffr_yoy_comparison",
            func=lambda args: ffr_analyzer.get_yoy_comparison(args["product"], args["month"]),
            description="FFR case同比分析。输入: product(产品名), month(年月)"
        ),
        Tool(
            name="ffr_product_distribution",
            func=lambda args: ffr_analyzer.get_product_distribution(args["month"]),
            description="FFR产品投诉占比分析。输入: month(年月)"
        ),
        Tool(
            name="ffr_protection_cases",
            func=lambda args: ffr_analyzer.get_protection_cases(
                args["start_month"],
                args.get("end_month"),
                args.get("compare_start"),
                args.get("compare_end")
            ),
            description="保护产品失效case分析。输入: start_month(开始年月), end_month(结束年月,可选), compare_start(对比开始年月,可选), compare_end(对比结束年月,可选)"
        ),
        Tool(
            name="ffr_supplier_analysis",
            func=lambda args: ffr_analyzer.get_supplier_cases(args["supplier_type"], args["month"]),
            description="供应商相关case分析。输入: supplier_type(IG/OG), month(年月)"
        ),
        Tool(
            name="ffr_top_components",
            func=lambda args: ffr_analyzer.get_top_components(
                args["start_month"],
                args.get("end_month"),
                args.get("k", 3)
            ),
            description="失效零部件TopK分析。输入: start_month(开始年月), end_month(结束年月,可选), k(前几名,默认3)"
        ),
        Tool(
            name="ffr_customer_complaints",
            func=lambda args: ffr_analyzer.get_customer_complaints(args["month"], args.get("top_n", 3)),
            description="客户投诉分析。输入: month(年月), top_n(前几名,默认3)"
        ),
        Tool(
            name="ffr_case_status",
            func=lambda args: ffr_analyzer.get_case_status(args["month"]),
            description="Case状态分析。输入: month(年月)"
        ),
        Tool(
            name="error",
            func=lambda args: error_tool(args),
            description="当解析失败或操作不支持时，返回错误提示。"
        )
    ]
initialize_ffr_tools()
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
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

@app.post("/api/upload-excel")
async def upload_excel(file: UploadFile = File(...)):
    global ffr_analyzer
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # 保存文件
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)
    
    # 初始化分析器
    try:
        ffr_analyzer = FFRDataAnalyzer(save_path)
        initialize_ffr_tools()
        return {"status": "success", "message": "文件上传并解析成功", "file_path": save_path}
    except Exception as e:
        return {"status": "error", "message": f"文件解析失败: {str(e)}"}

class Question(BaseModel):
    question: str

@app.post("/api/ask")
async def ask_question(q: Question):
    global ffr_analyzer
    if ffr_analyzer is None:
        return {"status": "error", "answer": "请先上传FFR数据文件"}
    
    try:
        # 构造包含上下文的问题
        full_question = f"""
        你是一个专业的FFR数据分析助手，请根据提供的FFR数据回答以下问题：
        问题：{q.question}
        
        要求：
        1. 回答要专业、准确
        2. 如果有数字结果，请明确给出具体数值
        3. 如果有比较，请说明增减趋势
        4. 如果有百分比，请保留两位小数
        """
        
        answer = agent.run(full_question)
        return {"status": "success", "answer": answer}
    except Exception as e:
        return {"status": "error", "answer": f"分析问题时出错: {str(e)}"}

# 启动服务
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)