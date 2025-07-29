import os
import pandas as pd
from openai import OpenAI
from typing import Any, List, Optional

from pydantic import BaseModel, Field
from langchain.llms.base import LLM
from langchain.agents import Tool, initialize_agent, AgentType
DASH_SCOPE_KEY="sk-b913df97180845a284251391b5b76fbb"

# ========== DashScope LLM Adapter ==========
class DashScope(LLM, BaseModel):
    """
    DashScope LLM wrapper using OpenAI-compatible client.
    硬编码或读取环境变量获取 API Key、Base URL 与模型。
    """
    api_key: str = Field(default=DASH_SCOPE_KEY)

    base_url: str = Field(
        default_factory=lambda: os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    )
    model: str = Field(
        default_factory=lambda: os.getenv(
            "DASHSCOPE_MODEL", "qwen-plus"
        )
    )
    client: OpenAI = Field(
        default_factory=lambda: OpenAI(
            api_key=DASH_SCOPE_KEY,
            base_url=os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        )
    )

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "dashscope"

    def _call(
        self, prompt: str, stop: Optional[List[str]] = None
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stop=stop
        )
        return response.choices[0].message.content.strip()

# ===== Tool implementations =====

def count_period_variable_occurrences(
    file_path: str,
    sheet_name: Any,
    date_column: str,
    period: str,
    column: str,
    variable: Any
) -> int:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df[date_column] = pd.to_datetime(df[date_column])
    mask = df[date_column].dt.to_period("M") == pd.to_datetime(period).to_period("M")
    return int((df.loc[mask, column] == variable).sum())


def compare_period_variable_counts(
    file_path: str,
    sheet_name: Any,
    date_column: str,
    column: str,
    variable: Any,
    period1: str,
    period2: str
) -> int:
    c1 = count_period_variable_occurrences(
        file_path, sheet_name, date_column, period1, column, variable
    )
    c2 = count_period_variable_occurrences(
        file_path, sheet_name, date_column, period2, column, variable
    )
    return c2 - c1


def top_k_variables_in_period(
    file_path: str,
    sheet_name: Any,
    date_column: str,
    period: str,
    column: str,
    k: int
) -> List[tuple]:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df[date_column] = pd.to_datetime(df[date_column])
    mask = df[date_column].dt.to_period("M") == pd.to_datetime(period).to_period("M")
    counts = df.loc[mask, column].value_counts().head(k)
    return list(counts.items())


def error_tool(args: dict) -> str:
    return f"错误：{args.get('message', '未知错误')}"


if __name__ == "__main__":
    # 实例化 LLM，无需自定义 __init__
    llm = DashScope()

    # 定义 tools
    tools = [
        Tool(
            name="count_period_variable",
            func=lambda args: count_period_variable_occurrences(
                file_path="demo.xlsx",
                sheet_name=args.get("sheet_name", 0),
                date_column=args["date_column"],
                period=args["period"],
                column=args["column"],
                variable=args["variable"],
            ),
            description="统计指定年月(period)下 Excel 中某列里变量出现次数。"
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
            description="比较两个年月下同列同变量出现次数的差值。"
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
            description="返回指定年月下列中出现次数最多的前 k 个变量及其计数。"
        ),
        Tool(
            name="error",
            func=lambda args: error_tool(args),
            description="解析失败或不支持操作时返回错误提示。"
        ),
    ]

    # 初始化 Agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    print("Agent ready. 输入您的查询 (输入 'exit' 或 'quit' 退出):")
    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break
        output = agent.run(user_input)
        print(output)