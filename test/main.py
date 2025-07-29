import os
import sys
import argparse
import pandas as pd
import requests
from datetime import datetime
from typing import Any, List, Optional

from langchain.llms.base import LLM
from langchain.agents import Tool, initialize_agent, AgentType


class DashScope(LLM):
    """
    A simple LangChain LLM wrapper for DashScope API.
    Requires DASHSCOPE_API_KEY environment variable or --api_key argument.
    """
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key

    @property
    def _llm_type(self) -> str:
        return "dashscope"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "prompt": prompt,
            "max_tokens": 512
        }
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
    period_dt = pd.to_datetime(period)
    mask = df[date_column].dt.to_period("M") == period_dt.to_period("M")
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
    count1 = count_period_variable_occurrences(
        file_path, sheet_name, date_column, period1, column, variable
    )
    count2 = count_period_variable_occurrences(
        file_path, sheet_name, date_column, period2, column, variable
    )
    return count2 - count1


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
    period_dt = pd.to_datetime(period)
    mask = df[date_column].dt.to_period("M") == period_dt.to_period("M")
    counts = df.loc[mask, column].value_counts().head(k)
    return list(counts.items())


def error_tool(args: dict) -> str:
    return f"错误：{args.get('message', '未知错误')}"


if __name__ == "__main__":


    # Instantiate LLM
    api_key = 'sk-51341d6718514f128743252bc13b5651'

    llm = DashScope(api_key=api_key)


    # Define tools
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
            description=(
                "统计指定年月(period)下 Excel 文件中某 sheet(sheet_name)，"
                "列(column)里变量(variable)出现的次数。"
                "参数：file_path(str)、sheet_name(str|int)、"
                "date_column(str)、period(str, YYYY-MM)、column(str)、variable(Any)"
            ),
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
            description=(
                "比较两个不同年月(period1, period2)下同一列(column)同一变量(variable)的出现次数差值。"
                "参数：file_path(str)、sheet_name(str|int)、"
                "date_column(str)、column(str)、variable(Any)、"
                "period1(str, YYYY-MM)、period2(str, YYYY-MM)"
            ),
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
            description=(
                "在指定年月(period)下，找出某列(column)中出现次数最多的前 k 个变量及其计数。"
                "参数：file_path(str)、sheet_name(str|int)、"
                "date_column(str)、period(str, YYYY-MM)、column(str)、k(int)"
            ),
        ),
        Tool(
            name="error",
            func=lambda args: error_tool(args),
            description="当解析失败或操作不支持时，返回错误提示。"
        ),
    ]

    # Initialize agent
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
        if user_input.lower() in ("exit", "quit"):
            print("退出。")
            break
        try:
            result = agent.run(user_input)
        except Exception as e:
            result = f"Error: {e}"
        print(result)