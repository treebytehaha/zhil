from langchain.llms import OpenAI
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv
import os, yaml
from langchain.tools import Tool
import json
from actions.excel_actions import (
    count_period_variable_occurrences,
    compare_period_variable_counts,
    top_k_variables_in_period,
)
load_dotenv()

llm= OpenAI(
    model_name=os.getenv("DASHSCOPE_MODEL", "qwen-max"),
    openai_api_key=os.getenv("DASHSCOPE_API_KEY") or 'sk-51341d6718514f128743252bc13b5651',
    openai_api_base=os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1" 
    ),
    temperature=0
)

tools = [
    Tool(
        name="count_period_variable",
        func=lambda args: count_period_variable_occurrences(
            file_path=args["file_path"],
            sheet_name=args.get("sheet_name", 0),
            date_column=args["date_column"],
            period=args["period"],        # 格式 "YYYY-MM"
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
            period1=args["period1"],      # e.g. "2025-06"
            period2=args["period2"],      # e.g. "2025-07"
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
            period=args["period"],        # e.g. "2025-07"
            column=args["column"],
            k=args["k"],                  # Top K
        ),
        description=(
            "在指定年月(period)下，找出某列(column)中出现次数最多的前 k 个变量及其计数。"
            "参数：file_path(str)、sheet_name(str|int)、"
            "date_column(str)、period(str, YYYY-MM)、column(str)、k(int)"
        ),
    ),
    Tool(
        name="error",
        func=lambda args: f"错误：{args['message']}",
        description="当解析失败或操作不支持时，返回错误提示。"
    ),
]
agent=(
    initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
)
result= agent.run(
    "What is the capital of France?"
)
print(result)