import pandas as pd
from typing import Any

def count_variable_occurrences(
    file_path: str,
    sheet_name: str,
    column: str,
    variable: Any
) -> int:
    """
    统计 Excel 文件中指定工作表(sheet_name)的某列(column)中，某个变量(variable)出现的次数。
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return int((df[column] == variable).sum())
import pandas as pd
from typing import Any, List, Dict

def count_period_variable_occurrences(
    file_path: str,
    sheet_name: str,
    date_column: str,
    period: str,        # 格式 "YYYY-MM"
    column: str,
    variable: Any
) -> int:
    """
    统计指定年月(period)下 Excel 文件中某 sheet(sheet_name)、
    列(column)里变量(variable)出现的次数。
    """
    # 1. 读表并解析日期
    df = pd.read_excel(
        file_path, sheet_name=sheet_name, parse_dates=[date_column]
    )
    # 2. 构造“YYYY-MM”列
    df['_period'] = df[date_column].dt.to_period('M').astype(str)
    # 3. 过滤出指定期、指定变量
    mask = (df['_period'] == period) & (df[column] == variable)
    return int(mask.sum())


def compare_period_variable_counts(
    file_path: str,
    sheet_name: str,
    date_column: str,
    column: str,
    variable: Any,
    period1: str,      # e.g. "2025-06"
    period2: str       # e.g. "2025-07"
) -> Dict[str, Any]:
    """
    比较同一文件同一 sheet、同一列(column)中同一个变量(variable)
    在 period1 和 period2 两期的出现次数及差值。
    返回：
    {
      "period1": period1,
      "count1": int,
      "period2": period2,
      "count2": int,
      "diff": count2 - count1
    }
    """
    df = pd.read_excel(
        file_path, sheet_name=sheet_name, parse_dates=[date_column]
    )
    df['_period'] = df[date_column].dt.to_period('M').astype(str)
    # 只保留这列等于 variable 的行
    df_var = df[df[column] == variable]
    # 按 period 计数
    grp = df_var.groupby('_period').size()
    c1 = int(grp.get(period1, 0))
    c2 = int(grp.get(period2, 0))
    return {
        "period1": period1,
        "count1": c1,
        "period2": period2,
        "count2": c2,
        "diff": c2 - c1
    }


def top_k_variables_in_period(
    file_path: str,
    sheet_name: str,
    date_column: str,
    period: str,      # e.g. "2025-07"
    column: str,
    k: int
) -> List[Dict[str, Any]]:
    """
    在指定年月(period)下，找出某列(column)中出现次数最多的前 k 个变量及其计数。
    返回格式：
      [
        {"variable": X1, "count": N1},
        {"variable": X2, "count": N2},
        ...
      ]
    """
    df = pd.read_excel(
        file_path, sheet_name=sheet_name, parse_dates=[date_column]
    )
    df['_period'] = df[date_column].dt.to_period('M').astype(str)
    # 先过滤到指定期
    dfp = df[df['_period'] == period]
    # 然后按 column 分组计数并排序
    counts = dfp.groupby(column).size().sort_values(ascending=False)
    top = counts.head(k)
    # 构造返回列表
    return [
        {"variable": idx, "count": int(cnt)}
        for idx, cnt in top.items()
    ]
