import json, re, os
from typing import Any, List, Tuple
import pandas as pd

# ---------- 公共辅助 ----------
def _clean_month_str(s: str) -> str:
    """把 '2024年3月' / '2024-3月' / '2024/03' 等转成 '2024-03'。"""
    s = str(s)
    s = re.sub(r"[年月/]", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    y, m, *_ = (s.split("-") + ["01"])
    return f"{y}-{m.zfill(2)}"

def _load_and_filter(
    file_path: str,
    sheet_name: Any,
    date_column: str,
    period: str,
) -> pd.DataFrame:
    """读取 Excel→清洗日期→筛选指定年月后返回 DataFrame。"""
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    if date_column not in df.columns:
        raise KeyError(f"日期列 '{date_column}' 不存在")

    df[date_column] = (
        df[date_column].astype(str)
                       .map(_clean_month_str)
                       .pipe(pd.to_datetime, format="%Y-%m")
    )
    period_dt = pd.to_datetime(period, format="%Y-%m")
    return df.loc[df[date_column].dt.to_period("M") == period_dt.to_period("M")]

def _match_rows(
    df: pd.DataFrame,
    column: str,
    regex: re.Pattern,
) -> pd.DataFrame:
    """返回包含 variable 的行（支持整表搜索）。"""
    if column in df.columns:
        mask = df[column].astype(str).str.contains(regex, na=False)
        return df.loc[mask]
    # 列名不存在 → 全表搜索
    mask = df.apply(lambda r: any(regex.search(str(x)) for x in r), axis=1)
    return df.loc[mask]

def _df_rows_to_str_records(df: pd.DataFrame) -> List[dict]:
    """将 DataFrame 转成字符串化的 records 列表，便于 JSON 序列化。"""
    return df.astype(str).to_dict(orient="records")

# ---------- 1) 单月计数 ----------
def count_period_variable_occurrences(
    file_path: str,
    sheet_name: Any,
    date_column: str,
    period: str,
    column: str,
    variable: Any,
) -> Tuple[int, List[dict]]:
    df_period = _load_and_filter(file_path, sheet_name, date_column, period)
    regex = re.compile(re.escape(str(variable)), re.IGNORECASE)
    df_hit = _match_rows(df_period, column, regex)
    return len(df_hit), _df_rows_to_str_records(df_hit)

def count_period_variable_wrapped(**kwargs) -> str:
    cnt, rows = count_period_variable_occurrences(**kwargs)
    payload = {
        "period": kwargs["period"],
        "column": kwargs["column"],
        "variable": kwargs["variable"],
        "count": cnt,
        "rows": rows,          # 整行内容
    }
    return json.dumps(payload, ensure_ascii=False)

# ---------- 2) 双月对比 ----------
def compare_period_variable_counts(
    file_path: str,
    sheet_name: Any,
    date_column: str,
    column: str,
    variable: Any,
    period1: str,
    period2: str,
) -> Tuple[int, List[dict], int, List[dict], int]:
    c1, rows1 = count_period_variable_occurrences(
        file_path, sheet_name, date_column, period1, column, variable
    )
    c2, rows2 = count_period_variable_occurrences(
        file_path, sheet_name, date_column, period2, column, variable
    )
    return c1, rows1, c2, rows2, c2 - c1

def compare_variable_counts_wrapped(**kwargs) -> str:
    c1, rows1, c2, rows2, diff = compare_period_variable_counts(**kwargs)
    payload = {
        "period1": kwargs["period1"], "count1": c1, "rows1": rows1,
        "period2": kwargs["period2"], "count2": c2, "rows2": rows2,
        "diff": diff
    }
    return json.dumps(payload, ensure_ascii=False)

# ---------- 3) 月内 Top‑K ----------
def top_k_variables_in_period(
    file_path: str,
    sheet_name: Any,
    date_column: str,
    period: str,
    column: str,
    k: int,
) -> Tuple[List[Tuple[Any, int]], List[dict]]:
    df_period = _load_and_filter(file_path, sheet_name, date_column, period)
    counts = df_period[column].value_counts().head(k)
    # 只返回命中前 K 的行
    top_values = set(counts.index.tolist())
    df_hit = df_period[df_period[column].isin(top_values)]
    return list(counts.items()), _df_rows_to_str_records(df_hit)

def top_k_variables_wrapped(**kwargs) -> str:
    topk, rows = top_k_variables_in_period(**kwargs)
    payload = {
        "period": kwargs["period"],
        "column": kwargs["column"],
        "top_k": topk,
        "rows": rows
    }
    return json.dumps(payload, ensure_ascii=False)
# def _clean_month_str(s: str) -> str:
#     """
#     将 '2024-2月' / '2024年3月' / '2024/03' 等格式统一成 '2024-02'
#     """
#     s = str(s)
#     s = re.sub(r"[年月/]", "-", s)          # 年/月/斜杠 → -
#     s = re.sub(r"-+", "-", s).strip("-")    # 合并多余的 -
#     y, m, *_ = (s.split("-") + ["01"])      # 只保留 年、月
#     return f"{y}-{m.zfill(2)}"              # 3 → 03


# ---------- 包装函数 ----------
# def count_period_variable_wrapped(**kwargs) -> str:
#     # kwargs 已经是 {"file_path": "...", "sheet_name": 0, ...}
#     return str(count_period_variable_occurrences(**kwargs))

# def count_period_variable_wrapped(**kwargs) -> str:
#     """LangChain 工具：返回 JSON 字符串，便于 LLM 读取。"""
#     cnt = count_period_variable_occurrences(**kwargs)
#     payload = {
#         "period":  kwargs["period"],
#         "column":  kwargs["column"],
#         "variable": kwargs["variable"],
#         "count":   cnt,
#     }
#     return json.dumps(payload, ensure_ascii=False)

# def compare_variable_counts_wrapped(**kwargs) -> str:
#     return str(compare_period_variable_counts(**kwargs))

# def compare_variable_counts_wrapped(**kwargs) -> str:
#     c1, c2, diff = compare_period_variable_counts(**kwargs)
#     payload = {
#         "period1": kwargs["period1"], "count1": c1,
#         "period2": kwargs["period2"], "count2": c2,
#         "diff": diff                 # period2 - period1
#     }
#     return json.dumps(payload, ensure_ascii=False)

# def top_k_variables_wrapped(**kwargs) -> str:
#     return str(top_k_variables_in_period(**kwargs))

# def top_k_variables_wrapped(**kwargs) -> str:
#     lst = top_k_variables_in_period(**kwargs)
#     payload = {
#         "period": kwargs["period"],
#         "column": kwargs["column"],
#         "top_k":  lst                # [(value, count), ...]
#     }
#     return json.dumps(payload, ensure_ascii=False)
# def count_period_variable_occurrences(
#     file_path: str,
#     sheet_name: Any,
#     date_column: str,
#     period: str,
#     column: str,
#     variable: Any,
# ) -> int:
#     """
#     统计指定 period 内，指定列（或全表）中 *包含* variable 子串的单元格个数。
#     - 如果 column 不在表头，则在所有列里搜索。
#     - variable、column 均不区分大小写。
#     """
#     df = pd.read_excel(file_path, sheet_name=sheet_name)

#     if date_column not in df.columns:
#         raise KeyError(f"日期列 '{date_column}' 不存在")

#     df[date_column] = (
#         df[date_column].astype(str)
#                     .map(_clean_month_str)               # ① 先清洗
#                     .pipe(pd.to_datetime, format="%Y-%m") # ② 再解析
#     )

#     period_dt = pd.to_datetime(period)
#     mask_period = df[date_column].dt.to_period("M") == period_dt.to_period("M")
#     df_period = df.loc[mask_period]

#     # -------- 处理变量匹配（子串、忽略大小写）--------
#     pat = re.escape(str(variable))                       # 转义特殊字符
#     regex = re.compile(pat, flags=re.IGNORECASE)         # 不区分大小写

#     if column in df_period.columns:
#         # 精确列存在：只在这一列里匹配
#         series = df_period[column].astype(str).str.contains(regex, na=False)
#         return int(series.sum())
#     else:
#         # 列不存在：在整张表里匹配，任何列命中即记 1
#         def row_has_substr(row) -> bool:
#             return any(bool(regex.search(str(x))) for x in row)

#         return int(df_period.apply(row_has_substr, axis=1).sum())
# def count_period_variable_occurrences(
#     file_path: str,
#     sheet_name: Any,
#     date_column: str,
#     period: str,
#     column: str,
#     variable: Any,
# ) -> int:
#     """统计某年月内，指定列(或全表)包含 variable 的行数（子串 & 不区分大小写）。"""
#     df = pd.read_excel(file_path, sheet_name=sheet_name)

#     if date_column not in df.columns:
#         raise KeyError(f"日期列 '{date_column}' 不存在")

#     df[date_column] = (
#         df[date_column].astype(str)
#                        .map(_clean_month_str)
#                        .pipe(pd.to_datetime, format="%Y-%m")
#     )

#     period_dt = pd.to_datetime(period, format="%Y-%m")
#     df = df.loc[df[date_column].dt.to_period("M") == period_dt.to_period("M")]

#     regex = re.compile(re.escape(str(variable)), re.IGNORECASE)

#     if column in df.columns:
#         return int(df[column].astype(str).str.contains(regex, na=False).sum())
#     # 列名不存在 → 全表搜索
#     return int(
#         df.apply(lambda r: any(regex.search(str(x)) for x in r), axis=1).sum()
#     )

# def compare_period_variable_counts(
#     file_path: str,
#     sheet_name: Any,
#     date_column: str,
#     column: str,
#     variable: Any,
#     period1: str,
#     period2: str,
# ) -> int:
#     c1 = count_period_variable_occurrences(
#         file_path, sheet_name, date_column, period1, column, variable
#     )
#     c2 = count_period_variable_occurrences(
#         file_path, sheet_name, date_column, period2, column, variable
#     )
#     return c2 - c1

# def compare_period_variable_counts(
#     file_path: str,
#     sheet_name: Any,
#     date_column: str,
#     column: str,
#     variable: Any,
#     period1: str,
#     period2: str,
# ) -> Tuple[int, int, int]:
#     """返回 (period1_count, period2_count, diff)。"""
#     c1 = count_period_variable_occurrences(
#         file_path, sheet_name, date_column, period1, column, variable
#     )
#     c2 = count_period_variable_occurrences(
#         file_path, sheet_name, date_column, period2, column, variable
#     )
#     return c1, c2, c2 - c1

# def top_k_variables_in_period(
#     file_path: str,
#     sheet_name: Any,
#     date_column: str,
#     period: str,
#     column: str,
#     k: int,
# ) -> List[tuple]:
#     df = pd.read_excel(file_path, sheet_name=sheet_name)
#     df[date_column] = pd.to_datetime(df[date_column])
#     period_dt = pd.to_datetime(period)
#     mask = df[date_column].dt.to_period("M") == period_dt.to_period("M")
#     counts = df.loc[mask, column].value_counts().head(k)
#     return list(counts.items())
# def top_k_variables_in_period(
#     file_path: str,
#     sheet_name: Any,
#     date_column: str,
#     period: str,
#     column: str,
#     k: int,
# ) -> List[Tuple[Any, int]]:
#     df = pd.read_excel(file_path, sheet_name=sheet_name)
#     df[date_column] = (
#         df[date_column].astype(str)
#                        .map(_clean_month_str)
#                        .pipe(pd.to_datetime, format="%Y-%m")
#     )
#     period_dt = pd.to_datetime(period, format="%Y-%m")
#     mask = df[date_column].dt.to_period("M") == period_dt.to_period("M")
#     counts = df.loc[mask, column].value_counts().head(k)
#     return list(counts.items())