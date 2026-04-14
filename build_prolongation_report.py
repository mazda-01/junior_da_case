from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"


MONTHS_RU = {
    "январь": 1,
    "февраль": 2,
    "март": 3,
    "апрель": 4,
    "май": 5,
    "июнь": 6,
    "июль": 7,
    "август": 8,
    "сентябрь": 9,
    "октябрь": 10,
    "ноябрь": 11,
    "декабрь": 12,
}
MONTHS_RU_REVERSE = {v: k for k, v in MONTHS_RU.items()}


@dataclass
class ParsedValue:
    amount: float
    is_stop_end: bool
    is_zero_token: bool


def parse_ru_month(text: str) -> pd.Timestamp:
    text = str(text).strip().lower()
    month_name, year = text.split()
    return pd.Timestamp(year=int(year), month=MONTHS_RU[month_name], day=1)


def format_ru_month(ts: pd.Timestamp) -> str:
    return f"{MONTHS_RU_REVERSE[ts.month].capitalize()} {ts.year}"


def parse_value(raw: object) -> ParsedValue:
    if pd.isna(raw):
        return ParsedValue(amount=0.0, is_stop_end=False, is_zero_token=False)

    value = str(raw).strip().lower()
    if not value:
        return ParsedValue(amount=0.0, is_stop_end=False, is_zero_token=False)
    if value in {"стоп", "end"}:
        return ParsedValue(amount=0.0, is_stop_end=True, is_zero_token=False)
    if value == "в ноль":
        return ParsedValue(amount=0.0, is_stop_end=False, is_zero_token=True)

    normalized = re.sub(r"[\u00a0\s]", "", value).replace(",", ".")
    return ParsedValue(amount=float(normalized), is_stop_end=False, is_zero_token=False)


def build_project_month_agg(financial_df: pd.DataFrame, month_cols: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in financial_df.iterrows():
        project_id = int(row["id"])
        for month_col in month_cols:
            parsed = parse_value(row[month_col])
            rows.append(
                {
                    "id": project_id,
                    "month_col": month_col,
                    "amount": parsed.amount,
                    "is_stop_end": parsed.is_stop_end,
                    "is_zero_token": parsed.is_zero_token,
                    "has_positive_amount": parsed.amount > 0,
                }
            )

    long_df = pd.DataFrame(rows)
    agg = (
        long_df.groupby(["id", "month_col"], as_index=False)
        .agg(
            amount=("amount", "sum"),
            is_stop_end=("is_stop_end", "max"),
            has_zero_token=("is_zero_token", "max"),
            has_positive_amount=("has_positive_amount", "max"),
        )
    )
    agg["all_parts_zero_token_only"] = (
        (agg["amount"] == 0) & agg["has_zero_token"] & (~agg["has_positive_amount"])
    )
    return agg


def calculate_mart(
    prolongations_df: pd.DataFrame,
    month_agg_df: pd.DataFrame,
    ordered_month_cols: list[str],
) -> pd.DataFrame:
    month_index = {m: idx for idx, m in enumerate(ordered_month_cols)}
    stats_by_key = month_agg_df.set_index(["id", "month_col"]).to_dict("index")

    mart_rows: list[dict] = []
    for _, row in prolongations_df.iterrows():
        project_id = int(row["id"])
        manager = row["AM"]
        completion_ts = parse_ru_month(row["month"])
        completion_col = format_ru_month(completion_ts)

        if completion_col not in month_index:
            # Если в файле есть нестандартный месяц, пропускаем запись.
            continue

        comp_idx = month_index[completion_col]
        prev_col = ordered_month_cols[comp_idx - 1] if comp_idx - 1 >= 0 else None
        t1_col = ordered_month_cols[comp_idx + 1] if comp_idx + 1 < len(ordered_month_cols) else None
        t2_col = ordered_month_cols[comp_idx + 2] if comp_idx + 2 < len(ordered_month_cols) else None

        # Исключаем проект, если stop/end встречается в последнем месяце реализации или раньше.
        stop_before_or_at_completion = False
        for mcol in ordered_month_cols[: comp_idx + 1]:
            info = stats_by_key.get((project_id, mcol))
            if info and bool(info["is_stop_end"]):
                stop_before_or_at_completion = True
                break

        completion_info = stats_by_key.get((project_id, completion_col), {})
        last_month_shipment = float(completion_info.get("amount", 0.0))
        used_prev_month_for_last = False

        # Правило "в ноль": для последнего месяца берем предыдущий только если все части 0.
        if (
            completion_info.get("all_parts_zero_token_only", False)
            and prev_col is not None
        ):
            prev_info = stats_by_key.get((project_id, prev_col), {})
            last_month_shipment = float(prev_info.get("amount", 0.0))
            used_prev_month_for_last = True

        t1_shipment = float(stats_by_key.get((project_id, t1_col), {}).get("amount", 0.0)) if t1_col else 0.0
        t2_shipment = float(stats_by_key.get((project_id, t2_col), {}).get("amount", 0.0)) if t2_col else 0.0

        mart_rows.append(
            {
                "id": project_id,
                "AM": manager,
                "completion_month_ts": completion_ts,
                "completion_month": completion_col,
                "last_month_shipment": last_month_shipment,
                "ship_t1": t1_shipment,
                "ship_t2": t2_shipment,
                "excluded_stop_end": stop_before_or_at_completion,
                "used_prev_month_for_last": used_prev_month_for_last,
            }
        )

    return pd.DataFrame(mart_rows)


def safe_ratio(numerator: float, denominator: float) -> float:
    return np.nan if denominator == 0 else numerator / denominator


def monthly_kpi_for_group(group_df: pd.DataFrame, report_months: list[pd.Timestamp]) -> pd.DataFrame:
    rows = []
    for report_month in report_months:
        month_m1 = report_month - pd.DateOffset(months=1)
        month_m2 = report_month - pd.DateOffset(months=2)
        m_label = format_ru_month(report_month)

        cohort_m1 = group_df[
            (group_df["completion_month_ts"] == month_m1) & (~group_df["excluded_stop_end"])
        ]
        denominator_1 = cohort_m1["last_month_shipment"].sum()
        numerator_1 = cohort_m1.loc[cohort_m1["ship_t1"] > 0, "ship_t1"].sum()
        coef_1 = safe_ratio(numerator_1, denominator_1)

        cohort_m2_base = group_df[
            (group_df["completion_month_ts"] == month_m2)
            & (~group_df["excluded_stop_end"])
            & (group_df["ship_t1"] == 0)
        ]
        denominator_2 = cohort_m2_base["last_month_shipment"].sum()
        numerator_2 = cohort_m2_base.loc[cohort_m2_base["ship_t2"] > 0, "ship_t2"].sum()
        coef_2 = safe_ratio(numerator_2, denominator_2)

        rows.append(
            {
                "report_month": m_label,
                "numerator_1m": numerator_1,
                "denominator_1m": denominator_1,
                "coef_1m": coef_1,
                "numerator_2m": numerator_2,
                "denominator_2m": denominator_2,
                "coef_2m": coef_2,
                "completed_projects_m1": len(cohort_m1),
                "completed_projects_m2_no1m": len(cohort_m2_base),
                "not_prolonged_1m_cnt": int((cohort_m1["ship_t1"] == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def yearly_weighted_kpi(monthly_df: pd.DataFrame, group_name: str, year: int = 2023) -> pd.DataFrame:
    num1 = monthly_df["numerator_1m"].sum()
    den1 = monthly_df["denominator_1m"].sum()
    num2 = monthly_df["numerator_2m"].sum()
    den2 = monthly_df["denominator_2m"].sum()
    return pd.DataFrame(
        [
            {
                "group": group_name,
                "year": year,
                "numerator_1m": num1,
                "denominator_1m": den1,
                "coef_1m_year": safe_ratio(num1, den1),
                "numerator_2m": num2,
                "denominator_2m": den2,
                "coef_2m_year": safe_ratio(num2, den2),
            }
        ]
    )


def add_excel_visuals(path: Path) -> None:
    from openpyxl import load_workbook
    from openpyxl.chart import LineChart, BarChart, Reference
    from openpyxl.formatting.rule import ColorScaleRule

    wb = load_workbook(path)

    # Лист визуализации с трендом отдела и рейтингом менеджеров по годовому KPI.
    ws_vis = wb.create_sheet("4_visuals")
    ws_dep = wb["2_department_kpi"]
    ws_mgr_year = wb["1_manager_kpi_yearly"]

    ws_vis["A1"] = "Department trend (1M/2M coefficients)"
    ws_vis["A20"] = "Managers yearly ranking (1M coefficient)"

    line_chart = LineChart()
    line_chart.title = "Department coefficients by month"
    line_chart.y_axis.title = "Coefficient"
    line_chart.x_axis.title = "Month"

    cats = Reference(ws_dep, min_col=1, min_row=2, max_row=13)
    data = Reference(ws_dep, min_col=4, max_col=7, min_row=1, max_row=13)
    line_chart.add_data(data, titles_from_data=True)
    line_chart.set_categories(cats)
    ws_vis.add_chart(line_chart, "A3")

    bar_chart = BarChart()
    bar_chart.title = "Managers yearly 1M coefficient"
    bar_chart.y_axis.title = "Coefficient"
    bar_chart.x_axis.title = "Manager"
    cats2 = Reference(ws_mgr_year, min_col=1, min_row=2, max_row=ws_mgr_year.max_row)
    data2 = Reference(ws_mgr_year, min_col=6, min_row=1, max_row=ws_mgr_year.max_row)
    bar_chart.add_data(data2, titles_from_data=True)
    bar_chart.set_categories(cats2)
    ws_vis.add_chart(bar_chart, "A22")

    # Heatmap для помесячных коэффициентов менеджеров.
    ws_mgr_month = wb["1_manager_kpi_monthly"]
    ws_mgr_month.conditional_formatting.add(
        f"F2:F{ws_mgr_month.max_row}",
        ColorScaleRule(
            start_type="min",
            mid_type="percentile",
            mid_value=50,
            end_type="max",
        ),
    )
    ws_mgr_month.conditional_formatting.add(
        f"I2:I{ws_mgr_month.max_row}",
        ColorScaleRule(
            start_type="min",
            mid_type="percentile",
            mid_value=50,
            end_type="max",
        ),
    )

    wb.save(path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prolongations = pd.read_csv(DATA_DIR / "prolongations.csv")
    financial = pd.read_csv(DATA_DIR / "financial_data.csv")

    month_cols = [c for c in financial.columns if c not in {"id", "Причина дубля", "Account"}]
    month_cols_sorted = sorted(month_cols, key=lambda x: parse_ru_month(x))

    month_agg = build_project_month_agg(financial, month_cols_sorted)
    mart = calculate_mart(prolongations, month_agg, month_cols_sorted)

    # KPI рассчитываем за месяцы отчета 2023.
    report_months = [pd.Timestamp(year=2023, month=m, day=1) for m in range(1, 13)]

    manager_monthly_parts = []
    manager_yearly_parts = []
    for manager, g in mart.groupby("AM"):
        monthly = monthly_kpi_for_group(g, report_months)
        monthly.insert(0, "AM", manager)
        manager_monthly_parts.append(monthly)
        manager_yearly_parts.append(yearly_weighted_kpi(monthly, group_name=manager, year=2023))

    manager_monthly = pd.concat(manager_monthly_parts, ignore_index=True).sort_values(["AM", "report_month"])
    manager_yearly = pd.concat(manager_yearly_parts, ignore_index=True).sort_values("group")
    manager_yearly = manager_yearly.rename(columns={"group": "AM"})

    dep_monthly = monthly_kpi_for_group(mart, report_months)
    dep_yearly = yearly_weighted_kpi(dep_monthly, group_name="Отдел", year=2023)

    # Диагностический блок.
    diagnostics = (
        mart.groupby("AM", as_index=False)
        .agg(
            total_projects=("id", "count"),
            excluded_stop_end=("excluded_stop_end", "sum"),
            used_prev_month_fallback=("used_prev_month_for_last", "sum"),
        )
        .sort_values("AM")
    )
    diagnostics["excluded_share"] = np.where(
        diagnostics["total_projects"] > 0,
        diagnostics["excluded_stop_end"] / diagnostics["total_projects"],
        np.nan,
    )

    # Сохраняем промежуточные данные для прозрачности проверки.
    mart.to_csv(OUT_DIR / "project_mart.csv", index=False)
    manager_monthly.to_csv(OUT_DIR / "manager_monthly_kpi.csv", index=False)
    manager_yearly.to_csv(OUT_DIR / "manager_yearly_kpi.csv", index=False)
    dep_monthly.to_csv(OUT_DIR / "department_monthly_kpi.csv", index=False)
    dep_yearly.to_csv(OUT_DIR / "department_yearly_kpi.csv", index=False)
    diagnostics.to_csv(OUT_DIR / "diagnostics.csv", index=False)

    # Финальный отчет для руководителя.
    report_path = OUT_DIR / "prolongation_report_2023.xlsx"
    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        manager_monthly.to_excel(writer, sheet_name="1_manager_kpi_monthly", index=False)
        manager_yearly.to_excel(writer, sheet_name="1_manager_kpi_yearly", index=False)
        dep_monthly.to_excel(writer, sheet_name="2_department_kpi", index=False)
        dep_yearly.to_excel(writer, sheet_name="2_department_yearly", index=False)
        diagnostics.to_excel(writer, sheet_name="3_diagnostics", index=False)

    add_excel_visuals(report_path)

    print("Done.")
    print(f"Report: {report_path}")
    print(f"Mart:   {OUT_DIR / 'project_mart.csv'}")


if __name__ == "__main__":
    main()
