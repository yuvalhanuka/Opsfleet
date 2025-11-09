import json
import os

from bq_client import BigQueryRunner

big_query_runner_instance = BigQueryRunner()

def get_tables_information() -> str:
    script_directory = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    combined_summary_path = os.path.join(script_directory, "SQL_tables_summary.txt")

    if os.path.exists(combined_summary_path):
        with open(combined_summary_path, "r", encoding="utf-8") as f:
            return f.read()

    tables = {
        "orders": "Customer order information",
        "order_items": "Individual items within orders",
        "products": "Product catalog and details",
        "users": "Customer demographics and information",
    }

    table_summaries = {}

    for table, description in tables.items():
        schema_info = big_query_runner_instance.get_table_schema(table_name=table)  # expected: list[dict]
        query_nonulls = f"""
        SELECT *
        FROM `bigquery-public-data.thelook_ecommerce.{table}` AS t
        WHERE TO_JSON_STRING(t) NOT LIKE '%:null%'
        LIMIT 1
        """
        df = big_query_runner_instance.execute_query(sql_query=query_nonulls)
        if df is None or df.empty:
            df = big_query_runner_instance.execute_query(
                sql_query=f"SELECT * FROM `bigquery-public-data.thelook_ecommerce.{table}` LIMIT 1"
            )
        lines = []
        lines.append(f"Table name: {table}")
        lines.append(f"Table description: {description}")
        lines.append("Table columns information:")
        for col in schema_info or []:
            name = col.get("name")
            typ = col.get("type")
            mode = col.get("mode")
            lines.append(f"name: {name}, type: {typ}, mode: {mode}")

        lines.append("Row example :")
        if df is not None and not df.empty:
            row_dict = df.head(1).to_dict(orient="records")[0]
            lines.append(json.dumps(row_dict, indent=2, default=str))
        else:
            lines.append("(no rows returned)")

        table_summaries[table] = "\n".join(lines)

    combined_summary = "\n\n".join(table_summaries.values())

    with open(combined_summary_path, "w", encoding="utf-8") as f:
        f.write(combined_summary)

    return combined_summary

