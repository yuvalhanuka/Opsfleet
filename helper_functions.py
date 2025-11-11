import json
import os
import logging
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

def _truncate_text_words(text: str, max_words: int = 50) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"

def truncate_message(msg, max_words: int = 50):
    """
    Return a new message of the same class with content truncated to `max_words`.
    Preserves id, additional_kwargs, response_metadata when available.
    If content is a list of parts (LangChain multi-part), we join their 'text'.
    """
    content = msg.content
    if isinstance(content, str):
        new_content = _truncate_text_words(content, max_words)
    else:
        # content could be a list of parts like [{"type":"text","text":"..."}]
        try:
            parts_text = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            ).strip()
            new_content = _truncate_text_words(parts_text, max_words)
        except Exception:
            # If unknown structure, return as-is
            return msg

    return msg.__class__(
        content=new_content,
        id=getattr(msg, "id", None),
        additional_kwargs=getattr(msg, "additional_kwargs", {}),
        response_metadata=getattr(msg, "response_metadata", {}),
    )

def setup_logging(path="logs/app.log", level=logging.ERROR):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(level)

    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(fh)