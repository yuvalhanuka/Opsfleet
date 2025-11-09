import logging
import re
from typing import Optional, List, Dict, Any

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv()

class InvalidSQLQueryError(Exception):
    """Custom exception raised when a SQL query contains disallowed commands."""
    pass


class BigQueryRunner:
    """A lean BigQuery client for executing SQL queries and returning DataFrame results."""
    
    def __init__(self, project_id: Optional[str] = None, dataset_id: Optional[str] = "bigquery-public-data.thelook_ecommerce") -> None:
        """Initialize BigQuery client.
        
        Args:
            project_id: Google Cloud project ID. If None, uses default credentials.
            dataset_id: BigQuery dataset ID. If None, uses default dataset.
        """
        logging.info("Initializing BigQuery client")
        try:
            self.client = bigquery.Client(project=project_id)
            self.dataset_id = dataset_id
            logging.info(f"BigQuery client initialized for dataset: {self.dataset_id}")
        except Exception as e:
            logging.error(f"Failed to initialize BigQuery client: {str(e)}")
            raise
    
    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """Execute a SQL query and return results as a DataFrame.
        
        Args:
            sql_query: The SQL query to execute.
            
        Returns:
            DataFrame containing the query results.
            
        Raises:
            Exception: If query execution fails.
        """
        try:
            logging.info(f"Executing BigQuery query")
            sql_query = self.validate_sql_query(query=sql_query)
            query_job = self.client.query(sql_query)
            df = query_job.result().to_dataframe()
            logging.info(f"Query completed successfully, returned {len(df)} rows")
            return df
        except Exception as e:
            logging.error(f"BigQuery execution failed: {str(e)}")
            raise 

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get schema information for a specific table.
        
        Args:
            table_name: Name of the table (orders, order_items, products, users).
            
        Returns:
            List of dictionaries containing column information.
        """
        try:
            table_ref = f"{self.dataset_id}.{table_name}"
            table = self.client.get_table(table_ref)
            schema_info = []
            for field in table.schema:
                schema_info.append({
                    "name": field.name,
                    "type": field.field_type,
                    "mode": field.mode,
                    "description": field.description or ""
                })
            logging.info(f"Retrieved schema for table {table_name}")
            return schema_info
        except Exception as e:
            logging.error(f"Failed to get schema for table {table_name}: {str(e)}")
            raise

    @staticmethod
    def validate_sql_query(query: str) -> str:
        """
        Validate an incoming SQL query string to ensure it's safe/allowed.
        Raise an InvalidSQLQueryError if the query includes disallowed syntax.

        Returns the original query if it's safe, or raises an exception otherwise.
        """
        disallowed = [
            # Data Definition Language (DDL) commands:
            "CREATE",  # Creates new objects (tables, views, indexes, procedures, etc.)
            "ALTER",  # Modifies existing objects (tables, columns, constraints, etc.)
            "DROP",  # Deletes objects from the database.
            "TRUNCATE",  # Removes all rows from a table, often non-transactionally.
            "RENAME",  # Renames database objects.
            "COMMENT",  # Adds or modifies object comments (can alter metadata).

            # Data Manipulation Language (DML) commands:
            "INSERT",  # Inserts new rows into a table.
            "UPDATE",  # Modifies existing rows in a table.
            "DELETE",  # Deletes rows from a table.
            "MERGE",  # Conditionally inserts, updates, or deletes rows (UPSERT functionality).
            "REPLACE",  # MySQL-specific upsert-like command.
            "UPSERT",  # Available in some dialects to insert or update rows.

            # Stored Procedure / Function Execution:
            "EXEC",  # Executes a stored procedure (SQL Server, etc.)
            "EXECUTE",  # Executes a stored procedure.
            "CALL",  # Calls a stored procedure (MySQL, Oracle, PostgreSQL, etc.)

            # Data Control Language (DCL) commands:
            "GRANT",  # Grants privileges on database objects.
            "REVOKE",  # Revokes privileges on database objects.
            "DENY",  # SQL Server-specific command to deny privileges.

            # Administrative and System Commands:
            "SHUTDOWN",  # Shuts down the database server.
            "BACKUP",  # Initiates a database backup.
            "RESTORE",  # Restores database objects from a backup.
            "DBCC",  # SQL Server Database Console Commands that can alter state.
            "RECONFIGURE",  # SQL Server command to change server settings.

            # Extended/System Procedures (typically in SQL Server):
            "XP_",  # Prefix for extended stored procedures (e.g., XP_CMDSHELL) that can modify system state.

            # Commands that alter the entire database context or settings:
            "ALTER DATABASE",  # Alters database-level settings.
            "USE",  # Changes the database context.
            "SET",
            # May change session settings that affect data (e.g., SET IDENTITY_INSERT, SET TRANSACTION ISOLATION LEVEL).

            # Miscellaneous and dialect-specific commands:
            "FLUSH",  # MySQL command to clear caches or logs.
            "LOCK",  # Commands like LOCK TABLES can affect transactional state.
            "UNLOCK",  # Releases locks; included as part of lock management.
        ]

        # Case-insensitive check for disallowed commands
        for keyword in disallowed:
            if re.search(rf"\b{keyword}\b", query, re.IGNORECASE):
                raise InvalidSQLQueryError(
                    f"The query contains a disallowed command: {keyword}"
                )

        return query