"""Lakebase connection pool with OAuth token rotation."""

import os
import uuid
import psycopg
from psycopg_pool import ConnectionPool
from databricks.sdk import WorkspaceClient

_pool = None
_ws_client = None


def _get_ws_client() -> WorkspaceClient:
    global _ws_client
    if _ws_client is None:
        _ws_client = WorkspaceClient()
    return _ws_client


class OAuthConnection(psycopg.Connection):
    """Connection subclass that generates a fresh Lakebase OAuth token on each connect.

    Supports both Lakebase APIs:
      - Legacy projects/branches/endpoints path (LAKEBASE_ENDPOINT_PATH).
      - New Database Instances API (LAKEBASE_INSTANCE_NAME).
    Whichever env var is set wins. The pool calls this on every connection
    acquire, so tokens never expire mid-use.
    """

    @classmethod
    def connect(cls, conninfo="", **kwargs):
        ws = _get_ws_client()
        endpoint_path = os.environ.get("LAKEBASE_ENDPOINT_PATH")
        instance_name = os.environ.get("LAKEBASE_INSTANCE_NAME")

        if endpoint_path:
            credential = ws.postgres.generate_database_credential(endpoint=endpoint_path)
        elif instance_name:
            credential = ws.database.generate_database_credential(
                request_id=str(uuid.uuid4()),
                instance_names=[instance_name],
            )
        else:
            raise RuntimeError(
                "Set LAKEBASE_ENDPOINT_PATH (legacy) or LAKEBASE_INSTANCE_NAME (new API)"
            )

        kwargs["password"] = credential.token
        return super().connect(conninfo, **kwargs)


def get_pool() -> ConnectionPool:
    """Get or create the Lakebase connection pool."""
    global _pool
    if _pool is None:
        host = os.environ["LAKEBASE_HOST"]
        database = os.environ.get("LAKEBASE_DATABASE", "agent_memory")
        user = os.environ.get("LAKEBASE_USER", "")
        port = os.environ.get("LAKEBASE_PORT", "5432")

        # If no user specified, get current user email
        if not user:
            me = _get_ws_client().current_user.me()
            user = me.user_name

        conninfo = (
            f"host={host} port={port} dbname={database} "
            f"user={user} sslmode=require"
        )
        _pool = ConnectionPool(
            conninfo=conninfo,
            connection_class=OAuthConnection,
            min_size=1,
            max_size=10,
            open=True,
        )
    return _pool


def execute_sql(sql: str, params: tuple = None, fetch: bool = True):
    """Execute SQL against Lakebase."""
    pool = get_pool()
    with pool.connection() as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(sql, params)
            if fetch and cur.description:
                columns = [desc.name for desc in cur.description]
                rows = cur.fetchall()
                return columns, rows
            return None, None
