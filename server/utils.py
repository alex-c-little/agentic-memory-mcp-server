import contextvars
import os

from databricks.sdk import WorkspaceClient

header_store = contextvars.ContextVar("header_store")


def get_workspace_client():
    return WorkspaceClient()


def get_user_authenticated_workspace_client():
    is_databricks_app = "DATABRICKS_APP_NAME" in os.environ

    if not is_databricks_app:
        return WorkspaceClient()

    headers = header_store.get({})
    token = headers.get("x-forwarded-access-token")

    if not token:
        raise ValueError(
            "Authentication token not found in request headers (x-forwarded-access-token). "
        )

    return WorkspaceClient(token=token, auth_type="pat")
