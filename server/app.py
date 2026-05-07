"""
FastAPI application configuration for the MCP Memory Server.

Sets up the FastMCP server with memory tools, combines with FastAPI routes,
serves a landing page with connection details, and adds middleware for user
auth header capture.
"""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastmcp import FastMCP

from .tools import load_tools
from .utils import header_store

mcp_server = FastMCP(name="custom-mcp-server")

STATIC_DIR = Path(__file__).parent / "../static"

# Load and register all memory tools
load_tools(mcp_server)

# Convert to streamable HTTP application
mcp_app = mcp_server.http_app()

# FastAPI for additional endpoints
app = FastAPI(
    title="MCP Memory Server",
    version="2.0.0",
    lifespan=mcp_app.lifespan,
)


@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve the landing page with connection details."""
    if STATIC_DIR.exists() and (STATIC_DIR / "index.html").exists():
        return FileResponse(STATIC_DIR / "index.html")
    return {"message": "MCP Memory Server is running", "status": "healthy"}


@app.get("/health")
def health():
    return {"status": "ok", "service": "mcp-memory-server", "version": "2.0.0"}


# Combine MCP routes with custom routes
combined_app = FastAPI(
    title="Combined MCP App",
    routes=[
        *mcp_app.routes,
        *app.routes,
    ],
    lifespan=mcp_app.lifespan,
)


@combined_app.middleware("http")
async def capture_headers(request: Request, call_next):
    """Capture request headers for on-behalf-of-user auth."""
    header_store.set(dict(request.headers))
    return await call_next(request)
