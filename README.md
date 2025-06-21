# Multi-Cloud VM MCP Server

Support for AWS EC2, Azure Virtual Machines and GCP Compute Engine.

[Demo link](https://x.com/neelabalan/status/1935339333478924397)

## Configuration

```
{
  "mcpServers": {
    "CloudComputeInfo": {
      "command": "~/.local/bin/uv",
      "args": [
        "run",
        "--with",
        "mcp[cli]",
        "--with",
        "requests",
        "--with",
        "pyyaml",
        "mcp",
        "run",
        "~/cloud-compute-mcp-server/main.py"
      ]
    }
  }
}
```