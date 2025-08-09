from mcp_agent.mcp.mcp_aggregator import MCPAggregator

orig_call_tool = MCPAggregator.call_tool

async def wrapped_call_tool(self, name, arguments, *args, **kwargs):
    my_log_list.append((name, arguments))
    return await orig_call_tool(self, name, arguments, *args, **kwargs)

MCPAggregator.call_tool = wrapped_call_tool
