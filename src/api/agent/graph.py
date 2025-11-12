from pydantic import BaseModel
from typing import Annotated, List, Dict, Any 
from operator import add 
from langgraph.graph import StateGraph, START, END
from api.agent.utils.utils import get_tool_descriptions 
from api.agent.agents import intent_router_node, product_qa_agent, shopping_cart_agent, ToolCall, RAGUsedContext
from api.agent.tools import get_formatted_item_context, get_formatted_reviews_context, add_to_shopping_cart, remove_from_cart, get_shopping_cart
from langgraph.prebuilt import ToolNode
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue 
import numpy as np
import json
from langgraph.checkpoint.postgres import PostgresSaver
from pydantic import Field

class AgentProperties(BaseModel): 
    iteration: int = 0 
    final_answer: bool = False 
    available_tools: List[Dict[str, Any]] = [] 
    tool_calls: List[ToolCall] = []

class State(BaseModel):
    messages: Annotated[List[Any], add] = []
    user_intent: str = ""
    product_qa_agent: AgentProperties = Field(default_factory=AgentProperties)
    shopping_cart_agent: AgentProperties = Field(default_factory=AgentProperties)
    answer: str = ""
    references: Annotated[List[RAGUsedContext], add] = []
    user_id: str = ""
    cart_id: str = ""
    trace_id: str = ""

#### Routers 


def product_qa_agenttool_router(state) -> str:
    """Decide whether to continue or end"""
    
    if state.product_qa_agent.final_answer:
        return "end"
    elif state.product_qa_agent.iteration > 4:
        return "end"
    elif len(state.product_qa_agent.tool_calls) > 0:
        return "tools"
    else:
        return "end"

def shopping_cart_agent_tool_router(state) -> str:
    """Decide whether to continue or end"""
    
    if state.shopping_cart_agent.final_answer:
        return "end"
    elif state.shopping_cart_agent.iteration > 2:
        return "end"
    elif len(state.shopping_cart_agent.tool_calls) > 0:
        return "tools"
    else:
        return "end"

def user_intent_router(state) -> str:
    """Decide whether to continue or end"""
    
    if state.user_intent == "product_qa":
        return "product_qa_agent"
    elif state.user_intent == "shopping_cart":
        return "shopping_cart_agent"
    else:
        return "end"

#### Workflow

workflow = StateGraph(State)

product_qa_agent_tools = [get_formatted_item_context, get_formatted_reviews_context]
product_qa_agent_tool_node = ToolNode(product_qa_agent_tools)
product_qa_agent_tool_descriptions = get_tool_descriptions(product_qa_agent_tools)

shopping_cart_agent_tools = [add_to_shopping_cart, remove_from_cart, get_shopping_cart]
shopping_cart_agent_tool_node = ToolNode(shopping_cart_agent_tools)
shopping_cart_agent_tool_descriptions = get_tool_descriptions(shopping_cart_agent_tools)

workflow.add_node("product_qa_agent", product_qa_agent)
workflow.add_node("shopping_cart_agent", shopping_cart_agent)
workflow.add_node("intent_router", intent_router_node)

workflow.add_node("product_qa_agent_tool_node", product_qa_agent_tool_node)
workflow.add_node("shopping_cart_agent_tool_node", shopping_cart_agent_tool_node)

workflow.add_edge(START, "intent_router")

workflow.add_conditional_edges( 
    "intent_router",
    user_intent_router,
    {
        "product_qa_agent": "product_qa_agent",
        "shopping_cart_agent": "shopping_cart_agent",
        "end": END
    }
)
workflow.add_conditional_edges(
    "product_qa_agent",
    product_qa_agenttool_router,
    {
        "tools": "product_qa_agent_tool_node",
        "end": END
    }
)

workflow.add_conditional_edges(
    "shopping_cart_agent",
    shopping_cart_agent_tool_router,
    {
        "tools": "shopping_cart_agent_tool_node",
        "end": END
    }
)

workflow.add_edge("product_qa_agent_tool_node", "product_qa_agent")
workflow.add_edge("shopping_cart_agent_tool_node", "shopping_cart_agent")

#### Agent Execution Function 

def run_agent(question: str, thread_id: str):  

    initial_state = {
        "messages": [{"role": "user", "content": question}], 
        "iteration": 0,
        "available_tools": tool_descriptions,  
    } 

    config = {"configurable": {"thread_id": thread_id}}

    with PostgresSaver.from_conn_string("postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db") as checkpointer: 

        graph = workflow.compile(checkpointer=checkpointer)

        result = graph.invoke(initial_state, config=config) 

    return result 

def run_agent_stream_wrapper(question: str, thread_id: str): 

    def _string_for_sse(message: str): 
        return f"data: {message}\n\n"
    
    def _process_graph_event(chunk):

        def _is_node_start(chunk):
            return chunk[1].get("type") == "task"

        def _is_node_end(chunk):
            return chunk[0] == "updates"

        def _tool_to_text(tool_call):
            if tool_call.name == "get_formatted_itemcontext":
                return f"Looking for items: {tool_call.arguments.get('query', '')}."
            elif tool_call.name == "get_formatted_reviews_context":
                return f"Fetching user reviews..."
            else:
                return "Unknown tool call..."

        if _is_node_start(chunk):
            if chunk[1].get("payload", {}).get("name") == "intent_router_node":
                return "Analysing the question..."
            if chunk[1].get("payload", {}).get("name") == "agent_node":
                return "Planning..."
            if chunk[1].get("payload", {}).get("name") == "tool_node":
                message = " ".join([_tool_to_text(tool_call) for tool_call in chunk[1].get('payload', {}).get('input', {}).tool_calls])
                return message  

        else: 
            return False

    qdrant_client = QdrantClient(url="http://qdrant:6333")  

    initial_state = {
        "messages": [{"role": "user", "content": question}], 
        "user_intent": "",
        "product_qa_agent": {
            "iteration": 0,
            "final_answer": False,
            "available_tools": product_qa_agent_tool_descriptions,
            "tool_calls": []
        },
        "shopping_cart_agent": {
            "iteration": 0,
            "final_answer": False,
            "available_tools": shopping_cart_agent_tool_descriptions,
            "tool_calls": []
        }, 
        "user_id": thread_id,
        "cart_id": thread_id
    } 

    config = {"configurable": {"thread_id": thread_id}}

    with PostgresSaver.from_conn_string("postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db") as checkpointer: 

        graph = workflow.compile(checkpointer=checkpointer)

        for chunk in graph.stream(
            initial_state, 
            config=config,
            stream_mode=["debug", "values"]
        ):
            processed_chunk = _process_graph_event(chunk) 

            if processed_chunk:
                yield _string_for_sse(processed_chunk)

            if chunk[0] == "values": 
                result = chunk[1] 

    used_context = []  
    dummy_vector = np.zeros(1536).tolist() 

    for item in result.get("references", []):
        payload = qdrant_client.query_points( 
            collection_name="Amazon-items-collection-01-hybrid-search", 
            query=dummy_vector, 
            using="text-embedding-3-small",
            limit=1, 
            with_payload=True, 
            query_filter = Filter( 
                must=[
                    FieldCondition( 
                        key = "parent_asin", 
                        match = MatchValue(value=item.id))
                ]
            )
        ).points[0].payload

        image_url = payload.get("image")
        price = payload.get("price")

        if image_url:  
            used_context.append({"image_url": image_url, "price": price, "description": item.description})

    yield _string_for_sse(json.dumps(
        {
            "type": "final_result", 
            "data": {
            "answer": result.get("answer"),
            "used_context": used_context, 
            "trace_id": result.get("trace_id")
            }
        }
    ))