from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import logging 

from api.api.models import AgentRequest
from api.agent.graph import run_agent_stream_wrapper

logger = logging.getLogger(__name__) 

rag_router = APIRouter() 

@rag_router.post("/") 
def rag( 
    request: Request, 
    payload: AgentRequest
) -> StreamingResponse: 

    return StreamingResponse( 
        run_agent_stream_wrapper(payload.query, payload.thread_id),
        media_type="text/event-stream"
    )

api_router = APIRouter() 
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])