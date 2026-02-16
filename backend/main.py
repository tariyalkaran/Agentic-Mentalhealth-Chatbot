from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn 

from ai_agent import graph


app = FastAPI()

class Query(BaseModel):
    message: str

@app.post("/ask")
async def ask(query: Query):
    # inputs = {"messages": [("system", SYSTEM_PROMPT), ("user", query.message)]}
    inputs = {"messages": [("user", query.message)]}
    # stream = graph.stream(inputs, stream_mode="updates")
    result = graph.invoke(inputs)

    # Final assistant response
    final_response = result["messages"][-1].content

    # Optional: detect tool usage
    tool_called_name = "None"
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_called_name = msg.tool_calls[0]["name"]

    # tool_called_name, final_response = parse_response(stream)

    # Step3: Send response to the frontend
    return {"response": final_response,
            "tool_called": tool_called_name}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
