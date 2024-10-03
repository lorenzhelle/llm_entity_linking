from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List
from app.entity_linking import EntityLinking
from app.foundation_models.chat_openai import AIModelType
from app.foundation_models.claude_oodt import ClaudeOODT
from app.foundation_models.mistral_oodt import MistralOOTD
from app.foundation_models.llama_oodt import LlamaOOTD
from app.foundation_models.openai_oodt import ChatOpenAIOutOfDomainDetection

# load_and_export.py
from dotenv import load_dotenv
import os

app = FastAPI()

# Load the environment variables from .env file
load_dotenv()


class FilterRequest(BaseModel):
    message: str
    schema: Dict[str, Any]
    model: AIModelType


class FilterResponse(BaseModel):
    filter_generator_output: List[Dict[str, Any]]
    recognized_filters: List[Dict[str, Any]]


@app.post("/recognize-filters", response_model=FilterResponse)
async def recognize_filters(request: FilterRequest):
    llm_module = EntityLinking(schema=request.schema, model=request.model)

    if (
        request.model == AIModelType.MISTRAL_LARGE
        or request.model == AIModelType.MISTRAL_MIXTRAL_8x22B
        or request.model == AIModelType.MISTRAL_SMALL
    ):
        filter_generator_output = llm_module.generate_sync(conversation=request.message)
    else:
        filter_generator_output = await llm_module.generate_async(
            conversation=request.message
        )

    recognized_filters = []

    for filter in filter_generator_output:
        filtered_dict = {
            k: v for k, v in filter.model_dump().items() if v is not None and v
        }

        recognized_filters.append(filtered_dict)

    return FilterResponse(
        filter_generator_output=filter_generator_output,
        recognized_filters=recognized_filters,
    )


class QueryRequest(BaseModel):
    query: str
    model: AIModelType
    domain: str


@app.post("/check_domain")
async def check_domain(request: QueryRequest):
    print(request)
    prompt = f"""
    Beantworte die Frage, ob diese Anfrage in deine Beratungsdomäne fällt oder nicht. Bedenke dabei, dass du nur für die Verkaufsberatung von {request.domain} zuständig bist.

    Query: "{request.query}"

    Gib die Antwort in folgender JSON-Struktur zurück:
    
    {{
    "query": "{request.query}",
    "outOfDomain": true/false
    }}
    """
    try:
        if request.model in [AIModelType.CLAUDE_OPUS, AIModelType.CLAUDE_SONNET]:
            chat_model = ClaudeOODT(model=request.model)
        elif request.model in [AIModelType.MISTRAL_LARGE, AIModelType.MISTRAL_SMALL]:
            chat_model = MistralOOTD(model=request.model)
        elif request.model in [AIModelType.LLAMA_3_8B, AIModelType.LLAMA_3_70B]:
            chat_model = LlamaOOTD(model=request.model)
        elif request.model in [
            AIModelType.GPT3,
            AIModelType.GPT4_TURBO,
            AIModelType.GPT4_O_MINI,
            AIModelType.GPT4_O,
        ]:
            chat_model = ChatOpenAIOutOfDomainDetection(model=request.model)
        else:
            raise HTTPException(status_code=400, detail="Unsupported model")

        response = await chat_model.generate_response(prompt=prompt)
        print(response)
        return {"inDomain": not response.get("outOfDomain", False)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
