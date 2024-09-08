from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List
from app.entity_linking import EntityLinking
from app.foundation_models.chat_openai import AIModelType

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
