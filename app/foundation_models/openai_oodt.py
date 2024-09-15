import json
from typing import Union
from app.foundation_models.chat_openai import (
    AIModelType,
    get_api_key,
    get_model_name,
)
from openai import AsyncOpenAI


class ChatOpenAIOutOfDomainDetection:
    temperature: float
    system_prompt: Union[str, None] = None
    model: AIModelType

    def __init__(
        self,
        temperature=0.7,
        system_prompt: Union[str, None] = None,
        model: AIModelType = AIModelType.GPT4_TURBO,
    ):
        self.useTurbo = model == AIModelType.GPT4_TURBO
        api_key = get_api_key(model)

        self.openai = AsyncOpenAI(
            api_key=api_key,
        )
        self.temperature = temperature
        self.model = model
        self.system_prompt = system_prompt

    async def generate_response(self, query: str) -> object:
        system_message = (
            self.system_prompt
            if self.system_prompt is not None
            else "Du bist ein KI-Assistent, der dabei hilft, Suchanfragen zu klassifizieren und zu erkennen, ob sie in den Bereich 'Out-of-Domain' fallen."
        )
        model = get_model_name(model=self.model)

        # Creating the German prompt to detect out-of-domain queries
        prompt = f"""
        Klassifiziere die folgende Suchanfrage als entweder 'in-domain' oder 'out-of-domain'.
        Du bist ein Produktberater im E-Commerce, spezialisiert auf Multimedia-Produkte (z. B. Smartphones, Laptops, Tablets). Deine Aufgabe ist es, Kunden dabei zu unterstützen, Produkte zu finden, die ihren Bedürfnissen entsprechen.
        Beantworte die Frage, ob diese Anfrage in deine Beratungsdomäne fällt oder nicht. Bedenke dabei, dass du nur für die Verkaufsberatung von Multimedia-Produkte zuständig bist.

        Query: "{query}"

        Gib die Antwort in folgender JSON-Struktur zurück:
        {{
          "query": "{query}",
          "outOfDomain": true/false,
        }}
        """

        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )

        # Extract and return the structured response as a JSON object
        try:
            structured_output = json.loads(response.choices[0].message.content)
        except (json.JSONDecodeError, KeyError):
            print("Invalid response format".response.choices[0].message.content)
            structured_output = {"error": "Invalid response format"}

        return structured_output
