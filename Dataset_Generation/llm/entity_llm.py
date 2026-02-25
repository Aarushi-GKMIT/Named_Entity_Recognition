from openai import OpenAI
from schemas.entities_schema import ResumeEntities
from llm.prompts.entities_prompt import SYSTEM_PROMPT_ENTITIES
from config import OPENAI_API_KEY, MODEL_NAME

client = OpenAI(api_key=OPENAI_API_KEY)

def extract_entities(resume_text: str) -> ResumeEntities:
    response = client.responses.parse(
        model=MODEL_NAME,
        temperature=0,
        max_output_tokens=6000,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT_ENTITIES},
            {"role": "user", "content": resume_text}
        ],
        text_format=ResumeEntities
    )
    return response.output_parsed

