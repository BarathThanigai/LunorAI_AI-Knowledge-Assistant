"""NVIDIA LLM service for the LunorAI RAG pipeline."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Create and cache the NVIDIA OpenAI-compatible client."""

    api_key = os.getenv("NVIDIA_API_KEY")
    base_url = os.getenv(
        "NVIDIA_BASE_URL",
        "https://integrate.api.nvidia.com/v1",
    )

    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is not configured."
        )

    return OpenAI(
        base_url=base_url,
        api_key=api_key,
    )


def generate_answer(
    question: str,
    context: str,
) -> str:
    """Generate a grounded answer using retrieved document context."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            "Question must be a non-empty string."
        )

    if not isinstance(context, str) or not context.strip():
        raise ValueError(
            "Context must be a non-empty string."
        )

    model = os.getenv("NVIDIA_MODEL")

    if not model:
        raise RuntimeError(
            "NVIDIA_MODEL is not configured."
        )

    system_prompt = """You are LunorAI, a knowledge assistant.

Answer the user's question using ONLY the information provided
in the retrieved knowledge-base context.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts that are not supported by the context.
3. If the answer cannot be found in the context, say:
   "I couldn't find that information in the knowledge base."
4. Keep the answer concise and directly answer the question.
5. Return only the final answer.
"""

    user_prompt = f"""Retrieved knowledge-base context:

{context}

User question:
{question}

Return only the final answer.
"""

    client = get_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.2,
        max_tokens=512,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        },
    )

    answer = response.choices[0].message.content

    if not answer:
        raise RuntimeError(
            "The LLM returned an empty response."
        )

    return answer.strip()


__all__ = [
    "generate_answer",
]