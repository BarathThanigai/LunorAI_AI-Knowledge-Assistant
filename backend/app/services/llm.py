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

    system_prompt = """You are LunorAI, a grounded knowledge assistant. Answer the user's question using ONLY information explicitly supported by the retrieved knowledge-base context. GROUNDING RULES: 1. You may directly paraphrase or summarize information stated in the context. 2. You may answer a question using a natural-language equivalent of something explicitly stated in the context. Example: Context: "Barath is so depressed :(" Question: "How does Barath feel?" Valid answer: "Barath is described as depressed." 3. Do NOT invent facts that are not present in the context. 4. Do NOT infer: - preferences from skills or usage - favorites from lists - opinions from actions - intentions from actions - causes unless explicitly stated - relationships unless explicitly stated - personal characteristics unless explicitly stated 5. A semantic connection alone is NOT enough. The retrieved context must contain evidence that actually supports the answer. 6. If the context does not contain enough information to answer the user's question, return exactly: "I couldn't find that information in the knowledge base." 7. Do not use outside knowledge. 8. Keep the answer concise. Return only the final answer. """

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