from app.services.llm import generate_answer


context = """
Barath has worked on several software and AI projects.

One project is a Concurrent Web Crawler built using
React, FastAPI, Python, asyncio, aiohttp, BeautifulSoup,
SQLite, Docker, and GitHub Actions.

Another project is a GitHub Developer Portfolio Analytics
System built using React, Node.js, Express.js, MySQL, and Ollama.
"""


answer = generate_answer(
    question="What projects has Barath worked on?",
    context=context,
)

print("\nAnswer:")
print(answer)