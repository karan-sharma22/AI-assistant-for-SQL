from dotenv import load_dotenv
from app.llm import get_llm

load_dotenv()

llm = get_llm()

response = llm.invoke("Say hello in one sentence.")

print(response.content)