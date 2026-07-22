from dotenv import load_dotenv

load_dotenv()

from app.sql_agent import create_sql_agent

print("Creating agent...")

agent = create_sql_agent(csv_path=None)

print("Agent created.")

question = "List all tables."

print(f"Question: {question}")

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": question,
            }
        ]
    }
)

print("\nRESULT\n")
print(result)