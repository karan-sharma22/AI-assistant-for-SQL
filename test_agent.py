from dotenv import load_dotenv

load_dotenv()

from app.sql_agent import create_sql_agent

print("Creating agent...")
agent = create_sql_agent()
print("✓ Agent created")

print("Invoking agent...")
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "List all tables in the database."
            }
        ]
    }
)

print(result)