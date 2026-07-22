from dotenv import load_dotenv

load_dotenv()

from app.sql_agent import create_sql_agent

csv_path = r"D:\sql_assistant\text-to-sql-agent\sample_sales.csv"   # <-- replace with your CSV

print("Creating agent...")

agent = create_sql_agent(csv_path=csv_path)

print("Agent created.")

question = "How many rows are there?"

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    }
)

print(result)