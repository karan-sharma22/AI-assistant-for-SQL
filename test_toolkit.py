from dotenv import load_dotenv

load_dotenv()

from app.database import get_database
from app.llm import get_llm
from langchain_community.agent_toolkits import SQLDatabaseToolkit

db = get_database()
llm = get_llm()

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

for tool in toolkit.get_tools():
    print(tool.name)