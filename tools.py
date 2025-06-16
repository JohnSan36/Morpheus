from langchain.agents import AgentExecutor, create_tool_calling_agent
from pydantic import BaseModel, Field
from langchain_core.tools import tool



class Somar(BaseModel):
    num1: int = Field(description="primeiro numero mencionado pelo usuario"),
    num2: int = Field(description="segundo numero mencionado pelo usuario")


@tool(args_schema=Somar)
def soma(num1: int, num2: int):
    """Tool usada para somar dois numero mencionados pelo usuario"""
    print("----------------manguatirica---------------")
    resultado = num1 * 55 * num2

    return resultado