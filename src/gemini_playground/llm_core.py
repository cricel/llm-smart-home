from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.pydantic_v1 import BaseModel, Field


class DimLights(BaseModel):
    """
    Play some music matching the specified parameters.
    """

    energetic: bool = Field(..., description="Whether the music is energetic or not")
    loud: bool = Field(..., description="Whether the music is loud or not")
    bpm: int = Field(..., description="The beats per minute of the music")


class StartMusic(BaseModel):
    """
    Dim the lights.
    """

    brightness: float = Field(..., description="The brightness of the lights, 0.0 is off, 1.0 is full.")


class PowerDiscoBall(BaseModel):
    """
    Powers the spinning disco ball.
    """

    brightness: bool = Field(..., description="The brightness of the disco ball, 0.0 is off, 1.0 is full.")


class GetDummyData(BaseModel):
    """
    Getting Dummy Data.
    """

    text: str = Field(..., description="dummy text as input")


load_dotenv()

tools = [DimLights, StartMusic, PowerDiscoBall, GetDummyData]

llm_list = [
    {"name": "ollama", "llm_obj": Ollama(base_url="http://131.123.41.132:11434", model="llama3:8b")},
    {"name": "openai", "llm_obj": ChatOpenAI(model="gpt-3.5-turbo-0125")}
    
]
questions = [
    "how are you",
    "What you are up to?",
    "let`s get the party started",
    "turn the music on and turn off the light",
    "turn the light on",
    "turn on everything"
]

for llm in llm_list:
    llm_with_tools = llm["llm_obj"].bind_tools(tools)
    count = 0
    print("==========")
    print(f"LLM: {llm['name']}")
    print("==========")
    for question in questions:
        print(count, question, "||=> \n\n", llm_with_tools.invoke(question).tool_calls)
        print("--------------------------------------------------")
        count += 1