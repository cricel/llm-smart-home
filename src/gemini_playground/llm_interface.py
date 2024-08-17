from abc import ABC, abstractmethod

class LLMInterface(ABC):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def ask(self, _msg: str) -> str:
        """Ask question to LLM"""

    @abstractmethod
    def ask_image(self, _msg: str, _path: str) -> str:
        """Ask question to LLM with image"""