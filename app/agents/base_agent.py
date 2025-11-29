from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Abstract base class for all RL agents"""
    
    @abstractmethod
    def act(self, state) -> Any:
        """
        Choose an action given the current state
        
        Args:
            state: Current environment state
            
        Returns:
            Action to take
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str):
        """
        Save the agent's model/parameters
        
        Args:
            filepath: Path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, filepath: str):
        """
        Load the agent's model/parameters
        
        Args:
            filepath: Path to load the model from
        """
        pass