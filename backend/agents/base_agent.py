"""
Base Agent class
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseAgent(ABC):
    """
    Abstract base class for all agents
    """
    
    def __init__(self, name: str, model_name: str):
        self.name = name
        self.model_name = model_name
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input and return output
        
        Args:
            input_data: Dictionary containing input parameters
            
        Returns:
            Dictionary containing output results
        """
        pass
    
    @abstractmethod
    def validate_output(self, output: Dict[str, Any]) -> bool:
        """
        Validate the output format
        
        Args:
            output: Output dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass