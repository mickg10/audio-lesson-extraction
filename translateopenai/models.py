import logging
from openai import OpenAI
from typing import List, Optional

from translateopenai.utils import read_openai_key

def get_available_models(api_key_file: str) -> List[str]:
    """Get available OpenAI models with fallback to common models.
    
    Args:
        api_key_file: Path to the OpenAI API key file
        
    Returns:
        List of available GPT model names
    """
    try:
        client = OpenAI(api_key=read_openai_key(api_key_file))
        models = [m for m in [x.id for x in client.models.list().data] if 'gpt' in m]
        if models:
            return models
    except Exception as e:
        logging.warning(f"Could not fetch models from OpenAI API: {str(e)}")
    
    # Fallback to common models
    return ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]
