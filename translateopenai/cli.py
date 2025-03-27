import argparse
import logging
import os
from typing import List, Optional, Tuple

from translateopenai.models import get_available_models

def add_translation_arguments(parser: argparse.ArgumentParser, default_api_key: str = "../aikeys/openai.key") -> argparse.ArgumentParser:
    """
    Add standard translation arguments to an ArgumentParser instance.
    
    Args:
        parser: ArgumentParser instance to add arguments to
        default_api_key: Default path to the OpenAI API key file
        
    Returns:
        Modified ArgumentParser with translation arguments added
    """
    parser.add_argument('--language', default='english', help="Target language")
    parser.add_argument('--context', type=int, default=40, help="How many lines to process together")
    parser.add_argument('--api_key_file', default=default_api_key, help="OPENAI_API_KEY file")
    
    return parser

def add_model_argument(parser: argparse.ArgumentParser, api_key_file: str, default_model: str = "gpt-3.5-turbo") -> argparse.ArgumentParser:
    """
    Add model selection argument with available models as choices.
    
    Args:
        parser: ArgumentParser instance to add arguments to
        api_key_file: Path to the OpenAI API key file
        default_model: Default model to use if not specified
        
    Returns:
        Modified ArgumentParser with model argument added
    """
    models = get_available_models(api_key_file)
    parser.add_argument('--model', default=default_model, choices=models,
                      help=f"What OpenAI model to use for translation")
    return parser

def create_translation_parser(description: str = "Translate text using OpenAI models", 
                             default_api_key: str = "../aikeys/openai.key") -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    """
    Create a pre-configured ArgumentParser for translation tools.
    
    Args:
        description: Description for the ArgumentParser
        default_api_key: Default path to the OpenAI API key file
        
    Returns:
        Tuple of (configured ArgumentParser, pre-parsed args with log level and API key)
    """
    # First parse just the API key file and log level to initialize logging
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--log_level", default=logging.INFO, type=lambda x: getattr(logging, x))
    pre_parser.add_argument('--api_key_file', default=default_api_key)
    temp_args, _ = pre_parser.parse_known_args()
    
    # Create main parser
    parser = argparse.ArgumentParser(description=description,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # Add common arguments
    parser.add_argument("--log_level", default=logging.INFO, type=lambda x: getattr(logging, x),
                      help=f"Configure the logging level: {list(logging._nameToLevel.keys())}")
    parser.add_argument('--api_key_file', default=default_api_key, help="OPENAI_API_KEY file")
    
    return parser, temp_args
