import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Callable, Union

# Export public API
from translateopenai.utils import timer, read_openai_key, configure_logging, TranslationProgressTracker
from translateopenai.models import get_available_models
from translateopenai.translation import translate_text, translate_dataframe, translate_file
from translateopenai.io import read_excel_data, write_data, append_data
from translateopenai.cli import add_translation_arguments, add_model_argument, create_translation_parser

__version__ = "1.0.0"
__all__ = [
    'timer', 'read_openai_key', 'configure_logging', 'TranslationProgressTracker',
    'get_available_models',
    'translate_text', 'translate_dataframe', 'translate_file',
    'read_excel_data', 'write_data', 'append_data',
    'add_translation_arguments', 'add_model_argument', 'create_translation_parser'
]
