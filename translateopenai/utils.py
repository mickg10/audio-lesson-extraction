import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Optional

@contextmanager
def timer(label: str = "Block of code"):
    """
    Context manager for timing code execution.
    
    Args:
        label: Description of the code block being timed
    """
    start_time = time.time()
    logging.log(logging.INFO, f"{label} starting to execute - measuring timing")
    try:
        yield
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.log(logging.INFO, f"{label} executed in {elapsed_time:.6f} seconds")

def read_openai_key(file_path: str) -> str:
    """
    Read OpenAI API key from a file.
    
    Args:
        file_path: Path to the file containing the API key
        
    Returns:
        API key as a string
    """
    with open(file_path, 'r') as file:
        return file.readline().strip()

def configure_logging(level: int = logging.INFO, format_str: str = '%(asctime)s:%(lineno)d %(message)s') -> None:
    """
    Configure standard logging format for translation tools.
    
    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        format_str: Format string for logging
    """
    logging.basicConfig(level=level, format=format_str)

class TranslationProgressTracker:
    """
    Track and report translation progress.
    """
    def __init__(self, total_items: int, description: str = "Translating"):
        """
        Initialize the progress tracker.
        
        Args:
            total_items: Total number of items to be processed
            description: Description of the translation process
        """
        self.total = total_items
        self.current = 0
        self.description = description
        self.start_time = time.time()
        logging.info(f"{description} {total_items} items")
    
    def update(self, count: int = 1, additional_info: str = "") -> None:
        """
        Update the progress counter and log progress.
        
        Args:
            count: Number of items processed in this update
            additional_info: Additional information to include in the log
        """
        self.current += count
        elapsed = time.time() - self.start_time
        items_per_sec = self.current / elapsed if elapsed > 0 else 0
        percent = (self.current / self.total) * 100 if self.total > 0 else 0
        
        message = f"{self.description} progress: {self.current}/{self.total} ({percent:.1f}%) "
        message += f"at {items_per_sec:.2f} items/sec"
        
        if additional_info:
            message += f" - {additional_info}"
            
        logging.info(message)
    
    def complete(self) -> None:
        """
        Mark the task as complete and log completion statistics.
        """
        elapsed = time.time() - self.start_time
        items_per_sec = self.total / elapsed if elapsed > 0 else 0
        
        logging.info(f"{self.description} completed {self.total} items in {elapsed:.2f} seconds ")
        logging.info(f"Average processing rate: {items_per_sec:.2f} items/sec")
