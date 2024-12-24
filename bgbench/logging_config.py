import logging
import sys
from pathlib import Path

def setup_logging(debug=False):
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Set the root logger to WARNING to suppress verbose logs from dependencies
    logging.getLogger().setLevel(logging.WARNING)
    
    # Set up our application logger
    bgbench_logger = logging.getLogger("bgbench")
    bgbench_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # Create formatters
    console_formatter = logging.Formatter('%(message)s')  # Remove prefix for console
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler - less verbose
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    # File handler - more detailed
    file_handler = logging.FileHandler(logs_dir / "bgbench.log")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # Add handlers to our logger
    bgbench_logger.addHandler(console_handler)
    bgbench_logger.addHandler(file_handler)
