import csv
import threading
import logging
from datetime import datetime
import os

class SafeCSVLogger:
    def __init__(self, filename):
        self.filename = filename
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Setup file logging
        handler = logging.FileHandler('logs/analogierechner.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Create CSV if it doesn't exist
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        with open(self.filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if f.tell() == 0:  # If file is empty
                writer.writerow(['timestamp', 'source', 'language', 'word1', 'word2', 'word3', 'output'])

    def log(self, source, language, word1, word2, word3, output):
        with self.lock:  # Thread-safe logging
            with open(self.filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    source,
                    language,
                    word1,
                    word2,
                    word3,
                    output
                ])

    # Standard logging methods
    def info(self, msg):
        self.logger.info(msg)

    def error(self, msg):
        self.logger.error(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def debug(self, msg):
        self.logger.debug(msg)

# Global logger instance
logger = SafeCSVLogger('logs/analogies_log.csv')