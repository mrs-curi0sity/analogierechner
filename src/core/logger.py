import csv
import threading
from datetime import datetime
import os

class SafeCSVLogger:
    def __init__(self, filename):
        self.filename = filename
        self.lock = threading.Lock()
        
        # Create file if it doesn't exist
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

# Global logger instance
logger = SafeCSVLogger('logs/analogies_log.csv')