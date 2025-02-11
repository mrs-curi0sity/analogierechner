import pandas as pd
import requests
import time
import os
from typing import List, Dict
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnalogyProcessor:
    def __init__(self, api_url: str = "http://localhost:8081"):
        self.api_url = api_url
        self.batch_endpoint = f"{api_url}/batch-analogy"
        self.single_endpoint = f"{api_url}/analogy"
        
        # Configure session with retry strategy
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504]
        )
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        
        # Error result patterns that should be retried
        self.error_patterns = [
            "no_result_found",
            "not_found",
            "connection_error",
            "timeout_error",
            "api_error",
            None,
            ""
        ]
        
        # Verify API connection
        self._check_api_connection()

    def _check_api_connection(self) -> None:
        """Verify API connection before processing with multiple retries"""
        max_retries = 3
        base_timeout = 60  # Erhöht auf 60 Sekunden für das initiale Laden
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to connect to API (attempt {attempt + 1}/{max_retries})")
                test_payload = {
                    "word1": "könig",  # Using a known word from the vocabulary
                    "word2": "königin",
                    "word3": "prinz",
                    "language": "de"
                }
                # Increase timeout with each retry
                timeout = base_timeout * (attempt + 1)
                logger.info(f"Using timeout of {timeout} seconds")
                
                response = self.session.post(self.single_endpoint, json=test_payload, timeout=timeout)
                if response.status_code == 200:
                    logger.info(f"Successfully connected to API at {self.api_url}")
                    return
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before next attempt")
                    time.sleep(wait_time)
                continue
                
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Waiting {wait_time} seconds before next attempt")
                    time.sleep(wait_time)
                continue
                
            except Exception as e:
                logger.error(f"Unexpected error during connection check: {str(e)}")
                if attempt < max_retries - 1:
                    continue
                raise
        
        # If we get here, all retries failed
        logger.error(f"Could not connect to API at {self.api_url} after {max_retries} attempts")
        logger.error("Please ensure:")
        logger.error("1. Docker container is running")
        logger.error("2. Port 8081 is correctly mapped")
        logger.error("3. API service is running inside the container")
        logger.error("4. Try checking the container logs: docker logs analogierechner-analogierechner-1")
        raise ConnectionError(f"Failed to connect to API at {self.api_url}")

    def should_process_row(self, row: pd.Series) -> bool:
        """Determine if a row needs processing based on its current result"""
        if 'result' not in row or pd.isna(row['result']):
            return True
        return any(error in str(row['result']).lower() for error in self.error_patterns)

    def process_single_analogy(self, row: Dict) -> str:
        """Process a single analogy request with error handling"""
        try:
            payload = {
                "word1": row["word1"],
                "word2": row["word2"],
                "word3": row["word3"],
                "language": row["language"]
            }
            
            response = self.session.post(self.single_endpoint, json=payload, timeout=10)
            response.raise_for_status()
            
            result = response.json().get("results")
            if result is None:
                return "no_result_found"
            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {row}: {str(e)}")
            error_msg = str(e).lower()
            
            if "not found in vocabulary" in error_msg:
                return f"{row['word1']}_not_found"
            elif "connection" in error_msg:
                return "connection_error"
            elif "timeout" in error_msg:
                return "timeout_error"
            return "api_error"

    def process_csv(self, input_path: str, output_path: str, batch_size: int = 3) -> None:
        """Process the entire CSV file with batch processing"""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Load existing results if available
            if os.path.exists(output_path):
                df = pd.read_csv(output_path)
                logger.info(f"Loaded existing results from {output_path}")
            else:
                df = pd.read_csv(input_path)
                logger.info(f"Started new processing with {len(df)} rows from {input_path}")

            # Count rows that need processing
            rows_to_process = df[df.apply(self.should_process_row, axis=1)]
            logger.info(f"Found {len(rows_to_process)} rows that need processing")
            
            if len(rows_to_process) == 0:
                logger.info("No rows need processing. All results are valid.")
                return

            successful_batches = 0
            failed_batches = 0
            processed_count = 0

            # Process in batches, but only for rows that need it
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i + batch_size]
                needs_processing = batch[batch.apply(self.should_process_row, axis=1)]
                
                if len(needs_processing) == 0:
                    continue
                
                logger.info(f"Processing batch with {len(needs_processing)} items")
                
                batch_requests = [
                    {
                        "word1": row["word1"],
                        "word2": row["word2"],
                        "word3": row["word3"],
                        "language": row["language"]
                    }
                    for _, row in needs_processing.iterrows()
                ]

                try:
                    response = self.session.post(
                        self.batch_endpoint, 
                        json=batch_requests,
                        timeout=30
                    )
                    response.raise_for_status()
                    results = response.json().get("results", [])
                    
                    # Update results in dataframe
                    for idx, result in zip(needs_processing.index, results):
                        df.loc[idx, 'result'] = result if result else "no_result_found"
                    
                    successful_batches += 1
                    processed_count += len(needs_processing)

                except requests.exceptions.RequestException as e:
                    logger.error(f"Batch processing failed, falling back to individual processing: {str(e)}")
                    failed_batches += 1
                    
                    # Fallback to individual processing for this batch
                    for idx, row in needs_processing.iterrows():
                        result = self.process_single_analogy(row)
                        df.loc[idx, 'result'] = result
                        processed_count += 1

                # Save intermediate results after each batch
                df.to_csv(output_path, index=False)
                logger.info(f"Saved intermediate results to {output_path}")
                
                # Add small delay to prevent overwhelming the API
                time.sleep(0.2)

            # Final save and summary
            df.to_csv(output_path, index=False)
            logger.info(f"Processing summary:")
            logger.info(f"- Successful batch requests: {successful_batches}")
            logger.info(f"- Failed batch requests: {failed_batches}")
            logger.info(f"- Total rows processed this run: {processed_count}")
            logger.info(f"- Total rows in dataset: {len(df)}")

        except Exception as e:
            logger.error(f"Error processing CSV: {str(e)}")
            raise

def main():
    # Configuration
    BASE_PATH = "../../data"  # Base path for data files
    INPUT_PATH = os.path.join(BASE_PATH, "analogies.csv")
    OUTPUT_PATH = os.path.join(BASE_PATH, "analogies_results.csv")
    API_URL = "http://localhost:8081"
    
    try:
        processor = AnalogyProcessor(API_URL)
        processor.process_csv(INPUT_PATH, OUTPUT_PATH)
        logger.info("Processing completed successfully")
    except ConnectionError:
        logger.error("Failed to connect to API. Please check your Docker setup.")
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")

if __name__ == "__main__":
    main()