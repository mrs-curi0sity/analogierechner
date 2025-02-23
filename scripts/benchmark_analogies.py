# scripts/benchmark_analogies.py

import sys
import os
from pathlib import Path
import time
import numpy as np
from typing import List, Tuple

# Projektroot zum Python-Pfad hinzufügen
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.core.embedding_handler import EmbeddingHandler
from src.core.logger import logger

def print_separator():
    print("\n" + "="*50 + "\n")

def run_single_test(handler, method, words):
    """Führt einen einzelnen Test durch und gibt Ergebnisse aus"""
    start_time = time.time()
    
    if method == 'original':
        results, _, debug_info = handler.find_analogy(*words)
    else:  # vectorized
        results, _, debug_info = handler.find_analogy_vectorized(*words)
    
    duration = time.time() - start_time
    
    print(f"\nTest case: {words}")
    print(f"Duration: {duration:.3f} seconds")
    print("Top 5 results:")
    if results:
        for word, score in results[:5]:
            print(f"  {word}: {score:.4f}")
    else:
        print("  No results found")
    
    return duration, results

def run_benchmark():
    """
    Führt Benchmark-Tests für beide Implementierungen durch
    """
    print_separator()
    print("Starting Analogy Benchmark")
    print_separator()
    
    # Test-Fälle
    test_cases = [
        ("könig", "königin", "mann"),
        ("hund", "katze", "groß"),
        ("berlin", "deutschland", "paris"),
        ("computer", "programm", "buch"),
        ("sonne", "warm", "eis")
    ]
    
    # Handler initialisieren
    print("Initializing EmbeddingHandler...")
    handler = EmbeddingHandler(language='de')
    print("Handler initialized successfully.")
    print_separator()
    
    # Original Version testen
    print("Testing original implementation:")
    orig_times = []
    for test_case in test_cases:
        duration, _ = run_single_test(handler, 'original', test_case)
        orig_times.append(duration)
    
    print(f"\nOriginal implementation statistics:")
    print(f"Average time: {np.mean(orig_times):.3f} seconds")
    print(f"Min time: {np.min(orig_times):.3f} seconds")
    print(f"Max time: {np.max(orig_times):.3f} seconds")
    print_separator()
    
    # Vektorisierte Version testen
    print("Testing vectorized implementation:")
    vec_times = []
    for test_case in test_cases:
        duration, _ = run_single_test(handler, 'vectorized', test_case)
        vec_times.append(duration)
    
    print(f"\nVectorized implementation statistics:")
    print(f"Average time: {np.mean(vec_times):.3f} seconds")
    print(f"Min time: {np.min(vec_times):.3f} seconds")
    print(f"Max time: {np.max(vec_times):.3f} seconds")
    print_separator()
    
    # Vergleich
    speedup = np.mean(orig_times) / np.mean(vec_times)
    print(f"Performance comparison:")
    print(f"Average speedup: {speedup:.2f}x")
    print(f"Original implementation average: {np.mean(orig_times):.3f} seconds")
    print(f"Vectorized implementation average: {np.mean(vec_times):.3f} seconds")
    print_separator()

if __name__ == "__main__":
    try:
        run_benchmark()
    except Exception as e:
        print(f"Error during benchmark: {str(e)}")
        raise