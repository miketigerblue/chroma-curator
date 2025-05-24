"""
sample_vectors.py

This utility reads your large export_for_edge.json vector file and writes a 
smaller test_vectors.json file (default: 20 randomly chosen records) for use 
in unit tests and CI. 

Usage:
    python sample_vectors.py

Author: Mike Harris (mike.harris@tigerblue.tech)
This file is part of the Chroma Curator project, which profiles and exports
curated vector datasets for on-device ML/AI applications.
Date: 2023-10-01

"""

import json
import random
import os

# Number of sample records to include in the test file
SAMPLE_SIZE = 20

# Input file: The full exported dataset (large)
INPUT_FILE = 'export_for_edge.json'

# Output file: The smaller, test-friendly file (for CI)
OUTPUT_FILE = 'tests/test_vectors.json'

def main():
    # Ensure the output directory exists
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the full export file (can be large!)
    print(f"Loading data from {INPUT_FILE} ...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
    # Make sure there are enough records to sample
    if len(data) < SAMPLE_SIZE:
        raise ValueError(f"Not enough records in {INPUT_FILE} to sample {SAMPLE_SIZE}.")
    
    # Randomly select SAMPLE_SIZE unique records
    sample = random.sample(data, SAMPLE_SIZE)
    print(f"Sampled {SAMPLE_SIZE} records for test/CI usage.")
    
    # Write the sample to the output file (pretty JSON)
    with open(OUTPUT_FILE, 'w') as out:
        json.dump(sample, out, indent=2)
    print(f"Wrote sample data to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
