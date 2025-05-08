#!/usr/bin/env python3

"""Test script to verify argument parsing in server.py"""

import shlex
import re

def test_arg_parsing(text):
    print(f"Testing argument parsing with: '{text}'")
    sanitized_text = re.sub(r'[^a-zA-Z0-9\s\-_\.,:;?!/\'"()\[\]{}]', '', text or "")
    args = shlex.split(sanitized_text)
    
    clean_args = []
    i = 0
    while i < len(args):
        # Parallel upsert workers
        if args[i] == "--parallel" and i + 1 < len(args):
            try:
                parallel_var = int(args[i+1])
                print(f"Parsed --parallel {parallel_var}")
            except ValueError:
                pass
            i += 2
            continue
        # Chunking parameters
        if args[i] == "--chunk-size" and i + 1 < len(args):
            try:
                chunk_size_var = int(args[i+1])
                print(f"Parsed --chunk-size {chunk_size_var}")
            except ValueError:
                pass
            i += 2
        elif args[i] == "--chunk-overlap" and i + 1 < len(args):
            try:
                chunk_overlap_var = int(args[i+1])
                print(f"Parsed --chunk-overlap {chunk_overlap_var}")
            except ValueError:
                pass
            i += 2
        elif args[i] in ("--crawl-depth", "--depth-crawl") and i + 1 < len(args):
            try:
                crawl_depth_var = int(args[i+1])
                print(f"Parsed --crawl-depth {crawl_depth_var}")
            except ValueError:
                pass
            i += 2
        # Purge flag
        elif args[i] == "--purge":
            clean_args.append(args[i])
            print(f"Parsed --purge flag")
            i += 1
        # Summarization flags
        elif args[i] == "--generate-summaries":
            print(f"Parsed --generate-summaries flag")
            i += 1
        elif args[i] == "--no-generate-summaries":
            print(f"Parsed --no-generate-summaries flag")
            i += 1
        # Quality-check flags - handle common typos
        elif args[i] in ("--quality-checks", "--quality-check", "--qualith-checks", "--qualty-checks"):
            print(f"Parsed --quality-checks flag")
            i += 1
        elif args[i] in ("--no-quality-checks", "--no-quality-check"):
            print(f"Parsed --no-quality-checks flag")
            i += 1
        # Rich metadata flags
        elif args[i] == "--rich-metadata":
            clean_args.append(args[i])
            print(f"Parsed --rich-metadata flag")
            i += 1
        elif args[i] == "--no-rich-metadata":
            print(f"Parsed --no-rich-metadata flag")
            i += 1
        # Handle all our new feature flags
        elif args[i] in ("--hierarchical-embeddings", "--no-hierarchical-embeddings", 
                        "--entity-extraction", "--no-entity-extraction",
                        "--enhance-text-with-entities", "--no-enhance-text-with-entities",
                        "--adaptive-chunking", "--no-adaptive-chunking",
                        "--deduplication", "--no-deduplication",
                        "--merge-duplicates", "--no-merge-duplicates",
                        "--validate-ingestion", "--no-validate-ingestion",
                        "--run-test-queries", "--no-run-test-queries"):
            clean_args.append(args[i])
            print(f"Parsed new feature flag: {args[i]}")
            i += 1
        # Handle flags with parameters
        elif args[i] in ("--doc-embedding-model", "--section-embedding-model", 
                         "--chunk-embedding-model", "--similarity-threshold") and i + 1 < len(args):
            clean_args.append(args[i])
            clean_args.append(args[i+1])
            print(f"Parsed param flag: {args[i]} {args[i+1]}")
            i += 2
        else:
            clean_args.append(args[i])
            print(f"Parsed as source: {args[i]}")
            i += 1
    
    print(f"Final clean args: {clean_args}")
    return clean_args

# Test with various combinations
test_cases = [
    "--adaptive-chunking",
    "--adaptive-chunking /path/to/source.txt",
    "--doc-embedding-model text-embedding-3-large --adaptive-chunking",
    "--hierarchical-embeddings /path/to/source.txt --chunk-size 200",
    "--adaptive-chunking --entity-extraction --similarity-threshold 0.9"
]

for test in test_cases:
    test_arg_parsing(test)
    print("-" * 50)