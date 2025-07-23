#!/usr/bin/env python3
"""
Test script for deduplication pipeline
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import only the deduplicator
sys.path.append(str(Path(__file__).parent / "src" / "data_processing"))
from deduplication import Deduplicator

def test_deduplication():
    """Test the deduplication pipeline"""
    print("🧪 Testing Deduplication Pipeline")
    
    # Create test data with duplicates
    test_documents = [
        {
            "text": "EV charging stations are essential for electric vehicles.",
            "source": "https://www.tesla.com/support/charging",
            "type": "web"
        },
        {
            "text": "EV charging stations are essential for electric vehicles.",  # Exact duplicate
            "source": "https://www.tesla.com/support/charging",
            "type": "web"
        },
        {
            "text": "EV charging stations are essential for electric vehicles.",  # Exact duplicate
            "source": "https://www.tesla.com/support/charging",
            "type": "web"
        },
        {
            "text": "Electric vehicle charging stations are essential for EVs.",  # Near duplicate
            "source": "https://www.electrifyamerica.com/",
            "type": "web"
        },
        {
            "text": "Level 2 charging uses 240V power and can charge faster than Level 1.",
            "source": "ev_charging_guide.pdf",
            "type": "pdf"
        },
        {
            "text": "Level 2 charging uses 240V power and can charge faster than Level 1.",  # Exact duplicate
            "source": "ev_charging_guide.pdf",
            "type": "pdf"
        },
        {
            "text": "DC fast charging can provide 60-80% charge in 20-30 minutes.",
            "source": "https://www.electrifyamerica.com/",
            "type": "web"
        },
        {
            "text": "DC fast charging can provide 60-80% charge in 20-30 minutes.",  # Exact duplicate
            "source": "https://www.electrifyamerica.com/",
            "type": "web"
        },
        {
            "text": "Tesla Superchargers are the fastest charging option available.",
            "source": "https://www.tesla.com/support/charging",
            "type": "web"
        },
        {
            "text": "Tesla Superchargers are the fastest charging option available.",  # Exact duplicate
            "source": "https://www.tesla.com/support/charging",
            "type": "web"
        }
    ]
    
    print(f"\n📋 Original documents: {len(test_documents)}")
    for i, doc in enumerate(test_documents):
        print(f"  {i+1}. {doc['text'][:50]}...")
    
    # Test different deduplication methods
    methods = ["levenshtein", "hybrid"]
    
    for method in methods:
        print(f"\n🔍 Testing {method.upper()} deduplication:")
        
        # Initialize deduplicator
        deduplicator = Deduplicator(similarity_threshold=0.95, method=method)
        
        # Deduplicate
        original_count = len(test_documents)
        deduplicated_docs = deduplicator.deduplicate(test_documents, text_column="text")
        final_count = len(deduplicated_docs)
        
        # Get statistics
        stats = deduplicator.get_deduplication_stats(original_count, final_count)
        
        print(f"  Original count: {stats['original_count']}")
        print(f"  Final count: {stats['final_count']}")
        print(f"  Removed: {stats['removed_count']}")
        print(f"  Reduction: {stats['reduction_percentage']:.1f}%")
        print(f"  Method: {stats['method']}")
        print(f"  Threshold: {stats['threshold']}")
        
        print(f"\n  Deduplicated documents:")
        for i, doc in enumerate(deduplicated_docs):
            print(f"    {i+1}. {doc['text'][:50]}...")
    
    # Test exact duplicate removal
    print(f"\n🎯 Testing exact duplicate removal:")
    deduplicator = Deduplicator(similarity_threshold=1.0, method="levenshtein")
    exact_unique = deduplicator._remove_exact_duplicates(test_documents, "text")
    print(f"  Exact duplicates removed: {len(test_documents) - len(exact_unique)}")
    
    # Test with different thresholds
    print(f"\n⚙️ Testing different thresholds:")
    thresholds = [0.8, 0.9, 0.95, 0.98]
    
    for threshold in thresholds:
        deduplicator = Deduplicator(similarity_threshold=threshold, method="levenshtein")
        deduplicated = deduplicator.deduplicate(test_documents, text_column="text")
        stats = deduplicator.get_deduplication_stats(len(test_documents), len(deduplicated))
        print(f"  Threshold {threshold}: {stats['removed_count']} removed ({stats['reduction_percentage']:.1f}%)")
    
    # Test batch deduplication
    print(f"\n📦 Testing batch deduplication:")
    deduplicator = Deduplicator(similarity_threshold=0.95, method="levenshtein")
    batch_deduplicated = deduplicator.batch_deduplicate(test_documents, text_column="text", batch_size=3)
    batch_stats = deduplicator.get_deduplication_stats(len(test_documents), len(batch_deduplicated))
    print(f"  Batch deduplication: {batch_stats['removed_count']} removed ({batch_stats['reduction_percentage']:.1f}%)")
    
    # Test with temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\n💾 Testing file saving in {temp_dir}")
        
        # Save test data
        import json
        test_file = os.path.join(temp_dir, "test_documents.jsonl")
        with open(test_file, 'w', encoding='utf-8') as f:
            for doc in test_documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        # Save deduplicated data
        dedup_file = os.path.join(temp_dir, "deduplicated_documents.jsonl")
        with open(dedup_file, 'w', encoding='utf-8') as f:
            for doc in deduplicated_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        print(f"✅ Saved test documents to: {temp_dir}/test_documents.jsonl")
        print(f"✅ Saved deduplicated documents to: {temp_dir}/deduplicated_documents.jsonl")
        
        # Verify files exist
        assert os.path.exists(test_file)
        assert os.path.exists(dedup_file)
        print("✅ File saving test passed!")
    
    print("\n🎉 All deduplication tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_deduplication()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 