from scripts.data_processing import deduplicate

def test_deduplication():
    docs = [{"content": "test"}, {"content": "test"}]
    assert len(deduplicate(docs)) == 1 