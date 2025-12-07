from utils import simple_text_split

def test_simple_text_split_basic():
    text = "one two three four five six seven eight nine ten"
    chunks = simple_text_split(text, chunk_size=4, overlap=1)
    assert len(chunks) == 3
    assert chunks[0].startswith("one")
    assert "four" in chunks[1]
    assert chunks[-1].endswith("ten")
