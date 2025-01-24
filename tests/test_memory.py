import os
import psutil
import pytest
from pathlib import Path

from openhands_aci.editor.editor import OHEditor


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


@pytest.fixture
def large_file(tmp_path):
    # Create a 10MB file with repeating pattern
    file_path = tmp_path / "large_file.txt"
    pattern = "abcdefghij" * 100 + "\n"  # 1KB per line
    with file_path.open('w') as f:
        for _ in range(10000):  # 10000 lines = ~10MB
            f.write(pattern)
    return file_path


def test_memory_usage_view(large_file):
    editor = OHEditor()
    
    # Force garbage collection to get accurate memory measurements
    import gc
    gc.collect()
    initial_memory = get_memory_usage()
    
    # View entire file
    editor(command="view", path=str(large_file))
    
    # Measure memory after viewing
    peak_memory = get_memory_usage()
    memory_increase = peak_memory - initial_memory
    
    # Memory increase should be reasonable for streaming read
    assert memory_increase < 15, f"Memory increase ({memory_increase:.2f}MB) too high for streaming read"


def test_memory_usage_view_range(large_file):
    editor = OHEditor()
    
    # Force garbage collection
    import gc
    gc.collect()
    initial_memory = get_memory_usage()
    
    # View only first 10 lines
    editor(command="view", path=str(large_file), view_range=[1, 10])
    
    # Measure memory after viewing
    peak_memory = get_memory_usage()
    memory_increase = peak_memory - initial_memory
    
    # Memory increase should be minimal when viewing specific lines
    assert memory_increase < 2, f"Memory increase ({memory_increase:.2f}MB) too high for viewing specific lines"


def test_memory_usage_str_replace(large_file):
    editor = OHEditor()
    
    # Create a unique string to replace (near the middle of the file)
    unique_str = "UNIQUE_MARKER_FOR_REPLACEMENT_TEST_" + "x" * 100
    line_num = 5000  # Middle of file
    
    # First insert the unique string
    content = large_file.read_text()
    pos = content.find('\n', line_num * 1024)  # Find next newline after our target position
    if pos == -1:
        pos = len(content)
    new_content = content[:pos] + unique_str + content[pos:]
    large_file.write_text(new_content)
    
    # Force garbage collection
    import gc
    gc.collect()
    initial_memory = get_memory_usage()
    
    # Replace the unique string
    editor(command="str_replace", path=str(large_file), old_str=unique_str, new_str="y" * len(unique_str))
    
    # Measure memory after replacement
    peak_memory = get_memory_usage()
    memory_increase = peak_memory - initial_memory
    
    # Memory increase should be reasonable for streaming replacement
    assert memory_increase < 15, f"Memory increase ({memory_increase:.2f}MB) too high for streaming replacement"