"""Unit tests for tool call cache behavior."""
import tempfile
import os
import json
from pathlib import Path

from errol import DIM, RESET, GREEN, RED


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name):
        self.passed += 1
        print(f"  {GREEN}✓{RESET} {DIM}{name}{RESET}")

    def fail(self, name, msg):
        self.failed += 1
        self.errors.append((name, msg))
        print(f"  {RED}✗{RESET} {name} {DIM}- {msg}{RESET}")


def make_cache_key(name: str, args: dict) -> tuple:
    """Create a cache key the same way agent_loop does."""
    try:
        return (name, json.dumps(args, sort_keys=True))
    except:
        return (name, str(args))


def invalidate_cache_for_path(cache: dict, edited_path: str) -> dict:
    """Invalidate cached reads for a path, same logic as agent_loop."""
    edited_path = str(Path(edited_path).expanduser().resolve())
    for key in list(cache.keys()):
        if key[0] == "read_file":
            try:
                cached_args = json.loads(key[1])
                cached_path = str(Path(cached_args.get("path", "")).expanduser().resolve())
                if cached_path == edited_path:
                    del cache[key]
            except:
                pass
    return cache


def test_cache_key_consistency(r: TestResults):
    """Cache keys should be consistent for same args."""
    args1 = {"path": "/tmp/test.py", "offset": 0, "limit": 100}
    args2 = {"path": "/tmp/test.py", "offset": 0, "limit": 100}

    key1 = make_cache_key("read_file", args1)
    key2 = make_cache_key("read_file", args2)

    if key1 == key2:
        r.ok("cache_key_consistency")
    else:
        r.fail("cache_key_consistency", f"Keys differ: {key1} vs {key2}")


def test_cache_key_different_args(r: TestResults):
    """Cache keys should differ for different args."""
    args1 = {"path": "/tmp/test.py", "offset": 0}
    args2 = {"path": "/tmp/test.py", "offset": 10}

    key1 = make_cache_key("read_file", args1)
    key2 = make_cache_key("read_file", args2)

    if key1 != key2:
        r.ok("cache_key_different_args")
    else:
        r.fail("cache_key_different_args", f"Keys should differ: {key1}")


def test_cache_key_sorted(r: TestResults):
    """Cache keys should be same regardless of arg order."""
    args1 = {"path": "/tmp/test.py", "offset": 0, "limit": 100}
    args2 = {"limit": 100, "path": "/tmp/test.py", "offset": 0}

    key1 = make_cache_key("read_file", args1)
    key2 = make_cache_key("read_file", args2)

    if key1 == key2:
        r.ok("cache_key_sorted")
    else:
        r.fail("cache_key_sorted", f"Keys should match: {key1} vs {key2}")


def test_cache_invalidation_exact_path(r: TestResults):
    """Cache should be invalidated for exact path match."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("test content")
        path = f.name

    try:
        cache = {}
        cache_key = make_cache_key("read_file", {"path": path})
        cache[cache_key] = "cached content"

        # Invalidate for this path
        invalidate_cache_for_path(cache, path)

        if cache_key not in cache:
            r.ok("cache_invalidation_exact_path")
        else:
            r.fail("cache_invalidation_exact_path", "Cache entry should be removed")
    finally:
        os.unlink(path)


def test_cache_invalidation_preserves_other_files(r: TestResults):
    """Cache invalidation should preserve entries for other files."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f1:
        f1.write("content 1")
        path1 = f1.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f2:
        f2.write("content 2")
        path2 = f2.name

    try:
        cache = {}
        key1 = make_cache_key("read_file", {"path": path1})
        key2 = make_cache_key("read_file", {"path": path2})
        cache[key1] = "cached 1"
        cache[key2] = "cached 2"

        # Invalidate only path1
        invalidate_cache_for_path(cache, path1)

        if key1 not in cache and key2 in cache:
            r.ok("cache_invalidation_preserves_other_files")
        else:
            r.fail("cache_invalidation_preserves_other_files",
                   f"key1 in cache: {key1 in cache}, key2 in cache: {key2 in cache}")
    finally:
        os.unlink(path1)
        os.unlink(path2)


def test_cache_invalidation_relative_vs_absolute(r: TestResults):
    """Cache invalidation should work with relative vs absolute paths."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='.') as f:
        f.write("content")
        abs_path = os.path.abspath(f.name)
        rel_path = os.path.basename(f.name)

    try:
        cache = {}
        # Cache with relative path
        key = make_cache_key("read_file", {"path": rel_path})
        cache[key] = "cached content"

        # Invalidate with absolute path
        invalidate_cache_for_path(cache, abs_path)

        if key not in cache:
            r.ok("cache_invalidation_relative_vs_absolute")
        else:
            r.fail("cache_invalidation_relative_vs_absolute", "Should invalidate relative path entry")
    finally:
        os.unlink(abs_path)


def test_cache_invalidation_ignores_glob_grep(r: TestResults):
    """Cache invalidation should only affect read_file entries, not glob/grep."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("content")
        path = f.name

    try:
        cache = {}
        read_key = make_cache_key("read_file", {"path": path})
        glob_key = make_cache_key("glob", {"pattern": "*.py", "path": os.path.dirname(path)})
        grep_key = make_cache_key("grep", {"pattern": "test", "path": path})

        cache[read_key] = "read cached"
        cache[glob_key] = "glob cached"
        cache[grep_key] = "grep cached"

        # Invalidate for this path
        invalidate_cache_for_path(cache, path)

        # Only read_file should be removed
        if read_key not in cache and glob_key in cache and grep_key in cache:
            r.ok("cache_invalidation_ignores_glob_grep")
        else:
            r.fail("cache_invalidation_ignores_glob_grep",
                   f"read: {read_key in cache}, glob: {glob_key in cache}, grep: {grep_key in cache}")
    finally:
        os.unlink(path)


def test_cache_invalidation_multiple_offsets(r: TestResults):
    """Cache invalidation should remove all cached reads of a file regardless of offset/limit."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("line1\nline2\nline3\n")
        path = f.name

    try:
        cache = {}
        key1 = make_cache_key("read_file", {"path": path, "offset": 0, "limit": 100})
        key2 = make_cache_key("read_file", {"path": path, "offset": 10, "limit": 50})
        key3 = make_cache_key("read_file", {"path": path})  # no offset/limit

        cache[key1] = "cached 1"
        cache[key2] = "cached 2"
        cache[key3] = "cached 3"

        # Invalidate for this path
        invalidate_cache_for_path(cache, path)

        # All should be removed
        if key1 not in cache and key2 not in cache and key3 not in cache:
            r.ok("cache_invalidation_multiple_offsets")
        else:
            r.fail("cache_invalidation_multiple_offsets",
                   f"key1: {key1 in cache}, key2: {key2 in cache}, key3: {key3 in cache}")
    finally:
        os.unlink(path)


def test_cache_is_local_to_agent_loop(r: TestResults):
    """Verify cache is initialized fresh in agent_loop (check source)."""
    import inspect
    from errol import agent_loop

    source = inspect.getsource(agent_loop)

    # Check that tool_call_cache is initialized as empty dict inside the function
    if "tool_call_cache = {}" in source:
        r.ok("cache_is_local_to_agent_loop")
    else:
        r.fail("cache_is_local_to_agent_loop", "tool_call_cache should be initialized as {} in agent_loop")


def test_cache_hit_returns_cached_content(r: TestResults):
    """Simulate cache hit behavior."""
    cache = {}
    args = {"path": "/tmp/test.py", "offset": 0}
    key = make_cache_key("read_file", args)

    # First call - cache miss
    cached_result = cache.get(key)
    if cached_result is None:
        # Simulate execution and caching
        result = "file content here"
        cache[key] = result

        # Second call - cache hit
        cached_result = cache.get(key)
        if cached_result == result:
            r.ok("cache_hit_returns_cached_content")
        else:
            r.fail("cache_hit_returns_cached_content", f"Expected cached result, got: {cached_result}")
    else:
        r.fail("cache_hit_returns_cached_content", "Should not have cached result initially")


def run_all_tests() -> TestResults:
    """Run all cache tests and return results."""
    r = TestResults()

    print("\nCache key tests:")
    test_cache_key_consistency(r)
    test_cache_key_different_args(r)
    test_cache_key_sorted(r)

    print("\nCache invalidation tests:")
    test_cache_invalidation_exact_path(r)
    test_cache_invalidation_preserves_other_files(r)
    test_cache_invalidation_relative_vs_absolute(r)
    test_cache_invalidation_ignores_glob_grep(r)
    test_cache_invalidation_multiple_offsets(r)

    print("\nCache behavior tests:")
    test_cache_is_local_to_agent_loop(r)
    test_cache_hit_returns_cached_content(r)

    return r


if __name__ == "__main__":
    results = run_all_tests()
    print(f"\n{results.passed} passed, {results.failed} failed")
    if results.failed > 0:
        print("\nFailures:")
        for name, msg in results.errors:
            print(f"  {name}: {msg}")
        exit(1)
