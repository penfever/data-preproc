#!/usr/bin/env python3
"""
Test runner for data-preproc package.
Runs all tests and provides a summary.
"""

import sys
import subprocess
from pathlib import Path

def run_test_file(test_file: Path) -> bool:
    """Run a single test file and return success status."""
    print(f"\n{'='*60}")
    print(f"Running {test_file.name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {test_file.name} PASSED")
            return True
        else:
            print(f"❌ {test_file.name} FAILED (exit code: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ {test_file.name} FAILED (timeout)")
        return False
    except Exception as e:
        print(f"❌ {test_file.name} FAILED (exception: {e})")
        return False

def main():
    """Run all tests."""
    print("Data-Preproc Test Suite")
    print("=" * 60)
    
    tests_dir = Path(__file__).parent / "tests"
    test_files = list(tests_dir.glob("test_*.py"))
    
    if not test_files:
        print("No test files found in tests/ directory")
        return False
    
    print(f"Found {len(test_files)} test files:")
    for test_file in test_files:
        print(f"  - {test_file.name}")
    
    results = []
    for test_file in test_files:
        success = run_test_file(test_file)
        results.append((test_file.name, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<30} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)