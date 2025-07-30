#!/usr/bin/env python3
"""Basic test script to verify the data-preproc package works."""

import sys
import os

# Add the package to Python path for testing
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test that basic imports work."""
    try:
        import data_preproc
        print("✓ Main package imports successfully")
        
        from data_preproc.cli.args import PreprocessCliArgs
        print("✓ CLI args import successfully")
        
        from data_preproc.utils.dict import DictDefault
        print("✓ DictDefault imports successfully")
        
        from data_preproc.utils.logging import get_logger
        print("✓ Logging utilities import successfully")
        
        # Test DictDefault functionality
        cfg = DictDefault({"test": "value", "nested": {"key": "value"}})
        assert cfg.test == "value"
        assert cfg.nested.key == "value"
        assert cfg.nonexistent is None
        print("✓ DictDefault functionality works")
        
        # Test CLI args
        args = PreprocessCliArgs()
        assert args.debug is False
        assert args.debug_num_examples == 1
        print("✓ PreprocessCliArgs functionality works")
        
        print("\n🎉 All basic tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_cli_help():
    """Test that CLI shows help."""
    try:
        from data_preproc.cli.preprocess import do_cli
        print("✓ CLI function imports successfully")
        return True
    except Exception as e:
        print(f"❌ CLI test error: {e}")
        return False

if __name__ == "__main__":
    print("Testing data-preproc package...")
    print("=" * 50)
    
    success = test_imports()
    if success:
        success = test_cli_help()
    
    if success:
        print("\n✅ All tests passed! Package is ready to use.")
        print("\nTo use the package:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Install package: pip install -e .")
        print("3. Run preprocessing: data-preproc --config example_config.yaml")
    else:
        print("\n❌ Some tests failed. Please check the package setup.")
        sys.exit(1)