#!/usr/bin/env python3
"""
Test script for HuggingFace upload functionality.
"""

import tempfile
import yaml
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from data_preproc.cli.config import load_cfg
from data_preproc.utils.dict import DictDefault

def test_hf_config_validation():
    """Test HuggingFace configuration validation."""
    print("Testing HF configuration validation...")
    
    try:
        from data_preproc.utils.hf_upload import validate_hf_config, should_upload_to_hf
        
        # Test valid configuration
        valid_config = {
            "organization": "my-org",
            "dataset_name": "my-dataset"
        }
        validate_hf_config(valid_config)
        print("✅ Valid HF configuration passed validation")
        
        # Test should_upload_to_hf
        cfg_with_upload = DictDefault({
            "hf_upload": {
                "enabled": True,
                "organization": "test-org",
                "dataset_name": "test-dataset"
            }
        })
        
        assert should_upload_to_hf(cfg_with_upload) == True
        print("✅ should_upload_to_hf correctly detects enabled upload")
        
        cfg_without_upload = DictDefault({
            "hf_upload": {"enabled": False}
        })
        
        assert should_upload_to_hf(cfg_without_upload) == False
        print("✅ should_upload_to_hf correctly detects disabled upload")
        
        return True
        
    except Exception as e:
        print(f"❌ HF configuration validation failed: {e}")
        return False

def test_readme_generation():
    """Test README generation functionality."""
    print("\nTesting README generation...")
    
    try:
        from data_preproc.utils.hf_upload import generate_dataset_readme
        
        config = {
            "base_model": "allenai/Molmo-7B-O-0924",
            "tokenizer_config": "allenai/Molmo-7B-O-0924",
            "sequence_len": 8192
        }
        
        stats = {
            "original_samples": 1000,
            "processed_samples": 950,
            "success_rate": 0.95,
            "avg_length": 250.5,
            "max_length": 512,
            "truncation_rate": 0.02
        }
        
        readme = generate_dataset_readme(
            "test-vl-dataset",
            "A test vision-language dataset",
            config,
            stats
        )
        
        # Check that README contains expected content
        assert "test-vl-dataset" in readme
        assert "allenai/Molmo-7B-O-0924" in readme
        assert "8192" in readme
        assert "1000" in readme  # original samples
        assert "95.0%" in readme  # success rate
        assert "load_dataset" in readme  # usage example
        
        print("✅ README generation successful")
        print(f"   Generated README length: {len(readme)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ README generation failed: {e}")
        return False

def test_config_integration():
    """Test integration with configuration system."""
    print("\nTesting configuration integration...")
    
    try:
        # Create test configuration with HF upload
        test_config = {
            "base_model": "microsoft/DialoGPT-medium",
            "tokenizer_config": "microsoft/DialoGPT-medium",
            "sequence_len": 1024,
            "datasets": [{
                "path": "test_data.json",
                "type": "alpaca"
            }],
            "hf_upload": {
                "enabled": True,
                "organization": "test-org",
                "dataset_name": "test-dataset",
                "private": False,
                "description": "Test dataset for upload functionality",
                "license": "mit",
                "tags": ["test", "demo"],
                "create_readme": True,
                "push_to_hub_kwargs": {
                    "commit_message": "Test upload",
                    "branch": "main"
                }
            }
        }
        
        # Write to temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f, default_flow_style=False)
            config_path = f.name
        
        try:
            # Load configuration
            cfg = load_cfg(config_path)
            
            # Verify HF upload configuration
            assert cfg.hf_upload.enabled == True
            assert cfg.hf_upload.organization == "test-org"
            assert cfg.hf_upload.dataset_name == "test-dataset"
            assert cfg.hf_upload.private == False
            assert cfg.hf_upload.license == "mit"
            assert "test" in cfg.hf_upload.tags
            
            print("✅ Configuration integration successful")
            print(f"   HF organization: {cfg.hf_upload.organization}")
            print(f"   HF dataset name: {cfg.hf_upload.dataset_name}")
            print(f"   HF tags: {cfg.hf_upload.tags}")
            
            return True
            
        finally:
            # Clean up
            Path(config_path).unlink()
        
    except Exception as e:
        print(f"❌ Configuration integration failed: {e}")
        return False

def test_upload_error_handling():
    """Test upload error handling."""
    print("\nTesting upload error handling...")
    
    try:
        from data_preproc.utils.hf_upload import upload_dataset_to_hf, HFUploadError
        
        # Test with non-existent dataset path
        try:
            upload_dataset_to_hf(
                dataset_path=Path("/non/existent/path"),
                organization="test-org",
                dataset_name="test-dataset",
                config={},
                hf_config={},
                stats=None,
                token="fake-token"
            )
            print("❌ Should have failed with non-existent path")
            return False
        except HFUploadError as e:
            if "does not exist" in str(e):
                print("✅ Correctly handles non-existent dataset path")
            else:
                print(f"❌ Unexpected error message: {e}")
                return False
        
        return True
        
    except ImportError:
        print("⚠️  HuggingFace dependencies not available, skipping error handling test")
        return True
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_cli_integration():
    """Test CLI argument integration."""
    print("\nTesting CLI integration...")
    
    try:
        from data_preproc.cli.args import PreprocessCliArgs
        
        # Test CLI args with HF token
        cli_args = PreprocessCliArgs(hf_token="test-token")
        assert cli_args.hf_token == "test-token"
        
        print("✅ CLI integration successful")
        print(f"   HF token field available: {hasattr(cli_args, 'hf_token')}")
        
        return True
        
    except Exception as e:
        print(f"❌ CLI integration failed: {e}")
        return False

def main():
    """Run all HF upload tests."""
    print("=" * 60)
    print("HuggingFace Upload Functionality Tests")
    print("=" * 60)
    
    tests = [
        test_hf_config_validation,
        test_readme_generation, 
        test_config_integration,
        test_upload_error_handling,
        test_cli_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    if all(results):
        print("🎉 All HuggingFace upload tests passed!")
        print("✅ HF upload functionality is ready for use")
        
        print("\n📝 Configuration Example:")
        print("hf_upload:")
        print("  enabled: true")
        print("  organization: 'my-org'")
        print("  dataset_name: 'my-dataset'")
        print("  private: false")
        
        print("\n🔐 Authentication:")
        print("export HF_TOKEN='hf_your_token_here'")
        
    else:
        print("❌ Some HuggingFace upload tests failed")
        failed_tests = [test.__name__ for test, result in zip(tests, results) if not result]
        print(f"Failed tests: {failed_tests}")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)