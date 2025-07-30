#!/usr/bin/env python3
"""
Test script to verify VL configuration integration works properly.
"""

import tempfile
import yaml
from pathlib import Path

from data_preproc.cli.config import load_cfg
from data_preproc.utils.dict import DictDefault

def test_vl_config_loading():
    """Test that VL configuration is properly loaded from YAML."""
    
    # Create a test VL configuration
    test_config = {
        "base_model": "allenai/Molmo-7B-O-0924",
        "tokenizer_config": "allenai/Molmo-7B-O-0924",
        "trust_remote_code": True,
        "sequence_len": 8192,
        "datasets": [
            {
                "path": "lmms-lab/multimodal-open-r1-8k-verified",
                "type": "vision_language",
                "split": "train[:100]",
                "mm_plugin": "base",
                "image_token": "<image>",
                "video_token": "<video>",
                "audio_token": "<audio>",
                "filter_corrupted_images": True,
                "max_image_size": [1024, 1024]
            }
        ],
        "vl_config": {
            "image_processing": {
                "verify_images": True,
                "convert_to_rgb": True,
                "max_size": [1024, 1024],
                "min_size": [32, 32]
            },
            "video_processing": {
                "max_frames": 16,
                "frame_sampling": "uniform",
                "target_fps": 1
            },
            "audio_processing": {
                "max_duration": 30,
                "sample_rate": 16000,
                "format": "mel_spectrogram"
            }
        },
        "memory_config": {
            "use_streaming": False,
            "prefetch_factor": 2,
            "num_workers": 4
        },
        "special_tokens": {
            "additional_tokens": ["<image>", "<video>", "<audio>"]
        }
    }
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f, default_flow_style=False)
        temp_config_path = f.name
    
    try:
        # Load configuration
        cfg = load_cfg(temp_config_path)
        
        # Verify basic settings
        assert cfg.base_model == "allenai/Molmo-7B-O-0924"
        assert cfg.sequence_len == 8192
        assert cfg.trust_remote_code == True
        
        # Verify dataset configuration
        assert len(cfg.datasets) == 1
        ds = cfg.datasets[0]
        assert ds["type"] == "vision_language"
        assert ds["mm_plugin"] == "base"
        assert ds["image_token"] == "<image>"
        assert ds["filter_corrupted_images"] == True
        assert ds["max_image_size"] == [1024, 1024]
        
        # Verify VL configuration
        vl_config = cfg.vl_config
        assert vl_config["image_processing"]["verify_images"] == True
        assert vl_config["image_processing"]["max_size"] == [1024, 1024]
        assert vl_config["video_processing"]["max_frames"] == 16
        assert vl_config["audio_processing"]["sample_rate"] == 16000
        
        # Verify memory configuration
        mem_config = cfg.memory_config
        assert mem_config["use_streaming"] == False
        assert mem_config["num_workers"] == 4
        
        # Verify special tokens
        assert "<image>" in cfg.special_tokens["additional_tokens"]
        
        print("✅ All VL configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ VL configuration test failed: {e}")
        return False
    finally:
        # Clean up
        Path(temp_config_path).unlink()

def test_mm_plugin_config():
    """Test MM plugin configuration options."""
    try:
        from data_preproc import get_mm_plugin
        
        # Test base plugin configuration
        base_plugin = get_mm_plugin(
            "base",
            image_token="<img>",
            video_token="<vid>", 
            audio_token="<aud>"
        )
        
        assert base_plugin.image_token == "<img>"
        assert base_plugin.video_token == "<vid>"
        assert base_plugin.audio_token == "<aud>"
        
        # Test different plugin types
        llava_plugin = get_mm_plugin("llava", image_token="<image>")
        assert llava_plugin.image_token == "<image>"
        
        print("✅ MM plugin configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ MM plugin test failed: {e}")
        return False

def test_vision_language_strategy_config():
    """Test that vision language strategy respects configuration."""
    try:
        from data_preproc.prompt_strategies.vision_language import load
        from transformers import AutoTokenizer
        
        # Mock tokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        
        # Create configuration
        cfg = DictDefault({
            "sequence_len": 2048,
            "vl_config": {
                "image_processing": {
                    "verify_images": True,
                    "max_size": [512, 512]
                }
            }
        })
        
        # Dataset configuration
        ds_cfg = {
            "mm_plugin": "base",
            "image_token": "<image>",
            "filter_corrupted_images": True,
            "max_image_size": [1024, 1024]
        }
        
        # Load strategy
        strategy = load(tokenizer, cfg, ds_cfg)
        
        # Verify strategy was created
        assert callable(strategy)
        
        print("✅ Vision language strategy configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Vision language strategy test failed: {e}")
        return False

def main():
    """Run all configuration tests."""
    print("Testing VL configuration integration...")
    print("=" * 50)
    
    results = []
    
    print("\n1. Testing VL configuration loading...")
    results.append(test_vl_config_loading())
    
    print("\n2. Testing MM plugin configuration...")
    results.append(test_mm_plugin_config())
    
    print("\n3. Testing vision language strategy configuration...")
    results.append(test_vision_language_strategy_config())
    
    print("\n" + "=" * 50)
    if all(results):
        print("🎉 All VL configuration tests passed!")
        print("✅ VL features are fully configurable from YAML")
        return True
    else:
        print("❌ Some VL configuration tests failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)