#!/usr/bin/env python3

"""Simple test to verify pyABSA works with fixed transformers version"""

def test_pyabsa_fix():
    """Test that pyABSA now works without tokenizer errors."""
    print("Testing pyABSA with transformers 4.29.0...")
    
    try:
        from pyabsa import ATEPCCheckpointManager
        print("✅ pyABSA imported successfully")
        
        print("Loading ATEPC extractor...")
        aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(
            checkpoint='multilingual',
            auto_device=True,
            task_code='ATEPC'
        )
        print("✅ ATEPC extractor loaded successfully")
        
        print("Testing aspect extraction...")
        results = aspect_extractor.extract_aspect(
            ["The interface is great but performance is slow"],
            pred_sentiment=True
        )
        print(f"✅ Results: {results}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_pyabsa_fix()
    if success:
        print("\n🎉 pyABSA tokenizer issue FIXED!")
        print("Compatible transformers version resolved the problem.")
    else:
        print("\n❌ Still having issues...")