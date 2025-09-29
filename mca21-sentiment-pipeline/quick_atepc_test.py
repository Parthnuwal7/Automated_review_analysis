#!/usr/bin/env python3

"""Quick test of the updated AspectExtractor with proper imports"""

def test_aspect_extractor_imports():
    """Test that AspectExtractor initializes properly with updated imports."""
    print("=== Quick AspectExtractor Import Test ===")
    
    try:
        print("\n1. Testing pyABSA import...")
        try:
            from pyabsa import ATEPCCheckpointManager, TaskCodeOption
            print("✅ pyABSA ATEPC imports successful")
        except ImportError as e:
            print(f"❌ pyABSA import failed: {e}")
            return False
        
        print("\n2. Testing AspectExtractor initialization...")
        from pipeline.aspect_extraction import AspectExtractor
        
        # This should now recognize pyABSA is available
        extractor = AspectExtractor()
        print("✅ AspectExtractor initialized")
        
        print(f"   • pyABSA available: {hasattr(extractor, 'aspect_extractor')}")
        print(f"   • Models loaded: {extractor._available}")
        print(f"   • Has fallback methods: {hasattr(extractor, '_fallback_aspect_extraction')}")
        
        # If pyABSA is loading, we can still test fallback
        print("\n3. Testing fallback extraction...")
        test_text = "The interface is good but performance is bad."
        aspects = extractor._fallback_aspect_extraction(test_text)
        
        print(f"✅ Fallback extraction found {len(aspects)} aspects:")
        for aspect in aspects[:3]:
            print(f"   • {aspect['aspect_label']}: {aspect['aspect_sentiment']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_aspect_extractor_imports()
    if success:
        print("\n🎉 AspectExtractor is ready!")
        print("pyABSA integration prepared - will use ATEPC API when models load")
    else:
        print("\n❌ AspectExtractor setup failed")