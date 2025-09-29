#!/usr/bin/env python3

"""Test AspectExtractor specifically"""

def test_aspect_extractor():
    """Test AspectExtractor with fallback methods."""
    print("Testing AspectExtractor...")
    
    try:
        # Import the AspectExtractor
        from pipeline.aspect_extraction import AspectExtractor
        print("✅ AspectExtractor imported")
        
        # Create instance
        extractor = AspectExtractor()
        print("✅ AspectExtractor initialized")
        
        # Test extraction
        test_text = "The food was great but the service was slow and the interface was confusing."
        print(f"Testing text: {test_text}")
        
        aspects = extractor.extract_aspects(test_text)
        print(f"✅ Found {len(aspects)} aspects:")
        
        for i, aspect in enumerate(aspects):
            print(f"  {i+1}. {aspect['aspect_label']}: {aspect['aspect_sentiment']} ({aspect['aspect_sentiment_score']:.2f})")
        
        return len(aspects) > 0
        
    except Exception as e:
        print(f"❌ Error testing AspectExtractor: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_aspect_extractor()
    if success:
        print("\n✅ AspectExtractor test passed!")
    else:
        print("\n❌ AspectExtractor test failed!")