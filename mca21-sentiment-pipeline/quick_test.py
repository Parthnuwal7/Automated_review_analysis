#!/usr/bin/env python3

"""Quick test for aspect extraction"""

import sys
import os
sys.path.append(os.getcwd())

def test_basic_extraction():
    """Test basic aspect extraction functionality."""
    print("Testing basic aspect extraction...")
    
    try:
        from pipeline.aspect_extraction import AspectExtractor
        
        # Initialize with simple config
        extractor = AspectExtractor()
        
        # Test text
        test_text = "The food was great but the service was slow and the ambiance was nice."
        
        print(f"Testing text: {test_text}")
        
        # Extract aspects
        aspects = extractor.extract_aspects(test_text)
        
        print(f"Found {len(aspects)} aspects:")
        for aspect in aspects:
            print(f"- {aspect['aspect_label']}: {aspect['aspect_sentiment']} ({aspect['aspect_sentiment_score']:.2f})")
        
        return len(aspects) > 0
        
    except Exception as e:
        print(f"Error in basic test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_extraction()
    if success:
        print("\n✅ Basic aspect extraction test passed!")
    else:
        print("\n❌ Basic aspect extraction test failed!")