#!/usr/bin/env python3

"""Minimal test for aspect extraction without heavy imports"""

def test_minimal():
    """Test minimal functionality."""
    print("Testing minimal aspect extraction...")
    
    try:
        # Test basic imports first
        print("Testing config import...")
        import config
        print(f"✅ Config imported. ASPECT_KEYWORDS has {len(config.ASPECT_KEYWORDS)} categories")
        
        # Test simple text processing
        from utils.text import TextProcessor
        print("✅ TextProcessor imported")
        
        processor = TextProcessor()
        test_text = "The food was great but the service was slow."
        cleaned = processor.clean_text(test_text)
        print(f"✅ Text processing works: '{cleaned}'")
        
        # Test aspect keyword detection
        keywords_found = []
        test_lower = test_text.lower()
        
        for aspect_label, keywords in config.ASPECT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in test_lower:
                    keywords_found.append((aspect_label, keyword))
        
        print(f"✅ Found {len(keywords_found)} aspect keywords: {keywords_found}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in minimal test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_minimal()
    if success:
        print("\n✅ Minimal test passed!")
    else:
        print("\n❌ Minimal test failed!")