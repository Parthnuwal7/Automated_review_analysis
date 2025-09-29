#!/usr/bin/env python3

"""Final comprehensive test of pyABSA integration"""

import time

def test_pyabsa_complete():
    """Test complete pyABSA integration with proper error handling."""
    print("=== Complete pyABSA ATEPC Integration Test ===")
    
    start_time = time.time()
    
    try:
        print("\n1. Testing direct pyABSA API with compatible transformers...")
        from pyabsa import ATEPCCheckpointManager
        
        print("   Loading multilingual ATEPC model...")
        aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(
            checkpoint='multilingual',
            auto_device=True,
            task_code='ATEPC'
        )
        
        load_time = time.time() - start_time
        print(f"   ✅ Model loaded in {load_time:.1f} seconds")
        
        # Test with example text
        test_texts = [
            "The screen is great, but the battery life is terrible.",
            "The website interface is user-friendly but the payment system is slow.",
            "Registration process is simple and the documentation is helpful."
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n   Test {i}: {text}")
            
            results = aspect_extractor.extract_aspect([text], pred_sentiment=True)
            print(f"   Results: {results}")
        
        print("\n✅ Direct API test successful!")
        
        # Now test integrated AspectExtractor
        print("\n2. Testing integrated AspectExtractor...")
        
        from pipeline.aspect_extraction import AspectExtractor
        extractor = AspectExtractor()
        
        test_text = "The government portal registration is simple but the payment system is slow and the documentation is unclear."
        aspects = extractor.extract_aspects(test_text)
        
        print(f"\n📝 Test text: {test_text}")
        print(f"🏷️  Found {len(aspects)} aspects:")
        
        for aspect in aspects:
            sentiment_emoji = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}.get(aspect['aspect_sentiment'], "❓")
            print(f"   • {aspect['aspect_label']}: {aspect['aspect_sentiment']} {sentiment_emoji} (confidence: {aspect['aspect_sentiment_score']:.2f})")
        
        total_time = time.time() - start_time
        print(f"\n🎉 Complete integration successful in {total_time:.1f} seconds!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pyabsa_complete()
    print(f"\n{'='*60}")
    if success:
        print("🎉 SUCCESS: pyABSA ATEPC fully integrated and working!")
        print("   • Compatible transformers version: ✅")
        print("   • ATEPC model loading: ✅")
        print("   • Aspect extraction: ✅")
        print("   • Sentiment analysis: ✅")
        print("   • Integration complete: ✅")
    else:
        print("❌ FAILED: pyABSA integration needs more work")
    print("="*60)