#!/usr/bin/env python3

"""Test the updated pyABSA ATEPC integration"""

def test_pyabsa_atepc():
    """Test pyABSA with ATEPC API."""
    print("=== Testing pyABSA ATEPC Integration ===")
    
    try:
        # First test the direct API like your working example
        print("\n1. Testing direct pyABSA ATEPC API...")
        try:
            from pyabsa import ATEPCCheckpointManager, TaskCodeOption

            aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(
                checkpoint='multilingual',     # or 'multilingual2', 'multilingual-original'
                auto_device=True,
                task_code='ATEPC'  # Specifying task code as a string literal
            )

            results = aspect_extractor.extract_aspect(
                ["The screen is great, but the battery life is terrible."],
                pred_sentiment=True
            )
            
            print(f"✅ Direct API works!")
            print(f"Results: {results}")
            
        except Exception as e:
            print(f"❌ Direct API failed: {e}")
            return False
        
        # Now test our integrated AspectExtractor
        print("\n2. Testing integrated AspectExtractor...")
        
        from pipeline.aspect_extraction import AspectExtractor
        extractor = AspectExtractor()
        print("✅ AspectExtractor initialized")
        
        test_text = "The website interface is great but the payment system is slow and the documentation is poor."
        
        aspects = extractor.extract_aspects(test_text)
        
        print(f"\n📝 Test text: {test_text}")
        print(f"🏷️  Found {len(aspects)} aspects:")
        
        for i, aspect in enumerate(aspects, 1):
            sentiment_emoji = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}.get(aspect['aspect_sentiment'], "❓")
            print(f"   {i}. {aspect['aspect_label']}: {aspect['aspect_sentiment']} {sentiment_emoji} (confidence: {aspect['aspect_sentiment_score']:.2f})")
        
        return len(aspects) > 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pyabsa_atepc()
    print(f"\n{'='*50}")
    if success:
        print("🎉 pyABSA ATEPC integration successful!")
    else:
        print("❌ pyABSA ATEPC integration failed")
    print("="*50)