"""Test script to verify pyABSA integration."""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from pipeline.aspect_extraction import AspectExtractor

def test_pyabsa_integration():
    """Test pyABSA integration with sample texts."""
    
    # Sample test texts (English and simple Hindi transliterations)
    test_texts = [
        "The website is very slow and the login process is confusing.",
        "I love the new interface design, it's so user-friendly and fast.",
        "The customer support is terrible but the payment system works well.",
        "Documentation is unclear and the mobile app crashes frequently.",
        "Great performance improvement, very satisfied with the new features."
    ]
    
    print("🔍 Testing pyABSA Integration")
    print("=" * 50)
    
    # Initialize aspect extractor
    try:
        extractor = AspectExtractor()
        print(f"✅ AspectExtractor initialized")
        
        service_info = extractor.get_service_info()
        print(f"📊 Service info: {service_info}")
        
        if service_info['service_type'] == 'Fallback':
            print("⚠️  Using fallback mode - pyABSA models not loaded")
            print("💡 This is expected for the first run. Testing fallback functionality...")
        else:
            print("🎯 Using pyABSA models")
        
        print()
        
    except Exception as e:
        print(f"❌ Error initializing AspectExtractor: {e}")
        return
    
    # Test each text
    for i, text in enumerate(test_texts, 1):
        print(f"Test {i}: {text}")
        print("-" * 60)
        
        try:
            aspects = extractor.extract_aspects_with_sentiment(text)
            
            if aspects:
                for aspect in aspects:
                    print(f"  🎯 Aspect: {aspect['aspect_label']}")
                    print(f"     Sentiment: {aspect['aspect_sentiment']} ({aspect['aspect_sentiment_score']:.2f})")
                    print(f"     Context: {aspect['matched_text'][:50]}{'...' if len(aspect['matched_text']) > 50 else ''}")
                    print()
            else:
                print("  ℹ️  No aspects extracted")
                print()
        
        except Exception as e:
            print(f"  ❌ Error processing text: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    # Test batch processing
    print("🔄 Testing Batch Processing")
    print("=" * 50)
    
    try:
        batch_results = extractor.analyze_batch(test_texts[:3])  # Test with first 3 texts
        print(f"✅ Batch processing completed: {len(batch_results)} results")
        
        for i, result in enumerate(batch_results):
            print(f"  Text {i+1}: {len(result)} aspects extracted")
    
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n📈 Service Statistics:")
    stats = extractor.get_aspect_statistics()
    print(stats)
    
    print(f"\n🔧 Available Status:")
    print(f"  - Service Available: {extractor.is_service_available()}")
    print(f"  - Models Loaded: {extractor._available if hasattr(extractor, '_available') else 'Unknown'}")

def test_simple_pyabsa():
    """Test pyABSA directly to understand the API."""
    print("\n🧪 Direct pyABSA API Test")
    print("=" * 50)
    
    try:
        from pyabsa import AspectPolarityClassification as APC
        from pyabsa import available_checkpoints
        
        print("✅ pyABSA imported successfully")
        
        # Check available checkpoints
        try:
            apc_checkpoints = available_checkpoints('apc')
            print(f"📋 Available APC checkpoints: {apc_checkpoints[:3] if apc_checkpoints else 'None'}")
        except Exception as e:
            print(f"⚠️  Could not get checkpoints: {e}")
        
        # Try to create a classifier
        try:
            classifier = APC.AspectPolarityClassifier(
                checkpoint='english',  # Use default english checkpoint
                auto_device=True
            )
            print("✅ APC Classifier created successfully")
            
            # Test prediction
            test_input = "The food is good but the service is bad$T$food"
            result = classifier.predict(test_input, save_result=False, print_result=False)
            print(f"🎯 Test prediction result: {result}")
            
        except Exception as e:
            print(f"❌ Error creating classifier: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"❌ Could not import pyABSA: {e}")

if __name__ == "__main__":
    test_pyabsa_integration()
    test_simple_pyabsa()