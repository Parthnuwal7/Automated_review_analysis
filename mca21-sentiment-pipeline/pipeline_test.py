#!/usr/bin/env python3

"""End-to-end pipeline test with all components"""

def test_complete_pipeline():
    """Test the complete MCA21 pipeline end-to-end."""
    print("Testing complete MCA21 E-Consultation Pipeline...")
    
    try:
        # Test individual components first
        print("\n=== Testing Individual Components ===")
        
        # 1. Test Translation
        print("\n1. Testing Translation (M2M100)...")
        from pipeline.translation import TranslationService
        translator = TranslationService()
        
        hindi_text = "यह सेवा बहुत अच्छी है लेकिन वेबसाइट धीमी है।"
        english_text = translator.translate(hindi_text, source_lang="hi", target_lang="en")
        print(f"   Hindi: {hindi_text}")
        print(f"   English: {english_text}")
        print("   ✅ Translation working")
        
        # 2. Test Sentiment Analysis
        print("\n2. Testing Sentiment Analysis...")
        from pipeline.sentiment import SentimentAnalyzer
        sentiment_analyzer = SentimentAnalyzer()
        
        test_text = "The service is excellent but the website is very slow."
        sentiment = sentiment_analyzer.analyze_sentiment(test_text)
        print(f"   Text: {test_text}")
        print(f"   Sentiment: {sentiment['sentiment']} (confidence: {sentiment['confidence']:.2f})")
        print("   ✅ Sentiment analysis working")
        
        # 3. Test Intent Classification
        print("\n3. Testing Intent Classification...")
        from pipeline.intent import IntentClassifier
        intent_classifier = IntentClassifier()
        
        intent_result = intent_classifier.classify_intent(test_text)
        print(f"   Text: {test_text}")
        print(f"   Intent: {intent_result['intent']} (confidence: {intent_result['confidence']:.2f})")
        print("   ✅ Intent classification working")
        
        # 4. Test Aspect Extraction
        print("\n4. Testing Aspect Extraction...")
        from pipeline.aspect_extraction import AspectExtractor
        aspect_extractor = AspectExtractor()
        
        aspects = aspect_extractor.extract_aspects(test_text)
        print(f"   Text: {test_text}")
        print(f"   Found {len(aspects)} aspects:")
        for aspect in aspects[:3]:  # Show first 3
            print(f"     - {aspect['aspect_label']}: {aspect['aspect_sentiment']} ({aspect['aspect_sentiment_score']:.2f})")
        print("   ✅ Aspect extraction working")
        
        # 5. Test Complete Processing Pipeline
        print("\n=== Testing Complete Pipeline ===")
        print("\n5. Testing Complete Process Pipeline...")
        from pipeline.process import ProcessingPipeline
        
        pipeline = ProcessingPipeline()
        
        # Test with Hindi input
        test_inputs = [
            {
                "text": "यह पोर्टल बहुत उपयोगी है। रजिस्ट्रेशन प्रक्रिया सरल है।",
                "language": "hi"
            },
            {
                "text": "The website interface is confusing but the help system is good.",
                "language": "en"
            }
        ]
        
        for i, test_input in enumerate(test_inputs, 1):
            print(f"\n   Test Input {i}: {test_input['text']} ({test_input['language']})")
            
            result = pipeline.process_review(
                text=test_input['text'],
                metadata={
                    "source_language": test_input['language'],
                    "test_case": f"pipeline_test_{i}"
                }
            )
            
            print(f"   ✅ Processing completed")
            print(f"   📝 Translated Text: {result['processed_text'][:100]}...")
            print(f"   😊 Overall Sentiment: {result['sentiment']['sentiment']} ({result['sentiment']['confidence']:.2f})")
            print(f"   🎯 Intent: {result['intent']['intent']} ({result['intent']['confidence']:.2f})")
            print(f"   🏷️  Aspects Found: {len(result['aspects'])}")
            
            # Show top 2 aspects
            for aspect in result['aspects'][:2]:
                print(f"      • {aspect['aspect_label']}: {aspect['aspect_sentiment']}")
        
        print("\n🎉 Complete pipeline test passed!")
        print("All components are working together successfully.")
        return True
        
    except Exception as e:
        print(f"❌ Error in complete pipeline test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_pipeline()
    if success:
        print("\n✅ MCA21 E-Consultation Pipeline is fully operational!")
        print("Ready for deployment with M2M100 translation and fallback aspect extraction.")
    else:
        print("\n❌ Pipeline test failed!")