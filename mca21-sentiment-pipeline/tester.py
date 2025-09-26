from pipeline.process import ReviewProcessor
processor = ReviewProcessor()
results = processor.process_reviews_from_csv('test_reviews_upload.csv')
for review in results['reviews'][:5]:
    print(review['review_id'], review['language'], review.get('translated_text'), review['sentiment'], review['intent'])
