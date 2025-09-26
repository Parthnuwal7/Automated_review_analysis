# # Test Gemini (run in Python console)
# from models.gemini_client import GeminiClient
# client = GeminiClient()
# print("Gemini available:", client.is_available())
# print("Client info:", client.get_client_info())

# from utils.db import DatabaseManager


# db = DatabaseManager()
# db.clear_all_data()
# print("✅ Cleared all data from database.")

def check_sentiment_distribution(reviews):
    from collections import Counter
    
    sentiments = [review.get("sentiment", "Neutral") for review in reviews]
    counter = Counter(sentiments)
    
    total = len(sentiments)
    
    positive = counter.get("Positive", 0)
    negative = counter.get("Negative", 0)
    neutral = counter.get("Neutral", 0)
    
    print(f"Total reviews: {total}")
    print(f"Positive: {positive} ({(positive/total)*100:.2f}%)")
    print(f"Negative: {negative} ({(negative/total)*100:.2f}%)")
    print(f"Neutral: {neutral} ({(neutral/total)*100:.2f}%)")
    
    return counter

# Example usage assuming data from your processing pipeline:
reviews_data = [...]  # This should be your list of processed review dicts
sentiment_counts = check_sentiment_distribution(reviews_data)
