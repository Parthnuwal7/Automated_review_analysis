"""Generate sample review data for testing the sentiment analysis pipeline."""

import csv
import random
from datetime import datetime, timedelta
from typing import List, Dict
import os
from pathlib import Path

def generate_sample_reviews() -> List[Dict[str, str]]:
    """Generate sample reviews in both Hindi and English."""
    
    # Sample English reviews
    english_reviews = [
        "The website is very slow and difficult to navigate. Forms take too long to load.",
        "Great interface! The consultation process was smooth and efficient.",
        "I had trouble logging in multiple times. Password reset feature doesn't work properly.",
        "Excellent service. The online portal made it very easy to submit my documents.",
        "The system crashed when I was uploading files. Very frustrating experience.",
        "Amazing platform! Customer support was very helpful and responsive.",
        "Form validation errors are confusing and don't provide clear guidance.",
        "Love the new dashboard design. Much more user-friendly than before.",
        "Website performance is poor. Takes forever to complete any task.",
        "Thank you for the excellent service. The consultation was very thorough.",
    ]
    
    # Extended Hindi reviews with more variety
    hindi_reviews = [
        "यह वेबसाइट बहुत धीमी है और फॉर्म भरना मुश्किल है।",
        "बहुत अच्छा सिस्टम है! सब कुछ आसानी से हो गया।",
        "लॉगिन करने में बहुत परेशानी हुई। पासवर्ड रीसेट काम नहीं कर रहा।",
        "शानदार सेवा! डॉक्यूमेंट अपलोड करना बहुत आसान था।",
        "सिस्टम बार-बार क्रैश हो रहा था। बहुत निराशाजनक अनुभव।",
        "बेहतरीन प्लेटफॉर्म! कस्टमर सपोर्ट टीम बहुत सहायक थी।",
        "फॉर्म की त्रुटियां समझ नहीं आतीं। स्पष्ट निर्देश नहीं हैं।",
        "नया डिज़ाइन बहुत अच्छा है। उपयोग करना आसान हो गया।",
        "वेबसाइट की गति में सुधार की जरूरत है। बहुत धीमी है।",
        "उत्कृष्ट सेवा के लिए धन्यवाद! परामर्श बहुत विस्तृत था।",
        "मोबाइल ऐप ठीक से काम नहीं कर रहा। कई समस्याएं हैं।",
        "सरल और उपयोग में आसान। मिनटों में काम पूरा हो गया।",
        "सिस्टम में तकनीकी समस्याएं हैं। सुधार की आवश्यकता है।",
        "नई सुविधाएं बहुत उपयोगी हैं। प्रक्रिया तेज हो गई है।",
        "लॉगिन सिस्टम में सुधार चाहिए। अक्सर लॉगआउट हो जाता है।",
        "मुझे इस पोर्टल से अपना काम करने में बहुत सुविधा हुई।",
        "फ़ाइल अपलोड करने में समय बहुत लगता है। कृपया इसे तेज़ करें।",
        "हेल्प सेक्शन में पर्याप्त जानकारी नहीं है।",
        "ऑनलाइन सपोर्ट चैट का रिस्पॉन्स टाइम बहुत अच्छा है।",
        "मैं इस वेबसाइट को अन्य लोगों को भी सुझाऊंगा।"
    ]
    
    # Generate reviews with metadata
    reviews = []
    
    # Add English reviews
    for i, text in enumerate(english_reviews):
        review_id = f"EN_{i+1:03d}"
        timestamp = (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d %H:%M:%S")
        reviews.append({
            "review_id": review_id,
            "text": text,
            "timestamp": timestamp
        })
    
    # Add Hindi reviews
    for i, text in enumerate(hindi_reviews):
        review_id = f"HI_{i+1:03d}"
        timestamp = (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d %H:%M:%S")
        reviews.append({
            "review_id": review_id,
            "text": text,
            "timestamp": timestamp
        })
    
    # Shuffle for random distribution
    random.shuffle(reviews)
    
    return reviews

def save_sample_data():
    """Save sample data to CSV file."""
    try:
        # Generate reviews
        reviews = generate_sample_reviews()
        
        # Ensure data directory exists
        data_dir = Path(__file__).parent.parent / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Save to CSV
        csv_path = data_dir / "sample_reviews.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['review_id', 'text', 'timestamp']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for review in reviews:
                writer.writerow(review)
        
        print(f"✅ Generated {len(reviews)} sample reviews")
        print(f"📁 Saved to: {csv_path}")
        
        # Count languages
        hindi_count = sum(1 for r in reviews if r['review_id'].startswith('HI_'))
        english_count = sum(1 for r in reviews if r['review_id'].startswith('EN_'))
        
        print(f"🇮🇳 Hindi reviews: {hindi_count}")
        print(f"🇺🇸 English reviews: {english_count}")
        
        # Print some examples
        print("\n📝 Sample Reviews:")
        for i, review in enumerate(reviews[:4]):
            lang = "🇮🇳 Hindi" if review['review_id'].startswith('HI_') else "🇺🇸 English"
            print(f"{i+1}. [{lang}] {review['text'][:50]}...")
        
        return str(csv_path)
        
    except Exception as e:
        print(f"❌ Error generating sample data: {e}")
        return None

if __name__ == "__main__":
    print("🔧 Generating sample review data for MCA21 sentiment analysis...")
    save_sample_data()
