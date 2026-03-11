from ml.transliteration_model import transliterate
from ml.devanagari import predict_sentiment
from comment_extractor import extract_comments

def nlp_pipeline(url):
    results = []
    
    # 1. Extract comments
    comments = extract_comments(url)
    
    for comment in comments:
        # 2. Transliterate
        devanagari_comment = transliterate(comment)
        
        # 3. Predict aspect and sentiment
        aspect, sentiment = predict_sentiment(devanagari_comment)
        
        results.append({
            "original_comment": comment,
            "devanagari_comment": devanagari_comment,
            "aspect": aspect,
            "sentiment": sentiment
        })
    
    return results