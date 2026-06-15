import re

def clean_resume(resume_text):
    """
    Cleans raw resume text by removing URLs, hashtags, mentions,
    special characters, non-ASCII characters, and extra whitespaces.
    """
    if not isinstance(resume_text, str):
        return ""
    
    # Remove URLs
    resume_text = re.sub(r'https?://\S+|www\.\S+', ' ', resume_text)
    
    # Remove RT and cc
    resume_text = re.sub(r'\b(RT|cc)\b', ' ', resume_text)
    
    # Remove hashtags
    resume_text = re.sub(r'#\S+', ' ', resume_text)
    
    # Remove mentions
    resume_text = re.sub(r'@\S+', ' ', resume_text)
    
    # Remove punctuations and special characters
    resume_text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~]', ' ', resume_text)
    
    # Remove non-ASCII characters
    resume_text = re.sub(r'[^\x00-\x7f]', ' ', resume_text)
    
    # Remove extra whitespaces
    resume_text = re.sub(r'\s+', ' ', resume_text)
    
    return resume_text.strip()
