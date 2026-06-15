import unittest
import sys
import os

# Add src to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessor import clean_resume
from src.extractor import extract_skills, extract_contact_info
from src.matcher import match_tfidf

class TestResumeScreening(unittest.TestCase):
    
    def test_clean_resume(self):
        raw_text = "Hello @world! Check this out http://example.com #NLP. It's awesome."
        expected = "Hello Check this out NLP It s awesome"
        # We replace multiple spaces with single space in clean_resume, and remove punctuations
        cleaned = clean_resume(raw_text)
        self.assertIn("Hello", cleaned)
        self.assertNotIn("http://example.com", cleaned)
        self.assertNotIn("@world", cleaned)
        self.assertNotIn("#NLP", cleaned)

    def test_extract_skills(self):
        text = "I am a Senior Developer skilled in Python, React, AWS, and Kubernetes."
        skills = extract_skills(text)
        self.assertIn("Python", skills)
        self.assertIn("React", skills)
        self.assertIn("AWS", skills)
        self.assertIn("Kubernetes", skills)
        # Verify case insensitivity matching but correct casing in output
        self.assertNotIn("Developer", skills) # not in skills DB

    def test_extract_contact_info(self):
        text = "Contact me at test.developer@email.com or call +1 123 456 7890."
        email, phone = extract_contact_info(text)
        self.assertEqual(email, "test.developer@email.com")
        self.assertTrue(phone != "Not Found")

    def test_match_tfidf(self):
        resume = "Python React Developer with AWS experience"
        job_desc_match = "React Developer with Python and AWS experience"
        job_desc_diff = "Financial Analyst with Excel accounting expertise"
        
        score_match = match_tfidf(resume, job_desc_match)
        score_diff = match_tfidf(resume, job_desc_diff)
        
        self.assertGreater(score_match, score_diff)
        self.assertGreater(score_match, 30.0)
        self.assertLess(score_diff, 10.0)

    def test_extract_text_from_docx(self):
        import zipfile
        import tempfile
        # Create a tiny mock docx file structure
        content = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:body><w:p><w:r><w:t>Hello from Word Document</w:t></w:r></w:p></w:body>'
            '</w:document>'
        )
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with zipfile.ZipFile(tmp_path, 'w') as docx:
                docx.writestr('word/document.xml', content)
            
            from src.parser import extract_text_from_docx
            text = extract_text_from_docx(tmp_path)
            self.assertEqual(text.strip(), "Hello from Word Document")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_weighted_match_score_dynamic_weights(self):
        from src.matcher import weighted_match_score
        resume = "python developer"
        jd = "python developer"
        
        # Test 1: 100% Skill Coverage, 0% Text Similarity
        score_data1 = weighted_match_score(resume, jd, ["Python"], ["Python"], skill_weight=1.0, text_weight=0.0)
        self.assertEqual(score_data1["composite_score"], 100.0)
        
        # Test 2: 0% Skill Coverage, 100% Text Similarity
        score_data2 = weighted_match_score(resume, jd, [], ["Python"], skill_weight=0.0, text_weight=1.0)
        self.assertAlmostEqual(score_data2["composite_score"], score_data2["text_similarity_pct"], places=1)

if __name__ == '__main__':
    unittest.main()
