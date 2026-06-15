import re

# Comprehensive list of technical and business skills
SKILLS_DB = [
    # Programming Languages
    'python', 'r', 'sql', 'java', 'c++', 'c#', 'javascript', 'typescript', 'php', 'ruby', 'go', 'rust',
    'kotlin', 'swift', 'scala', 'matlab', 'perl', 'shell', 'bash', 'assembly', 'html', 'css',
    
    # AI / Data Science / ML
    'machine learning', 'deep learning', 'nlp', 'natural language processing', 'computer vision',
    'data science', 'data analysis', 'data analytics', 'data visualization', 'big data', 'statistics',
    'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'keras', 'pytorch', 'nltk', 'spacy', 'huggingface',
    'opencv', 'matplotlib', 'seaborn', 'scipy', 'statsmodels', 'xgboost', 'lightgbm', 'catboost',
    
    # Web Frameworks & Libraries
    'react', 'angular', 'vue', 'next.js', 'node.js', 'express', 'django', 'flask', 'fastapi',
    'spring boot', 'asp.net', 'laravel', 'symfony', 'rails', 'jquery', 'bootstrap', 'tailwind',
    
    # Databases & Big Data Tools
    'mysql', 'postgresql', 'postgres', 'mongodb', 'redis', 'cassandra', 'sqlite', 'oracle',
    'sql server', 'firebase', 'elasticsearch', 'dynamodb', 'neo4j', 'spark', 'hadoop', 'hive', 'kafka',
    
    # Cloud & DevOps
    'aws', 'amazon web services', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'jenkins',
    'git', 'github', 'gitlab', 'ci/cd', 'terraform', 'ansible', 'prometheus', 'grafana', 'linux',
    
    # Business / Soft Skills & Tools
    'agile', 'scrum', 'project management', 'product management', 'communication', 'leadership',
    'teamwork', 'problem solving', 'critical thinking', 'excel', 'tableau', 'power bi', 'jira',
    'sales', 'marketing', 'finance', 'recruitment', 'hr', 'human resources'
]

def extract_contact_info(text):
    """
    Extracts email and phone numbers from resume text using regex.
    """
    email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
    emails = re.findall(email_pattern, text)
    
    # Match various phone formats (e.g., +1-234-567-8901, 0300-1234567, etc.)
    phone_pattern = r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|\b\d{10,12}\b|\b\d{3,4}[-.\s]\d{7}\b'
    phones = re.findall(phone_pattern, text)
    
    # Clean phone numbers list
    phones = [p.strip() for p in phones if len(re.sub(r'\D', '', p)) >= 10]
    
    email = emails[0] if emails else "Not Found"
    phone = phones[0] if phones else "Not Found"
    
    return email, phone

def extract_skills(text):
    """
    Extracts skills from text by matching against a predefined database.
    """
    text_lower = text.lower()
    extracted_skills = []
    
    for skill in SKILLS_DB:
        # Use word boundaries to prevent partial matching (e.g. 'c' matching in 'cat')
        # Handle cases with special characters like c++ or .net
        escaped_skill = re.escape(skill)
        # Regex boundary rules
        if skill in ['c', 'r', 'go']:
            pattern = r'\b' + escaped_skill + r'\b'
        elif '+' in skill or '.' in skill:
            # For skills like C++ or .NET, word boundary at end or start might fail, so we handle it:
            pattern = r'(?:^|\s|\b)' + escaped_skill + r'(?:\s|\b|$)'
        else:
            pattern = r'\b' + escaped_skill + r'\b'
            
        if re.search(pattern, text_lower):
            # Return proper casing from SKILLS_DB
            extracted_skills.append(skill.title() if len(skill) > 3 else skill.upper())
            
    return sorted(list(set(extracted_skills)))
