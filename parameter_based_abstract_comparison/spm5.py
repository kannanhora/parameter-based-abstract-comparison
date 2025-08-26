import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import fitz
import docx
import re
import nltk
from nltk.corpus import wordnet as wn

nltk.download("wordnet")
nltk.download("omw-1.4")  # For additional wordnet resources


# Load models
model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")

# Define keyword sets for different parameters
TECHNOLOGY_KEYWORDS = [
    "machine learning", "deep learning", "cloud computing", "blockchain", "python", "java", "aws", "tensorflow",
    "react", "django", "hadoop", "SQL", "rest api", "artificial intelligence", "big data", "apache spark", "kafka",
    "kubernetes", "docker", "graphql", "node.js", "angular", "vue.js", "flutter", "swift", "C++", "C#", "ruby", "golang"
]

DEVELOPMENT_TOOL_KEYWORDS = [
    "Visual Studio", "PyCharm", "Jupyter Notebook", "Git", "Docker", "Kubernetes", "Eclipse", "Postman", "Jenkins", 
    "Travis CI", "VS Code", "Android Studio", "Xcode", "Maven", "Gradle", "IntelliJ IDEA", "NetBeans", "Sublime Text",
    "Notepad++", "GitHub", "Bitbucket", "GitLab", "CircleCI", "Android SDK", "Flutter SDK", "Node Package Manager",
    "npm", "Composer", "Unity", "Unreal Engine", "Blender", "Jira", "Confluence", "Slack"
]

ARCHITECTURE_KEYWORDS = [
    "monolithic", "microservices", "serverless", "hexagonal", "layered", "event-driven", "client-server", "MVC", 
    "SOA", "RESTful", "distributed systems", "CQRS", "event sourcing", "service-oriented architecture", "API-first", 
    "cloud-native", "domain-driven design", "containerized", "serverless computing", "micro frontends"
]

FUNCTIONALITY_KEYWORDS = [
    "authentication", "data processing", "report generation", "user management", "payment processing", "task scheduling",
    "chatbot", "dashboard", "data visualization", "notifications", "search", "data analytics", "workflow automation", 
    "inventory management", "CRM", "ERP", "user notifications", "real-time updates", "recommendation system", 
    "file sharing", "multi-language support", "geolocation", "data streaming", "chat functionality", "content management"
]

QUALITY_KEYWORDS = [
    "scalability", "security", "performance", "usability", "reliability", "maintainability", "portability", 
    "efficiency", "testability", "modularity", "flexibility", "fault tolerance", "recoverability", "extensibility", 
    "availability", "accessibility", "internationalization", "localization", "compliance", "responsiveness", "stability"
]

ALGORITHM_KEYWORDS = [
    "neural network", "decision tree", "k-means", "SVM", "regression", "random forest", "PCA", "LSTM", "BERT", 
    "transformer", "GAN", "VAE", "boosting", "XGBoost", "KNN", "logistic regression", "Naive Bayes", "recurrent neural network",
    "reinforcement learning", "Q-learning", "decision trees", "gradient boosting", "clustering", "k-means clustering",
    "principal component analysis", "genetic algorithm", "Bayesian networks"
]

PROJECT_OBJECTIVE_KEYWORDS = [
    "healthcare management", "e-commerce platform", "fraud detection", "inventory management", "customer relationship management",
    "financial analysis", "recommendation system", "real-time analytics", "supply chain management", "chatbots", 
    "customer support", "data mining", "social media analysis", "education", "smart cities", "IoT solutions", "personalization",
    "financial forecasting", "enterprise resource planning", "cloud migration", "automated testing", "robotics"
]

PROTOCOL_KEYWORDS = [
    "HTTP", "HTTPS", "FTP", "TCP/IP", "SMTP", "IMAP", "POP3", "OAuth", "SSL/TLS", "REST", "SOAP", "WebSocket", "MQTT", 
    "SSH", "SNMP", "RDP", "IPsec", "ICMP", "DNS", "NFS", "LDAP", "VPN", "SNMP", "WebRTC", "XMPP", "WebSockets"
]

PROCESS_KEYWORDS = [
    "Agile", "Scrum", "Waterfall", "DevOps", "Kanban", "Lean", "Extreme Programming (XP)", "CI/CD", 
    "Test-Driven Development (TDD)", "Behavior-Driven Development (BDD)", "Rapid Application Development (RAD)", 
    "Continuous Integration", "Continuous Deployment", "Incremental Development", "Feature-Driven Development", 
    "Unified Process", "V-Model", "Design Thinking", "Prototype model", "Spiral model", "Agile Scrum"
]
# Function to extract title (first one or two words)
def extract_title(abstract):
    words = abstract.split()  # Split the abstract into words
    title = " ".join(words[:2])  # Take the first two words as the title
    return title.strip()

# Functions for extracting text from files
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as pdf:
        for page in pdf:
            text += page.get_text()
    return text.strip()

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs]).strip()

# Extract key details based on user-selected parameters
def extract_key_details(abstract, selected_parameters):
    #def extract_keywords(abstract, keywords):
       # return ", ".join([kw for kw in keywords if kw.lower() in abstract.lower()]) or "Not specified"
    # Enhance extract_keywords function with WordNet synonyms
    def extract_keywords(abstract, keywords):
        matched_keywords = set()
        abstract_lower = abstract.lower()

        for kw in keywords:
            kw_lower = kw.lower()
            # Check if the keyword itself is present
            if kw_lower in abstract_lower:
                matched_keywords.add(kw)
            else:
                # Check synonyms using WordNet
                synonyms = wn.synsets(kw_lower)
                for syn in synonyms:
                    for lemma in syn.lemmas():
                        synonym = lemma.name().replace('_', ' ').lower()
                        if synonym in abstract_lower:
                            matched_keywords.add(kw)
                            break
                    if kw in matched_keywords:
                        break

        return ", ".join(matched_keywords) if matched_keywords else "Not specified"

        
    details = {}
    if "Technology Used" in selected_parameters:
        details["Technology Used"] = extract_keywords(abstract, TECHNOLOGY_KEYWORDS)
    if "Development Tools Used" in selected_parameters:
        details["Development Tools Used"] = extract_keywords(abstract, DEVELOPMENT_TOOL_KEYWORDS)
    if "Project Objective" in selected_parameters:
        details["Project Objective"] = extract_keywords(abstract, PROJECT_OBJECTIVE_KEYWORDS)
    if "Architecture Style" in selected_parameters:
        details["Architecture Style"] = extract_keywords(abstract, ARCHITECTURE_KEYWORDS)
    if "System Functionalities" in selected_parameters:
        details["System Functionalities"] = extract_keywords(abstract, FUNCTIONALITY_KEYWORDS)
    if "System Quality Attributes" in selected_parameters:
        details["System Quality Attributes"] = extract_keywords(abstract, QUALITY_KEYWORDS)
    if "Algorithms Used" in selected_parameters:
        details["Algorithms Used"] = extract_keywords(abstract, ALGORITHM_KEYWORDS)
    if "Protocols Used" in selected_parameters:
        details["Protocols Used"] = extract_keywords(abstract, PROTOCOL_KEYWORDS)
    if "Processes Used" in selected_parameters:
        details["Processes Used"] = extract_keywords(abstract, PROCESS_KEYWORDS)
    
    return details

# Function to compute cosine similarity
def compute_similarity(text1, text2):
    embeddings1 = model.encode([text1])
    embeddings2 = model.encode([text2])
    return cosine_similarity(embeddings1, embeddings2)[0][0]

# Main function for the Streamlit app
def main():
    st.title("Abstract Comparison Tool")
    st.write("Compare abstracts by selecting specific parameters.")
    
    # Parameter selection
    parameters = [
        "Technology Used", "Development Tools Used", "Project Objective", "Architecture Style",
        "System Functionalities", "System Quality Attributes", "Algorithms Used", "Protocols Used", "Processes Used"
    ]
    selected_parameters = st.multiselect(
        "Select Parameters to Compare", options=parameters, default=parameters
    )
    
    # Input method selection
    option = st.selectbox("Choose input method", ["Upload Files", "Paste Text"])
    
    abstract1 = abstract2 = ""
    title1 = title2 = ""

    if option == "Upload Files":
        file1 = st.file_uploader("Upload Abstract 1 (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
        file2 = st.file_uploader("Upload Abstract 2 (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
        
        if file1:
            if file1.type == "application/pdf":
                abstract1 = extract_text_from_pdf(file1)
            elif file1.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                abstract1 = extract_text_from_docx(file1)
            else:
                abstract1 = file1.read().decode("utf-8").strip()
                
        if file2:
            if file2.type == "application/pdf":
                abstract2 = extract_text_from_pdf(file2)
            elif file2.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                abstract2 = extract_text_from_docx(file2)
            else:
                abstract2 = file2.read().decode("utf-8").strip()
                
        # Extract titles from abstracts
        title1 = extract_title(abstract1)
        title2 = extract_title(abstract2)

    elif option == "Paste Text":
        abstract1 = st.text_area("Paste Abstract 1", height=200)
        abstract2 = st.text_area("Paste Abstract 2", height=200)
        
        # Extract titles from pasted text
        title1 = extract_title(abstract1)
        title2 = extract_title(abstract2)
    
    # Compare abstracts button
    if st.button("Compare Abstracts"):
        if not abstract1.strip() or not abstract2.strip():
            st.error("Both abstracts are required!")
        else:
            details1 = extract_key_details(abstract1, selected_parameters)
            details2 = extract_key_details(abstract2, selected_parameters)
            
            # Display extracted details with extracted titles
            st.write(f"### Extracted Details for {title1 if title1 else 'Abstract 1'}:")
            st.json(details1)
            st.write(f"### Extracted Details for {title2 if title2 else 'Abstract 2'}:")
            st.json(details2)
            
            similarity_results = {}
            total_similarity = 0
            num_comparisons = 0
            
            st.write("### Similarity for Each Selected Parameter:")
            for parameter in selected_parameters:
                text1, text2 = details1.get(parameter, ""), details2.get(parameter, "")
                if "Not specified" in (text1, text2):
                    similarity_results[parameter] = "Not specified"
                    st.warning(f"**{parameter}:** Not specified in one or both abstracts.")
                else:
                    score = compute_similarity(text1, text2)
                    similarity_results[parameter] = score
                    total_similarity += score
                    num_comparisons += 1
                    
                    if score > 0.8:
                        st.success(f"**{parameter}:** {score:.4f} - Highly similar!")
                    elif score > 0.5:
                        st.warning(f"**{parameter}:** {score:.4f} - Moderately similar.")
                    else:
                        st.error(f"**{parameter}:** {score:.4f} - Not similar.")
            
            # Display cumulative similarity score
            cumulative_similarity = total_similarity / num_comparisons if num_comparisons > 0 else 0
            st.write("### Cumulative Similarity:")
            if cumulative_similarity > 0.8:
                st.success(f"**Cumulative Similarity Score:** {cumulative_similarity:.4f} - Highly similar overall!")
            elif cumulative_similarity > 0.5:
                st.warning(f"**Cumulative Similarity Score:** {cumulative_similarity:.4f} - Moderately similar overall.")
            else:
                st.error(f"**Cumulative Similarity Score:** {cumulative_similarity:.4f} - Not similar overall.")

if __name__ == "__main__":
    main()
