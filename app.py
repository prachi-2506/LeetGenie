import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Page configuration
st.set_page_config(
    page_title="LeetGenie - Code Problem Similarity Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "# LeetGenie 🧠\nIntelligent Code Problem Similarity Analyzer\n\nBuilt with Streamlit"
    }
)

# Custom styling
st.markdown("""
    <style>
        /* Hide deploy button only */
        [data-testid="stToolbar"] [data-testid="stActionButton"]:first-child {
            display: none;
        }
        
        footer {visibility: hidden;}
        
        /* Import modern font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global styles */
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Main container */
        .main {
            background-color: #FAFAFA;
            padding: 1rem 2rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        /* Logo and header */
        .logo-container {
            text-align: center;
            margin-bottom: 1.5rem;
            padding: 1rem 0;
        }
        
        .logo-text {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.25rem;
            letter-spacing: -0.5px;
        }
        
        .tagline {
            font-size: 0.9rem;
            color: #6B7280;
            font-weight: 400;
            margin-top: 0.25rem;
        }
        
        /* Card containers */
        .input-card {
            background: white;
            border-radius: 12px;
            padding: 1.25rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }
        
        .input-card:hover {
            box-shadow: 0 4px 6px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.05);
            transform: translateY(-2px);
        }
        
        /* Section headers */
        .section-header {
            font-size: 1rem;
            font-weight: 600;
            color: #1F2937;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        /* Text areas */
        .stTextArea textarea {
            border-radius: 12px !important;
            border: 1.5px solid #E5E7EB !important;
            font-size: 0.95rem !important;
            font-family: 'Inter', sans-serif !important;
            padding: 1rem !important;
            transition: all 0.2s ease !important;
        }
        
        .stTextArea textarea:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        }
        
        .stTextArea label {
            font-size: 0.9rem !important;
            font-weight: 500 !important;
            color: #4B5563 !important;
        }
        
        /* Sample buttons */
        .sample-container {
            background: white;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            margin: 1.25rem 0;
        }
        
        .stButton > button {
            border-radius: 10px !important;
            border: 1.5px solid #E5E7EB !important;
            background-color: white !important;
            color: #374151 !important;
            font-weight: 500 !important;
            font-size: 0.9rem !important;
            padding: 0.6rem 1.2rem !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
        }
        
        .stButton > button:hover {
            border-color: #667eea !important;
            background-color: #F9FAFB !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 2px 4px rgba(102, 126, 234, 0.15) !important;
        }
        
        /* Analyze button (special styling) */
        .analyze-button button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            padding: 0.85rem 2.5rem !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
        }
        
        .analyze-button button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4) !important;
        }
        
        /* Results container */
        .results-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin-top: 1.5rem;
            text-align: center;
        }
        
        .similarity-score {
            font-size: 3.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0.75rem 0;
            line-height: 1;
        }
        
        .score-label {
            font-size: 1rem;
            color: #6B7280;
            font-weight: 500;
            margin-bottom: 2rem;
        }
        
        /* Progress bar */
        .progress-container {
            width: 100%;
            height: 10px;
            background-color: #F3F4F6;
            border-radius: 999px;
            overflow: hidden;
            margin: 1.25rem 0;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 999px;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        /* Status badge */
        .status-badge {
            display: inline-block;
            padding: 0.75rem 1.5rem;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.95rem;
            margin: 1rem 0;
        }
        
        /* Feature boxes */
        .feature-box {
            background: #F9FAFB;
            border-radius: 10px;
            padding: 1rem;
            border: 1px solid #E5E7EB;
            margin: 0.75rem 0;
        }
        
        .feature-box h4 {
            font-size: 1rem;
            font-weight: 600;
            color: #1F2937;
            margin-bottom: 0.75rem;
        }
        
        .feature-box p {
            font-size: 0.9rem;
            color: #6B7280;
            line-height: 1.6;
        }
        
        /* Alerts */
        .stAlert {
            border-radius: 12px !important;
            border: none !important;
            font-weight: 500 !important;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            color: #9CA3AF;
            font-size: 0.9rem;
            margin-top: 4rem;
            padding: 2rem 0;
            border-top: 1px solid #E5E7EB;
        }
    </style>
""", unsafe_allow_html=True)

class SimpleSimilarityModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        """Preprocess text for similarity comparison"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize and remove stopwords
        tokens = text.split()
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def calculate_similarity(self, text1, text2):
        """Calculate similarity between two texts using multiple methods"""
        # Preprocess texts
        text1_clean = self.preprocess_text(text1)
        text2_clean = self.preprocess_text(text2)
        
        if not text1_clean or not text2_clean:
            return 0.0
        
        # TF-IDF Cosine Similarity
        tfidf_matrix = self.vectorizer.fit_transform([text1_clean, text2_clean])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Jaccard Similarity
        words1 = set(text1_clean.split())
        words2 = set(text2_clean.split())
        common_words = len(words1.intersection(words2))
        total_words = len(words1.union(words2))
        jaccard_sim = common_words / total_words if total_words > 0 else 0
        
        # Length-based similarity
        len1 = len(text1_clean.split())
        len2 = len(text2_clean.split())
        len_sim = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
        
        # Combined similarity score (weighted average)
        final_similarity = (0.6 * cosine_sim + 0.3 * jaccard_sim + 0.1 * len_sim)
        
        return max(0, min(1, final_similarity))

def load_sample_questions():
    """Load sample LeetCode questions for demonstration"""
    return [
        "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
        "You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.",
        "Given a string s, find the length of the longest substring without repeating characters.",
        "Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.",
        "Given a string s, return the longest palindromic substring in s."
    ]

def load_sample_pairs():
    """Load pre-paired sample problems for testing"""
    return [
        {
            "problem1": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
            "problem2": "Given a 1-indexed array of integers numbers that is already sorted in non-decreasing order, find two numbers such that they add up to a specific target number."
        },
        {
            "problem1": "You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.",
            "problem2": "Given a linked list, reverse the nodes of a linked list k at a time and return its modified list."
        },
        {
            "problem1": "Given a string s, find the length of the longest substring without repeating characters.",
            "problem2": "Given a string s, return the longest palindromic substring in s."
        },
        {
            "problem1": "Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.",
            "problem2": "There are two sorted arrays nums1 and nums2 of size m and n respectively. Find the median of the two sorted arrays."
        },
        {
            "problem1": "Given an array nums of n integers, return an array output such that output[i] is equal to the product of all the elements of nums except nums[i].",
            "problem2": "Given an integer array nums, find a contiguous non-empty subarray within the array that has the largest product, and return the product."
        }
    ]

def main():
    # Initialize model
    if 'model' not in st.session_state:
        st.session_state.model = SimpleSimilarityModel()
    
    # Header
    st.markdown("""
        <div class="logo-container">
            <div class="logo-text">🧠 LeetGenie</div>
            <div class="tagline">Intelligent Code Problem Similarity Analyzer</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Sample questions section
    st.markdown('<div class="sample-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">💡 Quick Start: Try Sample Problem Pairs</div>', unsafe_allow_html=True)
    
    sample_pairs = load_sample_pairs()
    sample_cols = st.columns(5)
    
    for i, col in enumerate(sample_cols):
        with col:
            if st.button(f"Sample {i+1}", key=f"sample_{i}"):
                st.session_state.problem1 = sample_pairs[i]["problem1"]
                st.session_state.problem2 = sample_pairs[i]["problem2"]
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create two columns for input
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📝 Problem 1</div>', unsafe_allow_html=True)
        problem1 = st.text_area(
            "Enter your first coding problem:",
            height=200,
            placeholder="Paste your first coding problem description here...\n\nExample: Given an array of integers, return indices of two numbers that add up to a target.",
            key="problem1",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📝 Problem 2</div>', unsafe_allow_html=True)
        problem2 = st.text_area(
            "Enter your second coding problem:",
            height=200,
            placeholder="Paste your second coding problem description here...\n\nExample: Given a sorted array, find two numbers that sum to a specific target.",
            key="problem2",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Analyze button
    st.markdown('<br>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        st.markdown('<div class="analyze-button">', unsafe_allow_html=True)
        analyze_clicked = st.button("🔍 Analyze Similarity", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Results section
    if analyze_clicked and problem1 and problem2:
        with st.spinner("Analyzing problem similarity..."):
            similarity = st.session_state.model.calculate_similarity(problem1, problem2)
            similarity_percentage = similarity * 100
        
        # Display results in modern card
        st.markdown('<div class="results-card">', unsafe_allow_html=True)
        
        # Similarity score with gradient styling
        st.markdown(f'<div class="similarity-score">{similarity_percentage:.1f}%</div>', unsafe_allow_html=True)
        st.markdown('<div class="score-label">Similarity Score</div>', unsafe_allow_html=True)
        
        # Progress bar
        progress_html = f"""
        <div class="progress-container">
            <div class="progress-bar" style="width: {similarity_percentage}%;"></div>
        </div>
        """
        st.markdown(progress_html, unsafe_allow_html=True)
        
        # Status badge with color coding
        if similarity_percentage >= 80:
            status_class = "status-badge" 
            status_color = "background-color: #D1FAE5; color: #065F46;"
            status_text = "🎯 High Similarity"
            status_desc = "These problems are very closely related!"
        elif similarity_percentage >= 60:
            status_class = "status-badge"
            status_color = "background-color: #DBEAFE; color: #1E40AF;"
            status_text = "🔍 Moderate Similarity"
            status_desc = "Problems share significant concepts"
        elif similarity_percentage >= 40:
            status_class = "status-badge"
            status_color = "background-color: #FEF3C7; color: #92400E;"
            status_text = "📝 Some Similarity"
            status_desc = "Partial overlap in concepts"
        else:
            status_class = "status-badge"
            status_color = "background-color: #FEE2E2; color: #991B1B;"
            status_text = "🚫 Low Similarity"
            status_desc = "Problems are quite different"
        
        st.markdown(f'<div class="{status_class}" style="{status_color}">{status_text}</div>', unsafe_allow_html=True)
        st.markdown(f'<p style="color: #6B7280; margin-top: 0.5rem;">{status_desc}</p>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed analysis section
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="section-header" style="text-align: center;">🔬 Detailed Analysis</div>', unsafe_allow_html=True)
        
        # Preprocess texts for analysis
        text1_clean = st.session_state.model.preprocess_text(problem1)
        text2_clean = st.session_state.model.preprocess_text(problem2)
        
        words1 = set(text1_clean.split())
        words2 = set(text2_clean.split())
        common_words = words1.intersection(words2)
        
        # Feature boxes in grid
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-box">
                <h4>Problem 1 Stats</h4>
                <p style="font-size: 1.5rem; font-weight: 700; color: #667eea; margin: 0.5rem 0;">{}</p>
                <p>Total Words</p>
                <p style="font-size: 1.5rem; font-weight: 700; color: #764ba2; margin: 0.5rem 0;">{}</p>
                <p>Unique Concepts</p>
            </div>
            """.format(len(text1_clean.split()), len(words1)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-box">
                <h4>Problem 2 Stats</h4>
                <p style="font-size: 1.5rem; font-weight: 700; color: #667eea; margin: 0.5rem 0;">{}</p>
                <p>Total Words</p>
                <p style="font-size: 1.5rem; font-weight: 700; color: #764ba2; margin: 0.5rem 0;">{}</p>
                <p>Unique Concepts</p>
            </div>
            """.format(len(text2_clean.split()), len(words2)), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-box">
                <h4>Common Ground</h4>
                <p style="font-size: 1.5rem; font-weight: 700; color: #667eea; margin: 0.5rem 0;">{}</p>
                <p>Shared Concepts</p>
                <p style="font-size: 1.5rem; font-weight: 700; color: #764ba2; margin: 0.5rem 0;">{:.1f}%</p>
                <p>Overlap Ratio</p>
            </div>
            """.format(len(common_words), (len(common_words) / max(len(words1), len(words2)) * 100) if max(len(words1), len(words2)) > 0 else 0), unsafe_allow_html=True)
        
        # Common concepts display
        if common_words:
            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.markdown('<h4 style="margin-bottom: 0.75rem;">🔗 Shared Concepts</h4>', unsafe_allow_html=True)
            common_words_list = sorted(list(common_words))[:15]
            st.markdown(f'<p style="color: #4B5563; line-height: 1.8;">{" • ".join(common_words_list)}{" ..." if len(common_words) > 15 else ""}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif analyze_clicked:
        st.warning("⚠️ Please enter both problem descriptions to analyze similarity.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; margin-top: 3rem;'>"
        "LeetGenie · Intelligent Code Problem Analysis · Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()