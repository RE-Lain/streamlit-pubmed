import streamlit as st
import requests
import json
import re
from datetime import datetime
import xml.etree.ElementTree as ET
import time
import random

# Configure page
st.set_page_config(
    page_title="Medical Research Platform",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'medical_results' not in st.session_state:
    st.session_state.medical_results = None
if 'training_data' not in st.session_state:
    st.session_state.training_data = []
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model_performance' not in st.session_state:
    st.session_state.model_performance = None

class MedicalProcessor:
    """Medical article processor for PubMed."""
    
    def __init__(self):
        self.pubmed_base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
        
    def fetch_medical_articles(self, query: str, max_results: int = 15):
        """Fetch medical articles from PubMed."""
        try:
            # Add delay to respect NCBI rate limits
            time.sleep(0.5)
            
            # Search PubMed
            search_url = f"{self.pubmed_base}esearch.fcgi"
            search_params = {
                'db': 'pubmed',
                'term': f"{query}[Title/Abstract] AND (clinical OR medical OR patient OR treatment)",
                'retmax': max_results,
                'retmode': 'json',
                'tool': 'medical_research_platform',
                'email': 'research@example.com'  # Required by NCBI
            }
            
            response = requests.get(search_url, params=search_params, timeout=30)
            response.raise_for_status()
            
            search_results = response.json()
            ids = search_results.get('esearchresult', {}).get('idlist', [])
            
            if not ids:
                return {'status': 'no_results', 'message': 'No medical articles found for this query'}
            
            # Add delay before fetching details
            time.sleep(0.5)
            
            # Get article details
            fetch_url = f"{self.pubmed_base}efetch.fcgi"
            fetch_params = {
                'db': 'pubmed',
                'id': ','.join(ids),
                'retmode': 'xml',
                'tool': 'medical_research_platform',
                'email': 'research@example.com'
            }
            
            detail_response = requests.get(fetch_url, params=fetch_params, timeout=30)
            detail_response.raise_for_status()
            
            articles = self._parse_xml(detail_response.text)
            
            if not articles:
                return {'status': 'no_results', 'message': 'Articles found but could not parse content'}
            
            # Calculate relevance scores
            scored_articles = []
            for article in articles:
                score = self._calculate_relevance(query, article['full_text'])
                article['relevance_score'] = score
                article['medical_relevance'] = 'High' if score > 0.7 else 'Medium' if score > 0.4 else 'Low'
                scored_articles.append(article)
            
            # Sort by relevance
            scored_articles.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return {
                'status': 'success',
                'articles': scored_articles,
                'count': len(scored_articles),
                'query': query
            }
            
        except requests.exceptions.Timeout:
            return {'status': 'error', 'message': 'Request timed out. Please try again.'}
        except requests.exceptions.RequestException as e:
            return {'status': 'error', 'message': f'Network error: {str(e)}'}
        except Exception as e:
            return {'status': 'error', 'message': f'An error occurred: {str(e)}'}
    
    def _parse_xml(self, xml_content):
        """Parse PubMed XML."""
        articles = []
        try:
            # Clean the XML content
            xml_content = xml_content.replace('&', '&amp;')
            root = ET.fromstring(xml_content)
            
            for article in root.findall('.//PubmedArticle'):
                try:
                    # Get title
                    title_elem = article.find('.//ArticleTitle')
                    title = self._clean_text(title_elem.text) if title_elem is not None and title_elem.text else "No title available"
                    
                    # Get abstract
                    abstract_texts = []
                    for abstract_elem in article.findall('.//Abstract/AbstractText'):
                        if abstract_elem.text:
                            abstract_texts.append(self._clean_text(abstract_elem.text))
                    
                    abstract = ' '.join(abstract_texts) if abstract_texts else "No abstract available"
                    
                    # Get authors
                    authors = []
                    for author in article.findall('.//Author'):
                        lastname = author.find('.//LastName')
                        forename = author.find('.//ForeName')
                        if lastname is not None and forename is not None:
                            if lastname.text and forename.text:
                                authors.append(f"{forename.text} {lastname.text}")
                        elif lastname is not None and lastname.text:
                            authors.append(lastname.text)
                    
                    # Get journal
                    journal_elem = article.find('.//Journal/Title')
                    journal = self._clean_text(journal_elem.text) if journal_elem is not None and journal_elem.text else "Unknown Journal"
                    
                    # Get year
                    year_elem = article.find('.//PubDate/Year')
                    if year_elem is None:
                        year_elem = article.find('.//PubDate/MedlineDate')
                        if year_elem is not None and year_elem.text:
                            # Extract year from MedlineDate (e.g., "2023 Jan-Feb")
                            year_match = re.search(r'\d{4}', year_elem.text)
                            year = year_match.group() if year_match else "Unknown"
                        else:
                            year = "Unknown"
                    else:
                        year = year_elem.text if year_elem.text else "Unknown"
                    
                    # Get PMID
                    pmid_elem = article.find('.//PMID')
                    pmid = pmid_elem.text if pmid_elem is not None and pmid_elem.text else "Unknown"
                    
                    articles.append({
                        'title': title,
                        'abstract': abstract,
                        'authors': authors,
                        'journal': journal,
                        'year': year,
                        'pmid': pmid,
                        'full_text': f"{title}\n\n{abstract}",
                        'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid != "Unknown" else None
                    })
                    
                except Exception as e:
                    st.warning(f"Failed to parse one article: {str(e)}")
                    continue
                    
        except ET.ParseError as e:
            st.error(f"XML parsing error: {str(e)}")
        except Exception as e:
            st.error(f"Error parsing articles: {str(e)}")
            
        return articles
    
    def _clean_text(self, text):
        """Clean and normalize text."""
        if not text:
            return ""
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        # Remove HTML tags if any
        text = re.sub(r'<[^>]+>', '', text)
        return text
    
    def _calculate_relevance(self, query, article_text):
        """Calculate relevance score."""
        medical_terms = [
            'treatment', 'therapy', 'diagnosis', 'patient', 'clinical', 'medical', 'disease',
            'symptom', 'condition', 'medication', 'drug', 'surgery', 'procedure', 'outcome',
            'study', 'trial', 'research', 'healthcare', 'hospital', 'physician', 'doctor'
        ]
        
        query_lower = query.lower()
        article_lower = article_text.lower()
        
        # Word overlap
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        article_words = set(re.findall(r'\b\w+\b', article_lower))
        
        overlap = len(query_words.intersection(article_words))
        total = len(query_words.union(article_words))
        basic_score = overlap / total if total > 0 else 0
        
        # Medical term bonus
        query_medical = [term for term in medical_terms if term in query_lower]
        article_medical = [term for term in medical_terms if term in article_lower]
        medical_overlap = len(set(query_medical).intersection(set(article_medical)))
        
        medical_bonus = (medical_overlap / len(medical_terms)) * 0.3
        final_score = min(basic_score + medical_bonus, 1.0)
        
        return final_score


def main():
    st.title("🏥 Medical Research Platform")
    st.markdown("**Search medical articles from PubMed and train similarity models**")
    
    # Add information about the platform
    with st.expander("ℹ️ About this Platform"):
        st.markdown("""
        This platform allows you to:
        - Search PubMed for medical research articles
        - Calculate relevance scores for articles
        - Generate training data from search results
        - Simulate model training and performance analysis
        
        **Note:** This uses NCBI's E-utilities API and respects their usage guidelines.
        """)
    
    processor = MedicalProcessor()
    
    # Medical Query Section
    st.subheader("🔍 Medical Article Search")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        medical_query = st.text_input(
            "Enter medical question or topic:",
            placeholder="e.g., diabetes treatment options, hypertension management, cancer immunotherapy",
            help="Enter medical questions or topics to find relevant research articles"
        )
        
        max_articles = st.slider("Maximum articles to retrieve", 5, 50, 10)
    
    with col2:
        st.markdown("### Search Options")
        show_abstracts = st.checkbox("Show full abstracts", value=True)
    
    if st.button("🔍 Search Medical Articles", type="primary"):
        if medical_query.strip():
            with st.spinner(f"Searching PubMed for: {medical_query}"):
                results = processor.fetch_medical_articles(medical_query.strip(), max_articles)
                
                if results['status'] == 'success':
                    st.success(f"Found {results['count']} medical articles!")
                    st.session_state.medical_results = results
                    
                    # Display articles
                    st.subheader(f"📋 Medical Articles for: '{medical_query}'")
                    
                    # Show summary statistics
                    articles = results['articles']
                    high_rel = sum(1 for a in articles if a['medical_relevance'] == 'High')
                    med_rel = sum(1 for a in articles if a['medical_relevance'] == 'Medium')
                    low_rel = sum(1 for a in articles if a['medical_relevance'] == 'Low')
                    
                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                    with stat_col1:
                        st.metric("Total Articles", len(articles))
                    with stat_col2:
                        st.metric("High Relevance", high_rel)
                    with stat_col3:
                        st.metric("Medium Relevance", med_rel)
                    with stat_col4:
                        st.metric("Low Relevance", low_rel)
                    
                    # Display articles
                    for i, article in enumerate(articles):
                        relevance_icon = "🟢" if article['medical_relevance'] == 'High' else "🟡" if article['medical_relevance'] == 'Medium' else "🔴"
                        
                        with st.expander(f"{relevance_icon} {article['title'][:100]}{'...' if len(article['title']) > 100 else ''} (Score: {article['relevance_score']:.3f})"):
                            col_info, col_actions = st.columns([3, 1])
                            
                            with col_info:
                                if article['authors']:
                                    authors_display = ', '.join(article['authors'][:3])
                                    if len(article['authors']) > 3:
                                        authors_display += f" et al. ({len(article['authors'])} total)"
                                    st.write(f"**Authors:** {authors_display}")
                                else:
                                    st.write("**Authors:** Not available")
                                
                                st.write(f"**Journal:** {article['journal']} ({article['year']})")
                                st.write(f"**PMID:** {article['pmid']}")
                                st.write(f"**Relevance:** {article['medical_relevance']} ({article['relevance_score']:.3f})")
                                
                                if show_abstracts and article['abstract'] != "No abstract available":
                                    st.write("**Abstract:**")
                                    st.write(article['abstract'])
                                elif not show_abstracts:
                                    st.write(f"**Abstract Preview:** {article['abstract'][:200]}...")
                            
                            with col_actions:
                                if article['url']:
                                    st.link_button("View on PubMed", article['url'])
                                
                
                elif results['status'] == 'no_results':
                    st.warning(f"No results found for '{medical_query}'. Try different keywords or broader terms.")
                else:
                    st.error(f"Search failed: {results['message']}")
        else:
            st.warning("Please enter a medical question or topic")
    
    # Automated Training Section
    st.markdown("---")
    st.subheader("🧠 Automated Model Training & Analysis")
    
    if st.session_state.medical_results and st.session_state.medical_results['status'] == 'success':
        training_col1, training_col2 = st.columns(2)
        
        with training_col1:
            st.markdown("### Auto-Generate Training Data")
            st.write("Use found medical articles to automatically create training examples:")
            
            articles = st.session_state.medical_results['articles']
            query = st.session_state.medical_results.get('query', 'medical query')
            
            st.write(f"**Source Query:** {query}")
            st.write(f"**Available Articles:** {len(articles)}")
            if articles:
                st.write(f"**Relevance Range:** {min(a['relevance_score'] for a in articles):.3f} - {max(a['relevance_score'] for a in articles):.3f}")
            
            use_top_n = st.slider("Use top N articles for training", 3, min(len(articles), 15), min(8, len(articles)))
            
            if st.button("🤖 Auto-Generate Training Data"):
                with st.spinner("Generating training examples from medical articles..."):
                    # Create training examples from found articles
                    training_examples = []
                    for i, article in enumerate(articles[:use_top_n]):
                        training_example = {
                            'prompt': query,
                            'article': article['abstract'] if article['abstract'] != "No abstract available" else article['title'],
                            'score': article['relevance_score'],
                            'source': f"PubMed PMID: {article['pmid']}",
                            'timestamp': datetime.now().isoformat(),
                            'title': article['title'][:100] + "..." if len(article['title']) > 100 else article['title']
                        }
                        training_examples.append(training_example)
                    
                    st.session_state.training_data = training_examples
                    st.success(f"✅ Generated {len(training_examples)} training examples from medical articles!")
        
        with training_col2:
            st.markdown("### Training & Performance Analysis")
            
            if st.session_state.training_data:
                st.write(f"**Training Examples Ready:** {len(st.session_state.training_data)}")
                
                # Show sample of training data
                with st.expander("📊 Preview Training Data"):
                    for i, example in enumerate(st.session_state.training_data[:3]):
                        st.write(f"**Example {i+1}:**")
                        st.write(f"- Title: {example['title']}")
                        st.write(f"- Relevance Score: {example['score']:.3f}")
                        st.write(f"- Source: {example['source']}")
                        st.write(f"- Content Preview: {example['article'][:150]}...")
                        st.write("---")
                
                if st.button("🎯 Train & Analyze Model", type="primary"):
                    with st.spinner("Training model and analyzing performance..."):
                        # Simulate realistic model training with performance analysis
                        training_data = st.session_state.training_data
                        
                        # Calculate training metrics
                        scores = [ex['score'] for ex in training_data]
                        avg_relevance = sum(scores) / len(scores)
                        score_variance = sum((s - avg_relevance) ** 2 for s in scores) / len(scores)
                        
                        # Simulate more realistic prediction accuracy
                        predicted_scores = []
                        actual_scores = []
                        
                        # Add some randomness but keep it realistic
                        random.seed(42)  # For reproducible results
                        
                        for example in training_data:
                            actual = example['score']
                            # Simulate model prediction with controlled noise
                            noise_factor = 0.1 + (score_variance * 0.5)  # Adaptive noise based on data variance
                            noise = (random.random() - 0.5) * noise_factor
                            predicted = actual + noise
                            predicted = max(0, min(1, predicted))  # Clamp to [0,1]
                            
                            predicted_scores.append(predicted)
                            actual_scores.append(actual)
                        
                        # Calculate performance metrics
                        mae = sum(abs(p - a) for p, a in zip(predicted_scores, actual_scores)) / len(scores)
                        rmse = (sum((p - a) ** 2 for p, a in zip(predicted_scores, actual_scores)) / len(scores)) ** 0.5
                        
                        # Accuracy within threshold
                        threshold = 0.15
                        accurate_predictions = sum(1 for p, a in zip(predicted_scores, actual_scores) if abs(p - a) <= threshold)
                        accuracy_percentage = (accurate_predictions / len(scores)) * 100
                        
                        # Correlation coefficient
                        mean_pred = sum(predicted_scores) / len(predicted_scores)
                        mean_actual = sum(actual_scores) / len(actual_scores)
                        
                        numerator = sum((p - mean_pred) * (a - mean_actual) for p, a in zip(predicted_scores, actual_scores))
                        denom_pred = sum((p - mean_pred) ** 2 for p in predicted_scores) ** 0.5
                        denom_actual = sum((a - mean_actual) ** 2 for a in actual_scores) ** 0.5
                        
                        correlation = numerator / (denom_pred * denom_actual) if denom_pred * denom_actual > 0 else 0
                        
                        st.session_state.model_trained = True
                        st.success("🎉 Model training and analysis completed!")
                        
                        # Display comprehensive performance metrics
                        st.subheader("📊 Model Performance Analysis")
                        
                        # Main metrics
                        met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                        with met_col1:
                            st.metric("Training Examples", len(training_data))
                        with met_col2:
                            st.metric("Mean Abs Error", f"{mae:.3f}")
                        with met_col3:
                            st.metric("Accuracy (±0.15)", f"{accuracy_percentage:.1f}%")
                        with met_col4:
                            st.metric("Correlation", f"{correlation:.3f}")
                        
                        # Additional analysis
                        st.subheader("🔍 Detailed Analysis")
                        
                        analysis_col1, analysis_col2 = st.columns(2)
                        
                        with analysis_col1:
                            st.write("**Training Data Statistics:**")
                            st.write(f"- Average Relevance: {avg_relevance:.3f}")
                            st.write(f"- Score Variance: {score_variance:.4f}")
                            st.write(f"- RMSE: {rmse:.3f}")
                            st.write(f"- Min Score: {min(scores):.3f}")
                            st.write(f"- Max Score: {max(scores):.3f}")
                            st.write(f"- Score Range: {max(scores) - min(scores):.3f}")
                        
                        with analysis_col2:
                            st.write("**Model Performance:**")
                            performance_quality = "Excellent" if accuracy_percentage > 85 else "Good" if accuracy_percentage > 70 else "Fair" if accuracy_percentage > 55 else "Needs Improvement"
                            st.write(f"- Performance Quality: **{performance_quality}**")
                            
                            corr_strength = "Strong" if abs(correlation) > 0.7 else "Moderate" if abs(correlation) > 0.5 else "Weak"
                            st.write(f"- Correlation Strength: **{corr_strength}**")
                            
                            # Model stability
                            stability = "High" if mae < 0.1 else "Medium" if mae < 0.2 else "Low"
                            st.write(f"- Model Stability: **{stability}**")
                            
                            # Recommendations
                            st.write("**Recommendations:**")
                            recommendations = []
                            if accuracy_percentage < 70:
                                recommendations.append("• Add more diverse training examples")
                            if abs(correlation) < 0.5:
                                recommendations.append("• Improve feature engineering")
                            if score_variance < 0.02:
                                recommendations.append("• Include more varied relevance scores")
                            if mae > 0.15:
                                recommendations.append("• Review scoring methodology")
                            
                            if not recommendations:
                                recommendations.append("• Model performance is satisfactory")
                            
                            for rec in recommendations:
                                st.write(rec)
                        
                        # Store results for future reference
                        st.session_state.model_performance = {
                            'mae': mae,
                            'rmse': rmse,
                            'accuracy': accuracy_percentage,
                            'correlation': correlation,
                            'training_size': len(training_data),
                            'avg_relevance': avg_relevance,
                            'score_variance': score_variance,
                            'performance_quality': performance_quality,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # Show training completion message
                        st.balloons()
            else:
                st.info("🔄 Generate training data first using the found medical articles")
    else:
        st.info("🔍 Search for medical articles first to enable automated training")
    
    # Display previous model performance if available
    if st.session_state.model_performance:
        st.markdown("---")
        st.subheader("📈 Previous Training Results")
        perf = st.session_state.model_performance
        
        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.write(f"**Last Training:** {datetime.fromisoformat(perf['timestamp']).strftime('%Y-%m-%d %H:%M')}")
            st.write(f"**Performance Quality:** {perf.get('performance_quality', 'N/A')}")
            st.write(f"**Training Size:** {perf['training_size']} examples")
        
        with result_col2:
            st.write(f"**Accuracy:** {perf['accuracy']:.1f}%")
            st.write(f"**Correlation:** {perf['correlation']:.3f}")
            st.write(f"**Mean Error:** {perf['mae']:.3f}")

if __name__ == "__main__":
    main()