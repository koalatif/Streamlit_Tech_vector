import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import streamlit as st

# Vérification des dépendances optionnelles
OPTIONAL_DEPS = {
    'sentence_transformers': False,
    'gensim': False
}

try:
    import gensim
    from gensim.models import Word2Vec
    OPTIONAL_DEPS['gensim'] = True
except ImportError:
    logging.warning("Gensim non installé - Word2Vec désactivé")

try:
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
    OPTIONAL_DEPS['sentence_transformers'] = True
except ImportError:
    logging.warning("sentence-transformers non installé - BERT désactivé")

# Téléchargement des ressources NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class TextPreprocessor:
    """Classe pour le prétraitement des textes"""
    
    def __init__(self, language='french'):
        self.language = language
        try:
            self.stop_words = set(stopwords.words(language))
        except:
            self.stop_words = set()
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess_text(self, text):
        """Nettoie et prétraite un texte"""
        if not text or not isinstance(text, str):
            return ""
        
        # Nettoyage basique
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenization et filtrage
        tokens = text.split()
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Lemmatisation
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)

def validate_texts(texts):
    """Valide et nettoie une liste de textes"""
    if not texts:
        return []
    
    validated = []
    for i, text in enumerate(texts):
        if isinstance(text, str) and text.strip():
            validated.append(text.strip())
        else:
            logging.warning(f"Texte invalide ignoré à l'index {i}")
    
    return validated

class AdvancedWebScraper:
    """Classe améliorée pour le scraping web avec gestion des erreurs et cache"""
    
    def __init__(self, max_workers=5, use_cache=True):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        
        self.max_workers = max_workers
        self.preprocessor = TextPreprocessor()
    
    def _is_valid_article_url(self, url):
        """Vérifie si l'URL correspond à un article"""
        article_patterns = [
            r'article', r'news', r'actu', r'blog',
            r'\d{4}/\d{2}/\d{2}', r'\d{4}-\d{2}-\d{2}'
        ]
        return any(re.search(pattern, url, re.IGNORECASE) for pattern in article_patterns)
    
    def extract_urls_from_links(self, base_url):
        """Extrait les URLs en parcourant les liens de la page d'accueil"""
        try:
            response = self.session.get(base_url, timeout=10, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            urls = set()
            for link in soup.find_all('a', href=True):
                href = urljoin(base_url, link['href'])
                if self._is_valid_article_url(href) and urlparse(href).netloc == urlparse(base_url).netloc:
                    urls.add(href)
            
            return list(urls)
        except Exception as e:
            logging.error(f"Erreur extraction URLs depuis liens: {e}")
            return []
    
    def extract_article_content(self, url):
        """Extrait le contenu d'un article"""
        try:
            response = self.session.get(url, timeout=10, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Suppression des éléments non désirés
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            title = soup.find('title')
            title_text = title.get_text(strip=True) if title else 'Sans titre'
            
            # Extraction du contenu principal
            content_selectors = [
                'article', '.article-content', '.post-content', 
                '.entry-content', 'main', '.content', '.post'
            ]
            
            content_text = ''
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content_text = ' '.join([elem.get_text(strip=True) for elem in elements])
                    break
            
            if not content_text:
                # Fallback: tous les paragraphes
                paragraphs = soup.find_all('p')
                content_text = ' '.join(p.get_text(strip=True) for p in paragraphs)
            
            return {
                'url': url,
                'title': title_text,
                'text': content_text,
                'publish_date': None,
                'authors': []
            }
            
        except Exception as e:
            logging.error(f"Erreur extraction {url}: {e}")
            return None
    
    def scrape_articles(self, base_url, max_articles=20):
        """Scrape les articles d'un site web"""
        logging.info(f"Scraping de {base_url}...")
        
        urls = self.extract_urls_from_links(base_url)
        
        if not urls:
            logging.warning("Aucune URL d'article trouvée")
            return []
        
        logging.info(f"{len(urls)} URLs trouvées, extraction du contenu...")
        
        articles = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {executor.submit(self.extract_article_content, url): url 
                           for url in urls[:max_articles]}
            
            for future in as_completed(future_to_url):
                result = future.result()
                if result and result['text'] and len(result['text']) > 100:  # Filtrer les textes trop courts
                    # Prétraitement du texte
                    result['processed_text'] = self.preprocessor.preprocess_text(result['text'])
                    articles.append(result)
                
                time.sleep(0.5)  # Respect politeness
        
        logging.info(f"{len(articles)} articles extraits avec succès")
        return articles

class EnhancedVectorizationComparator:
    """Classe améliorée pour la comparaison des techniques de vectorisation"""
    
    def __init__(self):
        self.bert_model = None
        self.preprocessor = TextPreprocessor()
    
    def bow_vectorization(self, texts, max_features=1000):
        """Vectorisation Bag of Words"""
        try:
            texts = validate_texts(texts)
            if not texts:
                raise ValueError("Aucun texte valide pour la vectorisation BoW")
            
            vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, 2))
            vectors = vectorizer.fit_transform(texts)
            return vectors, vectorizer
        except Exception as e:
            logging.error(f"Erreur BoW: {e}")
            return None, None
    
    def tfidf_vectorization(self, texts, max_features=1000):
        """Vectorisation TF-IDF"""
        try:
            texts = validate_texts(texts)
            if not texts:
                raise ValueError("Aucun texte valide pour la vectorisation TF-IDF")
            
            vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
            vectors = vectorizer.fit_transform(texts)
            return vectors, vectorizer
        except Exception as e:
            logging.error(f"Erreur TF-IDF: {e}")
            return None, None
    
    def word2vec_vectorization(self, texts):
        """Vectorisation Word2Vec avec Gensim"""
        try:
            if not OPTIONAL_DEPS['gensim']:
                logging.error("Gensim non disponible pour Word2Vec")
                return None, None
            
            texts = validate_texts(texts)
            if not texts:
                raise ValueError("Aucun texte valide pour la vectorisation Word2Vec")

            # Tokenization
            tokenized_texts = [text.lower().split() for text in texts]
            
            # Filtrer les listes vides
            tokenized_texts = [tokens for tokens in tokenized_texts if tokens]
            
            if not tokenized_texts:
                raise ValueError("Aucun texte tokenisé valide.")

            # Modèle Word2Vec avec paramètres par défaut
            model = Word2Vec(
                sentences=tokenized_texts,
                vector_size=100,
                window=5,
                min_count=1,
                workers=4,
                epochs=10
            )

            # Vectorisation moyenne des documents
            document_vectors = []
            for tokens in tokenized_texts:
                vectors = [model.wv[word] for word in tokens if word in model.wv.key_to_index]
                doc_vec = np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)
                document_vectors.append(doc_vec)

            return np.array(document_vectors), model

        except Exception as e:
            logging.error(f"Erreur Word2Vec: {e}")
            return None, None
    
    def bert_vectorization(self, texts):
        """Vectorisation BERT"""
        try:
            if not OPTIONAL_DEPS['sentence_transformers']:
                logging.error("sentence-transformers non disponible pour BERT")
                return None, None
                
            texts = validate_texts(texts)
            if not texts:
                raise ValueError("Aucun texte valide pour la vectorisation BERT")
            
            if self.bert_model is None:
                self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Limiter la taille des textes pour BERT
            truncated_texts = [text[:512] for text in texts]
            vectors = self.bert_model.encode(truncated_texts, show_progress_bar=False)
            return vectors, self.bert_model
            
        except Exception as e:
            logging.error(f"Erreur BERT: {e}")
            return None, None
    
    def calculate_similarity(self, texts, keywords, method='tfidf'):
        """Calcule la similarité entre les textes et les mots-clés"""
        if not texts or not keywords:
            return np.array([])
        
        if isinstance(keywords, str):
            keywords = [keywords]
        
        processed_keywords = [self.preprocessor.preprocess_text(kw) for kw in keywords]
        combined_keywords = ' '.join(processed_keywords)
        
        try:
            if method.lower() == 'bow':
                doc_vectors, vectorizer = self.bow_vectorization(texts)
                if doc_vectors is None:
                    return np.array([])
                kw_vectors = vectorizer.transform([combined_keywords])
            
            elif method.lower() == 'tfidf':
                doc_vectors, vectorizer = self.tfidf_vectorization(texts)
                if doc_vectors is None:
                    return np.array([])
                kw_vectors = vectorizer.transform([combined_keywords])
            
            elif method.lower() == 'word2vec':
                doc_vectors, model = self.word2vec_vectorization(texts)
                if doc_vectors is None:
                    return np.array([])
                
                # Vectoriser les mots-clés avec le même modèle Word2Vec
                kw_tokens = combined_keywords.lower().split()
                kw_vectors_list = []
                for word in kw_tokens:
                    if word in model.wv.key_to_index:
                        kw_vectors_list.append(model.wv[word])
                
                if kw_vectors_list:
                    kw_vectors = np.mean(kw_vectors_list, axis=0).reshape(1, -1)
                else:
                    kw_vectors = np.zeros((1, model.vector_size))
            
            elif method.lower() == 'bert':
                doc_vectors, model = self.bert_vectorization(texts)
                if doc_vectors is None:
                    return np.array([])
                kw_vectors = model.encode([combined_keywords], show_progress_bar=False)
            
            else:
                logging.error(f"Méthode inconnue: {method}")
                return np.array([])
            
            # Calcul de la similarité cosinus
            similarities = cosine_similarity(doc_vectors, kw_vectors)
            
            # Retourner les scores de similarité (colonne unique)
            return similarities.flatten()
        
        except Exception as e:
            logging.error(f"Erreur dans calculate_similarity pour {method}: {e}")
            return np.array([])
    
    def extract_keywords_from_texts(self, texts, top_n=10, language='french'):
        """Extraction automatique de mots-clés à partir d'un corpus de textes"""
        try:
            texts = validate_texts(texts)
            if not texts:
                return []
            
            # Utiliser les mots vides français si disponibles
            stop_words = None
            try:
                stop_words = list(stopwords.words('french'))
            except:
                stop_words = None
                
            vectorizer = TfidfVectorizer(
                stop_words=stop_words, 
                max_features=5000, 
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            tfidf = vectorizer.fit_transform(texts)
            scores = tfidf.mean(axis=0).A1
            terms = vectorizer.get_feature_names_out()
            scored = [(terms[i], scores[i]) for i in range(len(terms))]
            scored.sort(key=lambda x: x[1], reverse=True)
            keywords = [t for t, s in scored if len(t) > 2][:top_n]
            return keywords
        except Exception as e:
            logging.error(f"Erreur extraction automatique de keywords: {e}")
            return []

    def compare_all_methods(self, texts, keywords, top_k=10):
        """Compare toutes les méthodes de vectorisation disponibles"""
        
        # Validation des entrées
        texts = validate_texts(texts)
        if not texts:
            logging.error("Aucun texte fourni pour l'analyse")
            return pd.DataFrame(), {}
        
        if not keywords:
            logging.warning("Aucun mot-clé fourni, utilisation de mots par défaut")
            keywords = ["texte", "document"]
        
        # Méthodes disponibles selon les dépendances
        methods = ['bow', 'tfidf']
        if OPTIONAL_DEPS['gensim']:
            methods.append('word2vec')
        if OPTIONAL_DEPS['sentence_transformers']:
            methods.append('bert')
        
        results = {}
        
        for method in methods:
            try:
                similarities = self.calculate_similarity(texts, keywords, method=method)
                if len(similarities) > 0 and len(similarities) == len(texts):
                    results[method.upper()] = similarities
                else:
                    logging.warning(f"Résultats invalides pour {method}: longueur {len(similarities)} vs {len(texts)} textes")
            except Exception as e:
                logging.error(f"Erreur avec {method}: {e}")
        
        # Filtrer les méthodes qui ont échoué
        successful_methods = {k: v for k, v in results.items() if len(v) > 0}
        
        if not successful_methods:
            logging.error("Aucune méthode n'a fonctionné")
            return pd.DataFrame(), {}
        
        # Création du DataFrame de résultats
        df_results = pd.DataFrame(successful_methods)
        if texts:
            df_results['Text'] = [text[:100] + '...' for text in texts]
        
        # Classement par méthode
        rankings = {}
        for method in successful_methods.keys():
            if method in df_results.columns:
                df_sorted = df_results.nlargest(top_k, method)
                rankings[method] = df_sorted[['Text', method]]
        
        return df_results, rankings

def initialize_session_state():
    """Initialise les variables de session"""
    if 'articles' not in st.session_state:
        st.session_state.articles = []
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'use_demo' not in st.session_state:
        st.session_state.use_demo = False
    if 'example_data' not in st.session_state:
        # Données de démonstration
        st.session_state.example_data = [
            "L'éducation est fondamentale pour le développement de la société. Les écoles doivent offrir une formation de qualité.",
            "La formation professionnelle permet aux étudiants d'acquérir les compétences nécessaires pour le marché du travail.",
            "Le système éducatif doit s'adapter aux nouvelles technologies pour préparer les élèves au futur.",
            "Les enseignants jouent un rôle crucial dans l'apprentissage et la formation des jeunes générations.",
            "L'accès à une éducation de qualité est un droit fondamental pour tous les enfants."
        ]

def main():
    """Fonction principale de l'application Streamlit"""
    st.set_page_config(
        page_title="Comparaison des Techniques de Vectorisation",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Comparaison des Techniques de Vectorisation")
    st.markdown("""
    **Projet NLP - Comparaison BoW, TF-IDF, Word2Vec et BERT**
    """)
    
    # Initialisation des variables de session
    initialize_session_state()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    website_url = st.sidebar.text_input("URL du site web", "https://lefaso.net")
    max_articles = st.sidebar.slider("Nombre maximum d'articles", 5, 30, 10)
    keywords_input = st.sidebar.text_area("Mots-clés (un par ligne)", "éducation\nformation\nécole")
    top_k = st.sidebar.slider("Nombre de top résultats à afficher", 3, 15, 5)
    
    # Option: génération automatique des mots-clés
    st.sidebar.markdown("---")
    auto_keywords_checkbox = st.sidebar.checkbox("Générer automatiquement les mots-clés à partir des articles", value=True)
    auto_top_n = st.sidebar.slider("Top N mots-clés auto générés", 5, 20, 10, step=1)
    
  
        
    if st.sidebar.button("Lancer l'analyse complète"):
        st.session_state.use_demo = False
        with st.spinner("Scraping des articles en cours... Cela peut prendre quelques minutes."):
            try:
                scraper = AdvancedWebScraper()
                articles = scraper.scrape_articles(website_url, max_articles)
                st.session_state.articles = articles
                st.session_state.analysis_done = True
                st.rerun()
            except Exception as e:
                st.error(f"Erreur lors du scraping: {e}")
                st.session_state.analysis_done = False
    
    # Affichage des résultats
    if st.session_state.analysis_done:
        
        if st.session_state.use_demo:
            st.info("Mode démonstration activé")
            articles = []
            for i, text in enumerate(st.session_state.example_data):
                articles.append({
                    'url': f"demo_{i+1}",
                    'title': f"Document de démonstration {i+1}",
                    'text': text,
                    'processed_text': TextPreprocessor().preprocess_text(text)
                })
        else:
            articles = st.session_state.articles
        
        if not articles:
            st.error("Aucun article trouvé. Vérifiez l'URL ou essayez un autre site.")
            return
        
        st.success(f"{len(articles)} documents analysés avec succès!")
        
        # Affichage des articles
        st.subheader("Contenu analysé")
        for i, article in enumerate(articles):
            with st.expander(f"Document {i+1}: {article['title']}"):
                st.write(f"**Source:** {article['url']}")
                st.write(f"**Extrait:** {article['text'][:200]}...")
                if 'processed_text' in article and article['processed_text']:
                    st.write(f"**Texte prétraité:** {article['processed_text'][:150]}...")
        
        # Traitement des mots-clés
        user_keywords = [kw.strip() for kw in keywords_input.split('\n') if kw.strip()]
        st.write(f"**Mots-clés fournis par l'utilisateur:** {', '.join(user_keywords) if user_keywords else 'Aucun'}")
        
        comparator = EnhancedVectorizationComparator()
        
        # Définition des mots-clés à utiliser
        keywords = []
        if auto_keywords_checkbox:
            # Extraction automatique à partir des textes pré-traités
            texts_for_keywords = [article.get('processed_text', '') for article in articles if article.get('processed_text')]
            if texts_for_keywords:
                auto_keywords = comparator.extract_keywords_from_texts(
                    texts_for_keywords, top_n=auto_top_n, language='french'
                )
                st.write(f"**Mots-clés auto-générés:** {', '.join(auto_keywords)}")
                keywords.extend(auto_keywords)
        
        # Ajouter les mots-clés utilisateur
        keywords.extend(user_keywords)
        
        if not keywords:
            st.warning("Aucun mot-clé détecté ou fourni. L'analyse risque d'être faible.")
            keywords = ["éducation", "formation"]  # Mots-clés par défaut
        
        st.write(f"Mots-clés utilisés pour l'analyse:** {', '.join(keywords)}")
        
        with st.spinner("Calcul des similarités en cours..."):
            texts = [article.get('processed_text', '') for article in articles if article.get('processed_text')]
            if not texts:
                st.error("Aucun texte prétraité disponible pour l'analyse.")
                return
                
            df_results, rankings = comparator.compare_all_methods(texts, keywords, top_k)
        
        # Vérifier si des résultats sont disponibles
        if df_results.empty:
            st.error("Aucun résultat de similarité n'a pu être calculé. Vérifiez les dépendances.")
            st.info("Conseil: Installez sentence-transformers avec `pip install sentence-transformers` pour BERT")
            return
        
        # Résultats globaux
        st.subheader("Résultats de similarité")
        
        # Ajout des titres pour faciliter l'analyse
        df_display = df_results.copy()
        df_display['Titre'] = [article['title'][:50] + '...' for article in articles[:len(df_results)]]
        
        # Réorganiser les colonnes
        cols = ['Titre'] + [col for col in df_display.columns if col != 'Titre']
        df_display = df_display[cols]
        
        st.dataframe(df_display.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
        
        # Visualisation
        st.subheader("Visualisation des similarités")
        
        # Méthodes disponibles dans les résultats
        available_methods = [col for col in df_results.columns if col != 'Text']
        
        if available_methods:
            n_methods = len(available_methods)
            n_cols = min(2, n_methods)
            n_rows = (n_methods + 1) // 2
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_methods == 1:
                axes = [axes]
            elif n_rows > 1:
                axes = axes.flatten()
            elif n_cols > 1:
                axes = list(axes)
            else:
                axes = [axes]
            
            for i, method in enumerate(available_methods):
                if i < len(axes):
                    axes[i].bar(range(len(df_results)), df_results[method], color='skyblue', alpha=0.7)
                    axes[i].set_title(f'Similarité - {method}', fontsize=12, fontweight='bold')
                    axes[i].set_ylabel('Score de similarité')
                    axes[i].set_xlabel('Documents')
                    axes[i].tick_params(axis='x', rotation=45)
                    axes[i].grid(axis='y', alpha=0.3)
            
            # Cacher les axes vides
            for i in range(len(available_methods), len(axes)):
                if i < len(axes):
                    axes[i].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Aucune méthode disponible pour la visualisation")
        
        # Top résultats par méthode
        if rankings:
            st.subheader("Top résultats par méthode de vectorisation")
            
            for method, df_rank in rankings.items():
                st.markdown(f"**{method} - Top {top_k}:**")
                st.dataframe(df_rank, use_container_width=True)
        
        # Analyse comparative
        st.subheader("Analyse comparative des méthodes")
        
        if available_methods:
            # Meilleure méthode en moyenne
            avg_scores = df_results[available_methods].mean()
            best_method = avg_scores.idxmax()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Meilleure méthode", best_method)
            with col2:
                st.metric("Score moyen", f"{avg_scores[best_method]:.3f}")
            with col3:
                st.metric("Documents analysés", len(articles))
            
            # Tableau comparatif
            st.write("**Scores moyens par méthode:**")
            avg_df = pd.DataFrame({
                'Méthode': available_methods,
                'Score Moyen': [avg_scores[method] for method in available_methods]
            }).sort_values('Score Moyen', ascending=False)
            st.dataframe(avg_df, use_container_width=True)
        else:
            st.warning("Aucune méthode disponible pour l'analyse comparative")
        
        # Téléchargement des résultats
        st.subheader("Téléchargement des résultats")
        csv = df_display.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="Télécharger les résultats (CSV)",
            data=csv,
            file_name="resultats_vectorisation.csv",
            mime="text/csv"
        )
    
    else:
        # Page d'accueil
        st.markdown("""                  
    """)
        

if __name__ == "__main__":
    main()

                    