import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from docx import Document
import re
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Load spaCy model
nlp = spacy.load('/Users/u1984485/miniconda3/lib/python3.12/site-packages/en_core_web_sm/en_core_web_sm-3.8.0')

# Initialize tools
stop_words = set(stopwords.words('english'))
sentiment_analyzer = SentimentIntensityAnalyzer()

# Custom stopwords and keywords
custom_stopwords = {"please", "thank", "regards", "dear", "sir", "madam", "etc", "also", "however", "therefore"}
focus_keywords = {"violence", "abuse", "discrimination", "women", "children", "safety", "fear", "harassment",
                  "exploitation", "gender", "victim", "assault", "threat", "disabled", "disability", "vulnerable"}

# Function: Text Preprocessing
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
    text = re.sub(r'\s+', ' ', text).strip().lower()  # Normalize spaces and lowercase
    doc = nlp(text)
    clean_tokens = [
        token.lemma_ for token in doc
        if token.is_alpha and 
           token.text not in stop_words and 
           token.text not in custom_stopwords and
           (token.text in focus_keywords or len(token.text) > 3)
    ]
    return " ".join(clean_tokens)

# Function: Extract Named Entities
def extract_named_entities(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in {"PERSON", "GPE", "ORG", "EVENT"}]
    return entities

# Function: Generate Word Cloud
def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.show()

# Function: Topic Modeling
def perform_topic_modeling(corpus, n_topics=3, n_words=5):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    words = vectorizer.get_feature_names_out()
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        topic_words = [words[i] for i in topic.argsort()[:-n_words - 1:-1]]
        topics[f"Topic {topic_idx + 1}"] = topic_words
    return topics


# Function: Sentiment Analysis
def perform_sentiment_analysis(text):
    return sentiment_analyzer.polarity_scores(text)['compound']

# Function: Emotion Analysis
def perform_emotion_analysis(text):
    # Placeholder for more advanced emotion detection
    emotions = {
        "Anger": 0,
        "Fear": 0,
        "Joy": 0,
        "Sadness": 0,
        "Disgust": 0,
    }
    doc = nlp(text)
    for token in doc:
        if token.text.lower() in emotions:
            emotions[token.text.lower()] += 1
    return max(emotions, key=emotions.get)

def visualize_emotion_distribution(emotions):
    emotion_counts = Counter(emotions)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(emotion_counts.keys()), y=list(emotion_counts.values()), palette="coolwarm")
    plt.title("Emotion Distribution Across Documents")
    plt.xlabel("Emotions")
    plt.ylabel("Frequency")
    plt.show()

# Function: Dependency Parsing and Relationship Extraction
def extract_relationships(text):
    doc = nlp(text)
    relationships = []
    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            relationships.append((token.text, token.head.text, [child.text for child in token.head.children]))
    return relationships

def visualize_relationships(relationships):
    G = nx.DiGraph()
    for subj, verb, objs in relationships:
        G.add_node(subj, color="blue")
        G.add_node(verb, color="red")
        for obj in objs:
            G.add_node(obj, color="green")
            G.add_edge(verb, obj)
        G.add_edge(subj, verb)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color="skyblue")
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    plt.title("Relationship Network")
    plt.show()

# Function: Temporal Analysis
def extract_dates(text):
    doc = nlp(text)
    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    return dates

def visualize_dates(dates):
    date_counts = Counter(dates)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(date_counts.keys()), y=list(date_counts.values()), palette="viridis")
    plt.title("Frequency of Mentioned Dates")
    plt.xticks(rotation=45, ha="right")
    plt.show()

# Function: Clustering Similar Experiences
def cluster_documents(corpus, n_clusters=3):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    return clusters

def visualize_clusters(corpus, clusters):
    from sklearn.manifold import TSNE
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus).toarray()
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(X)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=clusters, palette="tab10")
    plt.title("Clustering of Documents")
    plt.show()

# Function: Semantic Similarity Analysis
def semantic_similarity(corpus):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(corpus)
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix

def visualize_similarity(similarity_matrix, corpus):
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, xticklabels=corpus, yticklabels=corpus, cmap="coolwarm", annot=False)
    plt.title("Document Similarity Heatmap")
    plt.show()
    
# Function: Visualize Sentiment Score Distribution
def visualize_sentiment_distribution(all_sentiments):
    plt.figure(figsize=(10, 6))
    sns.histplot(all_sentiments, kde=True, color="skyblue")
    plt.title("Sentiment Score Distribution Across All Documents")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.show()


# Function: Active vs Passive Voice Analysis
def analyze_voice(text):
    doc = nlp(text)
    active, passive = 0, 0
    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            active += 1
        elif token.dep_ == "nsubjpass":
            passive += 1
    return active, passive

def visualize_voice_analysis(active, passive):
    plt.figure(figsize=(6, 4))
    sns.barplot(x=["Active", "Passive"], y=[active, passive], palette="magma")
    plt.title("Active vs Passive Voice Usage")
    plt.ylabel("Frequency")
    plt.show()

# Function: Visualize Topics
def visualize_topics(topics):
    plt.figure(figsize=(12, 8))
    for topic, words in topics.items():
        plt.barh(words, range(len(words)), label=topic)
    plt.gca().invert_yaxis()
    plt.title("Topics and Key Words")
    plt.xlabel("Words")
    plt.legend()
    plt.show()

# Function: Polarity of Relationships
def polarity_relationships(text):
    doc = nlp(text)
    polarities = []
    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            sentiment = sentiment_analyzer.polarity_scores(token.head.text)['compound']
            polarities.append((token.text, token.head.text, sentiment))
    return polarities

def visualize_polarity(polarities):
    df = pd.DataFrame(polarities, columns=["Subject", "Action", "Polarity"])
    sns.barplot(data=df, x="Action", y="Polarity", hue="Subject", palette="viridis")
    plt.title("Polarity of Actions by Subjects")
    plt.show()

# Function: Social Network Analysis
def visualize_social_network(relationships):
    G = nx.DiGraph()
    for subj, verb, objs in relationships:
        G.add_edge(subj, verb)
        for obj in objs:
            G.add_edge(verb, obj)
    plt.figure(figsize=(12, 10))
    nx.draw_networkx(G, with_labels=True, node_size=5000, node_color="lightblue", font_size=10)
    plt.title("Social Network of Relationships")
    plt.show()

# Function: Bias or Discrimination Sentiment Tagging
def tag_bias_discrimination(text):
    bias_keywords = ["discrimination", "bias", "prejudice", "stereotype", "inequality"]
    doc = nlp(text)
    tagged_sentences = []
    for sentence in doc.sents:
        if any(keyword in sentence.text.lower() for keyword in bias_keywords):
            sentiment = sentiment_analyzer.polarity_scores(sentence.text)['compound']
            tagged_sentences.append((sentence.text, sentiment))
    return tagged_sentences

def visualize_bias_tags(tagged_sentences):
    sentiments = [sent[1] for sent in tagged_sentences]
    plt.figure(figsize=(8, 6))
    sns.histplot(sentiments, kde=True, bins=20, color="purple")
    plt.title("Sentiment Distribution for Bias/Discrimination Sentences")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.show()

# Function: Comparative Analysis of Subgroups
def analyze_subgroups(corpus, metadata):
    # Metadata could include attributes like location, age, or disability type
    subgroup_results = {}
    for subgroup, texts in metadata.items():
        combined_text = " ".join(texts)
        sentiment = perform_sentiment_analysis(combined_text)
        subgroup_results[subgroup] = sentiment
    return subgroup_results

def visualize_subgroup_analysis(subgroup_results):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(subgroup_results.keys()), y=list(subgroup_results.values()), palette="coolwarm")
    plt.title("Sentiment Analysis by Subgroup")
    plt.xlabel("Subgroup")
    plt.ylabel("Average Sentiment Score")
    plt.show()

# Main Function: Process Word Documents
def process_documents(input_folder, output_csv):
    results = []
    all_clean_texts = []
    all_sentiments = []
    all_emotions = []
    all_entities = []
    all_relationships = []
    all_dates = []
    all_topics = {}
    tagged_bias_sentences = []

    # Example metadata structure for subgroup analysis
    metadata = {"Disability Type": {"Physical": [], "Cognitive": []}}

    for filename in os.listdir(input_folder):
        if filename.endswith(".docx") and not filename.startswith("~$"):
            file_path = os.path.join(input_folder, filename)
            print(f"Processing {filename}...")
            doc = Document(file_path)
            text = " ".join([para.text for para in doc.paragraphs])

            # Preprocessing
            clean_text = preprocess_text(text)
            all_clean_texts.append(clean_text)

            # Sentiment Analysis
            sentiment = perform_sentiment_analysis(text)
            all_sentiments.append(sentiment)

            # Emotion Analysis
            emotion = perform_emotion_analysis(text)
            all_emotions.append(emotion)

            # Named Entity Recognition
            entities = extract_named_entities(text)
            all_entities.extend(entities)

            # Relationship Extraction
            relationships = extract_relationships(text)
            all_relationships.extend(relationships)

            # Temporal Analysis
            dates = extract_dates(text)
            all_dates.extend(dates)

            # Topic Modeling
            topics = perform_topic_modeling([clean_text], n_topics=3)
            for topic_name, words in topics.items():
                if topic_name not in all_topics:
                    all_topics[topic_name] = []
                all_topics[topic_name].extend(words)

            # Bias or Discrimination Sentiment Tagging
            tagged_bias = tag_bias_discrimination(text)
            tagged_bias_sentences.extend(tagged_bias)

            # Metadata Analysis (e.g., Disability Type)
            # Placeholder: Append the text to a subgroup based on custom metadata logic
            if "disability" in text.lower():
                metadata["Disability Type"]["Physical"].append(text)

            # Collect results
            results.append({
                "Filename": filename,
                "Sentiment Score": sentiment,
                "Emotion": emotion,
                "Named Entities": ", ".join(entities),
                "Topics": "; ".join([f"{k}: {', '.join(v)}" for k, v in topics.items()]),
                "Dates": ", ".join(dates)
            })

    # Save results to CSV
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

    # Aggregated Visualizations
    combined_text = " ".join(all_clean_texts)

    # Generate visualizations for each technique
    generate_wordcloud(combined_text, "Combined Word Cloud of Relevant Content")
    visualize_sentiment_distribution(all_sentiments)
    visualize_emotion_distribution(all_emotions)
    visualize_topics(all_topics)
    visualize_relationships(all_relationships)
    visualize_dates(all_dates)
    visualize_bias_tags(tagged_bias_sentences)
    visualize_social_network(all_relationships)
    subgroup_results = analyze_subgroups(all_clean_texts, metadata)
    visualize_subgroup_analysis(subgroup_results)

    print("Pipeline complete!")

# Run Script
if __name__ == "__main__":
    input_folder = "/Users/u1984485/Documents/BE/NLP/word_documents"  # Path to folder with Word files
    output_csv = "processed_results2.csv"
    process_documents(input_folder, output_csv)