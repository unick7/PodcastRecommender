import pickle
import streamlit as st
import pandas as pd
import numpy as np
import webbrowser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from gensim.models import Word2Vec
import itertools

st.set_page_config(layout="wide", initial_sidebar_state="collapsed",
                   page_title="Podcast Recommender", page_icon=":headphones:")

key_counter = itertools.count()


def load_data():
    podcasts_df = pd.read_csv("data/podcasts_with_images.csv")
    vectorizer = pickle.load(open("pickles/tfidf_vectorizer.pkl", "rb"))

    return podcasts_df, vectorizer


def get_recommendations(podcasts_df, vectorizer, podcast_title):
    # Get the index of the podcast title
    index = podcasts_df[podcasts_df.title == podcast_title].index[0]

    # Get the TF-IDF matrix for the podcasts
    tfidf_matrix = vectorizer.transform(podcasts_df['text'])

    # Calculate the cosine similarity between the given podcast and all other podcasts
    similarities = cosine_similarity(tfidf_matrix[index], tfidf_matrix)

    # Sort the indices of the podcasts based on the cosine similarity
    sorted_indices = np.argsort(similarities[0])[::-1]

    # Get the top 10 podcasts (excluding the given podcast)
    recommendations = []
    for i in sorted_indices:
        if i != index:
            recommendation = podcasts_df.iloc[i][[
                'title', 'link', 'rating', 'image_url', 'text']].to_dict()
            recommendations.append(recommendation)
        if len(recommendations) == 8:
            break

    return recommendations


podcasts_df, vectorizer = load_data()

image = './data/images/POD.png'
st.image(image, width=600)
st.markdown("<p style='font-family:monospace'>This is a content-based recommender system for podcasts.<br> It recommends podcasts based on their textual description and similarity to the selected podcast.</p>", unsafe_allow_html=True)

# Inject custom CSS to style podcast names
st.markdown("""
<style>
.podcast-name {
    display: inline-block;
    border: 2px solid #FFE026;
    padding: 4px 8px;
    border-radius: 4px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='font-family:monospace; font-size: 24px; font-weight: bold;'>Search using keyword:</h2>", unsafe_allow_html=True)


# Take keyword input from user
keyword = st.text_input("Enter a keyword to get podcast recommendations:")


def get_recommendations2(keyword):
    recommendations = []
    # create a TfidfVectorizer object and fit it to the podcast titles and descriptions
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(
        podcasts_df['title'] + ' ' + podcasts_df['text'])

    # transform the user input into a vector using the fitted vectorizer
    user_vector = tfidf_vectorizer.transform([keyword])

    # calculate the cosine similarity between the user vector and all the podcast vectors
    cosine_similarities = cosine_similarity(user_vector, tfidf).flatten()

    # get the indices of the top 10 podcasts with the highest cosine similarity scores
    top_podcasts_idx = cosine_similarities.argsort()[:-13:-1]

    # get the corresponding podcast titles and descriptions
    recommended_podcasts = podcasts_df.iloc[top_podcasts_idx][[
        'title', 'link', 'rating', 'image_url', 'text']].to_dict('records')
    for podcast in recommended_podcasts:
        recommendations.append(podcast)
    return recommendations


# Add a button to trigger the recommendation
if st.button("Get Recommendations"):
    if keyword:
        recommendations = get_recommendations2(keyword)

        def get_star_rating_html(rating, max_rating=5, filled_star="★", empty_star="☆"):
            filled_stars = filled_star * int(rating)
            empty_stars = empty_star * (max_rating - int(rating))
            return f'<span style="display: inline-flex; align-items: center;"><span style="color: gold; margin-right: 5px;">{filled_stars}{empty_stars}</span> [{rating}]</span>'

        cols = st.columns(4)
        for i, recommendation in enumerate(recommendations):
            with cols[i % 4]:
                if not pd.isnull(recommendation['image_url']):
                    st.image(str(recommendation['image_url']), width=int(200))
                else:
                    st.image('./data/images/nan_image.jpeg', width=int(200))
                st.markdown(
                    f'<h6 style="margin-top: 10px;">{recommendation["title"]}</h6>', unsafe_allow_html=True)
                star_rating_html = get_star_rating_html(
                    recommendation['rating'])
                st.markdown(f"<p>{star_rating_html}</p>",
                            unsafe_allow_html=True)
                key_x = recommendation['link']
                st.markdown(f'<a href="{key_x}" target="_blank" style="display: inline-block; padding: 5px px; background-color: #FFE026; color: black; text-decoration: none; border-radius: 2px; text-align: center; line-height: 30px; width: 130px;">Listen on iTunes</a>', unsafe_allow_html=True)

    else:
        st.warning("Please enter a keyword to get podcast recommendations.")


st.markdown("<h2 style='font-family:monospace; font-size: 24px; font-weight: bold;'>Select a podcast:</h2>",
            unsafe_allow_html=True)

selected_podcast = st.selectbox(
    '', options=podcasts_df.title.values, key=next(key_counter))

if st.button('Show Recommendations'):
    recommendations = get_recommendations(
        podcasts_df, vectorizer, selected_podcast)

    def get_star_rating_html(rating, max_rating=5, filled_star="★", empty_star="☆"):
        filled_stars = filled_star * int(rating)
        empty_stars = empty_star * (max_rating - int(rating))
        return f'<span style="display: inline-flex; align-items: center;"><span style="color: gold; margin-right: 5px;">{filled_stars}{empty_stars}</span> [{rating}]</span>'

    cols = st.columns(4)
    for i, recommendation in enumerate(recommendations):
        with cols[i % 4]:
            # st.markdown('<div style="border: 1px solid lightgray; border-radius: 10px; padding: 10px; text-align: center;">', unsafe_allow_html=True)
            if not pd.isnull(recommendation['image_url']):
                st.image(str(recommendation['image_url']), width=int(200))
            else:
                st.image('./data/images/nan_image.jpeg', width=int(200))
            # st.image(str(recommendation['image_url']), width=int(200))
            st.markdown(
                f'<h6 style="margin-top: 10px;">{recommendation["title"]}</h6>', unsafe_allow_html=True)
            star_rating_html = get_star_rating_html(recommendation['rating'])
            st.markdown(f"<p>{star_rating_html}</p>", unsafe_allow_html=True)
            key_x = recommendation['link']
            st.markdown(f'<a href="{key_x}" target="_blank" style="display: inline-block; padding: 5px px; background-color: #FFE026; color: black; text-decoration: none; border-radius: 2px; text-align: center; line-height: 30px; width: 130px;">Listen on iTunes</a>', unsafe_allow_html=True)


st.markdown("<h2 style='font-family:monospace; font-size: 24px; font-weight: bold;'>Select the Genre(s):</h2>", unsafe_allow_html=True)

# Define the genres selection box
genres_list = sorted(podcasts_df.genre.unique())
selected_genres = st.multiselect(
    label='Select genres:',
    options=genres_list,
)

# Define a function to recommend podcasts


def recommend_podcasts():
    selected_genres_str = "|".join(selected_genres)
    filtered_podcasts_df = podcasts_df[podcasts_df['genre'].str.contains(
        selected_genres_str)]

    if len(filtered_podcasts_df) == 0:
        st.warning(
            'No podcasts found for the selected genre(s). Please select different genre(s).')
    else:
        # Sort the podcasts by rating in descending order
        sorted_podcasts_df = filtered_podcasts_df.sort_values(
            'rating', ascending=False)

        # Recommend the top 10 podcasts
        recommended_podcasts_df = sorted_podcasts_df.head(8)

        # Display the recommended podcasts as cards in a grid format
        col1, col2, col3, col4 = st.columns(4)
        for i, (_, podcast) in enumerate(recommended_podcasts_df.iterrows()):
            with eval(f'col{i % 4 + 1}'):
                if not pd.isna(podcast['image_url']):
                    st.image(podcast['image_url'], width=200)
                else:
                    st.image(
                        'https://via.placeholder.com/200x200.png?text=No+Image', width=200)
                st.markdown(
                    f'<h6 style="margin-top: 10px;">{podcast["title"]}</h6>', unsafe_allow_html=True)
                with st.container():
                    st.markdown(
                        f'<p style="font-style: italic; margin-top: 5px;">{podcast["genre"]}</p>', unsafe_allow_html=True)
                st.markdown(f'<a href="{podcast["link"]}" target="_blank" style="display: inline-block; padding: 5px px; background-color: #FFE026; color: black; text-decoration: none; border-radius: 2px; text-align: center; line-height: 30px; width: 130px;">Listen on iTunes</a>', unsafe_allow_html=True)


# Define a button to trigger the recommendation process
recommend_button = st.button('Recommend podcasts')

# If the button is clicked, recommend the podcasts
if recommend_button:
    if len(selected_genres) > 0:
        st.write(
            f"You have selected the following genres: {', '.join(selected_genres)}")
        recommend_podcasts()
    else:
        st.warning("Please select at least one genre.")


with open("./pickles/w2v_model.pkl", "rb") as f:
    w2v_model = pickle.load(f)

with open("./pickles/w2v_cosine_sim.pkl", "rb") as f:
    w2v_cosine_sim = pickle.load(f)


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec=None):
        self.word2vec = word2vec
        if word2vec is not None:
            self.dim = len(word2vec.wv.vectors[0])
        else:
            self.word2vec = None
            self.dim = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.word2vec is None:
            raise ValueError('Word2Vec model has not been provided')
        X = MyTokenizer().fit_transform(X)
        return np.array([
            np.mean([self.word2vec.wv[w] for w in words if w in self.word2vec.wv]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

    def fit_transform(self, X, y=None):
        return self.transform(X)


def recommend_podcasts_w2v(title, cosine_sim=w2v_cosine_sim, n=8):
    # get the index of the podcast that matches the title
    indices = pd.Series(podcasts_df.index, index=podcasts_df['title'])
    idx = indices[title]

    # get the pairwise cosine similarity scores for the given podcast
    sim_scores = list(enumerate(cosine_sim[idx]))

    # sort the podcasts by their cosine similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # get the indices of the top n most similar podcasts
    sim_scores = sim_scores[1:n+1]
    podcast_indices = [i[0] for i in sim_scores]

    # create a dictionary for each recommended podcast with its title, link, rating, image_url, and text
    recommended_podcasts = []
    for index in podcast_indices:
        podcast_dict = {}
        podcast_dict['title'] = podcasts_df['title'].iloc[index]
        podcast_dict['link'] = podcasts_df['link'].iloc[index]
        podcast_dict['rating'] = podcasts_df['rating'].iloc[index]
        podcast_dict['image_url'] = podcasts_df['image_url'].iloc[index]
        podcast_dict['text'] = podcasts_df['text'].iloc[index]
        recommended_podcasts.append(podcast_dict)

    # return the list of recommended podcasts
    return recommended_podcasts


mean_embedding_vectorizer = MeanEmbeddingVectorizer(word2vec=w2v_model)

st.markdown("<h2 style='font-family:monospace; font-size: 24px; font-weight: bold;'>Select a podcast:</h2>",
            unsafe_allow_html=True)

selected_podcast = st.selectbox(
    '', options=podcasts_df.title.values, key=next(key_counter))

# podcast_title = st.text_input('Enter a podcast title:')
if st.button('Show Recommendations', key='show_recommendations_btn'):
    recommendations = recommend_podcasts_w2v(selected_podcast)

    def get_star_rating_html(rating, max_rating=5, filled_star="★", empty_star="☆"):
        filled_stars = filled_star * int(rating)
        empty_stars = empty_star * (max_rating - int(rating))
        return f'<span style="display: inline-flex; align-items: center;"><span style="color: gold; margin-right: 5px;">{filled_stars}{empty_stars}</span> [{rating}]</span>'

    cols = st.columns(4)
    for i, recommendation in enumerate(recommendations):
        with cols[i % 4]:
            # st.markdown('<div style="border: 1px solid lightgray; border-radius: 10px; padding: 10px; text-align: center;">', unsafe_allow_html=True)
            if not pd.isnull(recommendation['image_url']):
                st.image(str(recommendation['image_url']), width=int(200))
            else:
                st.image('./data/images/nan_image.jpeg', width=int(200))
            # st.image(str(recommendation['image_url']), width=int(200))
            st.markdown(
                f'<h6 style="margin-top: 10px;">{recommendation["title"]}</h6>', unsafe_allow_html=True)
            star_rating_html = get_star_rating_html(recommendation['rating'])
            st.markdown(f"<p>{star_rating_html}</p>", unsafe_allow_html=True)
            key_x = recommendation['link']
            st.markdown(f'<a href="{key_x}" target="_blank" style="display: inline-block; padding: 5px px; background-color: #FFE026; color: black; text-decoration: none; border-radius: 2px; text-align: center; line-height: 30px; width: 130px;">Listen on iTunes</a>', unsafe_allow_html=True)
