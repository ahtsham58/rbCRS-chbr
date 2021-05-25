import pandas as pd
import numpy as np
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import linear_kernel
import random

class Recommender_MF():

    def __init__(self):
        self.model =np.array([])
        self.movie_title_list = None
        self.df_ratings = pd.DataFrame()
        self.df_movies = pd.DataFrame()
        self.movies_mentions = pd.DataFrame()
        self.is_session_changed = False
        self.recommended_movies = []
        self.df_movies = pd.read_csv("../recommenders_item_data/movies_latest.csv")
        self.movies_mentions = pd.read_csv("../recommenders_item_data/movies_data.csv", encoding="utf-8")
        self.df_ratings = pd.read_csv("../recommenders_item_data/ratings_latest.csv", usecols=['userId', 'movieId', 'rating'])
        if self.model.size == 0:
            self.model, self.movie_title_list = self.data_initialization()


    def data_initialization(self):
        self.df_ratings = self.df_ratings[:2000000]
        combine_movie_rating = pd.merge(self.df_ratings, self.df_movies, on='movieId')
        combine_movie_rating = combine_movie_rating.dropna(axis = 0, subset = ['title'])
        movie_ratingCount = (combine_movie_rating.
             groupby(by = ['title'])['rating'].
             count().
             reset_index().
             rename(columns = {'rating': 'totalRatingCount'})
             [['title', 'totalRatingCount']]
            )
        rating_with_totalRatingCount = combine_movie_rating.merge(movie_ratingCount, left_on = 'title', right_on = 'title', how = 'left')
        user_rating = rating_with_totalRatingCount.drop_duplicates(['userId','title'])

        df_temp_rating_count = user_rating.drop_duplicates(['title'],keep='first')
        year_list = []
        for index, row in self.df_movies.iterrows():
            try:
                title = str(row['title'])
                if title.__contains__('(') and title.__contains__(')'):
                    year = int(title[len(title)-5:].replace(')',''))
                    year_list.append(year)
                else:
                    year = 2000
                    year_list.append(year)
            except:
                year_list.append(1500)
                continue

        self.df_movies['year'] = year_list
        self.df_movies= pd.merge(self.df_movies[['movieId','title','genres','year']], df_temp_rating_count[['title','totalRatingCount']], left_on = 'title', right_on = 'title', how = 'left').fillna(0)

        movie_user_rating_pivot = pd.pivot_table(user_rating, index = 'userId', columns = 'title', values = 'rating').fillna(0)
        #movie_user_rating_pivot = user_rating.pivot(index = 'userId', columns = 'title', values = 'rating').fillna(0)
        X = movie_user_rating_pivot.values.T


        SVD = TruncatedSVD(n_components=20, random_state=10)
        matrix = SVD.fit_transform(X)
        self.model = np.corrcoef(matrix)
        movie_title = movie_user_rating_pivot.columns
        self.movie_title_list = list(movie_title)
        return self.model, self.movie_title_list

    def get_similar_movies_based_on_content(self, input_movieId, identifier, dialog_mentioned_movies = []):
        try:
            title = self.movies_mentions.loc[self.movies_mentions['databaseId'] == int(input_movieId)]['title']
            title = title.iloc[0]
            if len(title) < 2:
                return '__unk__'
            movie_index = self.movie_title_list.index(title)
            sim_scores  = list(enumerate(self.model[movie_index]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:15]
            movie_indices = [i[0] for i in sim_scores]
            movie_sim_scores = [i[1] for i in sim_scores]
            scores = [round(x, 2) for x in movie_sim_scores]
            similar_movies = pd.DataFrame([self.movie_title_list[i] for i in movie_indices],columns=['title'])
            similar_movies = pd.merge(similar_movies[['title']],self.df_movies[['title','genres','year','totalRatingCount']],how='left',on='title')
            similar_movies = similar_movies.sort_values(by ='totalRatingCount', ascending=False)
            similar_movies  = similar_movies.sort_values(by ='year', ascending=False).head(3).reset_index()
            similar_movies = similar_movies[similar_movies.year != 1500]
            movies_list = similar_movies.values.tolist()
            n = random.randint(0,len(similar_movies)-1)

            #in case of just check movie id ..
            recommendation_title  = movies_list[n][1]
            recommendation_id = None

            #if the recommmendation request is actually about recommendation, check avoid any duplicate recommendation in a same dialogue
            if identifier != 0:
                #filter recommendation if already mentioned or recommended in the previous dialog history up to the point
                i = 1
                while i < 4:
                    n = random.randint(0,len(movies_list)-1)
                    recommendation_title  = movies_list[n][1]
                    try:
                        recommendation_id = self.movies_mentions.loc[self.movies_mentions['title'] == str(recommendation_title)]['databaseId']
                    except KeyError:
                        break
                    if len(recommendation_id) > 0 and recommendation_id is not None:
                        recommendation_id  = str(int(recommendation_id.values[0]))
                        if identifier != 0 and recommendation_id in dialog_mentioned_movies:
                            movies_list.pop(n)
                            i = i+1
                        else:
                            break
                    else:
                        break
                if self.is_session_changed:
                    self.recommended_movies=[]
                    self.is_session_changed = False
                if recommendation_title in self.recommended_movies:
                    movies_list.pop(n)
                    n = random.randint(0,len(movies_list)-1)
                    recommendation_title = movies_list[n][1]
                    self.recommended_movies.append(recommendation_title)
                else:
                    self.recommended_movies.append(recommendation_title)
        except:
            recommendation_title = '__unk__'
            recommendation_id = '0'

        return  recommendation_title, recommendation_id

if __name__ == '__main__':
    obj= Recommender_MF()
    recommendations, rec_id = obj.get_similar_movies_based_on_content(184418,0)
    print(recommendations)

