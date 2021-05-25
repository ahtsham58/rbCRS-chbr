from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import numpy as np
import random


class Recommender:

    def __init__(self):
        self.cosine_sim =np.array([])
        self.df_movie_content =pd.DataFrame()
        self.recommended_movies=[]
        self.movies_mentions = pd.DataFrame()
        self.is_session_changed = False
        if self.cosine_sim.size == 0:
            self.cosine_sim, self.df_movie_content = self.data_initialization()

    def data_initialization(self):
        movie_ratings = '../recommenders_item_data/movies_rating_data.csv'
        #reading the movies dataset
        movie_ratings_list = pd.read_csv(movie_ratings,encoding="Latin1")
        Recommender.movies_mentions = pd.read_csv("../recommenders_item_data/movies_data.csv", encoding="Latin1")
        movie_ratings_list = movie_ratings_list.reset_index()
        year_list = []
        for index, row in movie_ratings_list.iterrows():
            title = str(row['title'])
            if title.__contains__('(') and title.__contains__(')'):
                year = int(title[len(title)-5:].replace(')',''))
                year_list.append(year)
            else:
                year = 2000
                year_list.append(year)

        movie_ratings_list['year'] = year_list
        genre_list = ""
        for index,row in movie_ratings_list.iterrows():
                genre_list += row.genres + "|"
        #split the string into a list of values
        genre_list_split = genre_list.split('|')
        #de-duplicate values
        new_list = list(set(genre_list_split))
        #remove the value that is blank
        new_list.remove('')
        #Enriching the movies dataset by adding the various genres columns.
        movies_with_genres = movie_ratings_list.copy()

        for genre in new_list :
            movies_with_genres[genre] = movies_with_genres.apply(lambda _:int(genre in _.genres), axis = 1)

        #Getting the movies list with only genres like Musical and other such columns
        movie_content_df_temp = movies_with_genres.copy()
        movie_content_df_temp.set_index('databaseId')
        movie_content_df = movie_content_df_temp.drop(columns = ['movieId','rating_mean','title','genres', 'year','databaseId'])
        movie_content_df = movie_content_df.values
        movie_content_df = np.delete(movie_content_df, 0, 1)
        movie_content_df = np.delete(movie_content_df, 0, 1)
        # Compute the cosine similarity matrix
        Recommender.cosine_sim = linear_kernel(movie_content_df,movie_content_df)
        #write model for offline initialization
        #np.savetxt('model.txt', cosine_sim)
        #write Movie contents for runtime recommendations
        #movie_content_df_temp.to_csv ('movie_content.csv', index = True, header=True)
        Recommender.df_movie_content = movie_content_df_temp
        return Recommender.cosine_sim, Recommender.df_movie_content

    def read_model_content_data(self):
        modeldata = np.loadtxt(fname = "model.txt")
        df = pd.read_csv('movie_content.csv',encoding="Latin1")
        return modeldata, df


    #Gets the top 10 similar movies based on the content
    def get_similar_movies_based_on_content(self, input_movieId,identifier, is_session_changed, dialog_mentioned_movies=[]) :
        try:

            #cosine_sim, movie_content_df_temp =self.data_initialization(movies_data)
            #create a series of the movie id and title
            movie_content_df_temp = Recommender. df_movie_content.loc[:, ~Recommender.df_movie_content.columns.str.contains('^Unnamed')]
            indicies = pd.Series(movie_content_df_temp.index, movie_content_df_temp['title'])
            title = movie_content_df_temp.loc[movie_content_df_temp['databaseId'] == int(input_movieId)]['title']
            title = title.iloc[0]
            if len(title) < 2:
                return '__unk__'
            movie_index =indicies[title]
            sim_scores = list(enumerate(Recommender.cosine_sim[movie_index]))
            # Sort the movies based on the similarity scores
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            # Get the scores of the 5 most similar movies
            sim_scores = sim_scores[1:15]

            movie_sim_scores = [i[1] for i in sim_scores]
            # Get the movie indices
            movie_indices = [i[0] for i in sim_scores]
            similar_movies = pd.DataFrame(movie_content_df_temp[['title','genres', 'year','rating_mean']].iloc[movie_indices])
            #similar_movies = similar_movies.drop( similar_movies[ similar_movies['title'] == title ].index , inplace=True)
            similar_movies = similar_movies[similar_movies.title != title]
            similar_movies = similar_movies.sort_values(by ='year', ascending=False)
            similar_movies  = similar_movies.head(5).reset_index()
            similar_movies  = similar_movies.sort_values(by='rating_mean').head(3).reset_index()
            movies_list = similar_movies.values.tolist()
            n = random.randint(0,len(similar_movies)-1)
            n = random.randint(0,len(similar_movies)-1)

            #in case of just check movie id ..
            recommendation_title  = movies_list[n][2]
            recommendation_id = None

            if identifier != 0:
                #filter recommendation if already mentioned or recommended in the previous dialog history up to the point
                i = 1
                while i < 4:
                    recommendation_title  = movies_list[n][2]
                    try:
                        recommendation_id = Recommender.movies_mentions.loc[self.movies_mentions['title'] == str(recommendation_title)]['databaseId']
                    except KeyError:
                        break
                    if recommendation_id is not None and len(recommendation_id) > 0:
                        recommendation_id  = str(int(recommendation_id.values[0]))
                        if identifier != 0 and recommendation_id in dialog_mentioned_movies:
                            movies_list.pop(n)
                            n = random.randint(0,len(movies_list)-1)
                            i = i+1
                        else:
                            break
                    else:
                        break
                #if the recommmendation request is actually about recommendation, check avoid any duplicate recommendation in a same dialogue
                if is_session_changed:
                    self.recommended_movies=[]
                    self.is_session_changed = False

                if recommendation_title in self.recommended_movies:
                    movies_list.pop(n)
                    n = random.randint(0,len(movies_list)-1)
                    recommendation_title = movies_list[n][2]
                    self.recommended_movies.append(recommendation_title)
                else:
                    self.recommended_movies.append(recommendation_title)


            # if recommendation_title in self.recommended_movies:
            #     movies_list.pop(n)
            #     n = random.randint(0,len(movies_list)-1)
            #     recommendation_title = movies_list[n][2]
            #
            # self.recommended_movies.append(recommendation_title)
        except:
            recommendation_title = '__unk__'
            recommendation_id = '0'

        return recommendation_title, recommendation_id
if __name__ == '__main__':
    obj= Recommender()
    recommendations ,rec_id = obj.get_similar_movies_based_on_content(123395,None,True)
    print(recommendations)
