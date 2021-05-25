import pandas as pd
import numpy as np
from ast import literal_eval
import random

class Recommender_with_genre:

    def __init__(self):
            self.df_movie_content = pd.DataFrame()
            self.recommended_movies=[]
            self.C = 0
            self.m =0
            self.movies_mentions = pd.read_csv("../recommenders_item_data/movies_data.csv", encoding="utf-8")
            if self.df_movie_content.__len__() == 0:
                self.df_movie_content = self.data_initialization()

    def get_similar_movies_based_on_genre(self, genre,dialog_mentioned_movies=[], percentile=0.85):
        try:
            print('the genre is ' +genre)
            if genre.lower() == 'scary':
                genre ='Horror'
            elif genre.lower() =='romantic':
                genre='Romance'
            elif genre.lower() == 'preference':
                genre = 'Adventure'
            elif genre.lower() =='suspense':
                genre = 'Thriller'
            elif genre.lower() =='funny':
                genre = 'Comedy'
            elif genre.lower() == 'comedies':
                genre = 'Comedy'
            elif genre.lower() == 'scifi':
                genre = 'Science Fiction'
            elif genre.lower() == 'kids':
                genre = 'Comedy'


            df = pd.DataFrame()
            df = self.df_movie_content[self.df_movie_content['genre'] == genre]
            vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
            vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
            self.C = vote_averages.mean()
            self.m = vote_counts.quantile(.85)

            col_list = ['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genre']
            qualified = df[(df['vote_count'] >= self.m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][col_list]
            qualified['vote_count'] = qualified['vote_count'].astype('int')
            qualified['vote_average'] = qualified['vote_average'].astype('int')

            qualified['weighted_rating'] = qualified.apply(lambda x:
                                                           (x['vote_count']/(x['vote_count']+self.m) * x['vote_average']) + (self.m/(self.m+x['vote_count']) * self.C),
                                                           axis=1)
            qualified = qualified.sort_values("weighted_rating", ascending=False).head(10).reset_index()
            qualified = qualified.sort_values('year', ascending=False).head(3).reset_index()
            n = random.randint(0,len(qualified)-1)
            movies_list = qualified.values.tolist()


            #filter recommendation if already mentioned or recommended in the previous dialog history up to the point
            i = 1
            while i < 4:
                n = random.randint(0,len(movies_list)-1)
                recommendation_title  = str(str(movies_list[n][2]) +' ('+ movies_list[n][3]+')')
                recommendation_id = self.movies_mentions.loc[self.movies_mentions['title'] == str(recommendation_title)]['databaseId']
                if len(recommendation_id) > 0:
                    recommendation_id  = str(int(recommendation_id.values[0]))
                else:
                    recommendation_id = '00000'
                if recommendation_id in dialog_mentioned_movies:
                    movies_list.pop(n)
                    i = i+1
                else:
                    break
        except (RuntimeError, TypeError, NameError) as err:
            print(err)
            print("exception accured here")
            recommendation_title = '__unk__'
            recommendation_id = '0'
        return recommendation_title ,recommendation_id

    def weighted_rating(x):
        v = x['vote_count']
        R = x['vote_average']
        return (v/(v+self.m) * R) + (self.m/(self.m+v) * self.C)

    def data_initialization(self):
        m_df = pd.read_csv('../recommenders_item_data/movies_metadata.csv', low_memory=False)
        m_df['genres'] = m_df['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x]
                                                                               if isinstance(x, list) else [])
        # extracting release year from release_date
        m_df['year'] = pd.to_datetime(m_df['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0]
                                                                                   if x != np.nan else np.nan)

        #Now, building list for particular genres. For that, cutoff is relaxed to 85% instead of 95
        temp = m_df.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
        temp.name = 'genre'
        mges_df = m_df.drop('genres', axis=1).join(temp)
        return mges_df

if __name__ == '__main__':
    obj = Recommender_with_genre()
    rec, rec_id  = obj.get_similar_movies_based_on_genre('Drama')
    print(rec)



