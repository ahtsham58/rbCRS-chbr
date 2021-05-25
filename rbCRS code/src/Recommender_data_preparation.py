
import pandas as pd

movies = pd.read_csv('../recommenders_item_data/movies_data.csv', encoding='Latin')
movies_ratings = pd.read_csv('../recommenders_item_data/ratings_latest.csv', ',', encoding='utf-8', )

temp_rating = pd.DataFrame()


movie_rating_df =  pd.merge(movies, movies_ratings, on='movieId', how='inner')
#temp_rating = movie_rating_df.loc[movie_rating_df['title']=='Toy Story (1995)']['rating'].mean()
mean_dict = movie_rating_df.groupby('movieId')['rating'].mean().to_dict()
movies_ratings__list = []
for item in mean_dict:
    key = item
    mean= mean_dict[item]
    mean = round(mean,2)
    movies_ratings__list.append({'movieId':key, 'rating_mean': mean,})


df_mean_rating = pd.DataFrame(movies_ratings__list, columns=['movieId','rating_mean'])
movie_avg_rating_df =  pd.merge(df_mean_rating[['movieId','rating_mean' ]], movie_rating_df[['movieId','title','genres','databaseId']], on='movieId', how='inner')
movie_avg_rating_df = movie_avg_rating_df.drop_duplicates(subset='movieId', keep='first')
print(movie_avg_rating_df)

movie_avg_rating_df.to_csv('movies_rating_data.csv', header = True, index =True)
#print(final_df)
