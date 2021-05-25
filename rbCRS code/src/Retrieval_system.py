import pandas as pd
import re
import random
import numpy as np
from Retrieval_based_Recomender.src.Recommender import Recommender
from Retrieval_based_Recomender.src.Recommender_MF import Recommender_MF
from Retrieval_based_Recomender.src.Recommender_with_genre import Recommender_with_genre


class Retrieval_system:
    def __init__(self):
        self.rec = None
        self.genre_rec = None
        self.rec_only_genre = None
        self.mentioned_recommended_movies = []
        self.file_name = '../dialogs_data/dialogue_data.txt'
        # list of movie domain attributes
        self.pref_keywords = ['scary','horror','comedy', 'kids','funny', 'comedies','action','family','adventure','crime','fantasy','thriller','scifi', 'documentary', 'science fiction', 'drama','romance','romantic','mystery','history','no preference','suspense' ]

    def parse_generated_dialogues_sq(self, parsed_redial_dialog):
        chit_chat_bye, chit_chat_compliment, ask_preference, recommend, response_to_seeker, encourage, recommend_anything,chit_chat_response, confirmatory_responses = self.load_utterances_data()
        completed_dialog_list = []
        seeker_movies = []
        self.mentioned_recommended_movies = []
        ground_truth_movies = []
        previous_gt_pref_keyword = []
        is_movie_mentioned_GT = False
        is_already_movie_recommended = False

        # parsing the generated dialogues file and converting into sentences only
        for i, line in enumerate(parsed_redial_dialog):
            try:
                completed_dialog_list.append(line)
                is_movie_mentioned = False
                if line.__contains__('CONVERSATION'):
                    continue
                movie_title = '__unk__'
                if line.__contains__('SEEKER:'):
                    seeker_query = self.seeker_sentences_parser(line)
                    seeker_query = re.sub('[^A-Za-z0-9@]+', ' ', seeker_query)
                    movieid = self.check_if_movieid_mention(seeker_query)
                    if movieid:
                        is_movie_mentioned = True
                        seeker_movies.append(movieid)

                    keywords = ['hi', 'heya', 'hello', 'looking', 'suggestions', 'suggest', 'hey', 'check', 'night',
                                'out', 'thank', 'thanks','wanting', 'goodbye', 'bye','recommend', 'suggestion','recommendations', 'day',
                                'interested']
                    pref_keywords = self.pref_keywords
                    seekerk_query_list = seeker_query.split(' ')

                    #identify common keywords list for intent recognition
                    common_tokens = list(set(seekerk_query_list).intersection(keywords))
                    pref_common_tokens = list(set(seekerk_query_list).intersection(pref_keywords))

                    #Rules starts here
                    # chit-chat inquiry
                    if seeker_query.__contains__('how are you') and not is_movie_mentioned:
                        gen_sen = self.genenrate_sentence(chit_chat_response)
                        completed_dialog_list.append(gen_sen + '\n')
                        continue
                    # response to seeker questions about movie types, movie metadata, personal experience etc.
                    if seeker_query.__contains__('who is') or seeker_query.__contains__('who s') or seeker_query.__contains__('what is') or seeker_query.__contains__('what s') or seeker_query.__contains__('what it about') or seeker_query.__contains__('did you') or seeker_query.__contains__('how are'):
                        gen_sen= self.genenrate_sentence(response_to_seeker)
                        completed_dialog_list.append(gen_sen + '\n')
                        continue

                    # response to seeker's confirmatory utterances
                    if seeker_query.__contains__('is that') or seeker_query.__contains__('does it') or seeker_query.__contains__('are they') or seeker_query.__contains__('is it') or seeker_query.__contains__('do you think'):
                        gen_sen= self.genenrate_sentence(confirmatory_responses)
                        completed_dialog_list.append(gen_sen + '\n')
                        continue

                    #generate recommendation on seeker's no genre intent
                    if (seeker_query.__contains__('any movie') or seeker_query.__contains__('good movies') or seeker_query.__contains__('good movie') or seeker_query.__contains__('any genre')or seeker_query.__contains__('new movies') or seeker_query.__contains__('all movies') or seeker_query.__contains__('any type') or seeker_query.__contains__('no preference')) and not is_movie_mentioned and len(pref_common_tokens) < 1:
                        mentioned_genre= 'Action'
                        temp_genre ='action or thriller'
                        movie_title = self.get_recommendation_by_genre(mentioned_genre)
                        gen_sen = self.genenrate_sentence(recommend_anything)
                        gen_sen = gen_sen.split('$')[0] + temp_genre + gen_sen.split('$')[1] +'"'+ movie_title+'"'
                        completed_dialog_list.append(gen_sen + '\n')
                        is_already_movie_recommended = True
                        continue
                    # generate recommendation if the seeker's utterance contains movie ID
                    if is_movie_mentioned:
                        movie_title = self.get_recommendation(movieid)
                        gen_sen = self.genenrate_sentence(recommend) +'"'+ movie_title+'"'
                        completed_dialog_list.append(gen_sen + '\n')
                        is_already_movie_recommended = True
                        continue

                    # check if seeker's query have common tokens to find intent
                    if any(common_tokens) or any(pref_common_tokens):
                        res = np.array(common_tokens)
                        common_tokens = np.unique(res)

                        #ask recommendation intent handled here
                        recommend_keywords = ['looking', 'interested', 'suggestions', 'suggest', 'suggestion',
                                              'recommend', 'recommendations','wanting']
                        recommend_common_tokens = ([value for value in common_tokens if value in recommend_keywords])
                        #rule to check if seeker is asking for recommendation
                        if len(recommend_common_tokens) > 0 or len(pref_common_tokens) > 0:
                            if (seeker_query.__contains__('thanks for') or seeker_query.__contains__('thank')) and len(pref_common_tokens) < 1:
                                gen_sen = self.genenrate_sentence(chit_chat_compliment)
                                completed_dialog_list.append(gen_sen + '\n')

                            elif is_already_movie_recommended and len(ground_truth_movies) > 1 and len(pref_common_tokens) < 1:
                                movieid = self.get_movie_id(ground_truth_movies)
                                movie_title = self.get_recommendation(movieid)
                                gen_sen = self.genenrate_sentence(recommend) +'"'+ movie_title+'"'
                                completed_dialog_list.append(gen_sen + '\n')
                                is_already_movie_recommended = True
                            elif is_already_movie_recommended and len(seeker_movies) > 1 and len(pref_common_tokens) < 1:
                                movieid = self.get_movie_id(seeker_movies)
                                movie_title = self.get_recommendation(movieid)
                                gen_sen = self.genenrate_sentence(recommend) +'"'+ movie_title+'"'
                                completed_dialog_list.append(gen_sen + '\n')
                                is_already_movie_recommended = True
                            elif is_movie_mentioned_GT and len(pref_common_tokens) < 1:
                                movieid = self.get_movie_id(ground_truth_movies)
                                movie_title = self.get_recommendation(movieid)
                                gen_sen = self.genenrate_sentence(recommend) +'"'+ movie_title+'"'
                                completed_dialog_list.append(gen_sen + '\n')
                                is_already_movie_recommended = True
                            else:
                                if len(pref_common_tokens) > 0:
                                    mentioned_genre= pref_common_tokens[len(pref_common_tokens)-1]
                                    movie_title = self.get_recommendation_by_genre(mentioned_genre)
                                    gen_sen = self.genenrate_sentence(recommend_anything)
                                    gen_sen = gen_sen.split('$')[0] + mentioned_genre + gen_sen.split('$')[1] +'"'+ movie_title+'"'
                                    completed_dialog_list.append(gen_sen + '\n')
                                    is_already_movie_recommended = True
                                else:
                                    gen_sen = self.genenrate_sentence(ask_preference)
                                    completed_dialog_list.append(gen_sen + '\n')

                        # handling other types of situation in which seeker ask for recommendation
                        elif (seeker_query.__contains__('like to watch') or seeker_query.__contains__('help me')) and len(common_tokens) < 1:
                            if is_movie_mentioned_GT:
                                movieid = self.get_movie_id(ground_truth_movies)
                                movie_title = self.get_recommendation(movieid)
                                gen_sen = self.genenrate_sentence(recommend) +'"'+ movie_title+'"'
                                completed_dialog_list.append(gen_sen + '\n')
                                is_already_movie_recommended = True
                                is_movie_mentioned_GT = False
                            else:
                                if len(pref_common_tokens) > 0:
                                    mentioned_genre= pref_common_tokens[len(pref_common_tokens)-1]
                                    movie_title = self.get_recommendation_by_genre(mentioned_genre)
                                    gen_sen = self.genenrate_sentence(recommend_anything)
                                    gen_sen = gen_sen.split('$')[0] + mentioned_genre + gen_sen.split('$')[1] +'"'+ movie_title+'"'
                                    completed_dialog_list.append(gen_sen + '\n')
                                    is_already_movie_recommended = True
                                else:
                                    gen_sen = self.genenrate_sentence(ask_preference)
                                    completed_dialog_list.append(gen_sen + '\n')

                        # handling utterance having conversation initiation tokens
                        elif common_tokens.__contains__('hi') or common_tokens.__contains__('hello') or common_tokens.__contains__('hey') or common_tokens.__contains__('heya'):

                            if len(pref_common_tokens) > 0:
                                    mentioned_genre= pref_common_tokens[len(pref_common_tokens)-1]
                                    movie_title = self.get_recommendation_by_genre(mentioned_genre)
                                    gen_sen = self.genenrate_sentence(recommend_anything)
                                    gen_sen = gen_sen.split('$')[0] + mentioned_genre + gen_sen.split('$')[1] +'"'+ movie_title+'"'
                                    completed_dialog_list.append(gen_sen + '\n')
                                    is_already_movie_recommended = True
                            else:
                                    gen_sen = self.genenrate_sentence(ask_preference)
                                    completed_dialog_list.append(gen_sen + '\n')

                        # response to seeker's utterance to encourge the seeker on the made recommendation
                        elif (common_tokens.__contains__('check') and common_tokens.__contains__('out')) or seeker_query.__contains__('will try') or seeker_query.__contains__('will check') or seeker_query.__contains__('ll check')or seeker_query.__contains__('ill check'):
                                gen_sen = self.genenrate_sentence(encourage)
                                completed_dialog_list.append(gen_sen + '\n')

                        #check if the above cases are true and still we have common tokens
                        elif len(common_tokens) > 0:
                            tokens_list1 = ['thanks', 'thank', 'you']
                            tokens_list2 = ['bye', 'goodbye', 'day','good night']

                            # check if the movie is not already recommended
                            if not is_already_movie_recommended:
                                if len(pref_common_tokens) > 0:
                                    mentioned_genre= pref_common_tokens[len(pref_common_tokens)-1]
                                    movie_title = self.get_recommendation_by_genre(mentioned_genre)
                                    gen_sen = self.genenrate_sentence(recommend_anything)
                                    gen_sen = gen_sen.split('$')[0] + mentioned_genre + gen_sen.split('$')[1] +'"'+ movie_title+'"'
                                    completed_dialog_list.append(gen_sen + '\n')
                                    is_already_movie_recommended = True
                                elif is_movie_mentioned_GT:
                                    movieid = self.get_movie_id(ground_truth_movies)
                                    movie_title = self.get_recommendation(movieid)
                                    gen_sen = self.genenrate_sentence(recommend) +'"'+ movie_title+'"'
                                    completed_dialog_list.append(gen_sen + '\n')
                                    is_already_movie_recommended = True
                                    is_movie_mentioned_GT = False
                                else:
                                    gen_sen = self.genenrate_sentence(ask_preference)
                                    completed_dialog_list.append(gen_sen + '\n')

                            elif any([substring in common_tokens for substring in tokens_list1]):
                                # generate recommendation if the previous ground truth utterance have preference keyword
                                if len(previous_gt_pref_keyword) > 0:
                                    movie_title = self.get_recommendation_by_genre(previous_gt_pref_keyword[len(previous_gt_pref_keyword)-1])
                                    gen_sen = self.genenrate_sentence(recommend) +'"'+ movie_title+'"'
                                    completed_dialog_list.append(gen_sen + '\n')
                                    is_already_movie_recommended = True
                                    previous_gt_pref_keyword = []
                                else:
                                    gen_sen = self.genenrate_sentence(chit_chat_compliment)
                                    completed_dialog_list.append(gen_sen + '\n')

                            elif any([substring in common_tokens for substring in tokens_list2]):
                                gen_sen = self.genenrate_sentence(chit_chat_bye)
                                completed_dialog_list.append(gen_sen + '\n')

                            elif len(previous_gt_pref_keyword) > 0:
                                    movie_title = self.get_recommendation_by_genre(previous_gt_pref_keyword[len(previous_gt_pref_keyword)-1])
                                    gen_sen = self.genenrate_sentence(recommend) +'"'+ movie_title+'"'
                                    completed_dialog_list.append(gen_sen + '\n')
                                    is_already_movie_recommended = True
                                    previous_gt_pref_keyword = []
                            else:
                                gen_sen = self.genenrate_sentence(confirmatory_responses)
                                completed_dialog_list.append(gen_sen + '\n')

                        # if no movie is mentioned in the dialog history, ask for preferences
                        elif len(seeker_movies) == 0 and len(ground_truth_movies) == 0 and not is_already_movie_recommended:
                            if len(pref_common_tokens) > 0:
                                mentioned_genre= pref_common_tokens[len(pref_common_tokens)-1]
                                movie_title = self.get_recommendation_by_genre(mentioned_genre)
                                gen_sen = self.genenrate_sentence(recommend_anything)
                                gen_sen = gen_sen.split('$')[0] + mentioned_genre + gen_sen.split('$')[1] +'"'+ movie_title+'"'
                                completed_dialog_list.append(gen_sen + '\n')
                                is_already_movie_recommended = True
                            else:
                                gen_sen = self.genenrate_sentence(ask_preference)
                                completed_dialog_list.append(gen_sen + '\n')


                    # when no common keyword found in seeker part
                    # gennerate recommendation on asking for recommendation intent
                    elif (seeker_query.__contains__('like to watch') or seeker_query.__contains__('help me')) and len(common_tokens) < 1:
                            if is_movie_mentioned_GT:
                                movieid = self.get_movie_id(ground_truth_movies)
                                movie_title = self.get_recommendation(movieid)
                                gen_sen = self.genenrate_sentence(recommend) +'"'+ movie_title+'"'
                                completed_dialog_list.append(gen_sen + '\n')
                                is_already_movie_recommended = True
                                is_movie_mentioned_GT = False
                            elif len(previous_gt_pref_keyword) > 0:
                                    movie_title = self.get_recommendation_by_genre(previous_gt_pref_keyword[len(previous_gt_pref_keyword)-1])
                                    gen_sen = self.genenrate_sentence(recommend) +'"'+ movie_title+'"'
                                    completed_dialog_list.append(gen_sen + '\n')
                                    is_already_movie_recommended = True
                                    previous_gt_pref_keyword = []
                            else:
                                gen_sen = self.genenrate_sentence(response_to_seeker)
                                completed_dialog_list.append(gen_sen + '\n')

                    # generate chit-chat utterance
                    elif seeker_query.__contains__('have a'):
                        gen_sen = self.genenrate_sentence(chit_chat_bye)
                        completed_dialog_list.append(gen_sen + '\n')

                    #if no movie is mentioned in the dialog history, ask for preferences
                    elif len(seeker_movies) == 0 and len(ground_truth_movies) == 0 and not is_already_movie_recommended:
                        if len(pref_common_tokens) > 0:
                            mentioned_genre= pref_common_tokens[len(pref_common_tokens)-1]
                            movie_title = self.get_recommendation_by_genre(mentioned_genre)
                            gen_sen = self.genenrate_sentence(recommend_anything)
                            gen_sen = gen_sen.split('$')[0] + mentioned_genre + gen_sen.split('$')[1] +'"'+ movie_title+'"'
                            completed_dialog_list.append(gen_sen + '\n')
                            is_already_movie_recommended = True
                        else:
                            gen_sen = self.genenrate_sentence(ask_preference)
                            completed_dialog_list.append(gen_sen + '\n')

                    # if no case is true, handle here
                    else:
                        if is_movie_mentioned_GT:
                            movieid = self.get_movie_id(ground_truth_movies)
                            movie_title = self.get_recommendation(movieid)
                            gen_sen = self.genenrate_sentence(recommend) +'"'+ movie_title+'"'
                            completed_dialog_list.append(gen_sen + '\n')
                            is_already_movie_recommended = True
                            is_movie_mentioned_GT = False
                        elif len(previous_gt_pref_keyword) > 0:
                                    movie_title = self.get_recommendation_by_genre(previous_gt_pref_keyword[len(previous_gt_pref_keyword)-1])
                                    gen_sen = self.genenrate_sentence(recommend) +'"'+ movie_title+'"'
                                    completed_dialog_list.append(gen_sen + '\n')
                                    is_already_movie_recommended = True
                                    previous_gt_pref_keyword = []
                        else:
                             gen_sen = self.genenrate_sentence(confirmatory_responses)
                             completed_dialog_list.append(gen_sen + '\n')


                ##Ground Truth (GT) Sentence processing
                # preserve/store the GT context here
                # 1) movie IDs mentioned in ground truth sentences
                # 2) preserve preference keyword
                else:
                    gt_response = self.gt_sentence_parser(line)
                    gt_response = re.sub('[^A-Za-z0-9@]+', ' ', gt_response)
                    movieid = self.check_if_movieid_mention(gt_response)
                    pref_kwds = self.pref_keywords
                    gt_query_list = gt_response.split(' ')
                    previous_gt_pref_keyword = list(set(gt_query_list).intersection(pref_kwds))
                    if movieid:
                        ground_truth_movies.append(movieid)
                        is_movie_mentioned_GT = True
                    else:
                        is_movie_mentioned_GT = False

            # deal exception here, and skip the utterance
            except RuntimeError as err:
                print(err)
                print("exception accured here")
                continue
        return completed_dialog_list

    def parse_generated_dialogues_gt(self, parsed_redial_dialog):
        chit_chat_bye, chit_chat_compliment, ask_preference, recommend, response_to_seeker, encourage,recommend_anything,chit_chat_response, confirmatory_responses = self.load_utterances_data()
        completed_dialogs_list = []
        gt_mentioned_movies = []
        seeker_mentioned_movies = []
        self.mentioned_recommended_movies = []
        is_movie_mentioned_GT= False
        is_already_movie_recommended= False
        is_ask_preference = False
        for seeker_line in parsed_redial_dialog:
            try:
                completed_dialogs_list.append(seeker_line)
                is_movie_mentioned = False
                if seeker_line.__contains__('CONVERSATION:'):
                    continue
                movie_title = '__unk__'
                print('response is being retrieved on seeker query...')
                if seeker_line.__contains__('SEEKER:'):
                    seeker_query = self.seeker_sentences_parser(seeker_line)
                    seeker_query = re.sub('[^A-Za-z0-9@]+', ' ', seeker_query)
                    movieid = self.check_if_movieid_mention(seeker_query)
                    if movieid:
                        is_movie_mentioned = True
                        seeker_mentioned_movies.append(movieid)

                    keywords = ['hi', 'hello', 'looking', 'suggestions', 'suggest', 'hey', 'check', 'night',
                                'out', 'thank', 'thanks', 'goodbye','bye', 'recommend', 'recommendations', 'day']
                    pref_keywords = self.pref_keywords
                    seekerk_query_list = seeker_query.split(' ')

                    #identify common keywords list for intent recognition
                    common_tokens = list(set(seekerk_query_list).intersection(keywords))
                    pref_common_tokens = list(set(seekerk_query_list).intersection(pref_keywords))

                    #Rules starts here
                    # chit-chat inquiry
                    if seeker_query.__contains__('how are you') and not is_movie_mentioned:
                        gen_sen = self.genenrate_sentence(chit_chat_response)
                        completed_dialogs_list.append(gen_sen + '\n')
                        continue

                    # generate recommendation on asking for any recommendation
                    if (seeker_query.__contains__('any movie') or seeker_query.__contains__('good movies') or seeker_query.__contains__('good movie') or seeker_query.__contains__('any genre')or seeker_query.__contains__('new movies') or seeker_query.__contains__('all movies') or seeker_query.__contains__('any type') or seeker_query.__contains__('no preference')) and not is_movie_mentioned and len(pref_common_tokens) < 1:
                        mentioned_genre= 'Action'
                        movie_title = self.get_recommendation_by_genre(mentioned_genre)
                        gen_sen = self.genenrate_sentence(recommend_anything)
                        gen_sen = gen_sen.split('$')[0] + mentioned_genre + gen_sen.split('$')[1] +'"'+ movie_title+'"'
                        completed_dialogs_list.append(gen_sen + '\n')
                        is_already_movie_recommended = True
                        continue

                    # response to seeker questions about movie types, movie metadata, personal experience etc.
                    if seeker_query.__contains__('who is') or seeker_query.__contains__('who s') or seeker_query.__contains__('what is') or seeker_query.__contains__('what s') or seeker_query.__contains__('what it about') or seeker_query.__contains__('did you') or seeker_query.__contains__('how are'):
                        gen_sen= self.genenrate_sentence(response_to_seeker)
                        completed_dialogs_list.append(gen_sen + '\n')
                        continue

                    # response to seeker's confirmatory utterances
                    if seeker_query.__contains__('is that') or seeker_query.__contains__('does it') or seeker_query.__contains__('are they') or seeker_query.__contains__('is it') or seeker_query.__contains__('do you think'):
                        gen_sen= self.genenrate_sentence(confirmatory_responses)
                        completed_dialogs_list.append(gen_sen + '\n')
                        continue

                    # generate recommendation if movie id mentioned in seeker query
                    if is_movie_mentioned:
                        movie_title = self.get_recommendation(movieid)
                        gen_sen = self.genenrate_sentence(recommend) +'"'+ movie_title+'"'
                        completed_dialogs_list.append(gen_sen + '\n')
                        is_already_movie_recommended = True
                        continue

                    # when any common intent recongnition keyword found
                    if any(common_tokens) or any(pref_common_tokens):
                        res = np.array(common_tokens)
                        common_tokens = np.unique(res)
                        recommend_keywords = ['looking', 'suggestions', 'suggest', 'suggestion',
                                              'recommend', 'recommendations']
                        recommend_common_tokens = ([value for value in common_tokens if value in recommend_keywords])

                        #check if ask recommendation keywords found
                        if len(recommend_common_tokens) > 0 or len(pref_common_tokens) > 0:
                            if (seeker_query.__contains__('thanks') or seeker_query.__contains__('thank you')) and len(pref_common_tokens) <1:
                                gen_sen = self.genenrate_sentence(chit_chat_compliment)
                                completed_dialogs_list.append(gen_sen + '\n')

                            elif is_already_movie_recommended and len(gt_mentioned_movies) > 0 and len(pref_common_tokens) < 1:
                                movieid = self.get_movie_id(gt_mentioned_movies)
                                movie_title = self.get_recommendation(movieid)
                                gen_sen = self.genenrate_sentence(recommend) +'"'+ movie_title+'"'
                                completed_dialogs_list.append(gen_sen + '\n')
                                is_already_movie_recommended = True


                            elif is_already_movie_recommended and len(seeker_mentioned_movies) > 0 and len(pref_common_tokens) < 1:
                                movieid = self.get_movie_id(seeker_mentioned_movies)
                                movie_title = self.get_recommendation(movieid)
                                gen_sen = self.genenrate_sentence(recommend) +'"'+ movie_title+'"'
                                completed_dialogs_list.append(gen_sen + '\n')
                                is_already_movie_recommended = True

                            else:
                                if len(pref_common_tokens) > 0:
                                    mentioned_genre= pref_common_tokens[len(pref_common_tokens)-1]
                                    movie_title = self.get_recommendation_by_genre(mentioned_genre)
                                    gen_sen = self.genenrate_sentence(recommend_anything)
                                    gen_sen = gen_sen.split('$')[0] + mentioned_genre + gen_sen.split('$')[1] +'"'+ movie_title+'"'
                                    completed_dialogs_list.append(gen_sen + '\n')
                                    is_already_movie_recommended = True
                                else:
                                    gen_sen = self.genenrate_sentence(ask_preference).replace('Hi, ','').replace('Hello, ','').capitalize()
                                    completed_dialogs_list.append(gen_sen + '\n')

                        # check if seeker is asking for recommendation using non-trivial ways
                        elif (seeker_query.__contains__('like to watch') or seeker_query.__contains__('help me')) and len(common_tokens) < 1:
                            if is_movie_mentioned_GT:
                                movieid = self.get_movie_id(gt_mentioned_movies)
                                movie_title = self.get_recommendation(movieid)
                                gen_sen = self.genenrate_sentence(recommend) +'"'+'"'+ movie_title+'"'+'"'
                                completed_dialogs_list.append(gen_sen + '\n')
                                is_already_movie_recommended = True
                                is_movie_mentioned_GT = False
                            else:
                                gen_sen = self.genenrate_sentence(response_to_seeker)
                                completed_dialogs_list.append(gen_sen + '\n')

                        # generate response on seeker initiation request
                        elif common_tokens.__contains__('hi') or common_tokens.__contains__('hello') or common_tokens.__contains__('hey'):
                            if len(pref_common_tokens) > 0:
                                mentioned_genre= pref_common_tokens[len(pref_common_tokens)-1]
                                movie_title = self.get_recommendation_by_genre(mentioned_genre)
                                gen_sen = self.genenrate_sentence(recommend_anything)
                                gen_sen = gen_sen.split('$')[0] + mentioned_genre + gen_sen.split('$')[1] +'"'+ movie_title+'"'
                                completed_dialogs_list.append(gen_sen + '\n')
                                is_already_movie_recommended = True
                            else:
                                gen_sen = self.genenrate_sentence(ask_preference).replace('Hi, ','').replace('Hello, ','').capitalize()
                                completed_dialogs_list.append(gen_sen + '\n')

                        # generate encourage statements
                        elif (common_tokens.__contains__('check') and common_tokens.__contains__('out')) or seeker_query.__contains__('will try') or seeker_query.__contains__('will check') or seeker_query.__contains__('ll check')or seeker_query.__contains__('ill check'):
                                gen_sen = self.genenrate_sentence(encourage)
                                completed_dialogs_list.append(gen_sen + '\n')

                        elif len(common_tokens) > 0:
                            tokens_list1 = ['thanks', 'thank', 'you']
                            tokens_list2 = ['bye', 'goodbye', 'day' ,'good','night']
                            if any([substring in common_tokens for substring in tokens_list1]):
                                gen_sen = self.genenrate_sentence(chit_chat_compliment)
                                completed_dialogs_list.append(gen_sen + '\n')

                            elif any([substring in common_tokens for substring in tokens_list2]):
                                gen_sen = self.genenrate_sentence(chit_chat_bye)
                                completed_dialogs_list.append(gen_sen + '\n')
                            else:
                                gen_sen = self.genenrate_sentence(confirmatory_responses)
                                completed_dialogs_list.append(gen_sen + '\n')

                    ### when no common keyword found
                    elif is_ask_preference:
                        if is_movie_mentioned_GT:
                            movieid = self.get_movie_id(gt_mentioned_movies)
                            movie_title = self.get_recommendation(movieid)
                            gen_sen = self.genenrate_sentence(recommend) +'"'+ movie_title+'"'
                            completed_dialogs_list.append(gen_sen + '\n')
                            is_already_movie_recommended = True
                        else:
                            if len(pref_common_tokens) > 0:
                                mentioned_genre= pref_common_tokens[len(pref_common_tokens)-1]
                                movie_title = self.get_recommendation_by_genre(mentioned_genre)
                                gen_sen = self.genenrate_sentence(recommend_anything)
                                gen_sen = gen_sen.split('$')[0] + mentioned_genre + gen_sen.split('$')[1] +'"'+ movie_title+'"'
                                completed_dialogs_list.append(gen_sen + '\n')
                                is_already_movie_recommended = True
                            else:
                                gen_sen = self.genenrate_sentence(ask_preference).replace('Hi, ','').replace('Hello, ','').capitalize()
                                completed_dialogs_list.append(gen_sen + '\n')

                    elif (seeker_query.__contains__('like to watch') or seeker_query.__contains__('help me')) and len(common_tokens) < 1:
                            if len(gt_mentioned_movies) == 0:
                                if len(pref_common_tokens) > 0:
                                    mentioned_genre= pref_common_tokens[len(pref_common_tokens)-1]
                                    movie_title = self.get_recommendation_by_genre(mentioned_genre)
                                    gen_sen = self.genenrate_sentence(recommend_anything)
                                    gen_sen = gen_sen.split('$')[0] + mentioned_genre + gen_sen.split('$')[1] +'"'+ movie_title+'"'
                                    completed_dialogs_list.append(gen_sen + '\n')
                                    is_already_movie_recommended = True
                                else:
                                    gen_sen = self.genenrate_sentence(ask_preference).replace('Hi, ','').replace('Hello, ','').capitalize()
                                    completed_dialogs_list.append(gen_sen + '\n')
                            else:
                                movieid = self.get_movie_id(gt_mentioned_movies)
                                movie_title = self.get_recommendation(movieid)
                                gen_sen = self.genenrate_sentence(recommend) +'"'+ movie_title+'"'
                                completed_dialogs_list.append(gen_sen + '\n')
                                is_already_movie_recommended = True
                                is_movie_mentioned_GT = False

                    elif seeker_query.__contains__('have a'):
                        kwd =['great', 'nice','wonderful','wonderfully', 'good']
                        if any(list(set(seekerk_query_list).intersection(kwd))):
                            gen_sen = self.genenrate_sentence(chit_chat_bye)
                            completed_dialogs_list.append(gen_sen + '\n')
                        else:
                            gen_sen = self.genenrate_sentence(confirmatory_responses)
                            completed_dialogs_list.append(gen_sen + '\n')
                    else:
                        if is_movie_mentioned_GT:
                            movieid = self.get_movie_id(gt_mentioned_movies)
                            movie_title = self.get_recommendation(movieid)
                            gen_sen = self.genenrate_sentence(recommend) +'"'+ movie_title+'"'
                            completed_dialogs_list.append(gen_sen + '\n')
                            is_already_movie_recommended = True
                            is_movie_mentioned_GT = False
                        elif not is_already_movie_recommended:
                            if len(pref_common_tokens) > 0:
                                mentioned_genre= pref_common_tokens[len(pref_common_tokens)-1]
                                movie_title = self.get_recommendation_by_genre(mentioned_genre)
                                gen_sen = self.genenrate_sentence(recommend_anything)
                                gen_sen = gen_sen.split('$')[0] + mentioned_genre + gen_sen.split('$')[1] +'"'+ movie_title+'"'
                                completed_dialogs_list.append(gen_sen + '\n')
                                is_already_movie_recommended = True
                            else:
                                gen_sen = self.genenrate_sentence(ask_preference).replace('Hi, ','').replace('Hello, ','').capitalize()
                                completed_dialogs_list.append(gen_sen + '\n')
                        else:
                             gen_sen = self.genenrate_sentence(confirmatory_responses)
                             completed_dialogs_list.append(gen_sen + '\n')

                ## Ground truth sentences handled ####
                # preserve the ground truth utterance's context
                # 1) store mentioned movie IDs
                # 2) store previous GT preference keyword
                # 3) store previous recommendation token intent
                else:
                    previous_gt_sentence = seeker_line
                    gt_query = self.gt_sentence_parser(seeker_line)
                    gt_query = re.sub('[^A-Za-z0-9@]+', ' ', gt_query)
                    movieid = self.check_if_movieid_mention(gt_query)
                    is_ask_preference = False
                    if movieid:
                        is_movie_mentioned_GT = True
                        gt_mentioned_movies.append(movieid)

                    keywords = ['kind', 'type', 'category','looking', 'recommend', 'suggest']
                    gt_query_list = gt_query.split(' ')
                    common_tokens = list(set(gt_query_list).intersection(keywords))
                    # use this block of code if any token of ground truth response is matched with keywords list
                    if any(common_tokens):
                        is_ask_preference = True


            # if something goes wrong with the 'try' block, just skip this sentence and move on..
            except RuntimeError as err:
                print(err)
                continue
        return completed_dialogs_list

    #prase seeker utterance and return the actuall utterance
    def seeker_sentences_parser(self, line):
        if line:
            p = re.compile("SEEKER:(.*)").search(str(line))
            temp_line = p.group(1)
            m = re.compile('<s>(.*?)</s>').search(temp_line)
            seeker_line = m.group(1)
            seeker_line = seeker_line.lower().strip()
            return seeker_line

        #prase groud truth utterance and return the actuall utterance
    def gt_sentence_parser(self, line):
        try:
            if not line == '\n':
                p = re.compile("GROUND TRUTH:(.*)").search(str(line))
                temp_line = p.group(1)
                m = re.compile('<s>(.*?)</s>').search(temp_line)
                gt_line = m.group(1)
                gt_line = gt_line.lower().strip()
                # gt_line = re.sub('[^A-Za-z0-9]+', ' ', gt_line)
            else:
                gt_line = ""

            return gt_line
        except AttributeError as err:
            print('exception accured while parsing ground truth.. \n')
            print(line)
            print(err)

    #get recommendation based on movie id
    def get_recommendation(self, movie_id, identifier = None):
        try:
            if self.rec is None:
                self.rec = Recommender_MF()
                suggestion, suggestion_id = self.rec.get_similar_movies_based_on_content(movie_id,identifier, self.mentioned_recommended_movies)
            else:
                suggestion, suggestion_id = self.rec.get_similar_movies_based_on_content(movie_id,identifier,self.mentioned_recommended_movies)

            ##backup recommender in case movie is not present in movielens data, identifier means, recommendation request is actual
            if suggestion== '__unk__' and identifier != 0:
                if self.genre_rec is None:
                    self.genre_rec = Recommender()
                    suggestion, suggestion_id = self.genre_rec.get_similar_movies_based_on_content(movie_id,identifier,self.rec.is_session_changed, self.mentioned_recommended_movies)
                else:
                    suggestion, suggestion_id = self.genre_rec.get_similar_movies_based_on_content(movie_id,identifier,self.rec.is_session_changed,self.mentioned_recommended_movies)

            try:
                if identifier != 0 and len(suggestion_id) > 0: # means actuall recommendation request
                    self.mentioned_recommended_movies.append(suggestion_id)
                    return str(suggestion)
            except:
                return str(suggestion)
        except RuntimeError as err:
            print(err)
            suggestion = '__unk__'
        return suggestion

    #get recommendation by recommender system which take genre type as an input
    def get_recommendation_by_genre(self, genre):
        try:
            if self.rec_only_genre is None:
                self.rec_genre = Recommender_with_genre()
                suggestion, suggestion_id = self.rec_genre.get_similar_movies_based_on_genre(genre.title(),self.mentioned_recommended_movies)
        except RuntimeError as err:
            print(err)
            suggestion = '__unk__'
        self.mentioned_recommended_movies.append(suggestion_id)
        return suggestion


    #check if valid movie id in the utterance
    def check_if_movieid_mention(self, query):
        # check if there is a movie mention
        is_negation = False
        movieid = ''
        if query.__contains__('@'):
            IDs = re.findall('@\\S+', query)
            parsed_IDs = [id.replace('@', '') for id in IDs]
            self.mentioned_recommended_movies.extend(parsed_IDs)   # add all mentioned and recommended movie ids to the list

            #check if there is negation in the query, if yes, discard that ID
            if query.__contains__('dont like') or query.__contains__("don t like"):
                is_negation = True
            if is_negation:
                IDs = IDs[0:len(IDs)-1]

            if len(IDs) > 0:
                movieid = IDs[0].split('@')[1]
                for i, id in enumerate(IDs):
                    index = (len(IDs)-1)-i
                    id = IDs[index]
                    movieid = id.split('@')[1]
                    suggestions = self.get_recommendation(movieid, identifier = 0)
                    if suggestions != '__unk__':
                        return movieid
                    elif suggestions == '__unk__' and i == len(IDs) - 1:   # i: to handle multiple movie ids in a single utterance
                        return movieid
                    elif suggestions == '__unk__':
                        continue

                movieid =''
            else:
                movieid = ''

        return movieid

    #load utterances types
    # in total we have 53 unique sentences
    def load_utterances_data(self):
        recommend = []
        ask_preference = []
        chit_chat_compliment = []
        chit_chat_bye = []
        response_to_seeker_question = []
        encourage_seeker = []
        recommend_anything = []
        chit_chat_complimentary = []
        confirmatory_responses = []
        # based on ask for preference intent (4)
        ask_preference.append('Hello, what type of movies do you like?')
        ask_preference.append('Hi, what kind of movies are you looking for?')
        ask_preference.append('Hi, what type of movies are you interested in?')
        ask_preference.append('Hello, have you seen any good movies lately?')

        # based on recommendation intent (9)
        recommend.append('Have you seen ')
        recommend.append('If you like that one, you should try ')
        recommend.append('I think you might like ')
        recommend.append('Have you ever seen  ')
        recommend.append('How about  ')
        recommend.append("A really good one is ")
        recommend.append('I would recommend  ')
        recommend.append('Another great one is  ')
        recommend.append('I think you should watch  ')
        # chit-chat compliment intent (9)
        chit_chat_compliment.append('Welcome!')
        chit_chat_compliment.append('You are welcome. Have a good night!')
        chit_chat_compliment.append('You are welcome!')
        chit_chat_compliment.append('you are very welcome. Goodbye')
        chit_chat_compliment.append('You are welcome! Bye!')
        chit_chat_compliment.append('You are very welcome. You too!')
        chit_chat_compliment.append('Have a nice evening! bye.')
        chit_chat_compliment.append('Thank you. Have a nice evening.')
        chit_chat_compliment.append('Ok, have a nice day. bye.')

        #open genre recommendations (5)
        recommend_anything.append("If you like $ movies, you should try ")
        recommend_anything.append("If you like $, you should also check out ")
        recommend_anything.append("If you like $ movies, may be you would like ")
        recommend_anything.append("If you are looking for something $, I can recommend ")
        recommend_anything.append("Are you looking for $ movies? I like ")
        # seeker aid (6)
        encourage_seeker.append('I hope you have fun! enjoy it.')
        encourage_seeker.append('I am sure you will enjoy them!')
        encourage_seeker.append('I hope you will enjoy them.')
        encourage_seeker.append('I think you will enjoy them!')
        encourage_seeker.append('I hope you like them.')
        encourage_seeker.append('I hope I gave you some choices to start with.')

        # seeker response (4)
        confirmatory_responses.append("It's pretty good.")
        confirmatory_responses.append("Yes! It is good.")
        confirmatory_responses.append("It is a good one.")
        confirmatory_responses.append("Yes! It's really good.")

        # response to seeker questions (5)
        response_to_seeker_question.append("I have'nt seen it yet.")
        response_to_seeker_question.append("ok, I haven't seen that one yet but I heard it was good.")
        response_to_seeker_question.append("I have'nt seen that, but I heard really good things.")
        response_to_seeker_question.append('I heard about that, and yes I have to check it out.')
        response_to_seeker_question.append("I have seen that one. It is good.")

        # chit-chat quiet (8)
        chit_chat_bye.append('Bye')
        chit_chat_bye.append('Goodbye')
        chit_chat_bye.append('NO problem. Bye.')
        chit_chat_bye.append('You too! Good Bye!')
        chit_chat_bye.append('Have a nice day, bye.')
        chit_chat_bye.append('Hope you have a nice day.')
        chit_chat_bye.append('Bye! have a nice day.')
        chit_chat_bye.append('I hope you enjoy and have a nice day. Bye!')

        # complementory responses (3)
        chit_chat_complimentary.append("I am good. Thanks!")
        chit_chat_complimentary.append("I'm good. Thanks!")
        chit_chat_complimentary.append("I'm great. Thanks!")

        return chit_chat_bye, chit_chat_compliment, ask_preference, recommend, response_to_seeker_question, encourage_seeker, recommend_anything, chit_chat_complimentary, confirmatory_responses

    #generate sentences based on intent type
    def genenrate_sentence(self, intent_list, genre = ''):
        n = random.randint(0, len(intent_list) - 1)
        sentence = intent_list[n]
        return sentence

    def get_movie_id(self, movie_mentioned_list):
        n = random.randint(0, len(movie_mentioned_list) - 1)
        return movie_mentioned_list[len(movie_mentioned_list) - 1]

    #read and retrive dialogs from the input file
    def read_dialogs_input_file(self, file_name):
        is_visited = False
        redial_dialogues = []
        dialog = []
        with open(file_name, 'r') as input:
            for line in input:
                if not line.strip(): continue
                if 'CONVERSATION:' in line and is_visited:
                    redial_dialogues.append(dialog)
                    dialog = []
                    dialog.append(line)
                    is_visited = False
                else:
                    dialog.append(line)
                    is_visited = True
        if not dialog[0].__contains__('CONVERSATION:'):
            return
        redial_dialogues.append(dialog)
        return redial_dialogues

    def process_dialog(self,input_dialogs):
        final_dialogues = []
        for dlg in input_dialogs:
            print('new dialog is processing.......')
            if dlg[1].__contains__('SEEKER:'):
                processed_dialogue = self.parse_generated_dialogues_sq(dlg)
                final_dialogues.append(processed_dialogue)
            elif dlg[1].__contains__('GROUND TRUTH:'):
                processed_dialogue = self.parse_generated_dialogues_gt(dlg)
                final_dialogues.append(processed_dialogue)
            if self.rec is not None:
                self.rec.is_session_changed = True
        return final_dialogues
    #replace movie IDs with their titles
    def replace_movieIDs_with_titles(self, input_dialogue_data):
        df = pd.DataFrame()
        # reading csv data file
        df = pd.read_csv('../recommenders_item_data/movies_with_mentions.csv', ',', encoding='utf-8', dtype=object, low_memory=False)
        final_dialogues = []
        for dlg in input_dialogue_data:
            lines = []
            for row in dlg:
                try:
                    # remove substrings movie IDs e.g. @1234
                    if "@" in row:
                        ids = re.findall(r'@\S+', row)
                        for id in ids:
                            id = re.sub('[^0-9@]+', '', id)
                            onlyid = id.split('@')[1]
                            temp = df[df['databaseId'] == str(onlyid)]
                            if len(temp) > 0:
                                movieName = temp['movieName'].iloc[0]
                                # m = movieName.index('(')
                                # movieName = movieName[:m]
                                row = row.replace(id, '"' +movieName+'"')

                        lines.append(row)
                    else:
                        lines.append(row)
                except:
                    lines.append(row)
                    continue
            final_dialogues.append(lines)
        print('execution ends here')
        return final_dialogues


## main function to start the process and write the output of the program in the .csv file
if __name__ == '__main__':
    obj = Retrieval_system()
    redial_input_dialogues = obj.read_dialogs_input_file('../dialogs_data/input_dialogs_study.txt')
    if redial_input_dialogues != None:
        if len(redial_input_dialogues) > 0:
            new_retrieved_dialogs = obj.process_dialog(redial_input_dialogues)
            retrieval_based_dialog = obj.replace_movieIDs_with_titles(new_retrieved_dialogs)
            #write genenerated dialogs in the text file
            with open('../output_dialogs/Generated_Dialogues_temp.csv', 'w') as filehandle:
                for dia in retrieval_based_dialog:
                    filehandle.writelines("%s" % line for line in dia)
            print('Dialogs have been generated successfully.')
        else:
            print('Please insert dialog conversation in the input file (i.e. input_dalogs_study) first.')
    else:
        print('Please check the input dialog format.')
