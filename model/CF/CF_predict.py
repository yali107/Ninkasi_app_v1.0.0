'''
Recommendations based on CF
By Luke Chu and Nelson Chen 12/20/16 for Beer Recommendation project
'''
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import pairwise as pw

def order_list(seq):
    checked = []
    for e in seq:
        if e not in checked:
            checked.append(e)
    return checked

def CF_mat_preprocess(ratings_mat):
    global_avg = np.mean(ratings_mat)
    user_bias = np.sum(ratings_mat, axis=1) / ratings_mat.shape[1]
    ratings_mat = ratings_mat + global_avg
    ratings_mat = ratings_mat - np.expand_dims(user_bias, 1)
    return ratings_mat, global_avg

# this function sucks dick
def CF_user_preprocess(user_inp, ratings_mat, beer_names):

    beer_name_inp = [i[0] for i in user_inp]
    rating_inp = [i[1] for i in user_inp]
    # print 'list comprehension done'
    user_data = np.repeat(np.nan, ratings_mat.shape[1])
    # print 'np.repeat done   '
    beer_idx = [beer_names.keys()[beer_names.values().index(i)] for i in beer_name_inp]
    for i, j in enumerate(beer_idx):
        user_data[j] = rating_inp[i]

    return user_data


def CF_rec(user_data, ratings, global_avg, beer_names, neighbors=10, num_recs=5):

    # Find indices of observed and missing values
    index = np.where(~np.isnan(user_data))[0]
    missing_index = np.where(np.isnan(user_data))[0]

    # take items as columns
    ratings_mat_red = ratings[:, index]
    ratings_mat_miss = ratings[:, missing_index]

    user_data_new = user_data[index]

    # normalize via global avg mu
    user_data_new = user_data_new - user_data_new.mean() + global_avg
    user_data_new = user_data_new.reshape(1, -1)
    # print 'done'
    # print user_data_new
    # print user_data_new.reshape(-1, 1)
    # print user_data_new.reshape(1, -1)

    # compute euclidean distance and cosine similarity
    pw_cos = pw.cosine_similarity(ratings_mat_red,user_data_new).flatten()
    # print 'done2'

    # largest values are most similar users
    pw_cos_df = pd.DataFrame([pw_cos]).transpose()
    cos_topn = pw_cos_df.nlargest(neighbors,0)

    # turn distance/similarity into weights
    # might need to inspect distance again, right now I simply reversed the weights
    cos_weights = np.matrix(cos_topn / sum(cos_topn[0]))

    # get ratings from the top n users for the missing beers
    cos_miss_ratings = ratings_mat_miss[cos_topn.index,:]

    # weigh ratings
    cos_new_ratings = pd.DataFrame(cos_miss_ratings.transpose() * cos_weights)

    # find top rating(s)
    cos_new_index = cos_new_ratings.nlargest(num_recs, 0).index[:]

    beer_ind_cos = missing_index[cos_new_index]

    # match index with beer name
    return [beer_names[ind].decode("utf-8", "ignore").encode("ascii", "ignore") for ind in beer_ind_cos]

if __name__ == '__main__':

    # load matrix here
    ratings_mat = np.load('ratings_svdpp.npy')
    print('mat loading complete')
    # global_avg = np.mean(ratings_mat)
    # user_bias = np.sum(ratings_mat, axis=1) / ratings_mat.shape[1]
    # ratings_mat = ratings_mat + global_avg
    # ratings_mat = ratings_mat - np.expand_dims(user_bias, 1)

    # load beer dictionary here
    with open('beer_dict.pickle', 'rb') as f:
        beer_names = pickle.load(f)
    print('dict loading complete')

    test_users = [('Yazoo Embrace the Funk Series: Deux Rouges', 20), ("Iron Hill Bourbon Porter", 3)]

    rating_mat, global_avg = CF_mat_preprocess(ratings_mat)
    print('CF_mat_preprocess done')
    user_data = CF_user_preprocess(test_users, ratings_mat, beer_names)
    print('CF_user_preprocess done')
    # print sum(user_data)
    #print beer_names

    # # test case
    # homer = np.repeat(np.nan, ratings_mat.shape[1])
    # for i in list(np.random.randint(0, ratings_mat.shape[1], 10)):
    #     homer[i] = np.random.randint(1, 20, 1)[0]

    print(CF_rec(user_data, ratings_mat, global_avg, beer_names))





