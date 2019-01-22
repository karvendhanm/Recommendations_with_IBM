import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

data_dir = "C:/Users/John/PycharmProjects/Recommendations_with_IBM/data/"

df = pd.read_csv(data_dir + 'user-item-interactions.csv')
df_content = pd.read_csv(data_dir + 'articles_community.csv')

del df['Unnamed: 0']
del df_content['Unnamed: 0']

# Get an idea of the data
df.head()
df_content.head()

user_article_interaction = (df.groupby('email')['article_id'].count()).sort_values(ascending=True)
num_of_articles = max(user_article_interaction[:int(len(user_article_interaction)/2)])


# distribution of how many articles a user interacts with in the dataset
n_count_ = defaultdict(int)
for n_article in user_article_interaction.unique():
    n_count_[n_article] = sum(user_article_interaction == n_article)

df_n_articles_n_users = pd.DataFrame.from_dict({'number_of_articles':list(n_count_.keys()), 'number_of_users':list(n_count_.values())})
print(df_n_articles_n_users.head(10))

# TODO find a better way for visualization here
# distribution of how many articles a user interacts with in the dataset
# fig = plt.figure();
# ax = fig.add_subplot(1, 1, 1);
# ax.hist(user_article_interaction, normed = True, bins = 30);
# ax.set_xscale('log');
# plt.xlabel('Number of Articles');
# plt.ylabel('Percentage of Users');

max_n_articles_by_an_user = max(user_article_interaction)

# Remove any rows that have the same article_id - only keep the first
value, count = np.unique(df_content['article_id'], return_counts=True)

duplicates = []
for idx,n in enumerate(count):
    if n > 1:
        duplicates.append(value[idx])

dict_duplicate_index = {}
for duplicate in duplicates:
    dict_duplicate_index[duplicate] = df_content[df_content['article_id'].isin([duplicate])].index.tolist()
    df_content.drop(index = max(dict_duplicate_index[duplicate]), axis = 0, inplace = True)

# The number of unique users
n_unique_users = len(df['email'][~df['email'].isna()].unique())

# The number of unique articles on the IBM platform
total_articles = len(df_content['article_id'].unique())

# The number of unique articles that have at least one interaction
unique_articles = len(df['article_id'].unique())

# The number of user-article interactions
user_article_interactions = df.shape[0]

# Use the cells below to find the most viewed article_id, as well as how often it was viewed.
# After talking to the company leaders, the email_mapper function was deemed a reasonable way to map users to ids.
# There were a small number of null values, and it was found that all of these null values likely belonged to a single user
# (which is how they are stored using the function below).

most_viewed_article_id = str(df['article_id'].value_counts().index[0])
max_views = df['article_id'].value_counts()[df['article_id'].value_counts().index[0]]


def email_mapper():
    coded_dict = dict()
    cter = 1
    email_encoded = []

    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter += 1

        email_encoded.append(coded_dict[val])
    return email_encoded


email_encoded = email_mapper()
del df['email']
df['user_id'] = email_encoded

#df.to_csv(data_dir + 'new_user_item_interactions.csv', index = False, index_label = False)
# does same user interact with the same article multiple number of times - YES
series = df.groupby(['user_id'])['article_id'].unique()
df.groupby(['user_id'])['article_id'].count()
series.apply(lambda x: len(x)) == df.groupby(['user_id'])['article_id'].count()

# Rank-Based Recommendations
def get_top_articles(n, df=df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook

    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles

    '''
    top_articles = df['title'].value_counts().index[:n].tolist()

    return top_articles  # Return the top article titles from df (not df_content)


def get_top_article_ids(n, df=df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook

    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles

    '''
    top_articles = df['article_id'].value_counts().index[:n].tolist()

    return top_articles  # Return the top article ids

# User-User Based Collaborative Filtering
# we will use either pearson's correlation coefficient, spearman's correlation coefficient, or kendall's tau
# We could even use distance metrics like euclidean distance or manhattan distance

# create the user-article matrix with 1's and 0's
def create_user_item_matrix(df):
    '''
    INPUT:
    df - pandas dataframe with article_id, title, user_id columns

    OUTPUT:
    user_item - user item matrix

    Description:
    Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with
    an article and a 0 otherwise
    '''
    df['values'] = 1
    user_item = df.groupby(['user_id', 'article_id'])['values'].max().unstack().fillna(0)

    return user_item  # return the user_item matrix


user_item = create_user_item_matrix(df)


def find_similar_users(user_id, user_item=user_item):
    '''
    INPUT:
    user_id - (int) a user_id
    user_item - (pandas dataframe) matrix of users by articles:
                1's when a user has interacted with an article, 0 otherwise

    OUTPUT:
    similar_users - (list) an ordered list where the closest users (largest dot product users)
                    are listed first

    Description:
    Computes the similarity of every pair of users based on the dot product
    Returns an ordered

    '''
    idx = np.where(user_item.index == user_id)[0][0]

    # compute similarity of each user to the provided user
    dot_prod_users_articles = np.dot(user_item.values, np.transpose(user_item.values))

    # sort by similarity
    most_similar_users = dot_prod_users_articles[idx, :]

    # create list of just the ids
    most_similar_users = most_similar_users.argsort()
    most_similar_users = most_similar_users[::-1] + 1

    # remove the own user's id
    most_similar_users = most_similar_users.tolist()
    del most_similar_users[most_similar_users.index(user_id)]

    return most_similar_users  # return a list of the users in order from most to least similar


def get_article_names(article_ids, df=df):
    '''
    INPUT:
    article_ids - (list) a list of article ids
    df - (pandas dataframe) df as defined at the top of the notebook

    OUTPUT:
    article_names - (list) a list of article names associated with the list of article ids
                    (this is identified by the title column)
    '''
    article_names = df[df['article_id'].isin(article_ids)]['title'].unique().tolist()

    return article_names  # Return the article names associated with list of article ids


def get_user_articles(user_id, user_item=user_item):
    '''
    INPUT:
    user_id - (int) a user id
    user_item - (pandas dataframe) matrix of users by articles:
                1's when a user has interacted with an article, 0 otherwise

    OUTPUT:
    article_ids - (list) a list of the article ids seen by the user
    article_names - (list) a list of article names associated with the list of article ids
                    (this is identified by the doc_full_name column in df_content)

    Description:
    Provides a list of the article_ids and article titles that have been seen by a user
    '''
    idx = np.where(user_item.index == user_id)[0][0]
    np.where(user_item.iloc[idx, :] == 1)
    article_ids = user_item.columns[np.where(user_item.iloc[idx, :] == 1)].tolist()
    article_names = get_article_names(article_ids)
    article_ids = [str(i) for i in article_ids]
    return article_ids, article_names  # return the ids and names


def user_user_recs(user_id, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user

    OUTPUT:
    recs - (list) a list of recommendations for the user

    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found

    Notes:
    Users who are the same closeness are chosen arbitrarily as the 'next' user

    For the user where the number of recommended articles starts below m
    and ends exceeding m, the last items are chosen arbitrarily

    '''

    neighbors = find_similar_users(user_id)
    seen_article_ids, seen_article_names = get_user_articles(user_id)
    recs = np.array([])
    for user in neighbors:
        article_ids, article_names = get_user_articles(user)
        new_recs = np.setdiff1d(article_ids, seen_article_ids, assume_unique=True)
        recs = np.unique(np.concatenate([new_recs, recs], axis=0))

        if len(recs) > m - 1:
            break

    return recs[:m].tolist()  # return your recommendations for this user_id


def get_top_sorted_users(user_id, df=df, user_item=user_item):
    '''
    INPUT:
    user_id - (int)
    df - (pandas dataframe) df as defined at the top of the notebook
    user_item - (pandas dataframe) matrix of users by articles:
            1's when a user has interacted with an article, 0 otherwise


    OUTPUT:
    neighbors_df - (pandas dataframe) a dataframe with:
                    neighbor_id - is a neighbor user_id
                    similarity - measure of the similarity of each user to the provided user_id
                    num_interactions - the number of articles viewed by the user - if a u

    Other Details - sort the neighbors_df by the similarity and then by number of interactions where
                    highest of each is higher in the dataframe

    '''
    neighbors = find_similar_users(user_id)
    user_arr = user_item.iloc[np.where(user_item.index == user_id)[0][0], :]
    similarity_score = []
    for neighbour in neighbors:
        neighbor_arr = user_item.iloc[np.where(user_item.index == neighbour)[0][0], :]
        similarity_score.append(np.corrcoef(user_arr, neighbor_arr)[0][1])

    n_interactions = user_item[user_item.index.isin(neighbors)].sum(axis=1).tolist()

    neighbors_df = pd.DataFrame({"neighbor_id": neighbors, "similarity_score": similarity_score, "n_article_interaction": n_interactions})
    neighbors_df.sort_values(by=['similarity_score', 'n_article_interaction'], axis=0, inplace=True, ascending=[False, False])
    neighbors_df.reset_index(drop = True ,inplace = True)

    return neighbors_df  # Return the dataframe specified in the doc_string

def rank_articles(article_ids):
    '''

    :param article_ids:
    :return:
    '''
    df_temp = df.groupby('article_id')['user_id'].count().sort_values(ascending=False, axis=0)
    sorted_article_ids = df_temp[df_temp.index.isin(article_ids)].index.tolist()
    return sorted_article_ids


def user_user_recs_part2(user_id, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user

    OUTPUT:
    recs - (list) a list of recommendations for the user by article id
    rec_names - (list) a list of recommendations for the user by article title

    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found

    Notes:
    * Choose the users that have the most total article interactions
    before choosing those with fewer article interactions.

    * Choose articles with the articles with the most total interactions
    before choosing those with fewer total interactions.

    '''
    neighbors_df = get_top_sorted_users(1)
    seen_article_ids, seen_article_names = get_user_articles(user_id)
    recs = np.array([])
    for neighbor_id in neighbors_df['neighbor_id']:
        article_ids, article_names = get_user_articles(neighbor_id)
        new_recs = np.setdiff1d(article_ids, seen_article_ids, assume_unique=True)
        new_recs = rank_articles(list(new_recs))
        recs = np.unique(np.concatenate([new_recs, recs], axis=0))

        if len(recs) > m - 1:
            break

    return recs[:m].tolist(), get_article_names(recs[:m].tolist())


#  Matrix Factorization

# Load the matrix here
user_item_matrix = pd.read_pickle(data_dir + 'user_item_matrix.p')

# quick look at the matrix
user_item_matrix.head()

u, s, vt = np.linalg.svd(user_item_matrix, full_matrices=True)

num_latent_feats = np.arange(10, 700 + 10, 20)
sum_errs = []

for k in num_latent_feats:
    # restructure with k latent features
    s_new, u_new, vt_new = np.diag(s[:k]), u[:, :k], vt[:k, :]

    # take dot product
    user_item_est = np.around(np.dot(np.dot(u_new, s_new), vt_new))

    # compute error for each prediction to actual value
    diffs = np.subtract(user_item_matrix, user_item_est)

    # total errors and keep track of them
    err = np.sum(np.sum(np.abs(diffs)))
    sum_errs.append(err)

plt.plot(num_latent_feats, 1 - np.array(sum_errs) / df.shape[0]);
plt.xlabel('Number of Latent Features');
plt.ylabel('Accuracy');
plt.title('Accuracy vs. Number of Latent Features');








