# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: '182'
#     language: python
#     name: python3
# ---

# + [markdown] id="Vq31CdSRpgkI"
# # Customizing embeddings
#
# This notebook demonstrates one way to customize OpenAI embeddings to a particular task.
#
# The input is training data in the form of [text_1, text_2, label] where label is +1 if the pairs are similar and -1 if the pairs are dissimilar.
#
# The output is a matrix that you can use to multiply your embeddings. The product of this multiplication is a 'custom embedding' that will better emphasize aspects of the text relevant to your use case. In binary classification use cases, we've seen error rates drop by as much as 50%.
#
# In the following example, I use 1,000 sentence pairs picked from the SNLI corpus. Each pair of sentences are logically entailed (i.e., one implies the other). These pairs are our positives (label = 1). We generate synthetic negatives by combining sentences from different pairs, which are presumed to not be logically entailed (label = -1).
#
# For a clustering use case, you can generate positives by creating pairs from texts in the same clusters and generate negatives by creating pairs from sentences in different clusters.
#
# With other data sets, we have seen decent improvement with as little as ~100 training examples. Of course, performance will be better with  more examples.

# + [markdown] id="arB38jFwpgkK"
# # 0. Imports
# -



# + id="ifvM7g4apgkK"
# imports
from typing import List, Tuple  # for type hints
from scipy import stats
import os
import time
import json
import numpy as np  # for manipulating arrays
import pandas as pd  # for manipulating data in dataframes
import pickle  # for saving the embeddings cache
import plotly.express as px  # for plots
import random  # for generating run IDs
from sklearn.model_selection import train_test_split  # for splitting train & test data
import torch  # for matrix optimization
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity  # for embeddings
import math
# -

os.environ['OPENAI_API_KEY'] = "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"

# + [markdown] id="DtBbryAapgkL"
# ## 1. Inputs
#
# Most inputs are here. The key things to change are where to load your datset from, where to save a cache of embeddings to, and which embedding engine you want to use.
#
# Depending on how your data is formatted, you'll want to rewrite the process_input_data function.
# -

# dataset = {'text_1': [], 'text_2':[], 'label':[], 'readme':[]}
# for file in os.listdir("v1-alpha-full/"):
#     if file.endswith(".json"):
#         with open("v1-alpha-full/" + file) as f:
#             data_dict = json.load(f)
#         for example in data_dict['examples']:
#             for query in example['queries']:
#                 dataset['text_1'].append(query)
#                 dataset['text_2'].append(example['code'])
#                 dataset['label'].append(1)   
#                 dataset['readme'].append(data_dict['usable_readme'])
#         #df = pd.read_json("v1-alpha-full/" + file)

# df = pd.DataFrame.from_dict(dataset)
# df

# # df2 = pd.DataFrame.from_dict(dataset)
# # df2
df = pd.read_csv('data/df2.csv')
# + id="UzxcWRCkpgkM"
# input parameters
embedding_cache_path = "data/snli_embedding_cache.pkl"  # embeddings will be saved/loaded here
embedding_cache_path2 = "data/snli_embedding_cache2.pkl"  # embeddings will be saved/loaded here
default_embedding_engine = "babbage-similarity"  # choice of: ada, babbage, curie, davinci
num_pairs_to_embed = 100  # 1000 is arbitrary - I've gotten it to work with as little as ~100
local_dataset_path = "data/snli_1.0_train_2k.csv"  # download from: https://nlp.stanford.edu/projects/snli/

# def process_input_data(df: pd.DataFrame) -> pd.DataFrame:
#     # you can customize this to preprocess your own dataset
#     # output should be a dataframe with 3 columns: text_1, text_2, label (1 for similar, -1 for dissimilar)
#     # df["label"] = df["gold_label"]
#     # df = df[df["label"].isin(["entailment"])]
#     # where does the contradiction come from?
#     # df["label"] = df["label"].apply(lambda x: {"entailment": 1, "contradiction": -1}[x])
#     # df = df.rename(columns={"sentence1": "text_1", "sentence2": "text_2"})
#     # df = df[["text_1", "text_2", "label"]]
#     df = df.head(num_pairs_to_embed)
#     return df



# # + [markdown] id="aBbH71hEpgkM"
# # ## 2. Load and process input data

# # + id="kAKLjYG6pgkN" outputId="dc178688-e97d-4ad0-b26c-dff67b858966"
# # load data
# # df = pd.read_csv(local_dataset_path)

# # process input data
# df = process_input_data(df)  # this demonstrates training data containing only positives

# # view data
# df.head()


# # +
# df2 = process_input_data(df2)  # this demonstrates training data containing only positives

# # view data
# df2.head()


# # split data into train and test sets
# test_fraction = 0.5  # 0.5 is fairly arbitrary
# random_seed = 123  # random seed is arbitrary, but is helpful in reproducibility
# train_df2, test_df2 = train_test_split(
#     df2, test_size=test_fraction, stratify=df2["label"], random_state=random_seed
# )
# train_df2.loc[:, "dataset"] = "train"
# test_df2.loc[:, "dataset"] = "test"


# # + [markdown] id="MzAFkA2opgkP"
# # ## 4. Generate synthetic negatives
# #
# # This is another piece of the code that you will need to modify to match your use case.
# #
# # If you have data with positives and negatives, you can skip this section.
# #
# # If you have data with only positives, you can mostly keep it as is, where it generates negatives only.
# #
# # If you have multiclass data, you will want to generate both positives and negatives. The positives can be pairs of text that share labels, and the negatives can be pairs of text that do not share labels.
# #
# # The final output should be a dataframe with text pairs, where each pair is labeled -1 or 1.

# # + id="rUYd9V0zpgkP"
# # generate negatives
# def dataframe_of_negatives(dataframe_of_positives: pd.DataFrame) -> pd.DataFrame:
#     """Return dataframe of negative pairs made by combining elements of positive pairs."""
#     texts = set(dataframe_of_positives["text_1"].values) | set(
#         dataframe_of_positives["text_2"].values
#     )
#     all_pairs = {(t1, t2) for t1 in texts for t2 in texts if t1 < t2}
#     positive_pairs = set(
#         tuple(text_pair)
#         for text_pair in dataframe_of_positives[["text_1", "text_2"]].values
#     )
#     negative_pairs = all_pairs - positive_pairs
#     df_of_negatives = pd.DataFrame(list(negative_pairs), columns=["text_1", "text_2"])
    
    
#     df_of_negatives["label"] = -1
#     return df_of_negatives


# # + id="Rkh8-J89pgkP"
# negatives_per_positive = (
#     1  # it will work at higher values too, but more data will be slower
# )
# # generate negatives for training dataset
# train_df_negatives = dataframe_of_negatives(train_df)
# train_df_negatives["dataset"] = "train"
# # generate negatives for test dataset
# test_df_negatives = dataframe_of_negatives(test_df)
# test_df_negatives["dataset"] = "test"
# # sample negatives and combine with positives
# train_df = pd.concat(
#     [
#         train_df,
#         train_df_negatives.sample(
#             n=len(train_df) * negatives_per_positive, random_state=random_seed
#         ),
#     ]
# )
# test_df = pd.concat(
#     [
#         test_df,
#         test_df_negatives.sample(
#             n=len(test_df) * negatives_per_positive, random_state=random_seed
#         ),
#     ]
# )

# df = pd.concat([train_df, test_df])


# # +
# negatives_per_positive = (
#     1  # it will work at higher values too, but more data will be slower
# )
# # generate negatives for training dataset
# train_df2_negatives = dataframe_of_negatives(train_df2)
# train_df2_negatives["dataset"] = "train"
# # generate negatives for test dataset
# test_df2_negatives = dataframe_of_negatives(test_df2)
# test_df2_negatives["dataset"] = "test"
# # sample negatives and combine with positives
# train_df2 = pd.concat(
#     [
#         train_df2,
#         train_df2_negatives.sample(
#             n=len(train_df2) * negatives_per_positive, random_state=random_seed
#         ),
#     ]
# )
# test_df2 = pd.concat(
#     [
#         test_df2,
#         test_df2_negatives.sample(
#             n=len(test_df2) * negatives_per_positive, random_state=random_seed
#         ),
#     ]
# )

# df2 = pd.concat([train_df2, test_df2])

# # -

# df2.to_csv('data/df2.csv')

# + [markdown] id="8MVSLMSrpgkQ"
# ## 5. Calculate embeddings and cosine similarities
#
# Here, I create a cache to save the embeddings. This is handy so that you don't have to pay again if you want to run the code again.

# + id="R6tWgS_ApgkQ"
# establish a cache of embeddings to avoid recomputing
# cache is a dict of tuples (text, engine) -> embedding
embedding_cache_path = "data/snli_embedding_cache.pkl"  # embeddings will be saved/loaded here
try:
    with open(embedding_cache_path, "rb") as f:
        embedding_cache = pickle.load(f)
except FileNotFoundError:
    precomputed_embedding_cache_path = "https://cdn.openai.com/API/examples/data/snli_embedding_cache.pkl"
    embedding_cache = pd.read_pickle(precomputed_embedding_cache_path)

# this function will get embeddings from the cache and save them thereafterward
def get_embedding_with_cache(
    text: str,
    engine: str = default_embedding_engine,
    embedding_cache: dict = embedding_cache,
    embedding_cache_path: str = embedding_cache_path,
) -> list:
    if pd.isna(text):
        return None
    print(f"Getting embedding for {text}")
    if (text, engine) not in embedding_cache.keys():
        time.sleep(3)
        # if not in cache, call API to get embedding
        openai.api_key = os.getenv("OPENAI_API_KEY")
        # also delete the line 
        try: 
            embedding_cache[(text, engine)] = get_embedding(text, engine)
        except:
            return None

        # save embeddings cache to disk after each update
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(text, engine)]

def get_local_embeddings(str1, test_readme, N):
    stride = math.ceil(len(str1) / N)
    local_chunks = [str1[i:i+stride] for i in range(0, len(str1), stride)]
    local_embeddings = []
    glo, rea = get_embedding_with_cache(str1), get_embedding_with_cache(test_readme)
    if not glo or not rea:
        return None
    for chunk in local_chunks:
        chunk_em = get_embedding_with_cache(chunk)
        if not chunk_em:
            return None
        local_embeddings.append(chunk_em)
    global_embedding = np.vstack((glo, rea))
    global_embedding = global_embedding.T.mean(axis=1)
    local_embeddings, global_embedding = np.array(local_embeddings), np.array(global_embedding)
    local_global_cos = np.array([cosine_similarity(local_em, global_embedding) for local_em in local_embeddings])
    local_local_cos = np.array([[cosine_similarity(local_em1, local_em2) for local_em1 in local_embeddings] for local_em2 in local_embeddings])
    average_local_sim = local_local_cos.mean(axis=0) # uniqueness of code chunk in the code
    mixing_weights = np.exp(local_global_cos + average_local_sim) / np.sum(np.exp(local_global_cos + average_local_sim) )  # how much to mix in the code chunk
    final_embeddings = [mixing_weights[i] * global_embedding + (1 - mixing_weights[i]) * local_embeddings[i] for i in range(N)]
    final_embeddings = final_embeddings / np.sum(final_embeddings)
    return np.vstack(final_embeddings).T
df["text_2_embedding"] = df.apply(lambda row: get_local_embeddings(row['text_2'],row['readme'],10),axis=1)

# # create column of embeddings
# for column in ["text_1", "text_2"]:
#     df[f"{column}_embedding"] = df[column].apply(get_embedding_with_cache)

# # drop any invalid embeddings
# df = df.dropna()

# # create column of cosine similarity between embeddings
# df["cosine_similarity"] = df.apply(
#     lambda row: cosine_similarity(row["text_1_embedding"], row["text_2_embedding"]),
#     axis=1,
# )

# -

# df2 = pd.read_csv('data/df2.csv')
# df2

# +
# establish a cache of embeddings to avoid recomputing
# cache is a dict of tuples (text, engine) -> embedding
# try:
#     with open(embedding_cache_path, "rb") as f:
#         embedding_cache = pickle.load(f)
# except EOFError:
#     precomputed_embedding_cache_path = "https://cdn.openai.com/API/examples/data/snli_embedding_cache.pkl"
#     embedding_cache = pd.read_pickle(precomputed_embedding_cache_path)

# create column of embeddings
# for column in ["text_1","text_2", "readme"]:
#     df2[f"{column}_embedding"] = df2[column].apply(get_embedding_with_cache)

# # drop any invalid embeddings
# df2 = df2.dropna(subset=['text_1_embedding', 'text_2_embedding'])


# def compute_cosine_sim(row):
#     try: 
#         return cosine_similarity(row["text_2_embedding"], row["readme_embedding"])
#     except:
#         return None
# create column of cosine similarity between embeddings
# df2["cosine_similarity"] = df2.apply(
#     lambda row: cosine_similarity(row["text_1_embedding"], row["text_2_embedding"]),
#     axis=1,
# )

# df2["readme_1_cosine_similarity"] = df2.apply(
#     compute_cosine_sim,
#     axis=1,
# )
# df2["readme_2_cosine_similarity"] = df2.apply(
#     compute_cosine_sim,
#     axis=1,
# )
# -


# mar7 = pd.read_csv('data/df2_full.csv')
# mar7


# # + id="SoeDF8vqpgkQ" outputId="17db817e-1702-4089-c4e8-8ca32d294930"
# # calculate accuracy (and its standard error) of predicting label=1 if similarity>x
# # x is optimized by sweeping from -1 to 1 in steps of 0.01
# def accuracy_and_se(cosine_similarity: float, labeled_similarity: int) -> Tuple[float]:
#     accuracies = []
#     for threshold_thousandths in range(-1000, 2000, 1):
#         threshold = threshold_thousandths / 1000
#         total = 0
#         correct = 0
#         for cs, ls in zip(cosine_similarity, labeled_similarity):
#             total += 1
#             if cs > threshold:
#                 prediction = 1
#             else:
#                 prediction = -1
#             if prediction == ls:
#                 correct += 1
#         accuracy = correct / total
#         accuracies.append(accuracy)
#     a = max(accuracies)
#     n = len(cosine_similarity)
#     standard_error = (a * (1 - a) / n) ** 0.5  # standard error of binomial
#     return a, standard_error


# # check that training and test sets are balanced
# # px.histogram(
# #     mar7,
# #     x="cossim_augmented",
# #     color="label",
# #     barmode="overlay",
# #     width=500,
# #     facet_row="dataset",
# # ).show()

# # for dataset in ["train", "test"]:
# #     data = mar7[mar7["dataset"] == dataset]
# #     a, se = accuracy_and_se(data["cossim_augmented"], data["label"])
# #     print(f"{dataset} accuracy: {a:0.1%} ± {1.96 * se:0.1%}")


# # Attempt 2: Fusing local and global features.
# # Step 1: Chop up code.

# import matplotlib.pyplot as plt
# import math

# # + id="dPF-sczmpgkT"
# # -

# #

# N = 10

# # +
# # establish a cache of embeddings to avoid recomputing
# # cache is a dict of tuples (text, engine) -> embedding
# embedding_cache_path = "data/snli_embedding_cache.pkl"  # embeddings will be saved/loaded here
# try:
#     with open(embedding_cache_path, "rb") as f:
#         embedding_cache = pickle.load(f)
# except FileNotFoundError:
#     precomputed_embedding_cache_path = "https://cdn.openai.com/API/examples/data/snli_embedding_cache.pkl"
#     embedding_cache = pd.read_pickle(precomputed_embedding_cache_path)

# # this function will get embeddings from the cache and save them thereafterward
# def get_embedding_with_cache(
#     text: str,
#     engine: str = default_embedding_engine,
#     embedding_cache: dict = embedding_cache,
#     embedding_cache_path: str = embedding_cache_path,
# ) -> list:
#     if pd.isna(text):
#         return None
#     print(f"Getting embedding for {text}")
#     if (text, engine) not in embedding_cache.keys():
#         time.sleep(3)
#         # if not in cache, call API to get embedding
#         openai.api_key = os.getenv("OPENAI_API_KEY")
#         # also delete the line 
#         try: 
#             embedding_cache[(text, engine)] = get_embedding(text, engine)
#         except:
#             return None

#         # save embeddings cache to disk after each update
#         with open(embedding_cache_path, "wb") as embedding_cache_file:
#             pickle.dump(embedding_cache, embedding_cache_file)
#     return embedding_cache[(text, engine)]

# test_code = mar7.text_2[0]
# test_readme = mar7.readme[0]
# # def chop(str1, x):
# #     stride = math.ceil(len(str1) / x)
# #     parts = [str1[i:i+stride] for i in range(0, len(str1), stride)]
# #     return parts
# def get_local_embeddings(str1, N):
#     stride = math.ceil(len(str1) / N)
#     local_chunks = [str1[i:i+stride] for i in range(0, len(str1), stride)]
#     local_embeddings = []
#     for chunk in local_chunks:
#         local_embeddings.append(get_embedding_with_cache(chunk))
#     global_embedding = np.vstack((get_embedding_with_cache(test_code),  get_embedding_with_cache(test_readme)))
#     global_embedding = global_embedding.T.mean(axis=1)
#     local_embeddings, global_embedding = np.array(local_embeddings), np.array(global_embedding)
#     local_global_cos = np.array([cosine_similarity(local_em, global_embedding) for local_em in local_embeddings])
#     local_local_cos = np.array([[cosine_similarity(local_em1, local_em2) for local_em1 in local_embeddings] for local_em2 in local_embeddings])
#     average_local_sim = local_local_cos.mean(axis=0) # uniqueness of code chunk in the code
#     mixing_weights = np.exp(local_global_cos + average_local_sim) / np.sum(np.exp(local_global_cos + average_local_sim) )  # how much to mix in the code chunk
#     final_embeddings = [mixing_weights[i] * global_embedding + (1 - mixing_weights[i]) * local_embeddings[i] for i in range(N)]
#     final_embeddings = final_embeddings / np.sum(final_embeddings)
#     return np.vstack(final_embeddings).T


# cosine_similarity(row['text_1_embedding'],get_local_embeddings(row['text_2_embedding'], 10)).mean()

# cosine_sims = 
# df2['cosine_similarity'] = 