import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding

import pickle
import pandas as pd
import torch


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
        # self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size
        self.token_to_movieId = self.get_token_to_movieId()
        self.movie_to_genre_labels, self.num_genres = self.get_genre_map()
        self.genre_embeddings = nn.Embedding(self.num_genres+1, self.embed_size)  # +1 for unknown genre


    def forward(self, sequence):
        x = self.token(sequence) \
            + self.batch_convert_to_genre_embedding(sequence) \
            + self.position(sequence)  # + self.segment(segment_label)
        return self.dropout(x)

    def get_token_to_movieId(self):
        file_path = '/home/ubuntu/BERT4Rec/Data/preprocessed/ml-20m_min_rating3-min_uc5-min_sc0-splitleave_one_out'
        with open(file_path + '/inv_smap.pkl', 'rb') as f:
            smap = pickle.load(f)
        return smap

    def get_genre_map(self):
        movies_df = pd.read_csv('/home/ubuntu/BERT4Rec/Data/ml-20m/movies.csv')
        # Check for empty 'genres' entries
        if movies_df['genres'].isnull().any() or (movies_df['genres'] == '').any():
            print("Empty entries found in 'genres'.")
        else:
            print("No empty entries in 'genres'.")

            # Split the 'genres' string into a list
            movies_df['genres'] = movies_df['genres'].apply(lambda x: x.split('|'))

            # Get a unique list of genres and create a mapping to labels
            unique_genres = sorted(set(genre for sublist in movies_df['genres'] for genre in sublist))
            genre_to_label = {genre: label for label, genre in enumerate(unique_genres)}

            # Map movieId to a list of genre labels
            movie_to_genre_labels = {row['movieId']: [genre_to_label[genre] for genre in row['genres']] for index, row in movies_df.iterrows()}

            # Now movie_to_genre_labels is your desired map
            return movie_to_genre_labels, len(unique_genres)


    def get_movieIds(self, tensor_seq):
        movieIds = []
        for token in tensor_seq:
            try:
                movieIds.append(self.token_to_movieId[int(token)])
            except:
                movieIds.append(0)
        return movieIds
    
    def get_movie_genre_ids(self, movieIds):
        movie_genre_ids = []
        for movieId in movieIds:
            try:
                movie_genre_ids.append(self.movie_to_genre_labels[movieId])
            except:
                movie_genre_ids.append([self.num_genres])
        return movie_genre_ids
    
    def embed_genres(self, movie_genre_ids):
        genre_embeddings_list = [
            self.genre_embeddings(torch.tensor(genres, device=self.genre_embeddings.weight.device)) 
            for genres in movie_genre_ids]
        genre_embeddings_avg = [torch.mean(embeds, dim=0) for embeds in genre_embeddings_list]
        return torch.stack(genre_embeddings_avg)

    def convert_to_genre_embedding(self, tensor_seq_1d):
        return self.embed_genres(self.get_movie_genre_ids(self.get_movieIds(tensor_seq_1d)))
    
    def batch_convert_to_genre_embedding(self, tensor_seq_2d):
        return torch.stack([self.convert_to_genre_embedding(tensor_seq_1d) for tensor_seq_1d in tensor_seq_2d])