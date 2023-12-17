import torch.nn as nn
import torch
import pickle
import pandas as pd


class GenreEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        inv_smap = self.get_inv_smap()
        movie_to_genre_labels, self.num_genres = self.get_genre_map()
        
        super().__init__(self.num_genres + 1, embed_size)

        # Create the tensors and register as buffers
        movieId_lookup_tensor = self.get_movieId_lookup(inv_smap)
        genre_lookup_tensor, genre_len_lookup_tensor = self.create_genre_lookup_tensor(movie_to_genre_labels)
        
        self.register_buffer('movieId_lookup', movieId_lookup_tensor)
        self.register_buffer('genre_lookup', genre_lookup_tensor)
        self.register_buffer('genre_len_lookup', genre_len_lookup_tensor)




    def get_inv_smap(self):
        file_path = '/home/ubuntu/BERT4Rec/Data/preprocessed/ml-20m_min_rating3-min_uc5-min_sc0-splitleave_one_out'
        with open(file_path + '/inv_smap.pkl', 'rb') as f:
            inv_smap = pickle.load(f)
        return inv_smap

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

    def get_movieId_lookup(self, inv_smap):
        max_token_id = max(inv_smap.keys())
        lookup_tensor = torch.zeros(max_token_id + 3, dtype=torch.int32)
        for token_id, movieId in inv_smap.items():
            lookup_tensor[token_id] = movieId
        return lookup_tensor

    
    def create_genre_lookup_tensor(self, movie_to_genre_labels):
        # Determine the maximum number of genres any single movie has
        max_num_genres = max(len(genres) for genres in movie_to_genre_labels.values())
        
        # Initialize the lookup tensor with an extra label for 'no genre' or 'unknown genre'
        genre_lookup_tensor = torch.full((max(movie_to_genre_labels.keys()) + 1, max_num_genres), 
                                        fill_value=self.num_genres, # Use the num_genres as the 'unknown' genre label
                                        dtype=torch.int32)
    
        genre_len_lookup_tensor = torch.ones(max(movie_to_genre_labels.keys()) + 1, dtype=torch.int32)
        
        # Fill the tensor with the genre labels for each movie
        for movie_id, genres in movie_to_genre_labels.items():
            genre_lookup_tensor[movie_id, :len(genres)] = torch.tensor(genres, dtype=torch.int32)
            genre_len_lookup_tensor[movie_id] = len(genres)
        
        return genre_lookup_tensor, genre_len_lookup_tensor


    def get_movieIds(self, tensor_seq_2d):
        assert tensor_seq_2d.max() < self.movieId_lookup.size(0), "Index out of range"
        assert tensor_seq_2d.min() >= 0, "Index is negative"

        return self.movieId_lookup[tensor_seq_2d]
    
    def get_movie_genre_ids(self, movieIds):
        genre_output =  self.genre_lookup[movieIds]
        len_output = self.genre_len_lookup[movieIds]
        return genre_output, len_output
    
    def embed_genres(self, movie_genre_ids_len_tuple):
        movie_genre_ids, movie_genre_len = movie_genre_ids_len_tuple
        device = movie_genre_ids.device
        batch_size = movie_genre_ids.size(0)
        seq_len = movie_genre_ids.size(1)
        max_num_genres = movie_genre_ids.size(2)

        # Flatten the movie_genre_ids to 2D (batch_size*seq_len, max_num_genres) for embedding lookup
        flat_movie_genre_ids = movie_genre_ids.view(batch_size*seq_len, max_num_genres)
        flat_movie_genre_len = movie_genre_len.view(batch_size*seq_len)
        
        # Perform the embedding lookup in a batch
        # This will return a 3D tensor of shape (batch_size*seq_len, max_num_genres, embed_size)
        flat_genre_embeddings = super().forward(flat_movie_genre_ids)

        mask = torch.arange(max_num_genres, device=device).expand(batch_size*seq_len, -1) < flat_movie_genre_len.unsqueeze(-1)
        mask = mask.unsqueeze(-1).expand(-1, -1, self.embedding_dim)

        flat_genre_embeddings *= mask.float()

        # Shape (batch_size*seq_len, embed_size)
        flat_genre_embeddings_sum = torch.sum(flat_genre_embeddings, dim=1)

        flat_movie_genre_len[flat_movie_genre_len == 0] = 1  # Avoid division by zero
        flat_genre_embeddings_avg = flat_genre_embeddings_sum / flat_movie_genre_len.view(-1, 1).float()
        
        genre_embeddings_avg = flat_genre_embeddings_avg.view(batch_size, seq_len, -1)
        
        return genre_embeddings_avg

    def forward(self, tensor_seq_2d):
        return self.embed_genres(self.get_movie_genre_ids(self.get_movieIds(tensor_seq_2d)))