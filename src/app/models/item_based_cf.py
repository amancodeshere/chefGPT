import json
from sklearn.preprocessing import LabelEncoder
from typing_extensions import overload

from src.app.models.recommender import Recommender
from scipy.sparse import csr_matrix

class ItemBasedCF(Recommender):
    file_path = ''

    def __init__(self):
        super().__init__()
        self.data = self.filter_data(self.read_data(self.file_path))
        self.matrix, self.user_encoding, self.recipe_encodings = self.build_matrix()

    def read_data(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    def filter_data(self, data, min_user_ratings=1, min_recipe_ratings=1):
            """
            Optimized data filtering based on data analysis results
            In the system to be an active user it has to have more than 1 rating
            @:param data : The data to be filtered
            @:param min_user_ratings : The minimum number of ratings a user must have to be considered active
            @:param min_recipe_ratings : The minimum number of ratings a recipe must have to be considered popular
            """
            user_counts = data['userID'].value_counts()
            active_users = user_counts[user_counts >= min_user_ratings].index

            recipe_counts = data['recipeID'].value_counts()
            popular_recipes = recipe_counts[recipe_counts >= min_recipe_ratings].index

            filtered_data = data[
                data['userID'].isin(active_users) &
                data['recipeID'].isin(popular_recipes)
                ].copy()

            n_users = filtered_data['userID'].nunique()
            n_recipes = filtered_data['recipeID'].nunique()
            n_ratings = len(filtered_data)
            sparsity = 1 - n_ratings / (n_users * n_recipes)
            print(f"Sparsity after filtering: {sparsity}")

            return filtered_data

    def build_matrix(self):
        user_enc = LabelEncoder()
        recipe_enc = LabelEncoder()
        user_idx = user_enc.fit_transform(self.data['userID'])
        recipe_idx = recipe_enc.fit_transform(self.data['recipeID'])

        mat = csr_matrix(
            (self.data['rating'], (user_idx, recipe_idx)),
            shape=(user_enc.classes_.size, recipe_enc.classes_.size)
        )
        return mat, user_enc, recipe_enc

    @overload
    def recommend(self, user_id, top_k=10):
        pass