from src.app.models.item_based_cf import ItemBasedCF

ibcf_model = ItemBasedCF("../../cleaned_data/Colaborative Filtering Data/item_based_cf_data.json")

# Item to item
print(ibcf_model.similar_items("10000.0", k=10))

# User recommendations
print(ibcf_model.recommend("1153011.0", top_k=10))