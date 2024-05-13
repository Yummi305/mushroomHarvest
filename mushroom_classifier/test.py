import pickle

import numpy as np
import pandas as pd
from mushroomClassifier import MushroomClassifier
from sklearn.linear_model import LogisticRegression

# Load the model
mushroom_classifier = MushroomClassifier()


# def isPoisonous(mushroom_features, model):
#     # Create empty dataframe with columns from model.feature_names_in_
#     df_test = pd.DataFrame(columns=model.feature_names_in_)

#     # Convert mushroom dictionary to DataFrame
#     df_mushroom = pd.DataFrame(mushroom_features)
#     df_mushroom = pd.get_dummies(df_mushroom)

#     # Concat mushroom data to empty DataFrame
#     df_appended = pd.concat([df_test, df_mushroom], axis=0)

#     # Replace NaNs with 0
#     df_appended = df_appended.fillna(0)

#     poisonous = model.predict(df_appended)
#     return bool(poisonous)


# lets test this
# make a mushroom - Species: Amanita bisporigera (Destroying Angel)
# this is poisonous
amanita_bisporigera = {
    "cap-shape": ["x"],
    "cap-surface": ["s"],
    "cap-color": ["n"],
    "bruises": ["t"],
    "odor": ["p"],
    "gill-attachment": ["f"],
    "gill-spacing": ["c"],
    "gill-size": ["n"],
    "gill-color": ["w"],
    "stalk-shape": ["e"],
    "stalk-root": ["e"],
    "stalk-surface-above-ring": ["s"],
    "stalk-surface-below-ring": ["s"],
    "stalk-color-above-ring": ["w"],
    "stalk-color-below-ring": ["w"],
    "veil-type": ["p"],
    "veil-color": ["w"],
    "ring-number": ["o"],
    "ring-type": ["p"],
    "spore-print-color": ["k"],
    "population": ["s"],
    "habitat": ["u"]
}

# make a mushroom - Species: Agaricus bisporus (Button Mushroom)
# this is NOT poisonous
agaricus_bisporus = {
    "cap-shape": ["s"],  # Shape: Spherical
    "cap-surface": ["y"],  # Surface: Scaly
    "cap-color": ["n"],  # Color: Brown
    "bruises": ["t"],  # Bruises: Bruises
    "odor": ["a"],  # Odor: Almond
    "gill-attachment": ["f"],  # Gill Attachment: Free
    "gill-spacing": ["c"],  # Gill Spacing: Close
    "gill-size": ["b"],  # Gill Size: Broad
    "gill-color": ["w"],  # Gill Color: White
    "stalk-shape": ["t"],  # Stalk Shape: Tapering
    "stalk-root": ["b"],  # Stalk Root: Bulbous
    "stalk-surface-above-ring": ["s"],  # Stalk Surface Above Ring: Smooth
    "stalk-surface-below-ring": ["s"],  # Stalk Surface Below Ring: Smooth
    "stalk-color-above-ring": ["w"],  # Stalk Color Above Ring: White
    "stalk-color-below-ring": ["w"],  # Stalk Color Below Ring: White
    "veil-type": ["p"],  # Veil Type: Partial
    "veil-color": ["w"],  # Veil Color: White
    "ring-number": ["o"],  # Ring Number: One
    "ring-type": ["p"],  # Ring Type: Pendant
    "spore-print-color": ["n"],  # Spore Print Color: Brown
    "population": ["y"],  # Population: Abundant
    "habitat": ["d"]  # Habitat: Woods
}

# print("Amanita bisporigera is poisonous? {}".format(
#     isPoisonous(Amanita_bisporigera, logreg)))
# print("Lactarius sec is poisonous? {}".format(
#     isPoisonous(Agaricus_bisporus, logreg)))

amanita_is_poisonous = mushroom_classifier.is_poisonous(amanita_bisporigera)
agaricus_is_poisonous = mushroom_classifier.is_poisonous(agaricus_bisporus)

print("Amanita bisporigera is poisonous? {}".format(amanita_is_poisonous))
print("Agaricus bisporus is poisonous? {}".format(agaricus_is_poisonous))
