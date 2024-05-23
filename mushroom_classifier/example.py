from mushroomClassifier import MushroomClassifier

# Load the model
mushroom_classifier = MushroomClassifier()

# Species: Amanita bisporigera (Destroying Angel)
# Poisonous: True
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

# Species: Agaricus bisporus (Button Mushroom)
# Poisonous: False
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

amanita_is_poisonous = mushroom_classifier.is_poisonous(amanita_bisporigera)
agaricus_is_poisonous = mushroom_classifier.is_poisonous(agaricus_bisporus)

print("Amanita bisporigera is poisonous? {}".format(amanita_is_poisonous))
print("Agaricus bisporus is poisonous? {}".format(agaricus_is_poisonous))
