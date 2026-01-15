import pickle

with open("models/feature_names.pkl", "rb") as f:
    feats = pickle.load(f)

print(type(feats))
print(len(feats))
print(feats[:10])

print(type(feats))
print(len(feats))
print(feats[:10])
