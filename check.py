import joblib
model = joblib.load("Mabels_hybrid_model.pkl")
print("✅ Loaded without error.")

import os
print(os.path.getsize('Mabels_hybrid_model.pkl'))