from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

def load_data():
    dataset = load_dataset("tawhidmonowar/ufc-fighters-stats-and-records-dataset")
    df = pd.DataFrame(dataset['train'])
    df = df.dropna(subset=['age', 'height', 'weight', 'reach', 'win'])

    features = ['age', 'height', 'weight', 'reach']
    X = df[features]
    y = df['win'].astype(int)
    return X, y

def implied_probability(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

def is_value_bet(our_prob, bookmaker_odds):
    return our_prob > implied_probability(bookmaker_odds)

def run_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Simula quote bookmaker e predici probabilità
    sample = X_test.iloc[0:1]
    odds = 120  # es: +120 American odds

    our_prob = model.predict_proba(sample)[0][1]
    print(f"Probabilità stimata: {round(our_prob * 100, 2)}%")
    print(f"Probabilità implicita quota {odds}: {round(implied_probability(odds) * 100, 2)}%")

    if is_value_bet(our_prob, odds):
        print("✅ Value bet trovata!")
    else:
        print("❌ Nessuna value bet.")

    print(f"Accuracy del modello: {round(accuracy_score(y_test, model.predict(X_test)) * 100, 2)}%")

if __name__ == "__main__":
    run_model()
