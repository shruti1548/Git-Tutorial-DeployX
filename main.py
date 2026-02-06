import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


# Step 1: Create a dictionary and convert it to a pandas DataFrame
def create_dataframe():
    data = {
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "review": [
            "Great food and ambiance.",
            "Terrible service. ",
            "Amazing experience!",
            "Food was cold.",
            "Loved the desserts.",
            "Not worth the money.",
            "Excellent customer service.",
            "The place was too crowded.",
            "Best restaurant in town.",
            "Average experience."
        ]
    }

    df = pd.DataFrame(data)
    return df


# Step 2: Save dataframe to data/data.csv
def save_dataframe(df):
    if not os.path.exists("data"):
        os.makedirs("data")

    df.to_csv("data/data.csv", index=False)
    print("data.csv saved in 'data' folder.")


# Step 3: Load data.csv, vectorize text, and create K new columns
def process_data(k):
    # Load the saved dataframe
    df = pd.read_csv("data/data.csv")

    # Apply vectorization (CountVectorizer)
    vectorizer = CountVectorizer(max_features=k)
    vectorized_data = vectorizer.fit_transform(df["review"])
    feature_names = vectorizer.get_feature_names_out()

    # Create dataframe with K new columns
    vectorized_df = pd.DataFrame(
        vectorized_data.toarray(),
        columns=feature_names
    )

    # Combine original dataframe with vectorized features
    processed_df = pd.concat([df, vectorized_df], axis=1)

    # Save processed dataframe
    processed_df.to_csv("data/processed_data.csv", index=False)
    print(f"processed_data.csv saved in 'data' folder with {k} new columns.")

    return processed_df


# Main execution
if __name__ == "__main__":
    # Step 1
    df = create_dataframe()

    # Step 2
    save_dataframe(df)

    # Step 3
    k = 5  # Replace with your desired value of K
    processed_df = process_data(k)

    # Display shapes
    print(f"data shape: {df.shape}")
    print(f"processed data shape: {processed_df.shape}")