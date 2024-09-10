# Premier League Chatbot with Sentence Embeddings and GPT-2

This project creates a chatbot that answers questions about the Premier League using data from past seasons and matches. It uses **sentence embeddings** for retrieving relevant information and **GPT-2** to generate natural language responses.

## Installation

Before running the notebook, install the required libraries:

```bash
# Install the necessary libraries
pip install transformers sentence-transformers faiss-cpu pandas
```

Overview
The chatbot is built using two datasets:

Season-Level Data: Contains information about Premier League seasons (e.g., champions, total goals).
Match-Level Data: Contains detailed match information (e.g., teams, scores, and results).
The system uses sentence embeddings to encode these data into numerical representations and stores them in a FAISS index for efficient similarity search. GPT-2 is then used to generate responses based on the retrieved information.

Steps and Components
1. Data Loading and Preprocessing
First load two CSV files:
```
# Load the first CSV data (season-level data)
season_data = pd.read_csv('epl_season_1993_2024.csv')
season_data['text'] = season_data.apply(lambda row: f"In {row['Season_End_Year']}, {row['Champion']} won the championship with {row['Total_Goals']} goals. The runner-up was {row['Runners']}.", axis=1)

# Load the second CSV data (match-level data)
match_data = pd.read_csv('premier-league-matches.csv')
match_data['text'] = match_data.apply(lambda row: f"On {row['Date']}, {row['Home']} played against {row['Away']}. The match ended {row['HomeGoals']}-{row['AwayGoals']} with {row['FTR']} as the final result.", axis=1)

# Combine both datasets into one DataFrame
combined_data = pd.concat([season_data['text'], match_data['text']], ignore_index=True)
```

epl_season_1993_2024.csv: Contains season-level information.
premier-league-matches.csv: Contains match-level informatio

2. Creating Embeddings and FAISS Index
We use SentenceTransformer to create embeddings for each textual description. These embeddings are added to a FAISS index for efficient similarity search.
```
# Initialize the model for creating embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for the combined data
embeddings = embedding_model.encode(combined_data.tolist())

# Create a FAISS index for efficient similarity search
index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance-based index
index.add(embeddings)  # Add the embeddings to the FAISS index

```

3. Saving the FAISS Index and Data
The FAISS index and combined data (with embeddings) are saved for future use. Additionally, the embedding model is saved for reuse in later queries.
```
# Save the FAISS index
faiss.write_index(index, 'faiss_index.idx')

# Save the combined data with embeddings
combined_data.to_csv('combined_data_with_embeddings.csv', index=False)

# Save the embedding model (optional)
with open('model pickel/embedding_model.pkl', 'wb') as f:
    pickle.dump(embedding_model, f)

```

4. Loading Pre-Trained GPT-2 Model
We use GPT-2 as the language model to generate chatbot responses based on retrieved data.
```
# Load the pre-trained GPT-2 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Save the tokenizer and model for later use
tokenizer.save_pretrained('gpt2_tokenizer')
model.save_pretrained('gpt2_model')

```

5. Testing the Saved Components
We load the saved FAISS index, combined data, embedding model, and GPT-2 to ensure everything is working correctly.
```
# Test loading the FAISS index
loaded_index = faiss.read_index('faiss_index.idx')

# Test loading the combined data
loaded_combined_data = pd.read_csv('combined_data_with_embeddings.csv')

# Test loading the embedding model
with open('model pickel/embedding_model.pkl', 'rb') as f:
    loaded_embedding_model = pickle.load(f)

# Test loading the GPT-2 model and tokenizer
loaded_tokenizer = AutoTokenizer.from_pretrained("gpt2_tokenizer")
loaded_model = AutoModelForCausalLM.from_pretrained("gpt2_model")

# Print a sample output to ensure everything is working correctly
print(loaded_combined_data.head())

```

## How It Works

1. **User Query**: A user asks a question about the Premier League.
2. **Similarity Search**: The chatbot converts the query into an embedding using the `SentenceTransformer` model and retrieves the most similar piece of information from the FAISS index.
3. **Response Generation**: The retrieved information is passed to GPT-2, which generates a human-like response.

## Project Structure

- **`chatbot.ipynb`**: The main Jupyter Notebook where the chatbot system is built.
- **`epl_season_1993_2024.csv`**: Season-level data of the Premier League.
- **`premier-league-matches.csv`**: Match-level data of the Premier League.
- **`model pickel/`**: A folder containing the saved models and embeddings.
  - **`embedding_model.pkl`**: The saved SentenceTransformer model.
  - **`faiss_index.idx`**: The FAISS index for fast similarity search.
  - **`gpt2_tokenizer/`**: The saved GPT-2 tokenizer.
  - **`gpt2_model/`**: The saved GPT-2 model.

## Future Enhancements

- Improve the accuracy of responses by fine-tuning the GPT-2 model with more Premier League-specific data.
- Implement additional queries and conversation management to make the chatbot more interactive.
- Use more advanced transformers like GPT-3 or GPT-4 for even better responses.
