import pandas as pd
from langchain_community.chat_models import ChatGooglePalm  # Updated import
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings  # Correct import
from langchain_community.vectorstores import Chroma  # Correct import
from langchain.prompts import FewShotPromptTemplate, SemanticSimilarityExampleSelector
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Load your CSV data using pandas
try:
    df = pd.read_csv(r"G:\proejct\proejct-sc\project-llm\dataset\TnData-.csv") 
    print("CSV data loaded successfully.")
except FileNotFoundError:
    print("Error: CSV file not found. Please provide the correct path.")
    exit()  
except Exception as e:
    print(f"An error occurred while loading the CSV: {e}")
    exit()

# 2. Initialize Google Palm LLM using ChatGooglePalm
api_key = "AIzaSyCk1n8yw18tQDbnjX8va1aqPFqCfriMI2o"  # Replace with actual API key
# llm = ChatGooglePalm(google_api_key=api_key, temperature=0.2)
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key, temperature=0.2)

# 3. Define Few-Shot Examples (Crucial for good performance)
few_shots = [
    {
        "Question": "What are some popular tourist places in the dataset?",
        "Data": df.to_string(),  # Pass the DataFrame's string representation
        "Answer": "Based on the data, [mention specific places from the dataset or general categories like 'beaches', 'mountains', 'historical sites', etc.] are popular tourist destinations."
    },
    {
        "Question": "Which places have the highest number of visitors?",
        "Data": df.to_string(),
        "Answer": "[Mention specific places and the corresponding visitor counts from the dataset.  Example: 'The Taj Mahal had the most visitors with 100000, followed by the Eiffel Tower with 80000.']"
    },
    {
        "Question": "Are there any adventure tourism options mentioned?",
        "Data": df.to_string(),
        "Answer": "[Mention specific adventure activities and the corresponding locations from the dataset. Example: 'Yes, the data shows options for trekking in the Himalayas and scuba diving in the Andaman Islands.']"
    },
    # Add more examples relevant to your data and the types of questions you expect
]


# 4. Set up Embeddings and Vectorstore for Semantic Similarity
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
to_vectorize = [" ".join(example.values()) for example in few_shots]
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)

example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,  # Number of similar examples to retrieve
)

# 5. Create Prompt Template
prompt_template = """You are a helpful travel assistant. Given the following data from a CSV file and a user question, provide an informative and accurate answer.  Refer to the data directly to support your answer. If the data doesn't contain the answer, say 'I cannot answer this question based on the provided data.'

Data:
{Data}

Question: {Question}

Answer:"""

prompt = PromptTemplate(
    input_variables=["Data", "Question"],
    template=prompt_template,
)


from langchain.schema.runnable import RunnableSequence

# Update chain initialization
chain = RunnableSequence(prompt | llm)

# Use `invoke` instead of `run`
user_question = "What are the top-rated beaches in the dataset?"
answer = chain.invoke({"Data": df.to_string(), "Question": user_question}) 



# 6. Set up the LLMChain
# chain = LLMChain(llm=llm, prompt=prompt)

# 7. Example Usage
# user_question = "What are the top-rated beaches in the dataset?"
# answer = chain.run({"Data": df.to_string(), "Question": user_question})
print(answer)

user_question = "What is the average cost of a hotel in Goa?"  # Example question that might not be in the data
answer = chain.run({"Data": df.to_string(), "Question": user_question})
print(answer)


# Using Few Shot Learning (Optional, but recommended for more complex queries)

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=prompt,  # Use the same prompt as above
    prefix="""You are a helpful travel assistant. Given the following data from a CSV file and a user question, provide an informative and accurate answer.  Refer to the data directly to support your answer. If the data doesn't contain the answer, say 'I cannot answer this question based on the provided data.'""",
    suffix="""Data:
{Data}

Question: {Question}

Answer:""",
    input_variables=["Data", "Question"],
)

few_shot_chain = LLMChain(llm=llm, prompt=few_shot_prompt)

user_question = "Which places have the highest number of visitors?"
answer = few_shot_chain.run({"Data": df.to_string(), "Question": user_question})
print(answer)