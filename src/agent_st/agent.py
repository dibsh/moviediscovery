from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    HasIdCondition
)
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv
import os
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# Initialise Langfuse
lf = Langfuse()
langfuse_handler = CallbackHandler()

# Initialise QDRANT
qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
)

info = qdrant_client.get_collection(os.getenv("QDRANT_COLLECTION_NAME"))
print("Collection:", os.getenv("QDRANT_COLLECTION_NAME"))
print("Payload schema:", info.payload_schema)

# Initialise Embeddings        
embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# Initialise Vector Store
vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        embedding=embeddings,
)

qdrant_client.create_payload_index(
    collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
    field_name="metadata.title",
    field_schema=models.PayloadSchemaType.KEYWORD,
)

qdrant_client.create_payload_index(
    collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
    field_name="metadata.released_year",
    field_schema=models.PayloadSchemaType.KEYWORD,
)

# Initialise LLM Model
model = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# Create class
class AgentInput(BaseModel):
    """Input for movie search and recommendations."""
    query: str = Field(description="Movie search query or user's movie preferences")
    history: str = Field(description="Summary of previous conversation with the user")

# Create first tool for semantic movies search
@tool
def search_movies(query: str, k: int = 10):
    """
    Search for movies based on plot, storyline, themes, genre, mood, or user preferences.
    Use this tool when users describe the type of movie they want to watch in natural language.
    """

    results = vector_store.similarity_search_with_score(
        query=query,
        k=10
    )

    movies = []

    for doc, score in results:

        # Keep only sufficiently relevant results
        if score < 0.35:
            continue
        
        movie = {
            "title": doc.metadata.get("title", ""),
            "year": doc.metadata.get("released_year", ""),
            "genre": doc.metadata.get("genre", ""),
            "runtime": doc.metadata.get("runtime", ""),
            "rating": doc.metadata.get("imdb_rating", ""),
            "director": doc.metadata.get("director", ""),
            "cast": doc.metadata.get("cast", []),
            "overview": doc.page_content
        }

        movies.append(movie)

    return movies

# @tool
# def search_resume(query: str, k: int = 5) -> list[str]:
#     """Retrieve relevant resumes on the query."""


#     vector_store = QdrantVectorStore(
#         client=qdrant_client,
#         collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
#         embedding=embeddings,
#     )

#     docs = vector_store.similarity_search_with_score(query, k=k)
#     if docs:
#         formatted_results = []
#         for idx, result in enumerate(docs):
#             formatted_results.append(f"""
# Resume {idx + 1}:
# - ID: {result[0].metadata.get('row_index', 'N/A')}
# - Category: {result[0].metadata.get('category', 'N/A')}
# - Relevance Score: {result[1]:.3f}
# - Content Preview: {result[0].page_content[:300]}...
# """)
        
#         context = "\n".join(formatted_results)
#         return context
#     return "No relevant documents found."

@tool
def similar_movies(
    movie_title: str,
    year: int = None,
    limit: int = 5
):
    """
    Recommend movies similar to a selected movie.

    Finds the selected movie in Qdrant, retrieves its existing
    embedding, and searches for other movies with similar vectors.
    An optional release year can distinguish movies with the same title.
    """
    
    # Step 1: Find the selected movie in Qdrant
    conditions = [
        FieldCondition(
            key="metadata.title",
            match=MatchValue(value=movie_title)
        )
    ]

    if year is not None:
        conditions.append(
            FieldCondition(
                key="metadata.released_year",
                match=MatchValue(value=str(year))
            )
        )

    matching_movies, _ = qdrant_client.scroll(
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        scroll_filter=Filter(must=conditions),
        limit=10,
        with_payload=True,
        with_vectors=True
    )

    if not matching_movies:
        return f"No movie found with the title '{movie_title}'."

    # Step 2: Handle movies with duplicate titles
    if len(matching_movies) > 1:
        matches = [
            {
                "title": point.payload.get("metadata", {}).get("title", ""),
                "year": point.payload.get("metadata", {}).get("released_year", "")
            }
            for point in matching_movies
        ]

        return {
            "message": (
                "Multiple movies have this title. "
                "Please specify the release year."
            ),
            "matches": matches
        }

    source_movie = matching_movies[0]

    # Step 3: Retrieve the selected movie's existing vector
    source_vector = source_movie.vector

    # Step 4: Find similar movies, excluding the selected movie
    results = qdrant_client.query_points(
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        query=source_vector,
        query_filter=Filter(
            must_not=[
                HasIdCondition(has_id=[source_movie.id])
            ]
        ),
        limit=limit,
        with_payload=True,
        with_vectors=False
    ).points

    # Step 5: Format recommendations
    recommendations = []

    for point in results:
        metadata = point.payload.get("metadata", {})

        recommendations.append({
            "title": metadata.get("title", ""),
            "year": metadata.get("released_year", ""),
            "genre": metadata.get("genre", ""),
            "rating": metadata.get("imdb_rating", ""),
            "runtime": metadata.get("runtime", ""),
            "director": metadata.get("director", ""),
            "cast": metadata.get("cast", []),
            "similarity_score": point.score
        })

    return {
        "selected_movie": {
            "title": source_movie.payload.get("metadata", {}).get("title", ""),
            "year": source_movie.payload.get("metadata", {}).get("released_year", "")
        },
        "recommendations": recommendations
    }

# @tool
# def search_resume_skill(query: str, k: int = 10) -> list[str]:
#     """Retrieve relevant resumes on the query."""


#     vector_store = QdrantVectorStore(
#         client=qdrant_client,
#         collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
#         embedding=embeddings,
#     )

#     docs = vector_store.similarity_search_with_score(query, k=k)
#     if docs:
#         formatted_data = []
#         for idx, result in enumerate(docs):
#             formatted_data.append(f"""
# Resume {idx + 1} ({result[0].metadata.get('row_index', 'N/A')}):
# {result[0].page_content[:500]}...
# """)
        
#         context = "\n".join(formatted_data)
#         return context
#     return "No relevant documents found."


# Create agents
lf_search_movies = lf.get_prompt("search_movies_agent").get_langchain_prompt()

search_movies_agent = create_agent(
    model=model,
    tools=[search_movies],
    system_prompt=lf_search_movies
)

lf_similar_movies = lf.get_prompt("similar_movies_agent").get_langchain_prompt()

similar_movies_agent = create_agent(
    model=model,
    tools=[similar_movies],
    system_prompt=lf_similar_movies
)

# Turn agents into tools
@tool(
        args_schema=AgentInput
)
def movies_search(query: str, history: str) -> str:
    """Tool to search movies using search movies agent
    Use this when the user wants to  find/search for specific movies.

    query: "find movies by Christopher Nolan", "find movie with characters called Batman and Joker".
    history: chat history summary
    """
    result = search_movies_agent.invoke({
        "messages": [{"role": "user", "content": query + "history chat: " + history}]
    }, config={"callbacks": [langfuse_handler]})
    return result["messages"][-1].text

@tool(
        args_schema=AgentInput
)
def movie_similarity(query: str, history: str) -> str:
    """Tool to recommend similar movies once a certain movie is selected.
    Use when: User asks for similar movies.
    
    query: "5 movies similar to Inception", "movies close to the feeling of Joker"
    history: chat history
    """
    result = similar_movies_agent.invoke({
        "messages": [{"role": "user", "content": query + "history chat: " + history}]
    }, config={"callbacks": [langfuse_handler]})
    return result["messages"][-1].text

# supervisor agent
# lf_movie = lf.get_prompt("movie_agent").get_langchain_prompt()

# movie_agent = create_agent(
#     model=model,
#     tools=[search_movies, similar_movies],
#     system_prompt=lf_movie
# )

    # You are a movie recommendation assistant.

    # Your job is to help users discover movies.

    # Use recommend_similar_movies when users:
    # - Ask for movies similar to a specific movie.
    # - Mention a movie they enjoyed and want similar recommendations.
    # - Ask for alternatives to a particular movie.

    # Always use the tool to retrieve recommendations.
    # Do not invent movie recommendations.

    # Present the results clearly, including movie titles,
    # release years, genres and IMDb ratings when available.

# Test agent
# response = supervisor_agent.invoke({
#     "messages": [
#         {
#             "role": "user",
#             "content": "Movie by Christopher Nolan."
#         }
#     ]
# })

# print(response["messages"][-1].content)