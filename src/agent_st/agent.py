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
from contextvars import ContextVar
from langchain_core.messages import ToolMessage

load_dotenv()

# ----------------------------------
# INITIALISATIONS
# ----------------------------------

# Initialise Langfuse
lf = Langfuse()
langfuse_handler = CallbackHandler()

# Initialise QDRANT
qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
)

# Additional Info
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

# Initialise indexes for metadata scrolling
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

qdrant_client.create_payload_index(
    collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
    field_name="metadata.cast",
    field_schema=models.PayloadSchemaType.KEYWORD,
)

qdrant_client.create_payload_index(
    collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
    field_name="metadata.director",
    field_schema=models.PayloadSchemaType.KEYWORD,
)

# Initialise LLM Model
model = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# ----------------------------------
# TOKEN AND TOOLS USAGE TRACKING
# ----------------------------------

usage_tracker = ContextVar("usage_tracker", default=None)


def reset_usage_tracking():

    tracker = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "tools_used": [],
        "tool_messages": []
    }

    usage_tracker.set(tracker)


def record_agent_usage(result):

    tracker = usage_tracker.get()

    if tracker is None:
        reset_usage_tracking()
        tracker = usage_tracker.get()

    messages = result["messages"]

    # Extract Token Usage
    for message in messages:

        response_metadata = (
            getattr(message, "response_metadata", {}) or {}
        )

        # Newer LangChain usage format
        message_usage = getattr(message, "usage_metadata", None)

        # Alternative usage format
        if not message_usage:
            message_usage = response_metadata.get("usage_metadata")

        if message_usage:
            tracker["total_input_tokens"] += message_usage.get(
                "input_tokens", 0
            )

            tracker["total_output_tokens"] += message_usage.get(
                "output_tokens", 0
            )

        # Fallback for older or different model providers
        elif "token_usage" in response_metadata:
            token_usage = response_metadata["token_usage"]

            tracker["total_input_tokens"] += token_usage.get(
                "prompt_tokens", 0
            )

            tracker["total_output_tokens"] += token_usage.get(
                "completion_tokens", 0
            )

        # Extract Tool Calls
        if hasattr(message, "tool_calls") and message.tool_calls:

            for tool_call in message.tool_calls:
                tracker["tools_used"].append({
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {})
                })

        # Extract Tool Messages
        if isinstance(message, ToolMessage):
            tracker["tool_messages"].append(message.content)

    # Return final answer as a string
    final_message = messages[-1]

    return getattr(final_message, "text", None) or final_message.content


def get_usage_summary():

    tracker = usage_tracker.get()

    if tracker is None:
        tracker = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "tools_used": [],
            "tool_messages": []
        }

    input_tokens = tracker["total_input_tokens"]
    output_tokens = tracker["total_output_tokens"]

    # Estimated cost in Indonesian Rupiah
    price = 18_000 * (
        input_tokens * 0.15
        + output_tokens * 0.6
    ) / 1_000_000

    return {
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "price": price,
        "tools_used": tracker["tools_used"].copy(),
        "tool_messages": tracker["tool_messages"].copy()
    }

# ----------------------------------
# CREATE CLASS
# ----------------------------------
class AgentInput(BaseModel):
    """Input for movie search and recommendations."""
    query: str = Field(description="Movie search query or user's movie preferences")
    history: str = Field(description="Summary of previous conversation with the user")

# ----------------------------------
# CREATE TOOLS
# ----------------------------------

# Create TOOL NUMBER ONE for semantic movies search
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

# Create TOOL NUMBER TWO which finds similar movies
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

# Create TOOL NUMBER THREE to find top ranked movies by actor/director
@tool
def hall_of_fame_search(person_name: str, person_type: str) -> str:
    """
    Retrieve the five highest-rated movies of an actor or director.

    person_name: Name of the actor or director.
    person_type: Either actor or director.
    """

    # Retrieve movie records from Qdrant
    points, _ = qdrant_client.scroll(
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        limit=1000,
        with_payload=True,
        with_vectors=False
    )

    movies = []

    # Find movies featuring the actor or directed by the director
    for point in points:
        metadata = point.payload.get("metadata", {})

        if person_type.lower() == "actor":
            match = any(
                name.strip().casefold() == person_name.strip().casefold()
                for name in metadata.get("cast", [])
            )

        elif person_type.lower() == "director":
            match = (
                metadata.get("director", "").strip().casefold()
                == person_name.strip().casefold()
            )

        else:
            return "Please specify whether the person is an actor or director."

        if match:
            movies.append({
                "title": metadata.get("title", "Unknown"),
                "released_year": metadata.get("released_year", "Unknown"),
                "genre": metadata.get("genre", "Unknown"),
                "imdb_rating": float(metadata.get("imdb_rating", 0) or 0),
                "overview": point.payload.get("page_content", "")
            })

    # Check whether any movies were found
    if not movies:
        return f"No movies found for {person_name} in the database."

    # Sort movies by IMDb rating (highest first)
    movies.sort(
        key=lambda x: x["imdb_rating"],
        reverse=True
    )

    # Select the top five movies
    top_movies = movies[:5]

    # Format the results
    response = f"Top movies for {person_name}:\n\n"

    for i, movie in enumerate(top_movies, start=1):
        response += (
            f"{i}. {movie['title']} ({movie['released_year']})\n"
            f"   Genre: {movie['genre']}\n"
            f"   IMDb Rating: {movie['imdb_rating']}/10\n\n"
            f"   Overview: {movie['overview']}\n\n"
        )

    return response

# ----------------------------------
# CREATE AGENTS, CONNECT TO LANGFUSE
# ----------------------------------

# Agent for tool 1
lf_search_movies = lf.get_prompt("search_movies_agent").get_langchain_prompt()

search_movies_agent = create_agent(
    model=model,
    tools=[search_movies],
    system_prompt=lf_search_movies
)

# Agent for tool 2
lf_similar_movies = lf.get_prompt("similar_movies_agent").get_langchain_prompt()

similar_movies_agent = create_agent(
    model=model,
    tools=[similar_movies],
    system_prompt=lf_similar_movies
)

# Agent for tool 3
lf_hall_of_fame = lf.get_prompt("hall_of_fame_agent", type="text").get_langchain_prompt()

hall_of_fame_agent = create_agent(
    model=model,
    tools=[hall_of_fame_search],
    system_prompt=lf_hall_of_fame
)

# ----------------------------------
# TURN AGENTS INTO TOOLS
# ----------------------------------

# Movie search tool
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
        "messages": [
            {
                "role": "user", 
                "content": f"""
                Current user query: {query} + 
                
                Previous conversation: {history}
                """
            }
        ]    
    }, config={"callbacks": [langfuse_handler]})

    return record_agent_usage(result)

# Movie similar tool
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
                "messages": [
            {
                "role": "user", 
                "content": f"""
                Current user query: {query} + 
                
                Previous conversation: {history}
                """
            }
        ]    
    }, config={"callbacks": [langfuse_handler]})

    return record_agent_usage(result)


# Hall of Fame tool
@tool(args_schema=AgentInput)
def hall_of_fame(query: str, history: str) -> str:
    """Tool to explore the highest-rated movies of actors and directors."""

    result = hall_of_fame_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"""
                Current user query: {query}

                Previous conversation:
                {history}
                """
            }
        ]
    }, config={"callbacks": [langfuse_handler]})

    return record_agent_usage(result)

# ----------------------------------
# CREATE SUPERVISOR AGENT FOR CINEGUIDE
# ----------------------------------

lf_supervisor = lf.get_prompt("supervisor_agent", type="text").get_langchain_prompt()

supervisor_agent = create_agent(
    model=model,
    tools=[
        movies_search,
        movie_similarity,
        hall_of_fame
    ],
       system_prompt=lf_supervisor
)

def cineguide(query: str, history: str) -> str:
    result = supervisor_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"""
                Current user query: {query}

                Previous conversation:
                {history}
                """
            }
        ]
    }, config={"callbacks": [langfuse_handler]})

    return record_agent_usage(result)