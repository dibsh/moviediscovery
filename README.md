# CINEMIND APP

Movie Discovery and Recommendation App, Cinemind. Integrated with Langfuse and Streamlit.

## Flow

![Flow diagram agent][flow.png]

## Local Installation

1. Clone this repository
2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```
3. Set up environment variables in `.env`:
   - `OPENAI_API_KEY`
   - `QDRANT_URL`
   - `QDRANT_API_KEY`
   - `QDRANT_COLLECTION_NAME`
   - `EMBEDDING_MODEL`
   - `LLM_MODEL`

## Running Locally

### Streamlit Simulation
```bash
Streamlit link: https://moviediscovery-cinemind.streamlit.app/#cine-mind
```

## Contoh query

1. For Feature 1 (Discover Movies): `A lone astronaut standing on a massive water planet with towering tidal waves under a swirling, hyper-realistic black hole gargantua.`
2. For Feature 2 (Find Similar Movies): `Can you recommend movies similar to Inception?`
3. For Feature 3: (Hall of Fame): `List film terbaik Bong Joon Ho`
4. For Feature 4 (Cineguide): `Find Christopher Nolan's highest-rated movies and then recommend some movies similar to Interstellar`

[flow.png]: flow.png