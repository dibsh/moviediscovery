import streamlit as st
from langchain_core.messages import ToolMessage
from agent_st.agent import (
    movies_search,
    movie_similarity,
    hall_of_fame,
    cineguide,
    reset_usage_tracking,
    get_usage_summary
)
from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()

# ----------------------------------
# CONFIGURATIONS
# ----------------------------------

# Page configuration
st.set_page_config(
    page_title="CineMind",
    page_icon="🎬",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>

/* Overall page */
.stApp {
    background-color: #111315;
}

/* Main content spacing */
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Home screen feature cards */
.st-key-feature-discover div[data-testid="stButton"] button,
.st-key-feature-similar div[data-testid="stButton"] button,
.st-key-feature-hall div[data-testid="stButton"] button,
.st-key-feature-guide div[data-testid="stButton"] button {

    height: 155px;
    width: 100%;
    white-space: pre-wrap;
    text-align: left;
    padding: 20px;

    background-color: #222426;
    color: #FFFFFF;

    border: 1px solid #393939;
    border-radius: 14px;

    font-size: 16px;
    font-weight: 600;

    transition: all 0.2s ease;
}

/* Hover effect */
.st-key-feature-discover div[data-testid="stButton"] button:hover,
.st-key-feature-similar div[data-testid="stButton"] button:hover,
.st-key-feature-hall div[data-testid="stButton"] button:hover,
.st-key-feature-guide div[data-testid="stButton"] button:hover {

    border: 1px solid #D6AE68;
    background-color: #2B2C2D;
    transform: translateY(-3px);
}

/* Remove default focus border */
.st-key-feature-discover div[data-testid="stButton"] button:focus,
.st-key-feature-similar div[data-testid="stButton"] button:focus,
.st-key-feature-hall div[data-testid="stButton"] button:focus,
.st-key-feature-guide div[data-testid="stButton"] button:focus {
    border-color: #D6AE68;
    color: #FFFFFF;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------
# HOME PAGE AND FEATURE SELECTION
# ----------------------------------

# Initialise Session State
if "feature" not in st.session_state:
    st.session_state.feature = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {
        "Discover Movies": [],
        "Find Similar Movies": [],
        "Hall of Fame": [],
        "CineGuide": []
    }

# Home Screen
if st.session_state.feature is None:

    # Header
    st.markdown("""
    <div style="margin-bottom: 35px;">

        <div style="
            color: #D6AE68;
            font-size: 13px;
            font-weight: 600;
            letter-spacing: 3px;
            margin-bottom: 12px;
        ">
            YOUR PERSONAL CINEMA COMPANION
        </div>

        <h1 style="
            font-size: 42px;
            color: #FFFFFF;
            margin-bottom: 10px;
        ">
            🎬 CineMind
        </h1>

        <p style="
            font-size: 17px;
            color: #AAAAAA;
        ">
            Find your next favourite movie.
            Discover films, explore stories and celebrate cinema.
        </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h3 style="color: #FFFFFF; margin-bottom: 18px;">
        Explore CineMind
    </h3>
    """, unsafe_allow_html=True)

    # Feature cards
    col1, col2 = st.columns(2, gap="large")

    with col1:

        with st.container(key="feature-discover"):
            if st.button(
                "🎞️  Discover Movies\n\n"
                "Find films that match your mood, "
                "interests and preferences.",
                use_container_width=True
            ):
                st.session_state.feature = "Discover Movies"
                st.rerun()

        with st.container(key="feature-hall"):
            if st.button(
                "🏆  Hall of Fame\n\n"
                "Explore the highest-rated movies "
                "of actors and directors.",
                use_container_width=True
            ):
                st.session_state.feature = "Hall of Fame"
                st.rerun()

    with col2:

        with st.container(key="feature-similar"):
            if st.button(
                "🍿  Find Similar Movies\n\n"
                "Discover movies similar to "
                "your favourites.",
                use_container_width=True
            ):
                st.session_state.feature = "Find Similar Movies"
                st.rerun()

        with st.container(key="feature-guide"):
            if st.button(
                "✨  CineGuide\n\n"
                "Your AI-powered companion "
                "for all things cinema.",
                use_container_width=True
            ):
                st.session_state.feature = "CineGuide"
                st.rerun()

    # Footer
    st.markdown("""
    <div style="
        text-align: center;
        color: #777777;
        font-size: 12px;
        letter-spacing: 2px;
        margin-top: 50px;
    ">
        CINEMIND · DISCOVER MORE THAN JUST MOVIES
    </div>
    """, unsafe_allow_html=True)


# Display once a feature is chosen

else:

    feature = st.session_state.feature

    # Back to Home and Clear Chat button
    col1, col2 = st.columns([5, 1])

    with col1:
        if st.button("← Back to Home"):
            st.session_state.feature = None
            st.rerun()

    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history[feature] = []
            st.rerun()

    st.title(f"🎬 {feature}")

    # Get chat history for selected feature
    messages = st.session_state.chat_history[feature]

# ----------------------------------
# CHAT INTERFACE AND USER INPUT
# ----------------------------------

    # Display chat messages from history on app rerun
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input Placeholder
    if feature == "Discover Movies":
        placeholder = "Describe the type of movie you're looking for..."

    elif feature == "Find Similar Movies":
        placeholder = "Enter a movie title to find similar movies..."

    elif feature == "Hall of Fame":
        placeholder = "Enter an actor or director's name to find our their top ranked work..."

    else:
        placeholder = "Ask CineGuide anything about movies..."

    # User Input
    if prompt := st.chat_input(placeholder):

        # Build conversation history
        history = "\n".join(
            f'{message["role"]}: {message["content"]}'
            for message in messages
        )
        
        # Display user message
        messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        # Run selected agent
        with st.chat_message("assistant"):

                with st.spinner("CineMind is thinking..."):

                    # Reset token usage for this request
                    reset_usage_tracking()

                    if feature == "Discover Movies":
                        response = movies_search.invoke({
                            "query": prompt,
                            "history": history
                        })

                    elif feature == "Find Similar Movies":
                        response = movie_similarity.invoke({
                            "query": prompt,
                            "history": history
                        })

                    elif feature == "Hall of Fame":
                        response = hall_of_fame.invoke({
                            "query": prompt,
                            "history": history
                        })

                    else:
                        response = cineguide(prompt, history)

                    # Get complete usage summary
                    usage = get_usage_summary()

                    # Display response
                    st.markdown(response)

                    # Display Tool Usage
                    with st.expander("🛠️ Tools Used"):
                        if usage["tools_used"]:
                            for tool in usage["tools_used"]:
                                st.markdown(f"**{tool['name']}**")
                                st.json(tool["args"])
                        else:
                            st.caption("CineMind answered without using any tools.")

                    # Display Token Usage
                    with st.expander("📊 Usage Details"):

                        st.write(
                            f"Input tokens: "
                            f"{usage['total_input_tokens']:,}"
                        )

                        st.write(
                            f"Output tokens: "
                            f"{usage['total_output_tokens']:,}"
                        )

                        st.write(
                            f"Total tokens: "
                            f"{usage['total_tokens']:,}"
                        )

                        st.divider()

                        st.write(
                            f"Estimated cost: Rp {usage['price']:,.2f}"
                        )

                    # Display previous conversation
                    with st.expander("**History Chat:**"):
                        if history:
                            st.code(history)
                        else:
                            st.write("No previous conversation.")

        # Save assistant response
        messages.append({
            "role": "assistant",
            "content": response
        })