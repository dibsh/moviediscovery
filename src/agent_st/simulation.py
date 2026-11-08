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


import streamlit as st

# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="CineMind",
    page_icon="🎬",
    layout="wide"
)

# CUSTOM STYLING

st.markdown("""
<style>

/* Overall background */
.stApp {
    background-color: #F2F2F0;
    color: #292929;
}

/* Main content spacing */
.block-container {
    padding-top: 3rem;
    padding-bottom: 3rem;
    max-width: 1150px;
}

/* Main headings */
h1, h2, h3 {
    color: #292929 !important;
}

/* Body text */
p {
    color: #626262;
}

/* Feature cards */
div[data-testid="stButton"] button {
    min-height: 145px;
    width: 100%;
    white-space: pre-wrap;
    text-align: left;
    padding: 22px;

    background-color: #FFFFFF;
    color: #333333;

    border: 1px solid #E1E0DC;
    border-radius: 16px;

    font-size: 16px;
    font-weight: 600;

    box-shadow: 0 3px 10px rgba(0,0,0,0.035);
    transition: all 0.2s ease;
}

/* Hover effect */
div[data-testid="stButton"] button:hover {
    background-color: #FAF7F0;
    border: 1px solid #B99558;
    color: #292929;
    box-shadow: 0 5px 15px rgba(0,0,0,0.07);
}

/* Button focus */
div[data-testid="stButton"] button:focus {
    border-color: #B99558;
    color: #292929;
}

/* Chat input */
div[data-testid="stChatInput"] {
    background-color: #FFFFFF;
    border-radius: 12px;
}

/* Footer */
.cinemind-footer {
    text-align: center;
    color: #999999;
    font-size: 12px;
    letter-spacing: 2px;
    margin-top: 55px;
}

</style>
""", unsafe_allow_html=True)

# SESSION STATE

if "feature" not in st.session_state:
    st.session_state.feature = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {
        "Discover Movies": [],
        "Find Similar Movies": [],
        "Hall of Fame": [],
        "CineGuide": []
    }


# HOME SCREEN

if st.session_state.feature is None:

    # Header
    st.markdown(
        """
        <p style="
            color: #B99558;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 3px;
            margin-bottom: 5px;
        ">
            YOUR PERSONAL CINEMA COMPANION
        </p>
        """,
        unsafe_allow_html=True
    )

    st.title("🎬 CineMind")

    st.markdown(
        """
        <p style="
            color: #686868;
            font-size: 18px;
            margin-top: -8px;
        ">
            Think less. Watch better.
        </p>

        <p style="
            color: #888888;
            font-size: 15px;
        ">
            Your AI-powered movie discovery companion.
            Find your next favourite movie.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    st.write("")

    st.subheader("Explore CineMind")

    st.caption("Choose a feature to get started.")

    st.write("")

    # Feature cards
    col1, col2 = st.columns(2, gap="large")

    with col1:

        if st.button(
            "🎞️  Discover Movies\n\n"
            "Find films that match your mood, interests and preferences.",
            use_container_width=True
        ):
            st.session_state.feature = "Discover Movies"
            st.rerun()

        if st.button(
            "🏆  Hall of Fame\n\n"
            "Explore the highest-rated movies of actors and directors.",
            use_container_width=True
        ):
            st.session_state.feature = "Hall of Fame"
            st.rerun()

    with col2:

        if st.button(
            "🍿  Find Similar Movies\n\n"
            "Discover movies similar to your favourites.",
            use_container_width=True
        ):
            st.session_state.feature = "Find Similar Movies"
            st.rerun()

        if st.button(
            "✨  CineGuide\n\n"
            "Your AI-powered companion for all things cinema.",
            use_container_width=True
        ):
            st.session_state.feature = "CineGuide"
            st.rerun()

    # Footer
    st.markdown(
        """
        <div class="cinemind-footer">
            CINEMIND · DISCOVER MORE THAN JUST MOVIES
        </div>
        """,
        unsafe_allow_html=True
    )

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