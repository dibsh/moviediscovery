import streamlit as st
from langchain_core.messages import ToolMessage
from agent_st.agent import movies_search, movie_similarity
from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()

# def send_chat(question: str, history: str) -> dict:
#     result = agent.invoke(
#         {"messages": history + [{"role": "user", "content": question}]},
#         config={"callbacks": [langfuse_handler]}
#     )
#     answer = result["messages"][-1].content

#     total_input_tokens = 0
#     total_output_tokens = 0

#     for message in result["messages"]:
#         if "usage_metadata" in message.response_metadata:
#             total_input_tokens += message.response_metadata["usage_metadata"]["input_tokens"]
#             total_output_tokens += message.response_metadata["usage_metadata"]["output_tokens"]
#         elif "token_usage" in message.response_metadata:
#             # Fallback for older or different structures
#             total_input_tokens += message.response_metadata["token_usage"].get("prompt_tokens", 0)
#             total_output_tokens += message.response_metadata["token_usage"].get("completion_tokens", 0)

#     price = 18_000*(total_input_tokens*0.15 + total_output_tokens*0.6)/1_000_000

#     tool_messages = []
#     for message in result["messages"]:
#         if isinstance(message, ToolMessage):
#             tool_message_content = message.content
#             tool_messages.append(tool_message_content)

#     response = {
#         "answer": answer,
#         "price": price,
#         "total_input_tokens": total_input_tokens,
#         "total_output_tokens": total_output_tokens,
#         "tool_messages": tool_messages
#     }
#     return response

# Page configuration
st.set_page_config(
    page_title="CineMind",
    page_icon="🎬",
    layout="wide"
)

# App header
st.title("🎬 CineMind")
st.subheader("Think less. Watch better.")
st.write(
    "Your AI-powered movie discovery companion. "
    "Find your next favourite movie."
)

# Select movie feature
feature = st.radio(
    "What would you like to do?",
    [
        "Discover Movies",
        "Find Similar Movies"
    ],
    horizontal=True
)

# Initialise chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {
        "Discover Movies": [],
        "Find Similar Movies": []
    }

# Get history for selected feature
messages = st.session_state.chat_history[feature]

# Display chat messages from history on app rerun
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if feature == "Discover Movies":

    placeholder = (
        "Describe the type of movie you're looking for..."
    )

else:

    placeholder = (
        "Enter a movie title to find similar movies..."
    )

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

                if feature == "Discover Movies":

                    response = movies_search.invoke({
                        "query": prompt,
                        "history": history
                    })

                else:

                    response = movie_similarity.invoke({
                        "query": prompt,
                        "history": history
                    })

                st.markdown(response)

            with st.expander("**History Chat:**"):
                 st.code(history)
            
            # with st.expander("**Usage Details:**"):
            #      st.code(f'input token : {response["total_input_tokens"]}\noutput token : {response["total_output_tokens"]}')

    # Save assistant response
    messages.append({
        "role": "assistant",
        "content": response
    })


# Accept user input
# if prompt := st.chat_input("Ask me anything about Resumes!"):
#     messages_history = st.session_state.get("messages", [])[-20:]
#     history = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in messages_history]) or " "

#     # Display user message in chat message container
#     with st.chat_message("Human"):
#         st.markdown(prompt)
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
    
#     # Display assistant response in chat message container
#     with st.chat_message("AI"):
#         response = send_chat(prompt, messages_history)
#         answer = response["answer"]
#         st.markdown(answer)
#         st.session_state.messages.append({"role": "assistant", "content": answer})

#     with st.expander("**Tool Calls:**"):
#         st.code(response["tool_messages"])

    # with st.expander("**History Chat:**"):
    #     st.code(history)

    # with st.expander("**Usage Details:**"):
    #     st.code(f'input token : {response["total_input_tokens"]}\noutput token : {response["total_output_tokens"]}')