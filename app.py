# Import necessary libraries
import streamlit as st
from utils.video_processing import load_transcript, split_transcript
from utils.database import create_db
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from utils.prompts import CONDENSE_PROMPT, QA_PROMPT
openaiapikey = st.secrets["OPENAI_API_KEY"]

# Set Streamlit page configuration
st.set_page_config(page_title='🧠ChatBot🤖', layout='wide')

# Initialize session states
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "video_url" not in st.session_state:
    st.session_state["video_url"] = ""
if "video_loaded" not in st.session_state:
    st.session_state["video_loaded"] = False
if "chain" not in st.session_state:
    st.session_state["chain"] = None

# Sidebar options
with st.sidebar.expander("🛠️ Settings ", expanded=False):
    # Add some explanation here
    st.markdown("""To start a new chat, click the button below. This will clear your chat history and allow you to start a new conversation.""")
    # Option to start a new chat
    if st.button("New Chat"):
        st.session_state["chat_history"] = []
        st.session_state["video_url"] = ""
        st.session_state["video_loaded"] = False
        st.session_state["chain"] = None

# Set up the Streamlit app layout
st.title("🤖 Chat with your videos --By Juan Arfuch")
st.markdown(""" 
       Welcome to my unique ChatBot powered by LangChain, OpenAI, and Streamlit. This is a project for Makers, created with lots of ❤️ and ☕.
    This bot is not your ordinary bot, it's a super bot that uses the transcript of a YouTube video of your choice to answer your questions. 
    All you have to do is provide a YouTube video URL, and then you can ask any questions related to the video content.
    Let's make learning fun and interactive!
""") 

# Wrap transcript loading and splitting in try/except block
try:
    if not st.session_state["video_loaded"]:
        st.session_state["video_url"] = st.text_input("Please enter the YouTube video URL: ")
        if st.button('Load video'):
            if st.session_state["video_url"]:
                with st.spinner('Loading video transcript...'):
                    transcript = load_transcript(st.session_state["video_url"])
                    docs = split_transcript(transcript)
                    db = create_db(docs)

                st.success("Video transcript loaded successfully!")
                st.session_state["video_loaded"] = True

                llm = OpenAI(temperature=0.2)
                question_generator = LLMChain(llm=llm, prompt=CONDENSE_PROMPT)
                doc_chain = load_qa_chain(llm, prompt=QA_PROMPT)
                st.session_state["chain"] = ConversationalRetrievalChain(
                    retriever=db,
                    question_generator=question_generator,
                    combine_docs_chain=doc_chain,
                )
            else:
                st.warning("Please provide a valid YouTube URL.")

    else:
        st.success("Video transcript is already loaded. You can start asking questions.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}. Please try again with a different video URL.")

# Ask questions and display responses
try: 
    if st.session_state["chain"] is not None:
    user_question = st.text_input("Enter your question related to the video content")
    if st.button('Submit question'):
        if user_question and st.session_state["video_loaded"] and st.session_state["chain"]:
            with st.spinner('Processing your question...'):
                result = st.session_state["chain"]({"question": user_question, "chat_history": st.session_state["chat_history"]})
                st.session_state["chat_history"].append((user_question, result['answer']))

            # Display the conversation history
            with st.expander("Conversation History", expanded=True):
                for user, bot in st.session_state["chat_history"]:
                    st.markdown(f'**User**: {user}')
                    st.markdown(f'<span style="color:green">**Bot**: {bot}</span>', unsafe_allow_html=True)
except Exception as e:
    st.error(f'An error occurred: {e}')
    st.markdown("""
        Please make sure the YouTube URL is correct and accessible. If the problem persists, try to start a new chat.
    """)
    with st.form(key='question_form'):
        st.markdown("Ask a question related to the video content:")
        question = st.text_input("")
        submit_button = st.form_submit_button(label='Ask')

        if submit_button:
            if question:
                with st.spinner('Generating response...'):
                   response = st.session_state["chain"]({"question": user_question, "chat_history":st.session_state["chat_history"]})
                   st.session_state["chat_history"].append({'user': question, 'bot': response})
            else:
                st.warning("Please enter a question before pressing the Ask button.")

    # Display chat history
    if st.session_state["chat_history"]:
        for chat in st.session_state["chat_history"]:
            st.markdown(f"**You:** {chat['user']}")
            st.markdown(f"**ChatBot:** {chat['bot']}")  





