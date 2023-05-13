# Import necessary libraries
import streamlit as st
from utils.video_processing import load_transcript, split_transcript
from utils.database import create_db
###Trying another model
##from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from utils.prompts import CONDENSE_PROMPT, QA_PROMPT
openaiapikey = st.secrets["OPENAI_API_KEY"]

# Set Streamlit page configuration
st.set_page_config(page_title='🧠MAKERS TEST🤖', layout='wide')

# Initialize session states
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "video_url" not in st.session_state:
    st.session_state["video_url"] = ""
if "video_loaded" not in st.session_state:
    st.session_state["video_loaded"] = False
if "chain" not in st.session_state:
    st.session_state["chain"] = None

# Create a container for the header to make it more accessible
header = st.container()

with header:
    st.title("🤖 Chat with your videos --By Juan Arfuch")
    st.markdown(""" 
        Welcome to my unique ChatBot powered by LangChain, OpenAI, and Streamlit. This is a project for Makers, created with lots of ❤️ and ☕.
        This bot is not your ordinary bot, it's a super bot that uses the transcript of a YouTube video of your choice to answer your questions. 
        All you have to do is provide a YouTube video url includint the id, and then you can ask any questions related to the video content.
        Let's make learning fun and interactive!
    """) 
    # New chat button has been moved to the header for better accessibility

# Wrap transcript loading and splitting in try/except block
try:
    if not st.session_state["video_loaded"]:
        with st.form('Load video'):
            st.session_state["video_url"] = st.text_input("Please enter the YouTube video URL: ")
            submitted = st.form_submit_button("Load Video")
            if submitted:
                with st.spinner('Loading video transcript...'):
                    transcript = load_transcript(st.session_state["video_url"])
                    docs = split_transcript(transcript)
                    db = create_db(docs)

                st.success("Video transcript loaded successfully!")
                st.session_state["video_loaded"] = True
                ##Creating the conversational retrieval chain###
                ###Setting the chat model
                ##chat= ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo")
                chat=OpenAI(temperature=0.2, model_name="gpt-3.5-turbo")
                question_generator = LLMChain(llm=chat, prompt=CONDENSE_PROMPT)
                doc_chain = load_qa_chain(chat, prompt=QA_PROMPT)
                st.session_state["chain"] = ConversationalRetrievalChain(
                    retriever=db,
                    question_generator=question_generator,
                    combine_docs_chain=doc_chain,
                )
            else:
                st.warning("Please provide a valid YouTube URL, make sure you are including the video id, your url must be like this:https://www.youtube.com/watch?v=C_78DM8fG6E&t=7s")
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
        if st.button("Start new chat"):
            st.session_state["chat_history"] = []
            st.session_state["video_url"] = ""
            st.session_state["video_loaded"] = False
            st.session_state["chain"] = None

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


# Footer
st.markdown("""
---Built with ❤️ by [Juan Arfuch](https://github.com/juanarfuch) using [Streamlit](https://streamlit.io), [OpenAI](https://openai.com), and [LangChain](https://langchain.ai)
This is a project for Makers! Check out my [GitHub](https://github.com/juanarfuch) for more cool projects!
""")


