import streamlit as st
import re
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv


load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')


# Set page configuration
st.set_page_config(
    page_title="YouTube Video Summarizer",
    page_icon="📺",
    layout="centered"
)
st.secrets["GROQ_API_KEY"]

# App title and description
st.title("YouTube Video Summarizer")
st.markdown("Enter a YouTube video URL to get a concise summary of its content.")

# Function to extract YouTube video ID from URL
def extract_video_id(url):
    # Regular expression patterns for different YouTube URL formats
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([\w-]+)',
        r'(?:youtube\.com\/embed\/)([\w-]+)',
        r'(?:youtube\.com\/v\/)([\w-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

# Function to get video transcript
def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        return None

# Function to summarize transcript
def summarize_transcript(transcript_text):
    with st.spinner("Processing transcript..."):
        # Text splitting
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        documents = text_splitter.create_documents([transcript_text])
        
        # Store vectors in vector store
        try:
            embedding = OllamaEmbeddings(model="mxbai-embed-large:latest")
            db = FAISS.from_documents(documents, embedding)
            
            # Vector store as retriever
            retriever = db.as_retriever()
            
            # Get relevant documents
            docs = retriever.get_relevant_documents('What is the main idea of the video?')
            
            # Initialize the model for summarization
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0,
                api_key=groq_api_key,
            )
            
            # Prompt template
            prompt = PromptTemplate.from_template(
                "Summarize the following text:\n\n{text}\n\nSummary: and dont include unnecessary text start directlly by generating the summary"
            )
            
            # Create RAG chain
            summary_chain = LLMChain(llm=llm, prompt=prompt)
            retrieved_text = "\n\n".join([doc.page_content for doc in docs])  # Combine small docs
            summary = summary_chain.run({"text": retrieved_text})
            
            return summary
        except Exception as e:
            st.error(f"Error in summarization: {str(e)}")
            return None

# Main app
def main():
    # Input field for YouTube URL
    youtube_url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
    
    if st.button("Summarize", type="primary"):
        if youtube_url:
            # Extract video ID
            video_id = extract_video_id(youtube_url)
            
            if not video_id:
                st.error("Invalid YouTube URL. Please check and try again.")
                return
            
            # Get transcript
            with st.spinner("Fetching video transcript..."):
                transcript = get_transcript(video_id)
            
            if transcript:
                # Convert transcript to text
                transcript_text = " ".join([entry["text"] for entry in transcript])
                
                # Summarize transcript
                summary = summarize_transcript(transcript_text)
                
                if summary:
                    # Display summary
                    st.subheader("Video Summary")
                    st.markdown(summary)
                    
                    # Additional information
                    with st.expander("View Raw Transcript"):
                        st.text(transcript_text)
        else:
            st.warning("Please enter a YouTube video URL.")

if __name__ == "__main__":
    main()