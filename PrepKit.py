import streamlit as st
import PyPDF2
import os
import uuid
import tempfile
import requests
import re
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import urllib.parse

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# YouTube transcript handling
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

# Load environment variables
load_dotenv()

# Set Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your-api-key-here")
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "your-serpapi-key-here")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)

# Page configuration
st.set_page_config(layout="wide", page_title="PrepKit", page_icon="🎓")

# Initialize session state
if "documents" not in st.session_state:
    st.session_state.documents = []
if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "quiz_generated" not in st.session_state:
    st.session_state.quiz_generated = False
if "parsed_questions" not in st.session_state:
    st.session_state.parsed_questions = []
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}
if "results" not in st.session_state:
    st.session_state.results = None
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = []
if "combined_text" not in st.session_state:
    st.session_state.combined_text = ""

# Function to extract YouTube ID from URL
def extract_youtube_id(url):
    # Handle different YouTube URL formats
    query = urllib.parse.urlparse(url)
    if query.hostname in ('youtu.be'):
        return query.path[1:]
    if query.hostname in ('www.youtube.com', 'youtube.com'):
        if query.path == '/watch':
            p = urllib.parse.parse_qs(query.query)
            return p['v'][0]
        if query.path.startswith('/embed/'):
            return query.path.split('/')[2]
        if query.path.startswith('/v/'):
            return query.path.split('/')[2]
    return None

# Function to get transcript from YouTube video
def get_youtube_transcript(video_url):
    video_id = extract_youtube_id(video_url)
    if not video_id:
        return None, "Invalid YouTube URL"
    
    try:
        # Try with different language options
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        except:
            # Try with English if default language fails
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        
        # Directly concatenate text from entries
        transcript_text = ""
        for entry in transcript_list:
            transcript_text += entry.get('text', '') + " "
            
        return transcript_text, f"YouTube_video_{video_id}"
    except Exception as e:
        import traceback
        st.write(traceback.format_exc())  # Print full traceback for debugging
        return None, f"Error getting transcript: {str(e)}"

# Function to extract text from PDF
def read_and_textify(files):
    documents = []
    for file in files:
        pdfReader = PyPDF2.PdfReader(file)
        
        for i in range(len(pdfReader.pages)):
            pageObj = pdfReader.pages[i]
            text = pageObj.extract_text()
            if text:  # Only add if text was extracted successfully
                documents.append({
                    "content": text,
                    "source": f"{file.name}_page_{i+1}"
                })
            pageObj.clear()
    return documents

# Function to extract text from HTML
def load_html(html_file):
    file_content = html_file.read().decode("utf-8")
    soup = BeautifulSoup(file_content, "html.parser")
    
    # Extract the title of the HTML page
    title = soup.title.string if soup.title else "No title found"
    
    # Extract all the paragraphs <p> from the HTML
    paragraphs = soup.find_all("p")
    paragraph_text = "\n".join([para.get_text() for para in paragraphs])
    
    # Combine title and paragraphs into a single string
    full_text = f"Title: {title}\n\n{paragraph_text}"
    return full_text
# Function to extract text from PDF
def load_pdf(pdf_file):
    pdfReader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for i in range(len(pdfReader.pages)):
        pageObj = pdfReader.pages[i]
        text += pageObj.extract_text() + "\n"
    return text
# Function to extract text from plain text file
def load_text(txt_file):
    return txt_file.read().decode("utf-8")

# Create a document splitter function
def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    split_docs = []
    for doc in documents:
        chunks = text_splitter.split_text(doc["content"])
        for chunk in chunks:
            split_docs.append({
                "content": chunk,
                "source": doc["source"]
            })
    return split_docs

# Function to fetch study material links from Google
def get_study_material_links(query, api_key):
    url = "https://serpapi.com/search"
    params = {"engine": "google", "q": query, "api_key": api_key}
    response = requests.get(url, params=params)
    data = response.json()

    links = []
    if "organic_results" in data:
        for result in data["organic_results"]:
            links.append({"title": result.get("title", "No Title"), "link": result["link"]})
    return links

# Function to fetch YouTube video links
def get_youtube_links(query, api_key):
    url = "https://serpapi.com/search"
    params = {"engine": "youtube", "search_query": query, "api_key": api_key}
    response = requests.get(url, params=params)
    data = response.json()

    videos = []
    if "video_results" in data:
        for video in data["video_results"]:
            video_link = video.get("link", "")
            if video_link:
                videos.append({"title": video.get("title", "No Title"), "link": video_link})
    return videos

# Updated prompt to provide structured output that's easier to parse
quiz_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """Generate {num_questions} quiz questions with four multiple-choice options for each question.
        Format each question exactly as follows:
        
        Q1: [Question text]
        A) [Option A]
        B) [Option B]
        C) [Option C]
        D) [Option D]
        CORRECT: [Correct option letter only - A, B, C, or D]
        
        Q2: [Question text]
        ...and so on
        
        Do not include any explanations or additional text between questions. Follow this exact format.
        """),
        ("human", "Create quiz questions based on the following content:\n\nContent: {text}"),
    ]
)

# Function to generate quiz using LangChain
def generate_quiz(num_questions, text):
    Chain = quiz_prompt | llm | StrOutputParser()
    result = Chain.invoke({"num_questions": num_questions, "text": text})
    return result

# Function to parse quiz questions and answers
def parse_quiz(quiz_text):
    questions = []
    
    # Use regex to find questions and options
    pattern = r'Q(\d+): (.*?)\nA\) (.*?)\nB\) (.*?)\nC\) (.*?)\nD\) (.*?)\nCORRECT: ([A-D])'
    matches = re.findall(pattern, quiz_text, re.DOTALL)
    
    for match in matches:
        q_num, question, option_a, option_b, option_c, option_d, correct = match
        questions.append({
            'number': int(q_num),
            'question': question.strip(),
            'options': {
                'A': option_a.strip(),
                'B': option_b.strip(),
                'C': option_c.strip(),
                'D': option_d.strip()
            },
            'correct': correct.strip()
        })
    
    return questions

# Function to display an interactive quiz
def display_quiz(questions):
    st.subheader("Answer the questions below:")
    
    user_answers = {}
    
    for q in questions:
        st.write(f"**Question {q['number']}**: {q['question']}")
        
        # Create radio buttons for options
        options = [f"A) {q['options']['A']}", 
                  f"B) {q['options']['B']}", 
                  f"C) {q['options']['C']}", 
                  f"D) {q['options']['D']}"]
        
        answer = st.radio("Select your answer:", options, key=f"q{q['number']}")
        
        # Extract just the option letter (A, B, C, D)
        user_answers[q['number']] = answer[0]
        
        st.write("---")
    
    return user_answers

# Function to calculate quiz score
def calculate_score(questions, user_answers):
    total = len(questions)
    correct = 0
    
    results = []
    
    for q in questions:
        q_num = q['number']
        if q_num in user_answers and user_answers[q_num] == q['correct']:
            correct += 1
            results.append({
                'number': q_num,
                'result': 'Correct',
                'user_answer': user_answers[q_num],
                'correct_answer': q['correct']
            })
        else:
            results.append({
                'number': q_num,
                'result': 'Incorrect',
                'user_answer': user_answers.get(q_num, 'Not answered'),
                'correct_answer': q['correct']
            })
    
    return {
        'total': total,
        'correct': correct,
        'percentage': (correct / total) * 100 if total > 0 else 0,
        'details': results
    }

# Main application
st.title("🎓 PrepKit")
st.write("A comprehensive platform for document processing, knowledge base creation, search, and quiz generation")

# Main navigation
tabs = st.tabs(["Document Processing", "Study Resources", "Quiz Generator", "Knowledge Base"])

# Tab 1: Document Processing and QA
with tabs[0]:
    st.header("Document Processing & QA")
    st.write("Upload documents or add YouTube videos to create your knowledge base")
    
    # Sub-tabs for different input methods
    doc_tab1, doc_tab2 = st.tabs(["Document Upload", "YouTube Videos"])
    
    # File uploader in doc_tab1
    with doc_tab1:
        uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True, type=["txt", "pdf"])
        
        if uploaded_files:
            if st.button("Process Documents", key="process_docs"):
                with st.spinner("Processing documents..."):
                    pdf_documents = read_and_textify(uploaded_files)
                    st.session_state.documents.extend(pdf_documents)
                    st.success(f"Added {len(pdf_documents)} document chunks")
                    st.session_state.kb_ready = False
    
    # YouTube input in doc_tab2
    with doc_tab2:
        youtube_url = st.text_input("Enter YouTube URL", key="doc_youtube_url")
        
        if st.button("Add YouTube Video", key="add_youtube") and youtube_url:
            with st.spinner("Fetching YouTube transcript..."):
                transcript, source = get_youtube_transcript(youtube_url)
                if transcript:
                    youtube_doc = {"content": transcript, "source": source}
                    st.session_state.documents.append(youtube_doc)
                    st.success(f"Added transcript from {source}")
                    st.session_state.kb_ready = False
                else:
                    st.error(source)  # Display error message
    
    # Display current knowledge base status
    st.subheader("Knowledge Base Status")
    st.write(f"Total document chunks: {len(st.session_state.documents)}")
    
    # Sources summary
    if st.session_state.documents:
        sources = {}
        for doc in st.session_state.documents:
            source_key = doc["source"].split("_page_")[0] if "_page_" in doc["source"] else doc["source"]
            if source_key in sources:
                sources[source_key] += 1
            else:
                sources[source_key] = 1
        
        st.write("Sources:")
        for source, count in sources.items():
            st.write(f"- {source}: {count} chunks")
    
    # Button to build knowledge base
    if st.session_state.documents and not st.session_state.kb_ready:
        if st.button("Build Knowledge Base"):
            # Replace your current code block in the knowledge base building section
            # Replace your current code block in the knowledge base building section
            with st.spinner("Building knowledge base..."):
                try:
                    # Split documents into smaller chunks
                    split_docs = split_documents(st.session_state.documents)
                    
                    if not split_docs:
                        st.error("No content to process after splitting documents.")
                        
                    
                    # Prepare data for Chroma
                    texts = [doc["content"] for doc in split_docs]
                    metadatas = [{"source": doc["source"]} for doc in split_docs]
                    
                    # Create a temporary directory for the Chroma database
                    persist_directory = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
                    os.makedirs(persist_directory, exist_ok=True)
                    
                    # Extract embeddings using Google's Generative AI
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001",
                        google_api_key=GOOGLE_API_KEY
                    )
                    
                    # Create vector store with metadata
                    vectordb = Chroma.from_texts(
                        texts=texts,
                        embedding=embeddings,
                        metadatas=metadatas,
                        persist_directory=persist_directory
                    )
                    
                    # Create retriever
                    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
                    
                    # Define prompt template with the proper format for your LangChain version
                    template = """
                    Use the following pieces of context to answer the question at the end. 
                    If you don't know the answer, just say that you don't know, don't try to make up an answer.
                    
                    {context}
                    
                    Question: {question}
                    
                    Provide a detailed answer and cite the sources from which you obtained the information.
                    """
                    
                    # Create the RetrievalQA chain using the older format
                    from langchain.chains import RetrievalQA
                    from langchain_core.prompts import PromptTemplate
                    
                    PROMPT = PromptTemplate(
                        template=template,
                        input_variables=["context", "question"]
                    )
                    
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=retriever,
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": PROMPT}
                    )
                    
                    st.session_state.vectordb = vectordb
                    st.session_state.qa_chain = qa_chain
                    st.session_state.kb_ready = True
                    st.success("Knowledge base built successfully!")
                    
                except Exception as e:
                    st.error(f"Error building knowledge base: {e}")
                    import traceback
                    st.write(traceback.format_exc())  # Print full traceback for debugging
                    st.session_state.kb_ready = False
    
    # User interface for asking questions
    if st.session_state.kb_ready:
        st.header("Ask your knowledge base")
        user_q = st.text_area("Enter your questions here")
        
        if st.button("Get Response"):
            if user_q:
                try:
                    with st.spinner("Model is working on it..."):
                        # Get response from the model
                        result = st.session_state.qa_chain({"query": user_q})
                         
                        # Display results
                        st.subheader('Your response:')
                        st.write(result["result"])
                        
                        # Extract and display sources
                        sources = []
                        for doc in result["source_documents"]:
                            if doc.metadata["source"] not in sources:
                                sources.append(doc.metadata["source"])
                        
                        st.subheader('Sources:')
                        for source in sources:
                            st.write(source)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.error('Oops, the model response resulted in an error. Please try again with a different question.')
            else:
                st.warning("Please enter a question.")
    
    # Option to reset knowledge base
    if st.session_state.documents:
        if st.button("Reset Knowledge Base"):
            st.session_state.documents = []
            st.session_state.kb_ready = False
            st.session_state.vectordb = None
            st.session_state.qa_chain = None
            st.success("Knowledge base has been reset")
            st.rerun()

# Tab 2: Study Resources
with tabs[1]:
    st.header("📚 Study Materials & YouTube Video Gallery")
    query = st.text_input("🔍 Enter Search Topic:")
    
    if st.button("🔍 Search Now"):
        if not query:
            st.warning("⚠️ Please enter a search query!")
        else:
            with st.spinner("Fetching study materials and videos..."):
                study_links = get_study_material_links(query + " site:edu OR site:arxiv.org OR site:github.com", SERPAPI_KEY)
                youtube_links = get_youtube_links(query, SERPAPI_KEY)

            # Display study materials
            if study_links:
                st.subheader("📄 Study Materials Found")
                for item in study_links:
                    st.markdown(f"📘 [{item['title']}]({item['link']})")

            # Display YouTube video gallery
            if youtube_links:
                st.subheader("🎥 YouTube Video Gallery")

                # Create a 3-column grid layout
                cols = st.columns(3)  

                for index, item in enumerate(youtube_links):
                    video_id = item["link"].split("v=")[-1]
                    with cols[index % 3]:  # Distribute videos into columns
                        st.markdown(f"**{item['title']}**")
                        st.video(f"https://www.youtube.com/embed/{video_id}")

            if not study_links and not youtube_links:
                st.error("❌ No results found!")

# Tab 3: Quiz Generator
with tabs[2]:
    st.header("Interactive Quiz Generator")
    
    # Knowledge base collection
    quiz_tab1, quiz_tab2, quiz_tab3 = st.tabs(["Add Content", "Generate Quiz", "Review"])
    
    # Tab 1: Add Content to Knowledge Base
    with quiz_tab1:
        st.subheader("Add Content to Knowledge Base")
        
        content_source = st.radio(
            "Select content source to add:",
            ["Upload a file", "YouTube video"],
            key="content_source_radio"
        )
        
        # File upload option
        if content_source == "Upload a file":
            uploaded_file = st.file_uploader("Upload a file (PDF, Text, or HTML)", type=["pdf", "txt", "html"], key="quiz_file_uploader")
            
            if uploaded_file is not None:
                # Determine file type and load the text
                file_type = uploaded_file.type
                file_name = uploaded_file.name
                
                file_text = None
                if file_type == "application/pdf":
                    file_text = load_pdf(uploaded_file)
                    source_type = "PDF"
                elif file_type == "text/plain":
                    file_text = load_text(uploaded_file)
                    source_type = "Text"
                elif file_type == "text/html":
                    file_text = load_html(uploaded_file)
                    source_type = "HTML"
                else:
                    st.error("Unsupported file format.")
                
                if file_text:
                    with st.expander("Preview extracted content"):
                        st.write(file_text[:1000] + "..." if len(file_text) > 1000 else file_text)
                    
                    if st.button("Add to Knowledge Base", key="add_to_kb_file"):
                        # Add to knowledge base
                        st.session_state.knowledge_base.append({
                            "title": file_name,
                            "type": source_type,
                            "content": file_text,
                            "length": len(file_text)
                        })
                        st.success(f"Added {file_name} to knowledge base!")
                        # Update combined text
                        st.session_state.combined_text = "\n\n====================\n\n".join(
                            [f"Source: {item['title']}\n{item['content']}" for item in st.session_state.knowledge_base]
                        )
        
        # YouTube URL option
        else:  # content_source == "YouTube video"
            youtube_url = st.text_input("Enter YouTube video URL:", key="quiz_youtube_url")
            
            if youtube_url:
                video_id = extract_youtube_id(youtube_url)
                if video_id:
                    video_title = f"YouTube: {video_id}"
                    
                    if st.button("Fetch YouTube Transcript", key="fetch_youtube_transcript"):
                        with st.spinner("Fetching transcript..."):
                            transcript, error = get_youtube_transcript(youtube_url)
                            if error:
                                st.error(error)
                            else:
                                with st.expander("Preview transcript"):
                                    st.write(transcript[:1000] + "..." if len(transcript) > 1000 else transcript)
                                
                                # Add to knowledge base
                                st.session_state.knowledge_base.append({
                                    "title": video_title,
                                    "type": "YouTube",
                                    "content": transcript,
                                    "length": len(transcript),
                                    "url": youtube_url
                                })
                                st.success(f"Added {video_title} to knowledge base!")
                                # Update combined text
                                st.session_state.combined_text = "\n\n====================\n\n".join(
                                    [f"Source: {item['title']}\n{item['content']}" for item in st.session_state.knowledge_base]
                                )
                else:
                    st.error("Invalid YouTube URL.")
    
    # Tab 2: Generate Quiz
    with quiz_tab2:
        st.subheader("Generate Quiz")
        
        if not st.session_state.knowledge_base:
            st.warning("Your knowledge base is empty. Add content before generating a quiz.")
        else:
            st.write(f"Generating quiz from {len(st.session_state.knowledge_base)} sources in your knowledge base.")
            
            # Source selection for quiz
            source_options = ["All Sources"] + [item["title"] for item in st.session_state.knowledge_base]
            selected_source = st.selectbox("Select source for quiz:", source_options)
            
            # Number of quiz questions input
            num_questions = st.number_input("How many quiz questions would you like to generate?", 
                                           min_value=1, max_value=20, value=5)
            
            # Prepare content for quiz generation
            if selected_source == "All Sources":
                quiz_content = st.session_state.combined_text
            else:
                # Find the selected source in knowledge base
                for item in st.session_state.knowledge_base:
                    if item["title"] == selected_source:
                        quiz_content = item["content"]
                        break
            
            # Generate Quiz Button
            if not st.session_state.quiz_generated and st.button("Generate Quiz"):
                if quiz_content:
                    with st.spinner("Generating quiz..."):
                        quiz_text = generate_quiz(num_questions, quiz_content)
                        st.session_state.parsed_questions = parse_quiz(quiz_text)
                        st.session_state.quiz_generated = True
                        st.session_state.submitted = False
                        st.session_state.user_answers = {}
                        st.session_state.results = None
                        st.rerun()
                else:
                    st.error("No content available for quiz generation.")
            
            # Display quiz if it's been generated
            if st.session_state.quiz_generated:
                if not st.session_state.submitted:
                    st.session_state.user_answers = display_quiz(st.session_state.parsed_questions)
                    
                    if st.button("Submit Quiz"):
                        st.session_state.results = calculate_score(
                            st.session_state.parsed_questions, 
                            st.session_state.user_answers
                        )
                        st.session_state.submitted = True
                        st.rerun()
                
                # Display results if quiz has been submitted
                if st.session_state.submitted and st.session_state.results:
                    st.subheader("Quiz Results")
                    st.write(f"Score: {st.session_state.results['correct']}/{st.session_state.results['total']} " +
                            f"({st.session_state.results['percentage']:.1f}%)")
                    
                    # Display detailed results
                    st.write("### Question Review:")
                    for result in st.session_state.results['details']:
                        question_data = next((q for q in st.session_state.parsed_questions if q['number'] == result['number']), None)
                        
                        if question_data:
                            if result['result'] == 'Correct':
                                st.success(f"**Q{result['number']}**: {question_data['question']}")
                            else:
                                st.error(f"**Q{result['number']}**: {question_data['question']}")
                            
                            st.write(f"Your answer: {result['user_answer']} - " +
                                    f"{question_data['options'][result['user_answer']] if result['user_answer'] in question_data['options'] else 'Not answered'}")
                            st.write(f"Correct answer: {result['correct_answer']} - {question_data['options'][result['correct_answer']]}")
                            st.write("---")
                    
                    # Option to generate new quiz
                    if st.button("Generate New Quiz"):
                        st.session_state.quiz_generated = False
                        st.session_state.submitted = False
                        st.rerun()
    
    # Tab 3: Review Content
    with quiz_tab3:
        st.subheader("Review Knowledge Base")
        
        if not st.session_state.knowledge_base:
            st.info("Your knowledge base is empty. Add content from the 'Add Content' tab.")
        else:
            st.write(f"**{len(st.session_state.knowledge_base)} items in knowledge base:**")
            
            for idx, item in enumerate(st.session_state.knowledge_base):
                with st.expander(f"{item['title']} ({item['type']})"):
                    st.write(f"**Type:** {item['type']}")
                    st.write(f"**Content Length:** {item['length']} characters")
                    if 'url' in item:
                        st.write(f"**URL:** {item['url']}")
                    st.write("**Preview:**")
                    st.write(item['content'][:500] + "..." if len(item['content']) > 500 else item['content'])
                    
                    if st.button(f"Remove from Knowledge Base", key=f"remove_{idx}"):
                        st.session_state.knowledge_base.pop(idx)
                        st.success(f"Removed {item['title']} from knowledge base!")
                        # Update combined text
                        if st.session_state.knowledge_base:
                            st.session_state.combined_text = "\n\n====================\n\n".join(
                                [f"Source: {i['title']}\n{i['content']}" for i in st.session_state.knowledge_base]
                            )
                        else:
                            st.session_state.combined_text = ""
                        st.rerun()
            
            if st.button("Clear Knowledge Base"):
                st.session_state.knowledge_base = []
                st.session_state.combined_text = ""
                st.success("Knowledge base cleared!")
                st.rerun()

# Tab 4: Knowledge Base Status
with tabs[3]:
    st.header("Knowledge Base Overview")
    
    # Display Document Processing KB
    st.subheader("Document Processing Knowledge Base")
    st.write(f"Total document chunks: {len(st.session_state.documents)}")
    
    if st.session_state.documents:
        sources = {}
        for doc in st.session_state.documents:
            source_key = doc["source"].split("_page_")[0] if "_page_" in doc["source"] else doc["source"]
            if source_key in sources:
                sources[source_key] += 1
            else:
                sources[source_key] = 1
        
        st.write("Sources:")
        for source, count in sources.items():
            st.write(f"- {source}: {count} chunks")
    else:
        st.info("Document processing knowledge base is empty.")
    
    # Display Quiz KB
    st.subheader("Quiz Knowledge Base")
    st.write(f"Total items: {len(st.session_state.knowledge_base)}")
    
    if st.session_state.knowledge_base:
        st.write("Sources:")
        for item in st.session_state.knowledge_base:
            st.write(f"- {item['title']} ({item['type']}): {item['length']} characters")
    else:
        st.info("Quiz knowledge base is empty.")
