# PrepKit 🎓

PrepKit is an intelligent study preparation tool that helps users create and interact with educational content. It leverages the power of Google's Generative AI to process various types of content and generate interactive quizzes for effective learning.

## 🌟 Features

- 📚 **Document Processing**
  - PDF document analysis and text extraction
  - HTML content processing
  - Plain text file support
  - Smart document chunking for better comprehension

- 🎥 **YouTube Integration**
  - Automatic transcript extraction
  - Video content analysis
  - Seamless integration with study materials

- 🤖 **AI-Powered Learning**
  - Interactive quiz generation using Google's Generative AI
  - Intelligent content analysis
  - Smart search capabilities for study materials

- 📊 **Learning Assessment**
  - Multiple choice quiz format
  - Immediate feedback and scoring
  - Progress tracking
  - Performance analytics

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- pip (Python package manager)
- Google API key for Generative AI
- SerpAPI key for search functionality

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/prepkit.git
cd prepkit
```

2. Create and activate a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up environment variables
```bash
cp .env.example .env
```
Edit `.env` and add your API keys:
- `GOOGLE_API_KEY`: Your Google API key for Generative AI
- `SERPAPI_KEY`: Your SerpAPI key for search functionality

### Running the Application

1. Start the Streamlit server
```bash
streamlit run PrepKit.py
```

2. Open your browser and navigate to `http://localhost:8501`

## 💡 Usage Guide

### 1. Content Upload
- Upload PDF documents, HTML files, or text files
- Add YouTube video URLs for transcript analysis
- All content is processed and stored securely

### 2. Quiz Generation
- Select the content you want to be tested on
- Choose the number of questions
- Generate AI-powered quizzes

### 3. Learning Assessment
- Take interactive quizzes
- Get immediate feedback
- Review your performance
- Track your progress

## 🛠️ Development

### Project Structure
```
prepkit/
├── PrepKit.py          # Main application file
├── requirements.txt    # Python dependencies
├── .env.example       # Example environment variables
├── .gitignore         # Git ignore rules
└── LICENSE           # MIT license
```

### Contributing

1. Fork the repository
2. Create a feature branch
```bash
git checkout -b feature/amazing-feature
```
3. Commit your changes
```bash
git commit -m 'Add some amazing feature'
```
4. Push to the branch
```bash
git push origin feature/amazing-feature
```
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Support

If you encounter any issues or have questions:
- Open an issue in the GitHub repository
- Check existing issues for answers
- Contribute to the documentation