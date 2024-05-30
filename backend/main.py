from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from fastapi.middleware.cors import CORSMiddleware
from services.genai import YoutubeProcessor, GeminiProcessor

class VideoAnalysisRequest(BaseModel):
    youtube_link: HttpUrl
    # advanced settings

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/analyze_video")
def analyze_video(request: VideoAnalysisRequest):
    # # Doing the analysis
    # from langchain_community.document_loaders import YoutubeLoader
    # from langchain.text_splitter import RecursiveCharacterTextSplitter

    processor = YoutubeProcessor()
    result = processor.retrieve_youtube_documents(str(request.youtube_link), verbose=True)

    genai_processor = GeminiProcessor(
        model_name="gemini-pro",
        project="heroic-climber-423918-r1"
    )

    summary = genai_processor.generate_document_summary(
        result, verbose=True
    )
    return {
        "summary": summary
    }

    # loader = YoutubeLoader.from_youtube_url(str(request.youtube_link), add_video_info=True)
    # docs = loader.load()
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # result = text_splitter.split_documents(docs)

    # author = result[0].metadata['author']
    # length = result[0].metadata['length']
    # title = result[0].metadata['title']
    # total_size = len(result)

    # return {
    #     "author": author,
    #     "length": length,
    #     "title": title,
    #     "total_size": total_size
    # }