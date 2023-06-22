from fastapi import FastAPI
from pydantic import BaseModel
import logging

app = FastAPI()

class YouTubeWatchUrl(BaseModel):
    url: str

@app.get("/getsummary")
async def get_summary(youtube_url: YouTubeWatchUrl):
    logging.info(f"Received request for {youtube_url.url}")
    # Your code to generate the summary goes here
    summary = "This is a summary of the video"
    logging.info(f"Returning summary for {youtube_url.url}")
    return summary