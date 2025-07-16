# meeting-summarizer
This is a Python FastAPI implementation to summarize meetings using Gemini API. The program accepts both audio and video recordings of the meeting. Using OpenAI whisper, processes the audio, and outputs an AI generated summary. Alternatively, it also generates insights through text transcriptions of the meeting.

There are two files `main.py` and `index.html` which server as the backend and frontend of the program respectively.

To set up the program, you need the following installed:
1. FastAPI
2. OpenAI whisper (to process the files locally)
along with any other necessary libraries

Create you own GEMINI_API_KEY and set it as an Environment Variable

To run the program, use command `uvicorn main:app --reload` 
once the server is up and running, manually open the index.html file.
