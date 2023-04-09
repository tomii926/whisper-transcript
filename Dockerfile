FROM --platform=amd64 python:3.9
RUN python -m pip install --upgrade pip
RUN pip install pyannote.audio
RUN pip install openai-whisper
RUN apt-get update
RUN apt-get install -y libsndfile1
WORKDIR /app
