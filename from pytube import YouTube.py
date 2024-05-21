from pytube import YouTube
import whisper

def download_youtube_audio(youtube_url, output_path='audio.mp3'):
    yt = YouTube(youtube_url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_stream.download(filename=output_path)
    print(f"Downloaded audio to {output_path}")

def transcribe_and_translate(audio_path, language='en'):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language=language)
    print("Transcription result:")
    print(result['text'])
    return result['text']


youtube_url = 'https://www.youtube.com/watch?v=GxSqIF48XpI'

# Download the audio from YouTube
download_youtube_audio(youtube_url)

# Transcribe and translate the downloaded audio
transcription_text = transcribe_and_translate('audio.mp3')

print(transcription_text)
