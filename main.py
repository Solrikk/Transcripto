import pyaudio
import wave
import whisper

model = whisper.load_model("base")


def record_audio(duration=10, output_file="output.wav"):
  CHUNK = 1024
  FORMAT = pyaudio.paInt16
  CHANNELS = 1
  RATE = 44100

  p = pyaudio.PyAudio()

  stream = p.open(format=FORMAT,
                  channels=CHANNELS,
                  rate=RATE,
                  input=True,
                  frames_per_buffer=CHUNK)

  print("Recording...")

  frames = []

  for _ in range(0, int(RATE / CHUNK * duration)):
    data = stream.read(CHUNK)
    frames.append(data)

  print("Recording finished.")

  stream.stop_stream()
  stream.close()
  p.terminate()

  wf = wave.open(output_file, 'wb')
  wf.setnchannels(CHANNELS)
  wf.setsampwidth(p.get_sample_size(FORMAT))
  wf.setframerate(RATE)
  wf.writeframes(b''.join(frames))
  wf.close()


def transcribe_audio(audio_file):
  result = model.transcribe(audio_file)
  transcript = "\n".join([segment['text'] for segment in result['segments']])
  return transcript


if __name__ == "__main__":
  audio_filename = "output.wav"
  record_audio(duration=10, output_file=audio_filename)
  transcript = transcribe_audio(audio_filename)
  print("Transcript:", transcript)
