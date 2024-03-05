import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="currency-classification-409206-f52933d5ae63.json"

from google.cloud import texttospeech

# Instantiates a client
client = texttospeech.TextToSpeechClient()

def generateAudio(label, lang="nep"):

  labels = { "fifty": "पचास", "five": "पाँच", "five hundred": "पाँच सय", "hundred": "सय", "ten": "दस", "thousand": "हजार", "twenty": "बीस" }

  # Set the text input to be synthesized
  synthesis_input = texttospeech.SynthesisInput(text="Rupees " + label if lang=="eng" else "रुपैयाँ " + labels[label])

  # Build the voice request, select the language code ("en-US")
  # ****** the NAME
  # and the ssml voice gender ("neutral")
  voice = texttospeech.VoiceSelectionParams(
      language_code='en-US' if lang=="eng" else "hi-IN",
      ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)

  # Select the type of audio file you want returned
  audio_config = texttospeech.AudioConfig(
      audio_encoding=texttospeech.AudioEncoding.MP3)

  # Perform the text-to-speech request on the text input with the selected
  # voice parameters and audio file type
  response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

  # The response's audio_content is binary.
  with open('uploads/output.mp3', 'wb') as out:
      # Write the response to the output file.
      out.write(response.audio_content)
      print('Audio content written to file "output.mp3"')
