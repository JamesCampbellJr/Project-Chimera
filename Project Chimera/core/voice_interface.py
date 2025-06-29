# /core/voice_interface.py

import asyncio
import logging
import sounddevice as sd
import numpy as np
import whisper
import pyttsx3
from scipy.io.wavfile import write
import os

import config

# --- Module-level initializations ---
# TTS Engine
try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', config.TTS_ENGINE_RATE)
except Exception as e:
    logging.error(f"Failed to initialize pyttsx3 TTS engine: {e}. Voice output will be disabled.")
    tts_engine = None

# Whisper Model
try:
    whisper_model = whisper.load_model(config.WHISPER_MODEL)
    logging.info(f"Whisper model '{config.WHISPER_MODEL}' loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load Whisper model: {e}. Voice input will be disabled.")
    whisper_model = None

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
TEMP_AUDIO_FILE = "temp_user_audio.wav"

class VoiceInterface:
    def __init__(self):
        self.hotword_detected = asyncio.Event()
        self.is_listening = False
        self.recording_task = None

    def speak(self, text: str):
        """Converts text to speech and plays it."""
        if not tts_engine:
            logging.warning(f"TTS Engine not available. Cannot speak: {text}")
            print(f"[CHIMERA SPEAKS]: {text}") # Fallback to console
            return
            
        logging.info(f"Speaking: {text}")
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            logging.error(f"Error during TTS processing: {e}")

    async def transcribe(self, audio_file_path: str) -> str:
        """Transcribes audio file to text using Whisper."""
        if not whisper_model:
            logging.error("Whisper model not loaded. Transcription failed.")
            return ""
        
        logging.info("Transcribing audio...")
        try:
            loop = asyncio.get_running_loop()
            # whisper.transcribe is blocking, run in a thread
            result = await loop.run_in_executor(
                None, whisper.transcribe, whisper_model, audio_file_path
            )
            text = result['text'].strip()
            logging.info(f"Transcription result: '{text}'")
            return text
        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            return ""

    async def listen_for_hotword_and_command(self) -> str:
        """
        Listens for the hotword, then records a command and transcribes it.
        This is the main public method for this class.
        """
        print(f"Listening for hotword '{config.HOTWORD}'...")
        await self._listen_for_hotword()
        
        self.speak("Yes?")
        logging.info("Hotword detected. Recording command...")
        
        # Record audio after hotword
        audio_data = await self._record_audio(duration=7) # Record for up to 7 seconds
        
        # Save the recorded audio to a temporary file
        write(TEMP_AUDIO_FILE, SAMPLE_RATE, audio_data)
        
        # Transcribe the audio file
        transcribed_text = await self.transcribe(TEMP_AUDIO_FILE)
        
        # Clean up the temporary file
        if os.path.exists(TEMP_AUDIO_FILE):
            os.remove(TEMP_AUDIO_FILE)
            
        return transcribed_text

    def _audio_callback(self, indata, frames, time, status):
        """This function is called for each audio block from the stream."""
        if not self.is_listening:
            return
        # Simple energy-based silence detection
        volume_norm = np.linalg.norm(indata) * 10
        if volume_norm > self.silence_threshold:
            self.frames.extend(indata)
        else:
            self.silent_frames += 1

        # Check for end of speech
        if self.silent_frames > self.silence_limit:
            self.stop_recording_event.set()

    async def _record_audio(self, duration: int) -> np.ndarray:
        """Records audio from the microphone for a fixed duration or until silence."""
        logging.info("Recording...")
        recorded_frames = []

        def callback(indata, frames, time, status):
            if status:
                logging.warning(status)
            recorded_frames.append(indata.copy())

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback, dtype='float32'):
            await asyncio.sleep(duration)
        
        logging.info("Recording finished.")
        if not recorded_frames:
            return np.array([], dtype=np.float32)
        
        return np.concatenate(recorded_frames, axis=0)


    async def _listen_for_hotword(self):
        """
        Continuously listens in the background for a transcribed hotword.
        A more advanced implementation would use a dedicated hotword model like pvporcupine.
        This version uses a simplified listen->transcribe loop.
        """
        while True:
            # Short recording to check for hotword
            audio_data = await self._record_audio(duration=2.5)
            write(TEMP_AUDIO_FILE, SAMPLE_RATE, audio_data)
            
            # Transcribe the short audio
            text = await self.transcribe(TEMP_AUDIO_FILE)
            
            if config.HOTWORD.lower() in text.lower():
                logging.info("Hotword detected!")
                os.remove(TEMP_AUDIO_FILE)
                return
            await asyncio.sleep(0.1) # Small delay to prevent tight loop