from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import zlib, os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

app = FastAPI(title="CrypTalk Backend")

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1,
    device=-1
)

class Message(BaseModel):
    text: str

def compress(text):
    return zlib.compress(text.encode())

def decompress(data):
    return zlib.decompress(data).decode()

def encrypt(data):
    key = AESGCM.generate_key(bit_length=128)
    aes = AESGCM(key)
    nonce = os.urandom(12)
    encrypted = aes.encrypt(nonce, data, None)
    return encrypted, key, nonce

def decrypt(encrypted, key, nonce):
    aes = AESGCM(key)
    return aes.decrypt(nonce, encrypted, None)

@app.get("/")
def root():
    return {"status": "CrypTalk backend is running"}

@app.post("/send")
def send_message(msg: Message):
    emotion = emotion_classifier(msg.text)[0][0]["label"]
    tagged = f"[EMOTION: {emotion.upper()}] {msg.text}"

    compressed = compress(tagged)
    encrypted, key, nonce = encrypt(compressed)

    decrypted = decrypt(encrypted, key, nonce)
    final_text = decompress(decrypted)

    return {
        "emotion": emotion,
        "final_message": final_text
    }
