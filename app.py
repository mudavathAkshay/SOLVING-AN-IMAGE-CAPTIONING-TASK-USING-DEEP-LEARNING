import streamlit as st
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
import math
import os
from googletrans import Translator
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "image_caption_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' not found.")
    st.stop()

if not os.path.exists(TOKENIZER_PATH):
    st.error(f"Tokenizer file '{TOKENIZER_PATH}' not found.")
    st.stop()


model = load_model(MODEL_PATH, compile=False)
tokenizer = pickle.load(open(TOKENIZER_PATH, "rb"))

max_length = 51


base_model = InceptionV3(weights="imagenet")
cnn_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)


translator = Translator()
lang_map = {
    "English": "en",
    "Telugu": "te",
    "Hindi": "hi",
    "Tamil": "ta",
    "Bengali": "bn"
}

grammar_tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
grammar_model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")

def correct_caption_with_t5(caption: str) -> str:
    """Correct grammar and fluency of caption using a pretrained T5 model."""
    input_text = "gec: " + caption
    input_ids = grammar_tokenizer.encode(input_text, return_tensors='pt')
    outputs = grammar_model.generate(input_ids, max_length=64, num_beams=5, early_stopping=True)
    return grammar_tokenizer.decode(outputs[0], skip_special_tokens=True)


def extract_feature(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return cnn_model.predict(img, verbose=0)

def generate_caption_beam(photo, model, tokenizer, max_length, beam_index=5):
    start_seq = [tokenizer.word_index['startseq']]
    sequences = [[start_seq, 0.0]]

    while len(sequences[0][0]) < max_length:
        all_candidates = []
        for seq, score in sequences:
            padded = pad_sequences([seq], maxlen=max_length)
            yhat = model.predict([photo, padded], verbose=0)[0]
            top = np.argsort(yhat)[-beam_index:]
            for idx in top:
                candidate = seq + [idx]
                candidate_score = score + math.log(yhat[idx] + 1e-10)
                all_candidates.append([candidate, candidate_score])
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = ordered[:beam_index]

    final_seq = sequences[0][0]
    caption_words = [tokenizer.index_word.get(i, "") for i in final_seq]
    caption = " ".join(caption_words)
    caption = caption.replace("startseq", "").replace("endseq", "").strip()
    return caption

st.title("SOLVING AN IMAGE CAPTIONING TASK USING DEEP LEARNING")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)
with col1:
    language = st.selectbox("Select Language", ["English", "Telugu", "Hindi", "Tamil", "Bengali"])
with col2:
    emotion = st.selectbox("Select Emotion/Tone", ["Normal", "Joke", "Angry", "Sad", "Happy", "Romantic"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Caption"):
        temp_path = "temp.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    
        photo_feature = extract_feature(temp_path)

        raw_caption = generate_caption_beam(photo_feature, model, tokenizer, max_length, beam_index=5)

        caption = correct_caption_with_t5(raw_caption)

        
        if emotion != "Normal":
            caption = f"[{emotion}] {caption}"

        if language != "English":
            target_lang = lang_map[language]
            try:
                caption = translator.translate(caption, src="en", dest=target_lang).text
            except Exception as e:
                st.warning(f"Translation failed: {e}")

        st.success("Generated Caption: " + caption)

st.markdown("---")
st.markdown("**Project done by Group A**")
