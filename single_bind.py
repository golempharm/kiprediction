import streamlit as st
import pandas as pd
import numpy as np

st.title('GoLem Pharm')
st.write('')

#input box
int_put = st.text_input('Compound in SMILES format:   e.g. COc1ccc(NC(=O)Nc2nc3ccc(Cl)cc3c3nc(nn23)')
int_put2 =  st.text_input('Sequence of protein:   e.g. MLLETQDALYVALELVIAALSVAGNVLVCAAVG...')

if int_put2:
 review_lines = [str(int_put + ' ' + int_put2), str(int_put + ' ' + int_put2)]
 word_index1 = np.load('./word_index_100.npy',allow_pickle='TRUE').item()

 from tensorflow.keras.preprocessing.text import Tokenizer
 from keras.preprocessing.sequence import pad_sequences
 from tensorflow.keras.utils import to_categorical

 MAX_SEQUENCE_LENGTH = 600

 tokenizer = Tokenizer(lower = False, char_level=True)
 tokenizer.word_index = word_index1
 sequences = tokenizer.texts_to_sequences(review_lines)

 review_pad = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, truncating = 'post')

 x_test = review_pad


 from keras.models import load_model
 model2 = load_model('./final_modellog.h5')
 predict_x=model2.predict(x_test)

 st.write('')
 st.header ('Result:')
 st.header ( 'pKi = ' + str(round(float(predict_x[0][0]), 2)))
