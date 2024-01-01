import base64
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

#image processing libraries
import pytesseract
from PIL import Image,ImageFilter,ImageEnhance
from PIL import ImageOps

#Text processing Libraries
import pickle
import string
from nltk.corpus import stopwords
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Sentiment analysis libraries
import re
import seaborn as sns
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

#Customer conversion
import numpy as np

#recommender system
import import_ipynb


st.set_page_config(page_title = 'Final project',layout='wide') 

st.markdown('<h2 style="text-align: center;color:#800000;">Machine Learning and NLP Projects</h2>', unsafe_allow_html=True)
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
    

set_background('3.png')

with st.sidebar:
    selected = option_menu("Main Menu", ['Image Process','Text Process','Sentiment Analysis','Customer Conversion','Product Recommendation'], 
                icons=['image','file-earmark-text','chat-dots','repeat','diagram-3'], menu_icon="menu-up", default_index=0,orientation ="vertical")
if selected == 'Image Process':
        uploaded_file = st.file_uploader("Upload an Image file...",type=['jpg','png','jpeg'])
        if uploaded_file is not None:
                            image_file = Image.open(uploaded_file) 
                            col1, col2 = st.columns(2) 
                            with col1:
                                st.markdown("<h5 style='text-align: center; color: #800000;'>Image Uploaded </h5>", unsafe_allow_html=True)
                                st.divider()
                                st.image(image_file,width = 320)
                                
                            with col2:
                                st.markdown("<h5 style='text-align: center; color: #800000;'>Image Specifications</h5>", unsafe_allow_html=True)
                                st.divider()
                                st.markdown(f"<h4 style='text-align:left; color: #063F27;'>Mode  : {image_file.mode}</h4>", unsafe_allow_html=True) 
                                st.markdown(f"<h4 style='text-align:left; color: #063F27;'>Format  : {image_file.format}</h4>", unsafe_allow_html=True) 
                                st.markdown(f"<h4 style='text-align:left; color: #063F27;'>Size : {image_file.size}</h4>", unsafe_allow_html=True) 
                                
                            with st.container():
                                   c1,c2 = st.columns(2)
                                   with c1:
                                          st.markdown("<h5 style='text-align: center; color: #800000;'>Black and white</h5>", unsafe_allow_html=True)
                                          st.divider()
                                          grey = image_file.convert("L")
                                          st.image(grey,width=320)
                                   with c2:
                                          st.markdown("<h5 style='text-align: center; color: #800000;'>Rescaling</h5>", unsafe_allow_html=True)
                                          st.divider()
                                          resize = grey.resize((220,220))
                                          st.image(resize)
                                          st.markdown(f"<h5 style='text-align:left; color: #063F27;'>Size : {resize.size}</h5>", unsafe_allow_html=True) 
                            with st.container():
                                   a1,a2 = st.columns(2)
                                   with a1:
                                          st.markdown("<h5 style='text-align: center; color: #800000;'>Blurred Image</h5>", unsafe_allow_html=True)
                                          st.divider()
                                          blur_image = image_file.filter(ImageFilter.GaussianBlur(radius = 4))
                                          st.image(blur_image,width=320)
                                   with a2:
                                          st.markdown("<h5 style='text-align: center; color: #800000;'>Mirror Image</h5>", unsafe_allow_html=True)
                                          st.divider()
                                          mirror = ImageOps.mirror(image_file)
                                          st.image(mirror,width=320)
                            with st.container():
                                   b1,b2 = st.columns(2)
                                   with b1:
                                          st.markdown("<h5 style='text-align: center; color: #800000;'>Negative Image</h5>", unsafe_allow_html=True)
                                          st.divider()
                                          neg_image = ImageOps.invert(image_file)
                                          st.image(neg_image,width=320)
                                   with b2:
                                          st.markdown("<h5 style='text-align: center; color: #800000;'>Image Rotate</h5>", unsafe_allow_html=True)
                                          st.divider()
                                          st.image(image_file.rotate(90),width=320)
                            with st.container():
                                   d1,d2 = st.columns(2)
                                   with d1:
                                          st.markdown("<h5 style='text-align: center; color: #800000;'>Image Enhance</h5>", unsafe_allow_html=True)
                                          st.divider()
                                          bright = ImageEnhance.Brightness(image_file)
                                          bright_1 = bright.enhance(6)
                                          contrast = ImageEnhance.Contrast(image_file)
                                          contrast_1 = contrast.enhance(3)
                                          st.image(contrast_1,width=320)
                                   with d2:
                                          st.markdown("<h5 style='text-align: center; color: #800000;'>Edge Detection</h5>", unsafe_allow_html=True)
                                          st.divider()
                                          st.image(image_file.filter(ImageFilter.FIND_EDGES),width=320)
                            st.markdown("<h5 style='text-align: left; color: #800000;'>Text Extraction</h5>", unsafe_allow_html=True)
                            st.divider()
                            pytesseract.pytesseract.tesseract_cmd = "E:\\Program Files\\Tesseract-OCR\\tesseract.exe"
                            extracted_text = pytesseract.image_to_string(image_file)
                            if extracted_text:
                                    st.markdown(f"<h5 style='text-align: left; color: black;'>{extracted_text}</h5>", unsafe_allow_html=True)
if selected == 'Text Process':  
       txt = st.text_area("Text to analyze")   
       button = st.button('Enter')
       if button and txt:
              #Punctuations
              e1,e2 = st.columns([0.2,0.8])
              with e1:
                     st.markdown("<h5 style='text-align: left; color: black;'>Removing Punctuations :</h5>", unsafe_allow_html=True)
              with e2:
                     clean = [char for char in txt if char not in string.punctuation]
                     clean1 = ''.join(clean)
                     st.markdown(f"<h6 style='text-align: left; color: #000099;'>{clean1} </h6>", unsafe_allow_html=True)
              st.divider()            
              
              #Stopwords removal     
              f1,f2 = st.columns([0.2,0.8])
              with f1:
                     st.markdown("<h5 style='text-align: left; color: black;'>Keywords :</h5>", unsafe_allow_html=True)
              with f2:
                     stop = [word for word in clean1.split() if word.lower() not in stopwords.words('english')]
                     st.markdown(f"<h6 style='text-align: left; color: #000099;'>{stop} </h6>", unsafe_allow_html=True)
              st.divider()
              
              #Cleaned Text
              g1,g2 = st.columns([0.2,0.8])   
              with g1:
                     st.markdown(f"<h5 style='text-align: left; color: black;'>Processed Text  : </h5>", unsafe_allow_html=True)
              with g2:
                     stop1 = '  '.join(stop)
                     st.markdown(f"<h6 style='text-align: left; color: #000099;'>{stop1} </h6>", unsafe_allow_html=True)
              st.divider()
              
              #Sentiment Analysis
              h1,h2 = st.columns([0.2,0.8])
              with h1:
                     st.markdown(f"<h4 style='text-align: left; color: black;'>Sentiment Analysis : </h4>", unsafe_allow_html=True)
                     
              with h2:
                     result= TextBlob(stop1).sentiment.polarity
                     
                     if result > 0 :
                            st.markdown(f"<h3 style='text-align: left; color:purple;'>POSITIVE </h3>", unsafe_allow_html=True)
                     elif result == 0:
                            st.markdown(f"<h3 style='text-align: left; color:purple;'>NEUTRAL </h3>", unsafe_allow_html=True)
                     else:
                            st.markdown(f"<h3 style='text-align: left; color:purple;'>NEGATIVE </h3>", unsafe_allow_html=True)
              st.divider()
              
              #Word cloud
              
              st.markdown(f"<h5 style='text-align: left; color: black;'>WordCloud :</h5>", unsafe_allow_html=True)
              wordcloud = WordCloud(random_state=20,background_color= 'white',colormap='Paired_r').generate(stop1)
              plt.imshow(wordcloud, interpolation= 'bilinear')
              plt.axis('off')
              plt.show()
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.pyplot()
if selected ==  'Sentiment Analysis':
       st.header('Sentiment Analysis of Airlines Reviews',divider='rainbow')
       df = pd.read_csv('cleaned_text.csv')      
       st.dataframe(df[['text','cleaned_text','segments']].head(3),hide_index=True) 
       st.subheader('Visualizations',divider=True)  
       h1,h2 = st.columns(2)
       with h1:
              st.markdown("<h4 style='text-align: left; color: black;'>Polarity vs Subjectivity :</h4>", unsafe_allow_html=True) 
              sns.set_style('whitegrid')
              plot = sns.scatterplot(df, x = 'polarity', y ='subjectivity',hue ='segments',size = 50)
              st.pyplot(plot.get_figure())
       with h2:
              st.markdown("<h4 style='text-align: left; color: black;'>Count Plot :</h4>", unsafe_allow_html=True) 
              count = sns.countplot(df, x = 'segments',palette='viridis')
              st.pyplot(count.get_figure())
       st.markdown("<h4 style='text-align: left; color: black;'>Word Cloud :</h4>", unsafe_allow_html=True)     
       st.image('output.png')
       st.markdown("<h4 style='text-align: left; color: black;'>Sentiment Analysis:</h4>", unsafe_allow_html=True)
       classify = st.text_area('Text to analyse')
       button1 = st.button('Submit')

       if classify and button1:
              tfidf = TfidfVectorizer()
              data_text = df['text']
              data_targ = df['segments']
              train_data = tfidf.fit_transform(data_text)
              text_model = SVC(kernel="linear")
              text_model.fit(train_data,data_targ)
              with open('t_model.pkl','rb') as file:
                            text_mdl = pickle.load(file)
              vect_text = tfidf.transform([classify])
              result = text_mdl.predict(vect_text)[0]

              if result == 'postive':
                     st.markdown("<h3 style='text-align: left; color: black;'>Positive Review!!</h3>", unsafe_allow_html=True)
              elif result == 'negative':
                     st.markdown("<h3 style='text-align: left; color: black;'>Negative Review!!</h3>", unsafe_allow_html=True)
              else:
                     st.markdown("<h3 style='text-align: left; color: black;'>Neutral Review!!</h3>", unsafe_allow_html=True)
if selected == 'Customer Conversion':
       with open('model.pkl','rb') as file:
                     rf_model = pickle.load(file)
       st.header('Customer Conversion Prediction',divider = 'rainbow')
       i1,i2 = st.columns(2)
       with i1:
              revenue = st.text_input('Transaction Revenue',placeholder='0 to 999999999')
              interactions = st.text_input('Number of interactions',placeholder= '0 to 10000')
              session = st.text_input('Session Quality Dimension',placeholder='0 to 100')
       with i2:
              time = st.text_input('Time spent on site',placeholder='0 to 99999')
              count_hit = st.text_input('Count_hit',placeholder='0 to 9999')
              avg = st.text_input('Average session time',placeholder= '0.000 to 9999.9999')
       ip = [[revenue,interactions,session,time,count_hit,avg]]
       button2 = st.button('Predict')
       if button2:
              converted = rf_model.predict(np.array(ip))
              if converted:
                     st.markdown("<h3 style='text-align: left; color: black;'>Yes, The Customer will CONVERT</h3>", unsafe_allow_html=True)
              else:
                     st.markdown("<h3 style='text-align: left; color: black;'>No, Customer will NOT CONVERT</h3>", unsafe_allow_html=True)
if selected == 'Product Recommendation':
                st.header('Product Recommendation System',divider='rainbow')
                import recommender
                option = st.selectbox(
                     'Select an item',
                     ('HOME BUILDING BLOCK WORD',
                     'WHITE METAL LANTERN',
                     'RECIPE BOX WITH METAL HEART',
                     'DOORMAT NEW ENGLAND',
                     'JAM MAKING SET WITH JARS',
                     'HAND WARMER UNION JACK',
                     'TRAY, BREAKFAST IN BED',
                     'BLUE COAT RACK PARIS FASHION',
                     'PETIT TRAY CHIC',
                     'BLACK ORANGE SQUEEZER'),placeholder='Choose an item',index=None)
                if option:
                     recom =  recommender.get_recomm_items(option)
                     recom_list = recom[1:]['Description'].tolist()
                st.subheader("Products recommended:",divider='violet')
                try:
                     for i,item in enumerate(recom_list,1):
                            st.markdown(f"<h6 style='text-align: left; color:black;'>{i}. {recom_list[i]}</h6>", unsafe_allow_html=True)
                except:
                       pass     