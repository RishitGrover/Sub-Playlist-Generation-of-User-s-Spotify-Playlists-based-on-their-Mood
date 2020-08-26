from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.datastructures import ImmutableMultiDict
import time 
import sys
import json
import spotipy
import webbrowser
import spotipy.util as util
import urllib.parse
from json.decoder import JSONDecodeError
from sklearn.metrics import confusion_matrix
import pprint
from pprint import pprint
import tkinter as tk
from tkinter import messagebox
import subprocess
import pandas as pd
import re
import pprint
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
import requests
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
import string
from bs4 import BeautifulSoup
import datetime

from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import random

#APP_ROOT = os.path.dirname(os.path.abspath(__file__))

i = 0
progressBarValue = 0
token = 0
label = ""
emotion_probability = 0
username = ""
musicFactsList = ["In 2016, Mozart Sold More CDs than Beyoncé", "Singing in a Group Boosts Mood", "Listening to Music Enhances Physical Performance", "Finland Has the Most Metal Bands Per Capita",
                  "An Astronaut Released an Album with All Songs Recorded in Space", "The British Navy Uses Britney Spears Songs to Scare Off Pirates", "'Jingle Bells' Was Originally a Thanksgiving Song", 
                  "Music Helps Plants Grow Faster", "None of The Beatles Could Read or Write Music", "The Most Expensive Musical Instrument Sold for $15.9 Million", "Metallica is the First and Only Band to Have Played on All 7 Continents",
                  "Musical Education Leads to Better Test Score", "Listening to Music Utilizes the Entire Brain", "Michael Jackson Tried to Buy Marvel Comics", "The World's Longest Running Performance Will End in the 27th Century",
                  "Music is Physically Good for Your Heart", "A Sea Organ is Built Into the Coast of Croatia", "A Song That Gets Stuck in Your Head is Called an Earworm", "Cows Produce More Milk When Listening to Slow Music", 
                  "Music Helps People with Brain Injuries Recall Personal Memories", "Monaco's Army is Smaller Than Its Military Orchestra", "Prince Played 27 Instruments on His Debut Album"]

app = Flask(__name__)

@app.route("/")
def home():
	return render_template("index.html")

@app.route('/result',methods = ['POST', 'GET'])
def result():
	global username
	if request.method == 'POST':
		result = request.form
		#print("entered if loop")
		results = result.to_dict()
		print(results)
		#print(results['username'])
		username = results['username']
		if(username == 'RishitG'):
			username = 'ka4dq1jrgacp7z4jmb3cj5epz'
		print(username)

		scope = 'user-library-read playlist-read-private playlist-modify-public'
		global token 
		token = util.prompt_for_user_token(username,scope,client_id='2073cab1a4284c29bc38e88e40447393',client_secret='c02850d0f45645f4a80e1c6a1b13495e',redirect_uri='http://localhost:8888/callback/')
		if token:
			try:
				sp = spotipy.Spotify(auth=token)
				playlists = sp.user_playlists(username)
				#trial_real_time_video_final_func()
				return render_template("result.html", result = result)
			except:
				print("Invalid Username")
				quit()
			

@app.route('/updateVal')   
def updateVal():
	global progressBarValue
	strNo = str(progressBarValue)
	print(strNo)
	return strNo


@app.route('/result2',methods = ['POST', 'GET'])
def result2():
    return render_template("result2.html")


@app.route('/runningSystem', methods=['POST'])   
def runningSystem():

    global label
    global emotion_probability

    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()

    detection_model_path = '../haarcascade_files/haarcascade_frontalface_default.xml'
    emotion_model_path = '../models/_mini_XCEPTION.102-0.66.hdf5'
    face_detection = cv2.CascadeClassifier(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)
    
    EMOTIONS = ["angry" , "disgust", "scared", "happy", "sad", "surprised", "neutral"]
    new_emot = ["Angry" , "Angry", "Sad", "Happy", "Sad", "Relax", "Relax"]

    #cv2.namedWindow('your_face')
    #camera = cv2.VideoCapture(0)


    frame = cv2.imread("C:\\Users\\shivam\\Desktop\\Rishit_Trial\\Flask_Application\\opencv_frame_0.png")
    frame = imutils.resize(frame,width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    #print(faces)

    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
            # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
    else:
        print("No face was found!")
        print()
        print()
        print()
        print()
        print()

    new_emot_values = list()
    new_emotions = {"Angry": 0, "Sad": 0, "Happy": 0, "Relax": 0}
    try:
    	for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
        	new_emotions[new_emot[i]] = new_emotions[new_emot[i]] + prob
    except:
        print("No face was found in the clicked images")
        print()
        print()
        print()
        print()
        print()

                

    j = 0
    for i in new_emotions:
        text = "{}: {:.2f}%".format(i, new_emotions[i] * 100)
        w = int(new_emotions[i] * 300)
        
        cv2.rectangle(canvas, (4, (j * 35) + 5), (w, (j * 35) + 35), (0, 0, 255), -1)
        cv2.putText(canvas, text, (10, (j * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        j = j+1
    
    vals = list(new_emotions.values())
    keys = list(new_emotions.keys())
    max_emotion = keys[vals.index(max(vals))]
    label = max_emotion
    emotion_probability = new_emotions[max_emotion]

    cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
    
    #minute = datetime.datetime.now().time().minute

    #cv2.imwrite('C:\\Users\\shivam\\Desktop\\Rishit_Trial\\Flask Application\\static\\images\\User Face_'+str(minute)+'.jpg', frameClone)
    #cv2.imwrite('C:\\Users\\shivam\\Desktop\\Rishit_Trial\\Flask Application\\static\\images\\User Face Probability_'+str(minute)+'.jpg', canvas)

    cv2.imwrite('C:\\Users\\shivam\\Desktop\\Rishit_Trial\\Flask_Application\\static\\User_Face.jpg', frameClone)
    cv2.imwrite('C:\\Users\\shivam\\Desktop\\Rishit_Trial\\Flask_Application\\static\\User_Face_Probability.jpg', canvas)

    cv2.imshow('User Face', frameClone)
    cv2.imshow('User Face Probability', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #time.sleep(15)
    '''
    cam.release()
    target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    '''

    return jsonify({'success': "Yes"})
    #statusForFace = "Success"
    #return statusForFace
    #return render_template(url_for('result2'))

@app.route('/musicClassification', methods = ['POST', 'GET'])   
def musicClassification():
    return render_template("musicClassification.html")


@app.route('/musicClassificationRunning', methods = ['POST', 'GET'])   
def musicClassificationRunning():
    global label
    global emotion_probability

    user_emotion = label
    user_emotion_probability = emotion_probability
    #def acoustic_Function(user_emotion, user_emotion_probability):  


    plt.style.use('ggplot') # make plots look better
    df = pd.read_csv("C:\\Users\\shivam\\Desktop\\Rishit_Trial\\f1combfinal2.csv")
    df_feature_selected = df.drop(['f_name', 'a_name', 'title', 'lyrics', 'spot_id', 'sr_json', 'tr_json', "mood"], axis=1)

    labels = np.asarray(df.mood)

    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(labels)

    labels = le.transform(labels)

    df_features = df_feature_selected.to_dict( orient = 'records' )

    from sklearn.feature_extraction import DictVectorizer
    vec = DictVectorizer()
    features = vec.fit_transform(df_features).toarray()

    from sklearn.model_selection import train_test_split
    features_train, features_test, labels_train, labels_test = train_test_split( features, labels, test_size=0.20, random_state=91)

    # Random Forests Classifier
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier( min_samples_split=4, criterion="entropy" )


    # Support Vector Classifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    clf.fit(features_train, labels_train)

    acc_test = clf.score(features_test, labels_test)
    acc_train = clf.score(features_train, labels_train)
    print ("Test Accuracy:", acc_test)
    print("---------------------------------------------")

    # compute predictions on test features
    pred = clf.predict(features_test)

    #### Figure out what kind of mistakes it makes ####
    from sklearn.metrics import recall_score, precision_score

    precision = precision_score(labels_test, pred, average="weighted")
    recall = recall_score(labels_test, pred, average="weighted")

    print ("Precision:", precision)
    print ("Recall:", recall)


    CLIENT_ID = '2073cab1a4284c29bc38e88e40447393'
    CLIENT_SECRET = 'c02850d0f45645f4a80e1c6a1b13495e'
    REDIRECT_URI = 'http://localhost:8888/callback/'
    SCOPE = 'user-library-read playlist-read-private playlist-modify-public'

    predicted_emotions = list()

    def acousticFunctionInside():

        global username

        #username=rootNameText.get()
        #if(username == 'RishitG'):
        #    username = 'ka4dq1jrgacp7z4jmb3cj5epz'



        # MY USERNAME: ka4dq1jrgacp7z4jmb3cj5epz



        if token:
            sp = spotipy.Spotify(auth=token)

            # -------------------- TO GET USER'S PLAYLISTS ------------------------------------------------------
            print()
            playlists = sp.user_playlists(username)
            
            x = 0
            for playlist in playlists['items']:
                print("Playlist Name: ", playlist['name'])
               

            playlist_id = playlist['id']
            playlists_tracks = sp.user_playlist_tracks(username, playlist_id, fields='items, uri, name, id, total, artists', market='in')

         

            x=0
            for playlistTracks in playlists_tracks['items']:
                print()
                print(playlistTracks['track']['name'])

               
                track_id = playlistTracks['track']['id']
                acou_data = sp.audio_features(track_id)

                tempo = acou_data[0]['tempo']
                energy = acou_data[0]['energy']
                loudness = acou_data[0]['loudness']
                danceability = acou_data[0]['danceability']
                valence = acou_data[0]['valence']
                acousticness = acou_data[0]['acousticness']

               
                tr_json = acou_data[0]
                tr_json_d = json.dumps(tr_json, sort_keys=True, indent=1)

                music = [[tempo, energy, danceability, loudness, valence, acousticness]]

                # Prediction happening here
                class_code = clf.predict(music)
                confi = clf.predict_proba(music)
                
                

                print("-------------------------------------------------------------------------------------------")

                ha = round(confi[0][0] * 100, 4)
                print("happy %->", ha)
                an = round(confi[0][1] * 100, 4)
                print("angry %->", an)
                sa = round(confi[0][2] * 100, 4)
                print("sad %->", sa)
                rel = round(confi[0][3] * 100, 4)
                print("relax %->", rel)
                predicted_emotions.append([ha, an, sa, rel])

         
               

                print("----------------------------------------------------------------------------------------------")
                decoded_class = le.inverse_transform(class_code)

               
                if decoded_class == 0:
                    print("class-> happy ")
                elif decoded_class == 1:
                    print("class-> angry ")
                elif decoded_class == 2:
                    print("class-> sad ")
                elif decoded_class == 3:
                    print("class-> relax ")
                else:
                    print("failure state")

                print()
                global progressBarValue
                progressBarValue = progressBarValue + 1
                


            def Lyrical_classifier_Linked_With_SpotifyID_func(predicted_emotions, token, username, user_emotion, user_emotion_probability):    
                
                predicted_emotions_lyrics = list()


                df = pd.read_csv("C:\\Users\\shivam\\Desktop\\Rishit_Trial\\f1combfinal2.csv")
                stemmer = SnowballStemmer('english')
                words = stopwords.words("english")

                # Cleaning the lyrics waala column
                df['cleaned'] = df['lyrics'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
                print()

                X_train, X_test, y_train, y_test = train_test_split(df['cleaned'], df.mood, test_size=0.2 ,random_state=99)

                # Sequentially apply a list of transforms and a final estimator.
                pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                                     ('chi',  SelectKBest(chi2, k=10000)),
                                     ('clf', LogisticRegression())])

                model = pipeline.fit(X_train, y_train)

                vectorizer = model.named_steps['vect']
                chi = model.named_steps['chi']
                clf = model.named_steps['clf']

                feature_names = vectorizer.get_feature_names()
                feature_names = [feature_names[i] for i in chi.get_support(indices=True)]
                feature_names = np.asarray(feature_names)


                target_names = ['0', '1', '2', '3']
                
                pred = model.predict(X_test)
                print("Testing accuracy score: " + str(model.score(X_test, y_test)))

                # This function gets called at the end of this file
                def predict_song(model):

                    # Lyrics classification happense now
                    global username
                    if token:
                        # Create a Spotify() instance with our token
                        sp = spotipy.Spotify(auth=token)
                        playlists = sp.user_playlists(username)

                        x = 0
                        for playlist in playlists['items']:
                            print("Playlist Name: ", playlist['name'])


                        # -------------------- TO GET SONGS FROM USERS PLAYLIST ------------------------------------------------------

                        print()
                        

                        playlist_id = playlist['id']
                        playlists_tracks = sp.user_playlist_tracks(username, playlist_id, fields='items, uri, name, id, total, artists', market='in')

                        songsNameDict = dict()
                        songsNameDictWithoutTrackID = dict()

                        x=1
                        i=0
                        for playlistTracks in playlists_tracks['items']:
                            print()
                            # Removing the '(' part, like '(feat Drake)' etc. Because of this, previously, songs weren't being found on genius.com
                            song_name_test = playlistTracks['track']['name'].split("(")[0]

                            # Removing all the '!', ',', ''', '?', '.' and '-' from the songs so that they can be used in the url of genius.com  
                            song_name = song_name_test.translate(str.maketrans('', '', string.punctuation)).strip()
                            print(song_name)

                            artist = playlistTracks['track']['artists'][0]['name']
                            print(artist)

                            
                            # ----------------- Extracting lyrics from www.genius.com here. Provides more songs ka lyrics that 'Pylyrics' was providing -------------------------------
                            # create a valid url for web scrapping using song name and artist
                            song_url = '{}-{}-lyrics'.format(str(artist).strip().replace(' ', '-'),
                                                             str(song_name).strip().replace(' ', '-'))

                            #print('\nSong: {}\nArtist: {}'.format(song_name, artist))

                            # New request using song_url created before
                            request = requests.get("https://genius.com/{}".format(song_url))

                            # Verify status_code of request
                            if request.status_code == 200:
                                # BeautifulSoup library return an html code
                                html_code = BeautifulSoup(request.text, features="html.parser")
                                # Extract lyrics from beautifulsoup response using the correct prefix {"class": "lyrics"}
                                try:
                                    lyrics = html_code.find("div", {"class": "lyrics"}).get_text()
                                except:
                                    predicted_emotions_lyrics.append([0, 0, 0, 0])
                                    print("Sorry, the lyrics of this particular song could not be found")
                                    global progressBarValue
                                    progressBarValue = progressBarValue + 1
                                    continue

                                # Replacing newline with space
                                Cleaned_lyrics = lyrics.replace('\n', ' ')

                                # Genius.com used to mention stuff like '[Verse 1]' and '[Chorus]' so the below regular expression is being used to delete such stuff
                                inly = re.sub("[\(\[].*?[\)\]]", "", Cleaned_lyrics)
                                #print(inly)

                                # Prediction is taking place here
                                decoded_class=model.predict([inly])
                                confi=model.predict_proba([inly])

                                

                                print("-------------------------------------------------------------------------------------------")
                                ha=round(confi[0][0]*100, 4)
                                print("happy %->", ha)
                                an=round(confi[0][1]*100, 4)
                                print("angry %->", an)
                                sa=round(confi[0][2]*100, 4)
                                print("sad %->", sa)
                                rel=round(confi[0][3]*100, 4)
                                print("relax %->", rel)
                                predicted_emotions_lyrics.append([ha, an, sa, rel])
                               

                                print("--------------------------------------------------------------------------------------------")
                                if str(decoded_class) == "[0]":
                                    print("class-> happy ")
                                elif str(decoded_class) == "[1]":
                                    print("class-> angry ")
                                elif str(decoded_class) == "[2]":
                                    print("class-> sad ")
                                elif str(decoded_class) == "[3]":
                                    print("class-> relax ")
                                else:
                                    print("failure state")
                                print()

                                progressBarValue = progressBarValue + 1
                                
                            else:
                                # The 'else' condition will be triggered when the lyrics of a song aren't found on genius.com
                                predicted_emotions_lyrics.append([0, 0, 0, 0])
                                print("Sorry, the lyrics of this particular song could not be found")
                                progressBarValue = progressBarValue + 1
                            # ----------------- Extracting lyrics from www.genius.com here. Provides more songs ka lyrics that 'Pylyrics' was providing -------------------------------

                        
                        #print(predicted_emotions_lyrics)
                        #print(songsNameDict)

                        avgList = list()
                        onlyAcousticOnes = list()


                        # ----------------- Finding the average of the acoustic and the lyrics part of a song for each emotion -------------------------------
                        for i in range(len(predicted_emotions_lyrics)):
                            #print(i)
                            avgL = list()
                            for j in range(4):
                                if(predicted_emotions_lyrics[i][j] == 0):
                                    avg = predicted_emotions[i][j]
                                    avgL.append(avg)
                                    if(j==0):
                                        onlyAcousticOnes.append(i)
                                else:
                                    avg = (0.60*predicted_emotions[i][j]+0.40*predicted_emotions_lyrics[i][j])
                                    avgL.append(avg)
                            avgList.append(avgL)
                            print()
                        if(len(onlyAcousticOnes) != 0):
                            print("Song no(s)",onlyAcousticOnes,"lyrics' was/were not found. For the mentioned song no(s), Avg score contains prediction based on acousitc features only.\n")
                        # ----------------- Finding the average of the acoustic and the lyrics part of a song -------------------------------


                        print()

                        listOfEmotions = ['Happy', 'Angry', 'Sad', "Relax"]

                        songNum = 0


                        # ---------- Combining the name and artist of a song and the emotion that got the highest prediction % to give the format - 'ID: Song Name - Artist' ------------------
                        for playlistTracks in playlists_tracks['items']:
                            song_name = playlistTracks['track']['name']
                            artist = playlistTracks['track']['artists'][0]['name']
                            track_id = playlistTracks['track']['id']
                            emotion = ""
                            val = 0
                            for j in range(len(listOfEmotions)):
                                if(val<avgList[songNum][j]):
                                    val = avgList[songNum][j]
                                    emotion = listOfEmotions[j]
                            songsNameDict[track_id+": "+song_name+" - "+artist] = emotion
                            songsNameDictWithoutTrackID[song_name+" - "+artist] = emotion
                            songNum = songNum + 1

                        #print("Final Results (after Lyrical and Acousitic Averged)")
                        #pprint.pprint(songsNameDict)
                        
                        # ---------- Combining the name and artist of a song and the emotion that got the highest prediction % to give the format - 'ID: Song Name - Artist' ------------------

                        

                        # ---------- Segregating the songs into groups based on emotion. For ex. all songs that are predicted as 'Happy' are grouped ------------------
                        emotionsDict = {'Happy': [], 'Angry': [], 'Sad': [], 'Relax': []}
                        for i in songsNameDict:
                            #emotionsDict[songsNameDict[i]].append(i.split(": ")[1])
                            emotionsDict[songsNameDict[i]].append(i)
                        
                        #pprint.pprint(emotionsDict)

                        print()
                        print()
                        print()
                        print()
                        print()
                        print("The suggested songs are")
                        print(emotionsDict[user_emotion])
                        # Can be an empty list also

                        # ['50kpGaPAhYJ3sGmk6vplg0: Love Yourself - Justin Bieber', '2CX2fjnKQYwiLf7kPwNZne: The Scientist - Coldplay', 
                        # '4kLLWz7srcuLKA7Et40PQR: I Gotta Feeling - Black Eyed Peas', '7qEHsqek33rTcFNT9PFqLf: Someone You Loved - Lewis Capaldi']
                        print()
                        print("Because you are", user_emotion, "with a probability of", user_emotion_probability)
                        print()
                        print()
                        # ---------- Segregating the songs into groups based on emotion. For ex. all songs that are predicted as 'Happy' are grouped ------------------



                        flagHappy = 0
                        flagSad = 0
                        flagAngry = 0
                        flagRelax = 0

                        
                        # This section verifies whether playlists for the 4 emotions have been previously created for a playlist. Currently, this system labels any playlist that
                        # the user wants to divide into the format 'Happy Songs' (in general Emotion space 'Songs'). Check if you want to change this into 
                        # 'Playlist Name' Space Emotion spcae 'Songs'. Anyways, if the 4 playists have already been created, it doesn't recreate 4 more playlists everytime this program
                        # is run. But it also can't delete the playlist cause we couldn't find how to delete a playlist completely. So what the section below does is that it just deletes 
                        # every track present in those playlist that have already been made and places the new predicted tracks in their respective playlists
                        playlists = sp.user_playlists(username)
                        for playlist in playlists['items']:

                            # This condition should not be triggered when 'English Songs' is encountered.
                            if(playlist['name']=='Happy Songs' or playlist['name']=='Sad Songs' or playlist['name']=='Angry Songs' or playlist['name']=='Relax Songs'):

                                
                                if(playlist['name']=='Happy Songs'):
                                    # This flag is needed cause there are times your playlist has no happy songs at all. At such a time, there could be all other emotions
                                    # ka playlist and not a 'Happy Songs' playlist. This flag is later used to make sure that we just make a 'Happy' waala playlist
                                    flagHappy = 1


                                    # To be able to remove all the tracks from a playlist, we need a list of all the track ids in that playlist. 
                                    playlist_id_Happy = playlist['id']
                                    playlists_tracks = sp.user_playlist_tracks(username, playlist_id_Happy, fields='items, uri, name, id, total, artists', market='in')

                                    listOfTrackIdsHappy = list()

                                    # This for loop appends all the track ids of the tracks in the playlist to a list
                                    for playlistTracks in playlists_tracks['items']:
                                        #print(playlistTracks['track']['id'])
                                        track_id = playlistTracks['track']['id']
                                        listOfTrackIdsHappy.append(track_id)

                                    # The list of trackids is used here to remove all those tracks. We are then left with an empty playlist in which we later insert
                                    # the tracks which are newly predicted
                                    results = sp.user_playlist_remove_all_occurrences_of_tracks(username, playlist_id_Happy, listOfTrackIdsHappy)
                                    

                                elif(playlist['name']=='Sad Songs'):
                                    # This flag is needed cause there are times your playlist has no sad songs at all. At such a time, there could be all other emotions
                                    # ka playlist and not a 'Sad Songs' playlist. This flag is later used to make sure that we just make a 'Sad' waala playlist
                                    flagSad = 1

                                    # To be able to remove all the tracks from a playlist, we need a list of all the track ids in that playlist. 
                                    playlist_id_Sad = playlist['id']
                                    playlists_tracks = sp.user_playlist_tracks(username, playlist_id_Sad, fields='items, uri, name, id, total, artists', market='in')

                                    listOfTrackIdsSad = list()

                                    # This for loop appends all the track ids of the tracks in the playlist to a list
                                    for playlistTracks in playlists_tracks['items']:
                                        #print(playlistTracks['track']['id'])
                                        track_id = playlistTracks['track']['id']
                                        listOfTrackIdsSad.append(track_id)

                                    results = sp.user_playlist_remove_all_occurrences_of_tracks(username, playlist_id_Sad, listOfTrackIdsSad)
                                    

                                elif(playlist['name']=='Relax Songs'):
                                    # This flag is needed cause there are times your playlist has no Relax songs at all. At such a time, there could be all other emotions
                                    # ka playlist and not a 'Relax Songs' playlist. This flag is later used to make sure that we just make a 'Relax' waala playlist
                                    flagRelax = 1

                                    # To be able to remove all the tracks from a playlist, we need a list of all the track ids in that playlist. 
                                    playlist_id_Relax = playlist['id']
                                    playlists_tracks = sp.user_playlist_tracks(username, playlist_id_Relax, fields='items, uri, name, id, total, artists', market='in')

                                    listOfTrackIdsRelax = list()

                                    # This for loop appends all the track ids of the tracks in the playlist to a list
                                    for playlistTracks in playlists_tracks['items']:
                                        track_id = playlistTracks['track']['id']
                                        listOfTrackIdsRelax.append(track_id)

                                    results = sp.user_playlist_remove_all_occurrences_of_tracks(username, playlist_id_Relax, listOfTrackIdsRelax)
                                    

                                elif(playlist['name']=='Angry Songs'):
                                    # This flag is needed cause there are times your playlist has no Angry songs at all. At such a time, there could be all other emotions
                                    # ka playlist and not a 'Angry Songs' playlist. This flag is later used to make sure that we just make a 'Angry' waala playlist
                                    flagAngry = 1

                                    # To be able to remove all the tracks from a playlist, we need a list of all the track ids in that playlist. 
                                    playlist_id_Angry = playlist['id']
                                    playlists_tracks = sp.user_playlist_tracks(username, playlist_id_Angry, fields='items, uri, name, id, total, artists', market='in')

                                    listOfTrackIdsAngry = list()

                                    # This for loop appends all the track ids of the tracks in the playlist to a list
                                    for playlistTracks in playlists_tracks['items']:
                                        track_id = playlistTracks['track']['id']
                                        listOfTrackIdsAngry.append(track_id)

                                    results = sp.user_playlist_remove_all_occurrences_of_tracks(username, playlist_id_Angry, listOfTrackIdsAngry)
                            
                            
                    

                        # SOLVED - If no ‘Sad Songs’ ka playlist exists from before and your playlist has no songs that are predicted as ‘Sad’, 
                        # the program will still create a ‘Sad Songs’ playlist

                        # ----- THIS SECTION CREATES THE EMOTION PLAYLIST IF AN OLD ONE DOESN'T EXIST (FROM A PREVIOUS PREDICTION) BUT ONLY IF A SONG HAS BEEN PREDICTED WITH THAT EMOTION ---
                        # ----- FOR EX. IF A SAD SONGS PLAYLIST DIDN'T EXIST BEFORE AAANNDDD A SONG HAS BEEN PREDICTED AS A SAD SONG THIS TIME, A 'Sad Songs' PLAYLIST WILL BE CREATED -------
                        if(flagHappy == 0 and emotionsDict["Happy"]):
                            playlist_name = "Happy Songs"
                            new_playlists = sp.user_playlist_create(username, playlist_name)
                            #pprint.pprint(new_playlists)
                            playlist_id_Happy = new_playlists['id']

                        if(flagSad == 0 and emotionsDict["Sad"]):
                        
                            playlist_name = "Sad Songs"
                            new_playlists = sp.user_playlist_create(username, playlist_name)
                            #pprint.pprint(new_playlists)
                            playlist_id_Sad = new_playlists['id']

                        if(flagRelax == 0 and emotionsDict["Relax"]):
                        
                            playlist_name = "Relax Songs"
                            new_playlists = sp.user_playlist_create(username, playlist_name)
                            #pprint.pprint(new_playlists)
                            playlist_id_Relax = new_playlists['id']

                        if(flagAngry == 0 and emotionsDict["Angry"]):

                            playlist_name = "Angry Songs"
                            new_playlists = sp.user_playlist_create(username, playlist_name)
                            #pprint.pprint(new_playlists)
                            playlist_id_Angry = new_playlists['id']
                        # ----- THIS SECTION CREATES THE EMOTION PLAYLIST IF AN OLD ONE DOESN'T EXIST (FROM A PREVIOUS PREDICTION) BUT ONLY IF A SONG HAS BEEN PREDICTED WITH THAT EMOTION ---
                        # ----- FOR EX. IF A SAD SONGS PLAYLIST DIDN'T EXIST BEFORE AAANNDDD A SONG HAS BEEN PREDICTED AS A SAD SONG THIS TIME, A 'Sad Songs' PLAYLIST WILL BE CREATED -------
                        #songsSubDividedBasedOnEmotions = {'Angry': ['2tpWsVSb9UEmDRxAl1zhX1: Counting Stars - OneRepublic', '0pqnGHJpmpxLKifKRmU6WP: Believer - Imagine Dragons'], etc.....
                        




                        # ------------------ THIS SECTION ADDS THE SONGS OF A PARTICULAR EMOTION TO THE PLAYLIST THAT PERTAINS TO THAT EMOTION ---------------------------
                        if(emotionsDict["Angry"]):
                            if(user_emotion=="Angry"):
                                user_emotion_playlist_id = playlist_id_Angry
                            track_ids = list()
                            for i in emotionsDict["Angry"]:
                                track_ids.append(i.split(":")[0])
                            results = sp.user_playlist_add_tracks(username, playlist_id_Angry, track_ids)


                        if(emotionsDict["Happy"]):
                            if(user_emotion=="Happy"):
                                user_emotion_playlist_id = playlist_id_Happy
                            track_ids = list()
                            for i in emotionsDict["Happy"]:
                                track_ids.append(i.split(":")[0])
                            results = sp.user_playlist_add_tracks(username, playlist_id_Happy, track_ids)


                        if(emotionsDict["Sad"]):
                            if(user_emotion=="Sad"):
                                user_emotion_playlist_id = playlist_id_Sad
                            track_ids = list()
                            for i in emotionsDict["Sad"]:
                                track_ids.append(i.split(":")[0])
                            results = sp.user_playlist_add_tracks(username, playlist_id_Sad, track_ids)


                        if(emotionsDict["Relax"]):
                            if(user_emotion=="Relax"):
                                user_emotion_playlist_id = playlist_id_Relax
                            track_ids = list()
                            for i in emotionsDict["Relax"]:
                                track_ids.append(i.split(":")[0])
                            results = sp.user_playlist_add_tracks(username, playlist_id_Relax, track_ids)
                        # ------------------ THIS SECTION ADDS THE SONGS OF A PARTICULAR EMOTION TO THE PLAYLIST THAT PERTAINS TO THAT EMOTION ---------------------------


                        user_suggested_playlist = sp.playlist(user_emotion_playlist_id)
                        #print(user_suggested_playlist['external_urls']['spotify'])
                        webbrowser.open(user_suggested_playlist['external_urls']['spotify'])



                    else:
                        print("Can't get token for", username)

                
                predict_song(model)

                print("program over--------------------------------------------------------------")
                return

            Lyrical_classifier_Linked_With_SpotifyID_func(predicted_emotions, token, username, user_emotion, user_emotion_probability)

        else:
            print("Can't get token for", username)

        # Don't know if this is necessary
        return

    acousticFunctionInside()
    acousticStatus = "Complete"
    return acousticStatus

@app.route('/musicFactsProgram', methods = ['POST', 'GET'])   
def musicFactsProgram():
    global musicFactsList
    randNumber = random.randint(0, len(musicFactsList))
    musicFact = musicFactsList[randNumber]
    return musicFact

    #return 


if __name__ == "__main__":
    app.run(debug=True)
