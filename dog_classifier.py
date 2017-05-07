import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
import cv2
from keras.preprocessing import image
from tqdm import tqdm
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D,Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from extract_bottleneck_features import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    ResNet50_model = ResNet50(weights='imagenet')
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


def Resnet50_predict_breed_pred(img_path):
    # extract bottleneck features
    Resnet50_model = Sequential()
    Resnet50_model.add(GlobalAveragePooling2D(input_shape=(1, 1, 2048)))
    Resnet50_model.add(Dense(133, activation='softmax'))

    Resnet50_model.load_weights('saved_models/weights.best.Resnet50.hdf5')
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = Resnet50_model.predict(bottleneck_feature)
    # return predicted vector
    return predicted_vector

def detector(img_path):

    dog_names = ['Affenpinscher', 'Afghan_hound', 'Airedale_terrier', 'Akita', 'Alaskan_malamute',
                'American_eskimo_dog', 'American_foxhound', 'American_staffordshire_terrier',
                'American_water_spaniel', 'Anatolian_shepherd_dog', 'Australian_cattle_dog',
                'Australian_shepherd', 'Australian_terrier', 'Basenji', 'Basset_hound', 'Beagle',
                'Bearded_collie', 'Beauceron', 'Bedlington_terrier', 'Belgian_malinois', 'Belgian_sheepdog',
                'Belgian_tervuren', 'Bernese_mountain_dog', 'Bichon_frise', 'Black_and_tan_coonhound',
                'Black_russian_terrier', 'Bloodhound', 'Bluetick_coonhound', 'Border_collie', 'Border_terrier',
                'Borzoi', 'Boston_terrier', 'Bouvier_des_flandres', 'Boxer', 'Boykin_spaniel', 'Briard',
                'Brittany', 'Brussels_griffon', 'Bull_terrier', 'Bulldog', 'Bullmastiff', 'Cairn_terrier',
                'Canaan_dog', 'Cane_corso', 'Cardigan_welsh_corgi', 'Cavalier_king_charles_spaniel',
                'Chesapeake_bay_retriever', 'Chihuahua', 'Chinese_crested', 'Chinese_shar-pei', 'Chow_chow',
                'Clumber_spaniel', 'Cocker_spaniel', 'Collie', 'Curly-coated_retriever', 'Dachshund', 'Dalmatian',
                'Dandie_dinmont_terrier', 'Doberman_pinscher', 'Dogue_de_bordeaux', 'English_cocker_spaniel',
                'English_setter', 'English_springer_spaniel', 'English_toy_spaniel', 'Entlebucher_mountain_dog',
                'Field_spaniel', 'Finnish_spitz', 'Flat-coated_retriever', 'French_bulldog', 'German_pinscher',
                'German_shepherd_dog', 'German_shorthaired_pointer', 'German_wirehaired_pointer', 'Giant_schnauzer',
                'Glen_of_imaal_terrier', 'Golden_retriever', 'Gordon_setter', 'Great_dane', 'Great_pyrenees',
                'Greater_swiss_mountain_dog', 'Greyhound', 'Havanese', 'Ibizan_hound', 'Icelandic_sheepdog',
                'Irish_red_and_white_setter', 'Irish_setter', 'Irish_terrier', 'Irish_water_spaniel', 'Irish_wolfhound',
                'Italian_greyhound', 'Japanese_chin', 'Keeshond', 'Kerry_blue_terrier', 'Komondor', 'Kuvasz',
                'Labrador_retriever', 'Lakeland_terrier', 'Leonberger', 'Lhasa_apso', 'Lowchen', 'Maltese',
                'Manchester_terrier', 'Mastiff', 'Miniature_schnauzer', 'Neapolitan_mastiff', 'Newfoundland',
                'Norfolk_terrier', 'Norwegian_buhund', 'Norwegian_elkhound', 'Norwegian_lundehund', 'Norwich_terrier',
                'Nova_scotia_duck_tolling_retriever', 'Old_english_sheepdog', 'Otterhound', 'Papillon', 'Parson_russell_terrier',
                'Pekingese', 'Pembroke_welsh_corgi', 'Petit_basset_griffon_vendeen', 'Pharaoh_hound', 'Plott', 'Pointer',
                'Pomeranian', 'Poodle', 'Portuguese_water_dog', 'Saint_bernard', 'Silky_terrier', 'Smooth_fox_terrier',
                'Tibetan_mastiff', 'Welsh_springer_spaniel', 'Wirehaired_pointing_griffon', 'Xoloitzcuintli', 'Yorkshire_terrier']
    text =[]
    print('Processing... Wait a second')
    #Type4(Other objects)
    if(face_detector(img_path)==False and dog_detector(img_path)==False):
        print("Error")
        text_top ="Error"
        text_bottom=""
        text_breed1=""
        text_breed2=""
    else:
        #Type1(Dog)
        if (face_detector(img_path) == False and dog_detector(img_path)==True):
            #print text on the image
            text_top ='Your breed is'
            text_bottom = ''
        #Type2(Dog or human)
        elif(face_detector(img_path) == True and dog_detector(img_path)==True):
            #print text on the image
            text_top='Hello, you are predicted as...'
            text_bottom = ''
        #Type3(Human)
        elif(face_detector(img_path) == True and dog_detector(img_path)==False):
            #print text on the image
            text_top ='Hello, Human!'
            text_bottom = 'You look like a'
        print(text_top)
        print(text_bottom)


        #caluclate the predicted_vector
        predicted_vector = Resnet50_predict_breed_pred(img_path)

        # if the percentage is more than 80%, it will return only one dog breed
        if np.max(predicted_vector) > 0.8:
            text_breed1 = dog_names[np.argmax(predicted_vector)]
            text_breed2 = ""
            print(text_breed1)
            print(text_breed2)
        #Otherwise return two kinds of dog breeds
        else:
            #label is used to store the dog breed
            label = []
            #extract top 2 predicted_vector
            for ind in np.argsort(predicted_vector)[0,-3:]:
                label.append(dog_names[ind])
            #print the probability and the dog breeds
            text_breed1 = '%.3f' % (predicted_vector[0,-1]*100)+ '% of '+ label[0]
            text_breed2 = '%.3f' % (predicted_vector[0,-2]*100) + '% of ' + label[1]
            print(text_breed1)
            print(text_breed2)


    text.append(text_top)
    text.append(text_bottom)
    text.append(text_breed1)
    text.append(text_breed2)

    return text
