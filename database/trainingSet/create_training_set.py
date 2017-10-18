import cv2
import dlib
import numpy as np

import os
import pickle

landmark_predictor_path = '../model_file/shape_predictor_5_face_landmarks.dat'
face_recognition_model_path = '../model_file/dlib_face_recognition_resnet_model_v1.dat'

detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(landmark_predictor_path)
face_recognizer = dlib.face_recognition_model_v1(face_recognition_model_path)
print('finish initialing face detector, landmark predictor and face recognizer')


def compute_faces_representations(img):
    faces = detector(img, 1)
    faces_boxes = []
    landmarks = []
    faces_representations = []
    for i, face in enumerate(faces):
        face_box = [face.left(), face.top(), face.right(), face.bottom()]
        faces_boxes.append(face_box)
        
        landmark = landmark_predictor(img, face)
        # landmark.part(0)    --> point(x, y)    --> point.x, point.y
        # landmark.part(67)   
        landmarks.append(landmark)
        
        face_representations = face_recognizer.compute_face_descriptor(img, landmark)
        face_representations = np.array(face_representations)
        faces_representations.append(face_representations)
    
    return faces_boxes, landmarks, faces_representations
    
    

def build_faces_dataset(faces_path='./trainingSet', out_path=None):
    persons = {}
    with open('./trainingSet/trainingSet.txt','w') as f:
        for dirpath, dirnames, filenames in os.walk(faces_path):
            for subdirname in dirnames:
                person = subdirname
                print("computing {0}'s face vectors... ...".format(person))
                sub_path = os.path.join(dirpath, subdirname)
                for face_img_file in os.listdir(sub_path):
                    if face_img_file.split('.')[-1] not in ['jpg', 'png', 'JPEG'] or face_img_file.startswith('.'):
                        continue
                    face_img_path = os.path.join(sub_path, face_img_file)
                    face_img = cv2.imread(face_img_path, cv2.IMREAD_COLOR)
                    print("computing file: {0} ".format(face_img_file))
                    boxes, landmarks, faces_representations = compute_faces_representations(face_img)
                
                    if person not in persons:
                        persons[person] = []
                        persons[person].append(faces_representations[0])

                    f.write(face_img_path.split('/')[-2]+" ")
                    for i in faces_representations[0]:
                        f.write( str(i) +" ")
                    f.write("\n")

if __name__ == '__main__':
    build_faces_dataset()

    
