from __future__ import print_function, unicode_literals
from facepplib import FacePP, exceptions
import cloudinary.uploader
import cloudinary
import emoji
import cv2
from PIL import Image, ImageEnhance
import os
import glob
import requests
from numba import jit, cuda
face_detection = ""
faceset_initialize = ""
face_search = ""
face_landmarks = ""
dense_facial_landmarks = ""
face_attributes = ""
beauty_score_and_emotion_recognition = ""

# # define face comparing function
cloudinary.config(
  cloud_name = 'dxl1wjmzt',
  api_key = '336661672814477',
  api_secret = 'iZ0EPLsh6no1jXEJlUTeWLnmHls'
)

# cloudinary.uploader.upload("/media/fariborz/d35256bc-623b-42e1-b10d-48071cf615af/crop_training_data/Training_set9/n002622/0091_01.5.jpg", public_id="sample_id")  # upload
#
# cloudinary.uploader.destroy("sample_id")  # delete


def face_comparing(app, Image1, Image2):
    # print()
    # print('-' * 30)
    # print('Comparing Photographs......')
    # print('-' * 30)
    res = requests.head(Image1)
    res_ = requests.head(Image2)
    # image1 = url_img1
    # image2 = url_img2
    if res.ok and res_.ok:
        cmp_ = app.compare.get(image_url1=Image1,
                               image_url2=Image2)
        if len(cmp_.faces2._resources) > 0 and len(cmp_.faces1._resources) > 0:
        # print(cmp_.confidence)
        # Comparing Photos
        # if cmp_.confidence > 70:
        #     print('Both photographs are of same person......')
        # else:
        #     print('Both photographs are of two different persons......')
    # print('Photo1', '=', cmp_.image1)
    # print('Photo2', '=', cmp_.image2)

               return cmp_.confidence
        else:
               return 0.0
    else:
        return 0.0

    # Driver Code


if __name__ == '__main__':

    # api details
    api_key = 'xQLsTmMyqp1L2MIt7M3l0h-cQiy0Dwhl'
    api_secret = 'TyBSGw8NBEP9Tbhv_JbQM18mIlorY6-D'

    try:

        # # create a logo of app by using iteration,
        # # unicode and emoji module-------------
        # for i in range(1, 6):
        #
        #     for j in range(6, -i):
        #         print(" ", end=" ")
        #
        #     for j in range(1, i):
        #         print('\U0001F600', end=" ")
        #     # face_comparing_localphoto,
        #     # face_comparing_websitephoto,
        #     for j in range(i, 0, -1):
        #         print('\U0001F6A3', end=" ")
        #
        #     for j in range(i, 1, -2):
        #         print('\U0001F62B', end=" ")
        #
        #     print()
        #
        # print()
        #
        # # print name of the app--------
        # print("\t\t\t" + "Photo Comparing App\n")
        #
        # for i in range(1, 6):
        #
        #     for j in range(6, -i):
        #         print(" ", end=" ")
        #
        #     for j in range(1, i):
        #         print(emoji.emojize(":princess:"), end=" ")
        #
        #     for j in range(i, 0, -1):
        #         print('\U0001F610', end=" ")
        #
        #     for j in range(i, 1, -2):
        #         print(emoji.emojize(":baby:"), end=" ")
        #
        #     print()
        #
        #     # call api
        app_ = FacePP(api_key=api_key,
                      api_secret=api_secret)
        # funcs = [
        #     face_detection,
        #     # face_comparing_localphoto,
        #     # face_comparing_websitephoto,
        #     faceset_initialize,
        #     face_search,
        #     face_landmarks,
        #     dense_facial_landmarks,
        #     face_attributes,
        #     beauty_score_and_emotion_recognition
        # ]
        counter=1
        for filename in glob.glob('/media/fariborz/d35256bc-623b-42e1-b10d-48071cf615af/crop_training_data/*'):  # assuming gif
           # print(filename)
            head_father, tail_father = os.path.split(filename)
            for filename_i in glob.glob(filename+'/*'):
                #print(filename_i)
                head_, tail_ = os.path.split(filename_i)
                # os.mkdir('JPG95/'+ tail_)
                images_hr ='/media/fariborz/d35256bc-623b-42e1-b10d-48071cf615af/HGI_References/'+tail_+'/'+tail_+'.jpg'
                cloudinary.uploader.upload(images_hr, public_id=tail_)  # upload
                url_img1='https://res.cloudinary.com/dxl1wjmzt/image/upload/'+tail_+'.jpg'
                directory = '/media/fariborz/d35256bc-623b-42e1-b10d-48071cf615af/FacePP_Scores/' + tail_
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    for images in glob.glob(filename_i + '/*.jpg'):
                        head_i, tail_i = os.path.split(images)
                        # S = float(tail_i[-6:-4])
                        # if S!= 0.1 :
                        cloudinary.uploader.upload(images, public_id=tail_i[0:len(tail_i) - 4])  # upload
                        url_img2 = 'https://res.cloudinary.com/dxl1wjmzt/image/upload/' + tail_i
                        Score_ = face_comparing(app_, url_img1, url_img2)
                        myfile = open(directory + '/' + 'Verifier_score.text', 'a')
                        myfile.write(tail_father + '/' + tail_ + '/' + tail_i + ',' + str(Score_) + '\n')
                        myfile.close()
                        cloudinary.uploader.destroy(tail_i[0:len(tail_i) - 4])  # delete
                        counter = counter + 1
                        print(counter)
                cloudinary.uploader.destroy(tail_)  # delete
                Keep_Here = 1
    except exceptions.BaseFacePPError as e:
        print('Error:', e)




