import imageio
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import cv2
#matplotlib inline
import glob
image_list = []
file1 = open("list_test.txt",'w')
batch_size=64
List_=[]
U=[]
i=0
for filename in glob.glob('test_data/*.png'): #assuming gif
    image = imageio.imread(filename)

    print("Original:")
   # ia.imshow(image)
    cv2.imwrite('test_data/1.png', image)

    ia.seed(4)

    rotate = iaa.Affine(rotate=(-25, 25))
    image_aug = rotate(image=image)

    print("Augmented:")
   # ia.imshow(image_aug)
    cv2.imwrite('test_data/2.png', image_aug)

    images = [image, image, image, image]
    images_aug = rotate(images=images)

    print("Augmented batch:")
    cv2.imwrite('test_data/3.png', images_aug[0])
    cv2.imwrite('test_data/4.png', images_aug[1])
    cv2.imwrite('test_data/5.png', images_aug[2])
    cv2.imwrite('test_data/6.png', images_aug[3])
    #ia.imshow(images_aug[1])
    # for i in range(len(4)):
    #     cv2.imwrite('test_data/'+ str(i+5)+ '.png', image_aug[i])

    #ia.imshow(np.hstack(images_aug))

    seq = iaa.Sequential([
        iaa.Affine(rotate=(-25, 25)),
        iaa.AdditiveGaussianNoise(scale=(10, 60)),
        iaa.Crop(percent=(0, 0.2))
    ])

    images_aug = seq(images=images)

    print("Augmented:")
    cv2.imwrite('test_data/7.png', images_aug[0])
    cv2.imwrite('test_data/8.png', images_aug[1])
    cv2.imwrite('test_data/9.png', images_aug[2])
    cv2.imwrite('test_data/10.png', images_aug[3])
  #  ia.imshow(np.hstack(images_aug))

    seq = iaa.Sequential([
        iaa.Affine(rotate=(-25, 25)),
        iaa.AdditiveGaussianNoise(scale=(30, 90)),
        iaa.Crop(percent=(0, 0.4))
    ], random_order=True)

    images_aug = [seq(image=image) for _ in range(8)]

    print("Augmented:")

 #   ia.imshow(ia.draw_grid(images_aug, cols=4, rows=2))

#    if i<=63:
#        U.append(filename)
#
#        i=i+1
#    else:
#        i=0
#        List_.append(U)
#        U = []
#
#
# file1.close()
# print(List_[0])

