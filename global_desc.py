import numpy as np
import cv2
from matplotlib import pyplot as plt
import os 

def load_ds(path, print_moments=False):
    ds = os.listdir(path)
    ds_hu = []

    for img in ds:
        image = cv2.imread(f'./database/{img}', cv2.IMREAD_GRAYSCALE)
        moments = cv2.moments(image)
        hu_moments = cv2.HuMoments(moments).flatten()
        ds_hu.append(hu_moments)

    if print_moments:
        for i in range(len(ds)):
            print(f"Image: {ds[i]}")
            print(f"Moments de Hu: {ds_hu[i]}")
    
    return ds, ds_hu


ds, ds_hu = load_ds('./database')
ds_test = os.listdir('./test')

for img in ds_test:
    test_image = cv2.imread(f'./test/{img}', cv2.IMREAD_GRAYSCALE)
    test_moments = cv2.moments(test_image)
    test_hu_moments = cv2.HuMoments(test_moments).flatten()
    dist = np.sum(np.abs(ds_hu - test_hu_moments), axis=1)
    min_idx = np.argmin(dist)
    min_dist = dist[min_idx]
    
    print(f"Image de test : ./test/{img}")
    print(f"Meilleur match : {ds[min_idx]} avec une distance de {min_dist}")
    
    # Affichage des images de test et du meilleur match
    plt.subplot(121)
    plt.imshow(plt.imread(f'./test/{img}')), plt.title(f'Image : ./test/{img}'), plt.axis('off')
    plt.subplot(122)
    plt.imshow(plt.imread(f'./database/{ds[min_idx]}')), plt.title(f'Meilleur match : ./database/{ds[min_idx]}'), plt.axis('off')
    plt.suptitle(f'Distance : {min_dist}')
    plt.show()
