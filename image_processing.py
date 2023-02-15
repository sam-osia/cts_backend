import numpy as np
import cv2
import matplotlib.pyplot as plt
from operator import itemgetter
import math
import os
from time import time


def calculate_dist(p1, p2):
    dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return dist


def filter_mask(img, minDistCentroid=None, minDistEdge=None, minArea=None):
    # imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgGray = img.copy().astype('uint8')
    imgGray[imgGray > 0] = 255
    corrected_mask = np.ones(img.shape[:2], dtype='uint8') * 255

    cnts, hier = cv2.findContours(imgGray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    blob_features = []
    for cnt in cnts:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            centroid = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        else:
            # get the middle value of the cnt
            # centroid = (int(cnt[0][0][1]), int(cnt[0][0][0]))
            centroid = (0, 0)

        area = cv2.contourArea(cnt)
        blob_features.append((cnt, centroid, area))

    # sort the features by area
    blob_features = sorted(blob_features, key=itemgetter(2))
    # make it descending
    blob_features.reverse()

    # for cnt, centroid, area in blob_features:
    #     print(f'{area}, {centroid}')
    # drop the contour that fits around the entire image (should be the biggest one)
    blob_features = blob_features[1:]
    main_blob = blob_features[0]

    i = 0

    for cnt, centroid, area in blob_features:
        # cv2.circle(img, centroid, 3, (0, 255, 0), 1)
        # cv2.putText(img, str(i), centroid, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        # print(f'object {i}')
        # print('area', area)
        # print('centroid distance', calculate_dist(centroid, main_blob[1]))
        # print('edge distance', abs(cv2.pointPolygonTest(main_blob[0], centroid, True)))
        # print('')
        if minDistCentroid is not None:
            centroid_distance = calculate_dist(centroid, main_blob[1])
            if centroid_distance < minDistCentroid:
                cv2.drawContours(corrected_mask, [cnt], -1, 0, -1)
        if minDistEdge is not None:
            edge_distance = abs(cv2.pointPolygonTest(main_blob[0], centroid, True))
            print(edge_distance)
            if edge_distance < minDistEdge:
                cv2.drawContours(corrected_mask, [cnt], -1, 0, -1)
        if minArea is not None:
            if area > minArea:
                cv2.drawContours(corrected_mask, [cnt], -1, 0, -1)

        i += 1

    cv2.drawContours(corrected_mask, [main_blob[0]], -1, 0, -1)
    # plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.subplot(122), plt.imshow(corrected_mask, cmap='gray')
    # plt.show()

    return corrected_mask


if __name__ == '__main__':
    subject_id = '2022-08-31-i1'
    frame_num = 5
    threshold = 350
    image_parent_dir = f'/hpf/largeprojects/dsingh/cts/data_transfers/uploaded_scans/Android/{subject_id}/top_parsed'
    img = cv2.imread(os.path.join(image_parent_dir, f'depth_{frame_num}.png'), -1)
    img[img > threshold] = 0
    corrected_img = filter_mask(img, minDistCentroid=50)
    plt.subplot(121), plt.imshow(img)
    plt.subplot(122), plt.imshow(corrected_img)
    plt.show()
