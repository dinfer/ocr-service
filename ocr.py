import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import os
import json
import copy
from PIL import Image

debug = False
if __name__ == '__main__':
    debug = True

def pt_in_rect(pt, rect, offset = 20):
    (x,y) = pt
    if x >= (rect['left'] - offset) and x <= (rect['left'] + rect['width'] + offset) and y >= (rect['top'] - offset) and y <= (rect['top'] + rect['height'] + offset):
        print('in', x, y, rect)
        return True
    print('out', x, y, rect)
    return False

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    # print('four_point_transform', pts)
    rect = order_points(pts)
    (tl, bl, br, tr) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

def transform_image(image, pattern, regions, offset = 10):
    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(pattern,None)
    kp2, des2 = orb.detectAndCompute(image,None)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    # matches = sorted(matches, key = lambda x:x.distance)
    MAX_COUNT=50
    good = sorted(matches, key = lambda x:x.distance)
    if len(good) > MAX_COUNT:
        good = good[:MAX_COUNT]

    # Draw first 10 matches.
    # print(len(good))
    # img3 = cv.drawMatches(pattern,kp1,image,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # print('matches', matches, type(matches), dir(matches))
    # plt.imshow(img3),plt.show()

    MIN_MATCH_COUNT = 10
    if len(good) < MIN_MATCH_COUNT:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        return None

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    
    h,w,d = pattern.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    # print('dst', dst)
    # image = cv.polylines(image,[np.int32(dst)],True,255,3, cv.LINE_AA)

    flattern = four_point_transform(image, dst.reshape((4,2)))
    oh, ow, _ = pattern.shape
    flattern = cv.resize(flattern, (ow, oh), interpolation=cv.INTER_CUBIC)
    # plt.imshow(flattern),plt.show()

    target = np.zeros(pattern.shape, np.uint8)
    ocrs = regions['ocrs']
    for ocr in ocrs:
        # print('ocr', ocr)
        ox, oy, dx, dy = int(ocr['left'] - offset), int(ocr['top'] - offset), int(ocr['left'] + ocr['width'] + offset), int(ocr['top'] + ocr['height'] + offset)
        target[oy:dy, ox:dx] = flattern[oy:dy, ox:dx]

    if debug:
        # display
        f, axarr = plt.subplots(2,3)
        axarr[0,0].imshow(pattern)
        axarr[0,1].imshow(image)
        marked = cv.drawMatches(pattern,kp1,image,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        axarr[0,2].imshow(marked)
        axarr[1,0].imshow(flattern)
        axarr[1,1].imshow(target)
        plt.show()

    return target, M

# ocr

import tools.infer.utility as utility
import tools.infer.predict_det as predict_det
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_cls as predict_cls
from tools.infer.utility import draw_ocr_box_txt

def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                            [img_crop_width, img_crop_height],
                            [0, img_crop_height]])
    M = cv.getPerspectiveTransform(points, pts_std)
    dst_img = cv.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv.BORDER_REPLICATE,
        flags=cv.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img

class DefaultArgs(object):
    def __init__(self):
        self.det_model_dir="./inference/ch_ppocr_mobile_v1.1_det_infer"  
        self.rec_model_dir="./inference/ch_ppocr_mobile_v1.1_rec_infer/" 
        self.cls_model_dir="./inference/ch_ppocr_mobile_v1.1_cls_infer/" 
        self.use_angle_cls=True 
        self.use_space_char=True

        self.det_max_side_len = 960
        self.det_algorithm = 'DB'
        self.det_db_thresh = 0.3
        self.det_db_unclip_ratio = 1.6
        self.det_db_box_thresh = 0.5
        self.use_pdserving = False
        self.use_zero_copy_run = False
        self.use_gpu = False
        self.enable_mkldnn = False
        self.rec_image_shape = "3, 32, 320"
        self.rec_char_type = 'ch'
        self.rec_batch_num = 6
        self.rec_algorithm = 'CRNN'
        self.max_text_length = 25
        self.rec_char_dict_path = "./ppocr/utils/ppocr_keys_v1.txt"
        self.cls_image_shape = "3, 48, 192"
        self.label_list = ['0', '180']
        self.cls_thresh = 0.9

def run_ocr(image):
    ori_im = image.copy()

    args = DefaultArgs()
    # print('args', args)

    text_detector = predict_det.TextDetector(args)
    text_recognizer = predict_rec.TextRecognizer(args)
    text_classifier = predict_cls.TextClassifier(args)

    dt_boxes, elapse = text_detector(image)
    # print("dt_boxes num : {}, elapse : {}".format(len(dt_boxes), elapse))
    if dt_boxes is None:
        return None, None
    img_crop_list = []
    for bno in range(len(dt_boxes)):
        tmp_box = copy.deepcopy(dt_boxes[bno])
        img_crop = get_rotate_crop_image(ori_im, tmp_box)
        img_crop_list.append(img_crop)
    if args.use_angle_cls:
        img_crop_list, angle_list, elapse = text_classifier(
            img_crop_list)
        # print("cls num  : {}, elapse : {}".format(
        #     len(img_crop_list), elapse))
    rec_res, elapse = text_recognizer(img_crop_list)
    # print("rec_res num  : {}, elapse : {}".format(len(rec_res), elapse))
    # self.print_draw_crop_rec_res(img_crop_list, rec_res)
    return dt_boxes, rec_res

def get_ocr_result(image, pattern, regions):
    masked_image, M = transform_image(image, pattern, regions)
    boxes_all, texts_all = run_ocr(masked_image)
    boxes = []
    texts = []
    text_scores = []
    drop_score = 0.5
    for idx in range(len(texts_all)):
        if texts_all[idx][1] > drop_score:
            boxes.append(boxes_all[idx])
            texts.append(texts_all[idx][0])
            text_scores.append(texts_all[idx][1])
    if debug:
        print('results', boxes, texts, text_scores, M)
        display_image = draw_ocr_box_txt(
            Image.fromarray(masked_image),
            boxes,
            texts,
            text_scores,
            drop_score = drop_score,
        )
        plt.imshow(display_image),plt.show()
    results = []
    for ocr in regions['ocrs']:
        left = ocr['left']
        top = ocr['top']
        width = ocr['width']
        height = ocr['height']
        reigion_aera = np.float32([[left, top], [left + width, top], [left + width, top + height], [left, top + height]]).reshape(-1,1,2)
        reigion_aera = cv.perspectiveTransform(reigion_aera, M)
        item = { 'key': ocr['key'], 'values': [], 'region': ocr, 'aera': reigion_aera.reshape((4,2)).tolist() }
        results.append(item)
        for i in range(len(boxes)):
            ok = True
            for pt in boxes[i]:
                print('place box', texts[i], ocr['key'], pt)
                if not pt_in_rect(pt, ocr):
                    ok = False
                    break
            if ok:
                pts = np.float32(boxes[i]).reshape(-1,1,2)
                box_area = cv.perspectiveTransform(pts, M)
                print('sizes', box_area, box_area.shape, boxes[i], boxes[i].shape)
                item['values'].append({'box': box_area.reshape(boxes[i].shape).tolist(), 'text': texts[i], 'score': text_scores[i]})
    
    return results

if __name__ == '__main__':
    pattern_id = '82abf61a-3587-40d2-836b-f92386afec81'
    image = cv.imread('target.png')
    pattern = cv.imread('./patterns/' + pattern_id + '/pattern.png')
    regions = {}
    # print('sizes', image.shape, pattern.shape)
    with open('./patterns/' + pattern_id + '/regions.json') as f:
        regions = json.load(f)
    print('results', get_ocr_result(image, pattern, regions))
