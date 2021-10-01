import os
import numpy as np
import cv2
import pickle5 as pickle


# Given 2 bounding boxes, return their IoU
def bb_IoU(bb1,bb2):

    Area1 = abs(bb1[2] - bb1[0]) * abs(bb1[3]-bb1[1])
    Area2 = abs(bb2[2] - bb2[0]) * abs(bb2[3]-bb2[1])

    xA = max(bb1[0],bb2[0])
    yA = max(bb1[1],bb2[1])
    xB = min(bb1[2],bb2[2])
    yB = min(bb1[3],bb2[3])

    intersection = max(0, xB - xA) * max(0, yB - yA)
    IoU = intersection / (Area1 + Area2 - intersection)

    return(IoU)

def Area(bb1, image):
        area1 = abs(bb1[2] - bb1[0]) * abs(bb1[3]-bb1[1])
        return area1/image

def evaluation(image_path, gt_path, predict_path):

    with open("ln_data/yourefit/test_id.txt", "r") as f:
        test_id_list = f.readlines()
    test_id_list = [x.strip('\n') for x in test_id_list]

    TP= dict()
    TP['all'] = np.zeros((3,))
    TP['s'] = np.zeros((3,))
    TP['m'] = np.zeros((3,))
    TP['l'] = np.zeros((3,))

    FP= dict()
    FP['all'] = np.zeros((3,))
    FP['s'] = np.zeros((3,))
    FP['m'] = np.zeros((3,))
    FP['l'] = np.zeros((3,))
    gt_boxes = []
    for ind, pattern in enumerate(test_id_list):
        # print(ind, pattern)
        img = cv2.imread(os.path.join(image_path, pattern+'.jpg'))
        H,W,_ = img.shape

        pickle_name = os.path.join(gt_path, pattern+'.p')
        gt = pickle.load(open( pickle_name, "rb" ))

        ground_truth_box = gt['bbox']
        gt_boxes.append(ground_truth_box)

    # read prediction file (Need to change based on input)
        pred_pickle = os.path.join(predict_path, pattern+'.jpg.p')
        pred = pickle.load(open(pred_pickle, "rb" ))
        predicted_box = pred[0]
    #

        for ind, IoU in enumerate([0.25, 0.5, 0.75]):
            if bb_IoU(predicted_box,ground_truth_box) >= IoU:
                TP['all'][ind] +=1
                if 100*Area(ground_truth_box, H*W) < 0.48:
                    TP['s'][ind] += 1
                else:
                    if 100*Area(ground_truth_box, H*W) < 1.75:
                        TP['m'][ind] += 1
                    else:
                        TP['l'][ind] += 1
            else:
                FP['all'][ind] +=1
                if 100*Area(ground_truth_box, H*W) < 0.48:
                    FP['s'][ind] += 1
                else:
                    if 100*Area(ground_truth_box, H*W) < 1.75:
                        FP['m'][ind] += 1
                    else:
                        FP['l'][ind] += 1

    for ind, IoU in enumerate([0.25, 0.5, 0.75]):
        print('Accuracy =',TP['all'][ind]/(TP['all'][ind]+FP['all'][ind]))
        print('Small Accuracy =',TP['s'][ind]/(TP['s'][ind]+FP['s'][ind]), 'in', TP['s'][ind]+FP['s'][ind], 'samples')
        print('Medium Accuracy =',TP['m'][ind]/(TP['m'][ind]+FP['m'][ind]), 'in', TP['m'][ind]+FP['m'][ind], 'samples')
        print('Large Accuracy =',TP['l'][ind]/(TP['l'][ind]+FP['l'][ind]), 'in', TP['l'][ind]+FP['l'][ind], 'samples')

if __name__ == "__main__":
    image_path= 'ln_data/yourefit/images'
    gt_path= 'ln_data/yourefit/pickle'
    predict_path = 'test/test_final'
    evaluation(image_path, gt_path, predict_path)
