from torchvision import transforms
import torch
import cv2
import numpy as np
from metrics import get_similarity, weight_distance
from drawing import draw_keypoints_and_limbs_for_one_person


def compare_two_images(im0: np.ndarray, im1: np.ndarray, thres: float=0.8) -> [list, float, float]:
    """Match poses on two images

    Args:
        im0 (np.ndarray): first image
        im1 (np.ndarray): second image
        thres (float, optional): threshold for keypoints confidence. Defaults to 0.8.

    Returns:
        [list, float, float]: retutns prediction from keypoint detection model, cosine symilarity
        and weighted distance between predicted poses on both images
    """
    trans = transforms.Compose([transforms.ToTensor()])
    tensor_im0 = trans(im0.copy()).cuda()
    tensor_im1 = trans(im1.copy()).cuda()
    model.eval()
    res = model([tensor_im0, tensor_im1])
    score0 = res[0]['scores'][0]
    score1 = res[1]['scores'][0]
    if score0 >= thres and score1 >= thres:
        points0 = res[0]['keypoints'][0].cpu().detach().numpy()[:, :-1]
        points1 = res[1]['keypoints'][0].cpu().detach().numpy()[:, :-1]
        sim = get_similarity(points0, points1)
        conf0 = res[0]['keypoints_scores'][0].to('cpu').detach().numpy()
        wd = weight_distance(points0, points1, conf0)
    else:
        sim = 0, 
        wd = 100
    return res, sim, wd


def compare_two_videos(model: any, source0: str, source1: str, sim_low_thres: float=0.99, 
                     sim_high_thres: float=0.999, wd_low_thres: float=10, 
                     wd_high_thres: float=30, show_video: bool=False) -> None:
    """compare two videos by each frame a save videofile with results

    Args:
        model (any): keypoints prediction vodel
        source0 (str): source of first video
        source1 (str): source of second video
        sim_low_thres (float, optional): similarity lower threshold accept not 
        similar poses. Defaults to 0.99.
        sim_high_thres (float, optional): lower threshold accept highly similar 
        poses. Defaults to 0.999.
        wd_low_thres (float, optional): weighted distance lower threshold accept 
        highly similar poses. Defaults to 10.
        wd_high_thres (float, optional): weighted distance lower threshold accept 
        not similar poses. Defaults to 30.
        show_video (bool, optional): show video wihle processing flag. Defaults to False.
    """
    name0 = source0.split('/')[-1].split('.')[0]
    name1 = source1.split('/')[-1].split('.')[0]
    vcap0 = cv2.VideoCapture(source0)
    size = (int(vcap0.get(3)), int(vcap0.get(4)))
    total_size = (size[0] * 2, size[1])
    vcap1 = cv2.VideoCapture(source1)
    out = cv2.VideoWriter(f'{name0}-{name1}.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 24, total_size)
    ok_flag = True
    model.eval()
    i = 0
    sim_total = 0
    wd_total = 0
    while ok_flag:
        ret0, frame0 = vcap0.read()
        ret1, frame1 = vcap1.read()
        if not ret0 or not ret1:
            print("Frame is empty")
            break
        else:
            frame1 = cv2.resize(frame1, size)
            res, sim, wd = compare_two_images(frame0, frame1)
            sim_text = f"similarity: {round(sim, 6)}"
            wd_text =  f"WD: {round(wd, 2)}"
            points0 = res[0]['keypoints'].detach().to('cpu')[0][:, :2].int().tolist()
            points1 = res[1]['keypoints'].detach().to('cpu')[0][:, :2].int().tolist()
            frame0 = draw_keypoints_and_limbs_for_one_person(frame0, points0)
            frame1 = draw_keypoints_and_limbs_for_one_person(frame1, points1)
            if sim > sim_high_thres:
                sim_color = (0, 255, 0)
            elif sim > sim_low_thres and sim <= sim_high_thres:
                sim_color = (0, 255, 255)
            else:
                sim_color = (0, 0, 255)
            if wd < wd_low_thres:
                wd_color = (0, 255, 0)
            elif wd < wd_high_thres and wd >= wd_low_thres:
                wd_color = (0, 255, 255)
            else:
                wd_color = (0, 0, 255)
            frame0 = cv2.putText(frame0, 'Teacher', (250, 930), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            frame1 = cv2.putText(frame1, sim_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, sim_color, 2)
            frame1 = cv2.putText(frame1, wd_text, (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, wd_color, 2)
            frame1 = cv2.putText(frame1, 'Studient', (250, 930), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            frame = np.concatenate([frame0, frame1], axis=1)
            out.write(frame)
            if show_video:
                cv2.imshow('VIDEO', frame)
            if cv2.waitKey(1) == 27:
                ok_flag = False
            sim_total += sim
            wd_total += wd
        i += 1
    sim_total /= i
    wd_total /= i
    last_frame = np.zeros((total_size[1], total_size[0], 3)).astype('uint8')
    text = f'Your result is: similarity - {round(sim_total, 6)}, WD - {round(wd_total, 2)}'
    last_frame = cv2.putText(last_frame, text, (100, 480), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
    for i in range(30):
        out.write(last_frame)
    out.release()
    if show_video:
        cv2.destroyAllWindows()