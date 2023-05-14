import time

from pytube import YouTube
import os
import subprocess
import pandas as pd
import cv2


def download_vid_yt(youtube_id, save_dir):
    vid_url = f"https://www.youtube.com/watch?v={youtube_id}"
    name = f"{youtube_id}.mp4"
    yt = YouTube(vid_url)
    yt = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    yt.download(output_path=save_dir, filename=name)

    vid_path = os.path.join(save_dir, name)
    return vid_path


def get_data_vid(csv_path, num_row):
    data = {}
    reader = pd.read_csv(csv_path, nrows=num_row)
    data_raw = reader.iloc[:].values.tolist()
    for i in range(len(data_raw)):
        key = data_raw[i][0]
        data[key] = []
    for i in range(len(data_raw)):
        key = data_raw[i][0]
        data[key].append(data_raw[i][1:])

    return data


def video_to_img(youtube_id, video_path, data_info, save_dir):
    # cap = cv2.VideoCapture(video_path)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    track_id = -1
    for info in data_info:
        timestamp = info[0]
        obj_id = info[1]
        obj_name = info[2]
        box_id = info[3]
        status = info[4]
        x_min, x_max, y_min, y_max = info[5], info[6], info[7], info[8]
        if status == "present":
            track_id += 1
            x_center, y_center = (x_min + x_max)/2, (y_min + y_max)/2
            w, h = x_max - x_min, y_max - y_min
            img_name = f"{track_id}_{round(x_center, 3)}_{round(y_center, 3)}_{round(w, 3)}_{round(h, 3)}.jpg"
            obj_dir = f"{save_dir}/{youtube_id}_{box_id}"
            os.makedirs(obj_dir, exist_ok=True)
            img_path = f"{obj_dir}/{img_name}"
            print(img_path)

            h = timestamp//3600000
            m = timestamp//60000 - h*60
            s = timestamp//1000 - h*3600 -m*60
            ms = timestamp % 1000
            command = f"ffmpeg -ss {h}:{m}:{s}.{ms:03} -i {video_path} -vframes 1 -q:v 2 {img_path}"
            subprocess.call(command, shell=True)

    # os.remove(video_path)


if __name__ == "__main__":
    # # get data for train:
    # save_dir = "dataset/train"
    # save_vid_dir = "vid_yt_download/train"
    # data = get_data_vid('yt_bb_detection/youtube_boundingboxes_detection_train.csv', num_row=10000)
    # print(len(data))
    # num_vid = 0
    # for youtube_id in data.keys():
    #     if youtube_id+"_0" in os.listdir(save_dir):
    #         num_vid += 1
    #         print(f"num {num_vid}: {youtube_id} done before!")
    #     else:
    #         try:
    #             vid_path = download_vid_yt(youtube_id, save_vid_dir)
    #         except Exception as e:
    #             num_vid += 1
    #             print(f"num {num_vid}: {youtube_id} {str(e)}!")
    #             continue
    #         data_info = data[youtube_id]
    #         print(data_info)
    #         video_to_img(youtube_id, vid_path, data_info, save_dir)
    #         num_vid += 1
    #         print(f"num {num_vid}: {youtube_id} ok!")

    # get data for validation:
    save_dir = "dataset/val"
    save_vid_dir = "vid_yt_download/val"
    data = get_data_vid('yt_bb_detection/youtube_boundingboxes_detection_validation.csv', num_row=2500)
    print(len(data))
    num_vid = 0
    for youtube_id in data.keys():
        if youtube_id + "_0" in os.listdir(save_dir):
            num_vid += 1
            print(f"num {num_vid}: {youtube_id} done before!")
        else:
            try:
                vid_path = download_vid_yt(youtube_id, save_vid_dir)
            except Exception as e:
                num_vid += 1
                print(f"num {num_vid}: {youtube_id} {str(e)}!")
                continue
            data_info = data[youtube_id]
            print(data_info)
            video_to_img(youtube_id, vid_path, data_info, save_dir)
            num_vid += 1
            print(f"num {num_vid}: {youtube_id} ok!")