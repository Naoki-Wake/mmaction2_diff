import cv2
import decord
import numpy as np
import torch
import webcolors
from mmcv import Config, DictAction

from mmaction.apis import inference_recognizer, init_recognizer

import os.path as osp
import json
import os
from pathlib import Path
import shutil
import tempfile
import datetime
import time

import numpy as np
import fastapi
from fastapi import Body, FastAPI, File, Form, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

from pybsc.fastapi_utils import save_upload_file_to_tmp

__version__ = '0.0.1'

app = fastapi.FastAPI()

SERVICE = {
    "name": "action_recognition_mmaction_v1",
    "version": __version__,
    "libraries": {
        "action_recognition_mmaction_v1": __version__
    },
}


@app.get("/")
async def get_root():
    return {
        "service": SERVICE,
        "time": int(datetime.datetime.now().timestamp() * 1000),
    }


#----settings-----
checkpoint = '/mmaction2/pretrained_models/tsm_r50_256h_1x1x8_50e_sthv2_rgb_20210816-032aa4da.pth'
fp_config = '/mmaction2/configs/recognition/arr_tsm/tsm_r50_video_inference_1x1x8_50e_sthv2_rgb.py'
#fp_config = '/mmaction2/configs/recognition/tsm/tsm_r50_1x1x8_50e_sthv2_rgb.py'
label = '/mmaction2/data/sthv2/annotations/something-something-v2-labels.txt'
# assign the desired device.
device = torch.device('cuda:0')
cfg = Config.fromfile(fp_config)
cfg_options = {}
cfg.merge_from_dict(cfg_options)
# build the recognizer from a config file and checkpoint file/url
model = init_recognizer(fp_config, checkpoint, device=device)
#----settings-----
app = FastAPI()

def get_output(video_path,
               out_filename,
               label,
               fps=30,
               font_scale=0.5,
               font_color='white',
               target_resolution=None,
               resize_algorithm='bicubic',
               use_frames=False):
    """Get demo output using ``moviepy``.

    This function will generate video file or gif file from raw video or
    frames, by using ``moviepy``. For more information of some parameters,
    you can refer to: https://github.com/Zulko/moviepy.

    Args:
        video_path (str): The video file path or the rawframes directory path.
            If ``use_frames`` is set to True, it should be rawframes directory
            path. Otherwise, it should be video file path.
        out_filename (str): Output filename for the generated file.
        label (str): Predicted label of the generated file.
        fps (int): Number of picture frames to read per second. Default: 30.
        font_scale (float): Font scale of the label. Default: 0.5.
        font_color (str): Font color of the label. Default: 'white'.
        target_resolution (None | tuple[int | None]): Set to
            (desired_width desired_height) to have resized frames. If either
            dimension is None, the frames are resized by keeping the existing
            aspect ratio. Default: None.
        resize_algorithm (str): Support "bicubic", "bilinear", "neighbor",
            "lanczos", etc. Default: 'bicubic'. For more information,
            see https://ffmpeg.org/ffmpeg-scaler.html
        use_frames: Determine Whether to use rawframes as input. Default:False.
    """

    if video_path.startswith(('http://', 'https://')):
        raise NotImplementedError

    try:
        from moviepy.editor import ImageSequenceClip
    except ImportError:
        raise ImportError('Please install moviepy to enable output file.')

    # Channel Order is BGR
    if use_frames:
        frame_list = sorted(
            [osp.join(video_path, x) for x in os.listdir(video_path)])
        frames = [cv2.imread(x) for x in frame_list]
    else:
        video = decord.VideoReader(video_path)
        frames = [x.asnumpy()[..., ::-1] for x in video]

    if target_resolution:
        w, h = target_resolution
        frame_h, frame_w, _ = frames[0].shape
        if w == -1:
            w = int(h / frame_h * frame_w)
        if h == -1:
            h = int(w / frame_w * frame_h)
        frames = [cv2.resize(f, (w, h)) for f in frames]

    textsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_scale,
                               1)[0]
    textheight = textsize[1]
    padding = 10
    location = (padding, padding + textheight)

    if isinstance(font_color, str):
        font_color = webcolors.name_to_rgb(font_color)[::-1]

    frames = [np.array(frame) for frame in frames]
    for frame in frames:
        cv2.putText(frame, label, location, cv2.FONT_HERSHEY_DUPLEX,
                    font_scale, font_color, 1)

    # RGB order
    frames = [x[..., ::-1] for x in frames]
    video_clips = ImageSequenceClip(frames, fps=fps)

    out_type = osp.splitext(out_filename)[1][1:]
    if out_type == 'gif':
        video_clips.write_gif(out_filename)
    else:
        video_clips.write_videofile(out_filename, remove_temp=True)

def inference(fp_video, output_flag=False, output_layer_names=None):
    returned_feature = None
    # test a single video or rawframes of a single video
    if output_layer_names is not None:
        results, returned_feature = inference_recognizer(
            model, fp_video, outputs=output_layer_names, all_label=True)
    else:
        results = inference_recognizer(model, fp_video, all_label=True)
    #import pdb; pdb.set_trace()
    labels = open(label).readlines()
    labels = [x.strip() for x in labels]
    results = [(labels[k[0]], k[1]) for k in results]

    #print('The top-5 labels with corresponding scores are:')
    #for result in results:
    #    print(f'{result[0]}: ', result[1])

    if output_flag:
        # check resolution of the video file
        cap = cv2.VideoCapture(fp_video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        target_resolution = tuple((width, height))
        cap.release()
        fp_out_filename = fp_video.replace('.mp4', '_out.mp4')
        get_output(
            fp_video,
            fp_out_filename,
            results[0][0],
            fps=fps,
            target_resolution=target_resolution)
        return results, fp_out_filename, returned_feature
    return results, None, returned_feature
@app.post("/uploadfile/")
async def create_upload_file(upload_file: UploadFile = File(None)):
    print(upload_file.filename)
    if not ('.avi' in upload_file.filename) and (not '.mp4' in upload_file.filename):
        return {"detail": "upload data shoulb be avi or mp4"}
    video_file_path = save_upload_file_to_tmp(upload_file)
    # check time to process
    start_time = time.time()
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = 'cls_head'
    results, fp_out_video, returned_feature = inference(str(video_file_path), output_layer_names=output_layer_names)
    returned_feature = returned_feature['cls_head'].tolist()[0]
    end_time = time.time()
    print(f'Time to process: {end_time - start_time}')

    # remove the tmp file
    os.remove(video_file_path)
    if fp_out_video is not None:
        os.remove(fp_out_video)
    
    info_frame = {}
    for i, item in enumerate(results):
        info_frame['top_'+str(i)] = {"label": item[0], "score": str(item[1])}
    if returned_feature is not None:
        info_frame['feature'] = returned_feature
    return JSONResponse(content=jsonable_encoder(info_frame))
