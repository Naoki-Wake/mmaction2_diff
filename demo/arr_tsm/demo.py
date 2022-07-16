import argparse
import os
import os.path as osp

import torch

from mmaction.apis import inference_recognizer, init_recognizer


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file/url')
    parser.add_argument('video', help='video file/url or rawframes directory')
    parser.add_argument('label', help='label file')
    parser.add_argument(
        '--use-frames',
        default=False,
        action='store_true',
        help='whether to use rawframes as input')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--fps',
        default=30,
        type=int,
        help='specify fps value of the output video when using rawframes to '
        'generate file')
    parser.add_argument(
        '--font-size',
        default=20,
        type=int,
        help='font size of the label test in output video')
    parser.add_argument(
        '--font-color',
        default='white',
        help='font color of the label test in output video')
    parser.add_argument(
        '--target-resolution',
        nargs=2,
        default=None,
        type=int,
        help='Target resolution (w, h) for resizing the frames when using a '
        'video as input. If either dimension is set to -1, the frames are '
        'resized by keeping the existing aspect ratio')
    parser.add_argument(
        '--resize-algorithm',
        default='bicubic',
        help='resize algorithm applied to generate video')
    parser.add_argument(
        '--split-time',
        default=None,
        type=float,
        help='split a video into windows, then apply the recognition, then concatnate the results. Split or chunk size in seconds, for example 10')
    parser.add_argument('--out-filename', default=None, help='output filename')
    args = parser.parse_args()
    return args


def get_output(video_path,
               out_filename,
               label,
               fps=30,
               font_size=20,
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
        font_size (int): Font size of the label. Default: 20.
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
        from moviepy.editor import (CompositeVideoClip, ImageSequenceClip,
                                    TextClip, VideoFileClip)
    except ImportError:
        raise ImportError('Please install moviepy to enable output file.')

    if use_frames:
        frame_list = sorted(
            [osp.join(video_path, x) for x in os.listdir(video_path)])
        video_clips = ImageSequenceClip(frame_list, fps=fps)
    else:
        # revert the order to suit ``VideoFileClip``.
        # (weight, height) -> (height, weight)
        target_resolution = (target_resolution[1], target_resolution[0])
        video_clips = VideoFileClip(
            video_path,
            target_resolution=target_resolution,
            resize_algorithm=resize_algorithm)

    duration_video_clip = video_clips.duration
    #print(duration_video_clip)
    text_clips = TextClip(label, fontsize=font_size, color=font_color, bg_color='white', font='Helvetica-Bold')
    text_clips = (
        text_clips.set_position(
            ('left', 'top'),
            relative=True).set_duration(duration_video_clip))

    #print(video_clips.duration)
    #print(text_clips.duration)
    video_clips = CompositeVideoClip([video_clips, text_clips])
    #print(video_clips.duration)
    out_type = osp.splitext(out_filename)[1][1:]
    if out_type == 'gif':
        video_clips.write_gif(out_filename)
    else:
        video_clips.write_videofile(out_filename, remove_temp=True)


def main():
    args = parse_args()
    # assign the desired device.
    device = torch.device(args.device)
    # build the recognizer from a config file and checkpoint file/url
    model = init_recognizer(
        args.config,
        args.checkpoint,
        device=device,
        use_frames=args.use_frames)

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    # test a single video or rawframes of a single video
    if args.split_time is None:
        if output_layer_names:
            results, returned_feature = inference_recognizer(
                model,
                args.video,
                args.label,
                use_frames=args.use_frames,
                outputs=output_layer_names)
        else:
            results = inference_recognizer(
                model, args.video, args.label, use_frames=args.use_frames)

        print('The top-5 labels with corresponding scores are:')
        for result in results:
            print(f'{result[0]}: ', result[1])

        if args.out_filename is not None:

            if args.target_resolution is not None:
                if args.target_resolution[0] == -1:
                    args.target_resolution[0] = None
                if args.target_resolution[1] == -1:
                    args.target_resolution[1] = None
                args.target_resolution = tuple(args.target_resolution)
            else:
                args.target_resolution = (None, None)
            label_show = ''
            for result in results:
                label_show = label_show + result[0]+ ': {:.2g}'.format(result[1]) + '\n'

            get_output(
                args.video,
                args.out_filename,
                label_show[:-1],
                fps=args.fps,
                font_size=args.font_size,
                font_color=args.font_color,
                target_resolution=args.target_resolution,
                resize_algorithm=args.resize_algorithm,
                use_frames=args.use_frames)

    if args.split_time is not None:
        #https://stackoverflow.com/questions/28884159/using-python-script-to-cut-long-videos-into-chunks-in-ffmpeg
        #https://nico-lab.net/segment_muxer_with_ffmpeg/
        import re
        import math
        length_regexp = 'Duration: (\d{2}):(\d{2}):(\d{2})\.\d+,'
        re_length = re.compile(length_regexp)

        from subprocess import check_call, PIPE, Popen
        import shlex
        import os
        if args.split_time <= 0:
            print("Split length can't be 0")
            raise SystemExit

        p1 = Popen(["ffmpeg", "-i", args.video], stdout=PIPE, stderr=PIPE, universal_newlines=True)
        # get p1.stderr as input
        output = Popen(["grep", 'Duration'], stdin=p1.stderr, stdout=PIPE, universal_newlines=True)
        p1.stdout.close()
        matches = re_length.search(output.stdout.read())
        if matches:
            video_length = int(matches.group(1)) * 3600 + \
                        int(matches.group(2)) * 60 + \
                        int(matches.group(3))
            print("Video length in seconds: {}".format(video_length))
        else:
            print("Can't determine video length.")
            raise SystemExit
        split_count = math.ceil(video_length / args.split_time)
        if split_count == 1:
            print("Video length is less than the target split length.")
            raise SystemExit

        fname = os.path.basename(args.video)
        dirname = os.path.dirname(args.video)
        fname_base, ext = fname.rsplit(".", 1)
        tmp_path = os.path.join(dirname,'tmpdir')
        dummy_filenames = []
        if not os.path.isdir(tmp_path):
            os.makedirs(tmp_path)

        #copied_fname = "{}.{}".format(os.path.join(tmp_path,fname_base), ext)
        #cmd = "ffmpeg -i {} -vf scale=640:360  -y {}".\
        #    format(args.video, copied_fname)
        #check_call(shlex.split(cmd), universal_newlines=True)
        #print(split_count)
        '''for n in range(split_count):
            split_start = args.split_time * n
            cmd = "ffmpeg -i {} -vcodec copy  -strict -2 -ss {} -t {} -y {}-{}.{}".\
                format(args.video, split_start, args.split_time, os.path.join(tmp_path,fname_base), n, ext)
            dummy_filenames.append("{}-{}.{}".format(os.path.join(tmp_path,fname_base), n, ext))
            print("About to run: {}".format(cmd))
            check_call(shlex.split(cmd), universal_newlines=True)
            tmp_fname = "{}-{}.{}".format(os.path.join(tmp_path,fname_base), n, ext)'''


        cmd = "ffmpeg -i {} -map 0 -c copy -flags +global_header -f segment -segment_time {} -y -segment_list {} -segment_format_options movflags=+faststart -reset_timestamps 1 {}-%02d.{}".\
            format(args.video, args.split_time, os.path.join(tmp_path,'list_gen.txt'), os.path.join(tmp_path,fname_base), ext)
        print("About to run: {}".format(cmd))
        check_call(shlex.split(cmd), universal_newlines=True)
        #    cmd = "ffmpeg -i {} -vf scale=640:360 -y {}".\
        #        format(tmp_fname,tmp_fname)
        #    print("About to run: {}".format(cmd))
        #    check_call(shlex.split(cmd), universal_newlines=True)
        
        with open(os.path.join(tmp_path,'list_gen.txt'), 'r') as tmp_file:
            lines = tmp_file.readlines()
        for line in lines:
            dummy_filenames.append(os.path.join(tmp_path,line.replace('\n','')))
        #print(dummy_filenames)

        import pandas as pd
        
        with open(args.label, 'r') as f:
            label = [line.strip() for line in f]
        list_df = pd.DataFrame(columns=label, index = range(len(dummy_filenames)))
        #index_time = 0
        for i, video_block in enumerate(dummy_filenames):
            video_block_out = os.path.join(os.path.dirname(video_block), 'out_' + os.path.basename(video_block))
            output_layer_names =('cls_head', )
            if output_layer_names:
                results, returned_feature = inference_recognizer(
                    model,
                    video_block,
                    args.label,
                    use_frames=args.use_frames,
                    outputs=output_layer_names)
                ret_feature = returned_feature['cls_head'].cpu().detach().numpy()
                #list_df = list_df.append( ret_feature, ignore_index=True )
                #list_df = list_df.append(pd.DataFrame(ret_feature, columns=label, index= index_time)
                '''from prettytable import PrettyTable

                def count_parameters(model):
                    table = PrettyTable(["Modules", "Parameters"])
                    total_params = 0
                    for name, parameter in model.named_parameters():
                        if not parameter.requires_grad: continue
                        print(parameter.requires_grad)
                        param = parameter.numel()
                        table.add_row([name, param])
                        total_params+=param
                    print(table)
                    print(f"Total Trainable Params: {total_params}")
                    return total_params

                def fix_parameters(model):
                    table = PrettyTable(["Modules", "Parameters"])
                    total_params = 0
                    for name, parameter in model.named_parameters():
                        if not parameter.requires_grad: continue
                        if not 'cls_head' in name:
                            parameter.requires_grad = False
                count_parameters(model)
                fix_parameters(model)                        
                count_parameters(model)
                import pdb;pdb.set_trace()'''
                list_df.iloc[i, :] = ret_feature[0,:len(label)]
                #index_time = index_time + args.split_time
            else:
                results = inference_recognizer(
                    model, video_block, args.label, use_frames=args.use_frames)

            if args.out_filename is not None:
                if args.target_resolution is not None:
                    if args.target_resolution[0] == -1:
                        args.target_resolution[0] = None
                    if args.target_resolution[1] == -1:
                        args.target_resolution[1] = None
                    args.target_resolution = tuple(args.target_resolution)
                else:
                    args.target_resolution = (None, None)
                print('The top-5 labels with corresponding scores are:')
                for result in results:
                    print(f'{result[0]}: ', result[1])
                label_show = ''
                for result in results:
                    label_show = label_show + result[0]+ ': {:.2g}'.format(result[1]) + '\n'
                get_output(
                    video_path=video_block,
                    out_filename=video_block_out,
                    label=label_show[:-1],
                    fps=args.fps,
                    font_size=args.font_size,
                    font_color=args.font_color,
                    target_resolution=args.target_resolution,
                    resize_algorithm=args.resize_algorithm,
                    use_frames=args.use_frames)
        # concatnate files
        with open(os.path.join(tmp_path,'list.txt'), 'w') as tmp_file:
            for video_block in dummy_filenames:
                tmp_file.write("file " +   'out_' + os.path.basename(video_block) + "\n")
        cmd = "ffmpeg -f concat -i {} -c copy -y {}".\
            format(os.path.join(tmp_path,'list.txt'), args.out_filename)
        #cmd = "ffmpeg -i {} -c copy -segment_format_options movflags=+faststart {}".\
        #    format(os.path.join(tmp_path,'list.txt'), args.out_filename)
        print("About to run: {}".format(cmd))
        check_call(shlex.split(cmd), universal_newlines=True)
        import shutil
        #import pdb
        #pdb.set_trace()
        shutil.rmtree(tmp_path)
        import matplotlib
        import matplotlib.pyplot as plt
        plt.figure()
        list_df.plot(y=label)#, x=range(0, args.split_time*len(dummy_filenames),args.split_time)
        plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1))
        plt.subplots_adjust(right=0.7)
        plt.grid()
        fig_outdir = os.path.dirname(args.out_filename)
        fig_outname = os.path.basename(args.out_filename)
        fig_outname  = fig_outname.rsplit(".", 1)[0]
        plt.savefig(os.path.join(fig_outdir,fig_outname+'.png'))
        plt.close('all')
        list_df.to_csv(os.path.join(fig_outdir,fig_outname+'.csv'), index=False)
if __name__ == '__main__':
    main()
