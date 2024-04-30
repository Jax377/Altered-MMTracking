# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
import mmcv
import time
import pickle
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info
from mmdet.core import encode_mask_results
from mmtrack.apis import inference_mot, init_model
from mmtrack.datasets import build_dataset

def main():
    # Getting arguments and configs.
    parser = ArgumentParser()
    parser.add_argument('config', help='config file')
    parser.add_argument('--input', help='input video file or folder')
    parser.add_argument(
        '--output', help='output video file (mp4 format) or folder')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.0,
        help='The threshold of score to filter bboxes.')
    parser.add_argument(
        '--device', default='cuda:0', help='device used for inference')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether show the results on the fly')
    parser.add_argument('--eval', type=str, nargs='+', help='eval types')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--backend',
        choices=['cv2', 'plt'],
        default='cv2',
        help='the backend to visualize the results')
    parser.add_argument('--fps', help='FPS of the output video')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be kwargs for dataset.evaluate() function')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)  # annotated frames in the test_videos
    # list of the annotated frames in the videos. dim(video_ids, frames)
    annotated_frames = [[0, 1, 17, 37, 57, 77, 97, 117, 137, 157, 177, 197, 217, 237, 257, 277, 297, 317, 337, 357, 377, 397, 417], [0, 1, 49, 103, 158, 212, 267, 321, 375, 430, 484, 539, 593, 648, 702, 757, 811, 866, 920, 975, 1029, 1084, 1138, 1193], [0, 1, 10, 22, 34, 46, 58, 70, 82, 94, 106, 118, 130, 142, 154, 166, 178, 190, 202], [1, 31, 66, 102, 137, 172, 208, 243, 278, 313, 349, 384, 419, 454, 490, 525, 560, 595, 631, 666, 701, 736, 772, 807, 842, 877, 913], [1, 33, 71, 108, 146, 183, 221, 258, 295, 333, 370, 408, 445, 483, 520, 558, 595, 633, 670], [1, 67, 142, 217, 292, 367, 442, 516, 591, 666, 741, 816, 891, 966, 1041, 1116, 1191, 1266, 1341, 1416, 1491, 1565, 1640, 1715]]
    # video_id = 0 # Test a specific Video.
    # start_frames = 0
    # # for i in range(video_id):
    # #     start_frames += len(annotated_frames[i])
    # end_frames = 0
    # for i in range(video_id + 1):
    #     end_frames += len(annotated_frames[i])
    # dataset.data_infos = dataset.data_infos[start_frames:end_frames]
    # dataset.img_ids = dataset.img_ids[start_frames:end_frames]
    # dataset.img_ids = dataset.img_ids[0:24]

    # Did we input a video or a directory?
    if osp.isdir(args.input):
        video_paths = [osp.join(args.input, video) for video in os.listdir(args.input) if osp.isfile(osp.join(args.input, video))]
    elif args.input.endswith('.mp4'):
        video_paths = [args.input]
    else:
        raise TypeError("Only a directory of videos or a video(.mp4) is allowed.")
    IN_VIDEO = True

    # Collecting the outputs of the annotated frames of all videos tested.
    outputs = defaultdict(list)
    outputs['det_bboxes'] = []
    outputs['track_bboxes'] = []

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    print(f"There are {len(video_paths)} videos.")
    for video_id, video_path in enumerate(video_paths):

        # load images
        imgs = mmcv.VideoReader(video_path)
        # define output
        if args.output is not None:
            if args.output.endswith('.mp4'):
                OUT_VIDEO = True
                out_dir = tempfile.TemporaryDirectory()
                out_path = out_dir.name
                _out = args.output.rsplit(os.sep, 1)
                if len(_out) > 1:
                    os.makedirs(_out[0], exist_ok=True)
            elif osp.isdir(args.output):
                args_output_directory = args.output
                OUT_VIDEO = True
                timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
                args.output = args.output + "\\" + osp.basename(video_path)[:-4] + f"_track_{timestamp}.mp4"
                out_dir = tempfile.TemporaryDirectory()
                out_path = out_dir.name
                _out = args.output.rsplit(os.sep, 1)
                if len(_out) > 1:
                    os.makedirs(_out[0], exist_ok=True)
            else:
                OUT_VIDEO = False
                out_path = args.output
                os.makedirs(out_path, exist_ok=True)
        else:
            OUT_VIDEO = False

        fps = args.fps
        if args.show or OUT_VIDEO:
            if fps is None and IN_VIDEO:
                fps = imgs.fps
            if not fps:
                raise ValueError('Please set the FPS for the output video.')
            fps = int(fps)

        # build the model from a config file and a checkpoint file
        model = init_model(args.config, args.checkpoint, device=args.device)

        print(f"\nTesting video {video_id}.")
        prog_bar = mmcv.ProgressBar(len(imgs))
        # test and show/save the images
        results = defaultdict(list)
        for i, img in enumerate(imgs):
            img = img.astype(np.float32)
            if isinstance(img, str):
                img = osp.join(args.input, img)
            result = inference_mot(model, img, frame_id=i)
            if args.output is not None:
                if IN_VIDEO or OUT_VIDEO:
                    out_file = osp.join(out_path, f'{i:06d}.jpg')
                else:
                    out_file = osp.join(out_path, img.rsplit(os.sep, 1)[-1])
            else:
                out_file = None

            model.show_result(
                img,
                result,
                score_thr=args.score_thr,
                show=args.show,
                wait_time=int(1000. / fps) if fps else 0,
                out_file=out_file,
                backend=args.backend)

            # encode mask results
            for key in result:
                if 'mask' in key:
                    result[key] = encode_mask_results(result[key])

            for k, v in result.items():
                if k != 'count_tracks':
                    results[k].append(v)

            prog_bar.update()

        outputs['det_bboxes'] += [results['det_bboxes'][l] for l in annotated_frames[video_id]]
        outputs['track_bboxes'] += [results['track_bboxes'][l] for l in annotated_frames[video_id]]

        if args.output and OUT_VIDEO:
            print(f'\nMaking the output video of video {video_id} at {args.output} with a FPS of {fps}.')
            mmcv.frames2video(out_path, args.output, fps=fps, fourcc='mp4v')
            out_dir.cleanup()
            args.output = args_output_directory

    rank, _ = get_dist_info()
    if rank == 0:
        # print(result['count_tracks'])
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            eval_hook_args = [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule', 'by_epoch'
            ]
            for key in eval_hook_args:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset)

            temp_file = tempfile.NamedTemporaryFile(mode='wb', delete=False)
            try:
                pickle.dump(outputs, temp_file)
            finally:
                temp_file.close()
                print(f"Defaultdict is saved in temporary file: {temp_file.name}")
            # with open(r'C:\Users\jac41744\AppData\Local\Temp\tmpqnm5gwie', 'rb') as file:  # bytetrack
            # with open(r'C:\Users\jac41744\AppData\Local\Temp\tmpsvywo7je', 'rb') as file:  # sort
            # with open(r'C:\Users\jac41744\AppData\Local\Temp\tmpix4vk1kl', 'rb') as file:  # deepsort
            # with open(r'C:\Users\jac41744\AppData\Local\Temp\tmp5ls7s_pl', 'rb') as file:  # sort+ 0.25
            # with open(r'C:\Users\jac41744\AppData\Local\Temp\tmps96bv0wl', 'rb') as file:  # deepsort+ 0.25
            # with open(r'C:\Users\jac41744\AppData\Local\Temp\tmp9031nvk4', 'rb') as file:  # sort+ 0.2
            # with open(r'C:\Users\jac41744\AppData\Local\Temp\tmp98nkzmvw', 'rb') as file:  # deepsort+ 0.2
                # outputs = pickle.load(file)

            metric = dataset.evaluate(outputs, **eval_kwargs)
            print(metric)
            metric_dict = dict(
                config=args.config, mode='test', epoch=cfg.total_epochs)
            metric_dict.update(metric)
            if args.work_dir is not None:
                timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
                json_file = osp.join(args.work_dir, f'eval_{timestamp}.log.json')
                mmcv.dump(metric_dict, json_file)


if __name__ == '__main__':
    main()
