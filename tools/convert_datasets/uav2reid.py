import json
import numpy as np
import os
from PIL import Image
import mmcv

def create_imgs(train):
    work_dir = r'C:\Users\jac41744\smartvisiontoolbox\examples\data\uav_cocovid_with_video_id\reid\\'
    if train:
        print(r"Converting train images. The directory 'reid\imgs' should exist already.")
        filename_list = ['row_4.3_2', 'row_4.4_2', 'row_4.4_4', 'row_6.1_3',
                         'row_6.1_4', 'row_6.2_1', 'row_6.2_2', 'row_6.3', 'row_7.1_3',
                         'row_7.1_4', 'row_7.2_1', 'row_7.2_2', 'row_7.2_3', 'row_7.2_4', 'row_7.3_1', 'row_7.3_2',
                         'row_7.3_3', 'row_7.3_4', 'row_7.4_1', 'row_7.4_2', 'row_8_2', 'row_8_3',
                         'row_8_4']  # 'row_7.2_2', instances 11 - 17 missing # 'row_7.3_4', instance 12 is missing
        annotated_frames = [[0, 1, 13, 28, 43, 58, 73, 88, 103, 118, 133, 148, 163, 178, 193],
                            [0, 1, 17, 37, 57, 77, 97, 117, 137, 157, 177, 197, 217, 237, 257, 277, 297, 317, 337, 357],
                            [0, 1, 17, 37, 57, 77, 97, 117, 137, 157, 177, 197, 217, 237, 257, 277, 297, 317, 337, 357, 377,
                             397, 417, 437],
                            [0, 1, 21, 45, 69, 93, 117, 141, 165, 189, 213, 237, 261, 285, 309, 333, 357, 381, 405, 429,
                             453, 477, 501], [0, 1, 10, 22, 34, 46, 58, 70, 82, 94, 106, 118, 130, 142, 154, 166, 178, 190],
                            [0, 1, 24, 51, 79, 106, 133, 160, 187, 215, 242, 269, 296, 324, 351, 378, 405, 433, 460, 487,
                             514, 542, 569],
                            [0, 1, 29, 63, 96, 129, 163, 196, 229, 263, 296, 329, 362, 396, 429, 462, 496, 529, 562, 596,
                             629, 662, 695],
                            [0, 1, 31, 66, 102, 137, 172, 208, 243, 278, 313, 349, 384, 419, 454, 490, 525, 560, 595, 631,
                             666, 701],
                            [1, 49, 103, 158, 212, 267, 321, 375, 430, 484, 539, 593, 648, 702, 757, 811, 866, 920, 975,
                             1029, 1084, 1138, 1193, 1247, 1302, 1356],
                            [1, 25, 54, 82, 111, 139, 168, 196, 225, 254, 282, 311, 339, 368, 396, 425, 453, 482, 510, 539,
                             568, 596, 625, 653],
                            [1, 41, 87, 133, 179, 225, 272, 318, 364, 410, 456, 502, 548, 594, 640, 687, 733, 779, 825, 871,
                             917, 963, 1009],
                            [1, 53, 113, 173, 233, 293, 353, 413, 473, 533, 1012, 1072, 1132, 1192, 1252, 1312],
                            [0, 1, 77, 162, 248, 333, 419, 505, 590, 676, 762, 847, 933, 1018, 1104, 1190, 1275, 1361, 1447,
                             1532, 1618, 1704, 1789, 1875, 1960],
                            [0, 1, 53, 113, 173, 233, 293, 353, 413, 473, 533, 593, 653, 713, 773, 833, 893, 953, 1012,
                             1072, 1132, 1192, 1252, 1312],
                            [0, 1, 26, 55, 84, 114, 143, 172, 201, 230, 260, 289, 318, 347, 377, 406, 435, 464, 494, 523,
                             552, 581, 611, 640, 669],
                            [0, 1, 53, 113, 173, 233, 293, 353, 413, 473, 533, 593, 653, 713, 773, 833, 893, 953, 1012,
                             1072, 1132, 1192, 1252, 1312],
                            [0, 1, 41, 87, 133, 179, 225, 272, 318, 364, 410, 456, 502, 548, 594, 640, 687, 733, 779, 825,
                             871, 917, 963, 1009],
                            [0, 1, 51, 108, 165, 222, 279, 336, 393, 450, 508, 565, 679, 736, 793, 850, 907, 964, 1021,
                             1078, 1136, 1193, 1250, 1307],
                            [0, 1, 22, 48, 73, 99, 124, 150, 175, 201, 227, 252, 278, 303, 329, 354, 380, 405, 431, 456,
                             482, 507, 533, 558, 584, 609, 635],
                            [1, 49, 103, 158, 212, 267, 321, 375, 430, 484, 539, 593, 648, 702, 757, 811, 866, 920, 975,
                             1029, 1084, 1138, 1193, 1247, 1302, 1356],
                            [0, 1, 41, 87, 133, 179, 225, 272, 318, 364, 410, 456, 502, 548, 594, 640, 687, 733, 779, 825,
                             871, 917, 963, 1009],
                            [0, 1, 26, 56, 86, 116, 146, 176, 206, 236, 266, 296, 326, 356, 386, 416, 446, 476, 506, 536,
                             566, 596, 626, 656],
                            [1, 53, 113, 173, 233, 293, 353, 413, 473, 533, 593, 653, 713, 773, 833, 893, 953, 1012, 1072,
                             1132, 1192, 1252, 1312, 1372]]
        json_path = r'C:\Users\jac41744\smartvisiontoolbox\examples\data\uav_cocovid_with_video_id\train\coco_train.json'
        root = r'C:\Users\jac41744\smartvisiontoolbox\examples\data\uav_cocovid_with_video_id\train\\'

    else:
        print(r"Converting test images. The directory 'reid\imgs' should exist already.")
        filename_list = ['row_4.2_1', 'row_6.1_1', 'row_6.1_2', 'row_7.1_1', 'row_7.1_2', 'row_8_1']
        annotated_frames = [
            [0, 1, 17, 37, 57, 77, 97, 117, 137, 157, 177, 197, 217, 237, 257, 277, 297, 317, 337, 357, 377, 397, 417],
            [0, 1, 49, 103, 158, 212, 267, 321, 375, 430, 484, 539, 593, 648, 702, 757, 811, 866, 920, 975, 1029, 1084,
             1138, 1193], [0, 1, 10, 22, 34, 46, 58, 70, 82, 94, 106, 118, 130, 142, 154, 166, 178, 190, 202],
            [1, 31, 66, 102, 137, 172, 208, 243, 278, 313, 349, 384, 419, 454, 490, 525, 560, 595, 631, 666, 701, 736, 772,
             807, 842, 877, 913],
            [1, 33, 71, 108, 146, 183, 221, 258, 295, 333, 370, 408, 445, 483, 520, 558, 595, 633, 670],
            [1, 67, 142, 217, 292, 367, 442, 516, 591, 666, 741, 816, 891, 966, 1041, 1116, 1191, 1266, 1341, 1416, 1491,
             1565, 1640, 1715]]
        json_path = r'C:\Users\jac41744\smartvisiontoolbox\examples\data\uav_cocovid_with_video_id\test\coco_test.json'
        root = r'C:\Users\jac41744\smartvisiontoolbox\examples\data\uav_cocovid_with_video_id\test\\'

    progress_bar_length = 0
    for i in range(len(annotated_frames)):
        progress_bar_length += len(annotated_frames[i])

    # Opening JSON file
    f = open(json_path)
    # returns JSON object as
    # a dictionary
    data = json.load(f)
    f.close()
    list = []
    prog_bar = mmcv.ProgressBar(progress_bar_length)
    for video_id, row in enumerate(filename_list):
        instance_dict = {}
        for annotation_id, annotation in enumerate(data["annotations"]):
            if annotation["video_id"] == video_id:
                if annotation["instance_id"] in instance_dict.keys():
                    image_id = annotation["image_id"]
                    instance_dict[annotation["instance_id"]].append([annotation_id, image_id])
                else:
                    image_id = annotation["image_id"]
                    instance_dict[annotation["instance_id"]] = [[annotation_id, image_id]]
        list.append(instance_dict)

    for video_id, instance_id_dict in enumerate(list):
        for instance_id in instance_id_dict.keys():
            if train:
                new_instance_directory = 'UAV-' + str(video_id) + '-train_' + str(instance_id)
            else:
                new_instance_directory = 'UAV-' + str(video_id) + '-test_' + str(instance_id)
            new_instance_directory = os.path.join(work_dir + 'imgs', new_instance_directory)
            os.mkdir(new_instance_directory)
            for ann_and_img_id in instance_id_dict[instance_id]:
                file_name = data["images"][ann_and_img_id[1]]["file_name"]
                img_path = os.path.join(root, file_name)
                img = Image.open(img_path)
                bbox = data["annotations"][ann_and_img_id[0]]["bbox"]
                crop_box = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                croped_img = img.crop(crop_box)
                save_path = os.path.join(new_instance_directory, file_name)
                croped_img.save(save_path)
                prog_bar.update()


def create_meta():
    print("Creating the meta text files. The directory 'reid\meta should exist already.")
    image_root = r'C:\Users\jac41744\smartvisiontoolbox\examples\data\uav_cocovid_with_video_id\reid\imgs'
    train_txt_path = r'C:\Users\jac41744\smartvisiontoolbox\examples\data\uav_cocovid_with_video_id\reid\meta\train.txt'
    test_txt_path = r'C:\Users\jac41744\smartvisiontoolbox\examples\data\uav_cocovid_with_video_id\reid\meta\test.txt'
    train_instance_counter = 0
    test_instance_counter = 0
    print(f"Creating train.txt at {train_txt_path}")
    open(train_txt_path, 'x')
    print(f"Creating test.txt at {test_txt_path}")
    open(test_txt_path, 'x')
    prog_bar = mmcv.ProgressBar(len(os.listdir(image_root)))
    for instance in os.listdir(image_root):
        instance_root = os.path.join(image_root, instance)
        if 'train' in instance:
            for image in os.listdir(instance_root):
                image_path = os.path.join(instance_root, image)
                with open(train_txt_path, "a") as train_txt:
                    train_txt.write(f"{image_path} {train_instance_counter}\n")
            train_instance_counter += 1
        elif 'test' in instance:
            for image in os.listdir(instance_root):
                image_path = os.path.join(instance_root, image)
                with open(test_txt_path, "a") as test_txt:
                    test_txt.write(f"{image_path} {test_instance_counter}\n")
            test_instance_counter += 1
        else:
            print("Error")
        prog_bar.update()


def main():
    create_meta()


if __name__ == "__main__":
    main()
