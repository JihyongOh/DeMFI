from __future__ import division
import os, glob, sys, torch, shutil, random, math, time, cv2
import numpy as np
import torch.utils.data as data
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt

from datetime import datetime
from PIL import Image
from time import gmtime, strftime
from six.moves import xrange
from torch.nn import init
from skimage.metrics import structural_similarity
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms


class save_manager():
    def __init__(self, args):
        self.args = args
        self.model_dir = self.args.net_type + '_exp' + str(self.args.exp_num)
        print("model_dir:", self.model_dir)
        # ex) model_dir = "DeFInet_exp1"

        self.checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
        # './checkpoint_dir/DeFInet_exp1"
        check_folder(self.checkpoint_dir)

        print("checkpoint_dir:", self.checkpoint_dir)

        self.text_dir = os.path.join(self.args.text_dir, self.model_dir)
        print("text_dir:", self.text_dir)

        """ Save a text file """
        if not os.path.exists(self.text_dir + '.txt'):
            self.log_file = open(self.text_dir + '.txt', 'w')
            # "w" - Write - Opens a file for writing, creates the file if it does not exist
            self.log_file.write('----- Model parameters -----\n')
            for arg in vars(self.args):
                self.log_file.write('{} : {}\n'.format(arg, getattr(self.args, arg)))
            # ex) ./text_dir/DeFInet_exp1.txt
            self.log_file.close()


        else:
            self.log_file = open(self.text_dir + '.txt', 'a')
            self.log_file.write('----- Model parameters -----\n')
            for arg in vars(self.args):
                self.log_file.write('{} : {}\n'.format(arg, getattr(self.args, arg)))
            self.log_file.close()

    # "a" - Append - Opens a file for appending, creates the file if it does not exist

    def write_num_param(self, num_param, flops):
        sys.stdout.flush()
        self.log_file = open(self.text_dir + '.txt', 'a')
        self.log_file.write('Total # of parameters: ' + str(num_param) + '\n')
        self.log_file.write('Total Flops for patch size ' + str(self.args.patch_size) + ': ' + str(flops) + '\n')
        self.log_file.flush()
        self.log_file.close()
        print('Flops for patch size %d: %d' % (self.args.patch_size, flops))
        print('Total numbers of model parameters: {}'.format(
            num_param))

    def write_info(self, strings):
        self.log_file = open(self.text_dir + '.txt', 'a')
        self.log_file.write(strings)
        self.log_file.close()

    def save_best_model(self, combined_state_dict, best_PSNR_flag, best_SSIM_flag):
        file_name = self.checkpoint_dir + '/' + self.model_dir + '_latest.pt'
        # file_name = "./checkpoint_dir/DeFInet_exp1/DeFInet_exp1_latest.ckpt
        torch.save(combined_state_dict, file_name)
        if best_PSNR_flag:
            shutil.copyfile(file_name, self.checkpoint_dir + '/' + self.model_dir + '_best_PSNR.pt')
        # file_path = "./checkpoint_dir/DeFInet_exp1/DeFInet_exp1_best_PSNR.ckpt
        if best_SSIM_flag:
            shutil.copyfile(file_name, self.checkpoint_dir + '/' + self.model_dir + '_best_SSIM.pt')

    def save_epc_model(self, combined_state_dict, epoch):
        file_name = self.checkpoint_dir + '/' + self.model_dir + '_epc' + str(epoch) + '.pt'
        # file_name = "./checkpoint_dir/DeFInet_exp1/DeFInet_exp1_epc10.ckpt
        torch.save(combined_state_dict, file_name)

    def load_epc_model(self, epoch):
        checkpoint = torch.load(self.checkpoint_dir + '/' + self.model_dir + '_epc' + str(epoch - 1) + '.pt')
        print("load model '{}', epoch: {}, best_PSNR: {:3f}".format(
            self.checkpoint_dir + '/' + self.model_dir + '_epc' + str(epoch - 1) + '.pt', checkpoint['last_epoch'] + 1,
            checkpoint['best_PSNR']))
        return checkpoint

    def load_model(self, ):
        # checkpoint = torch.load(self.checkpoint_dir + '/' + self.model_dir + '_latest.pt', map_location='cuda:0')
        checkpoint = torch.load(self.checkpoint_dir + '/' + self.model_dir + '_latest.pt')
        # print("load model '{}', epoch: {}, best_PSNR: {:3f}, best_SSIM: {:3f}".format(
        #     self.checkpoint_dir + '/' + self.model_dir + '_latest.pt', checkpoint['last_epoch'] + 1,
        #     checkpoint['best_PSNR'], checkpoint['best_SSIM'])) # when 'best_PSNR' & 'best_SSIM' exists
        print("load model '{}', epoch: {},".format(
            self.checkpoint_dir + '/' + self.model_dir + '_latest.pt', checkpoint['last_epoch'] + 1))
        return checkpoint

    def load_best_PSNR_model(self, ):
        checkpoint = torch.load(self.checkpoint_dir + '/' + self.model_dir + '_best_PSNR.pt')
        print("load _best_PSNR model '{}', epoch: {}, best_PSNR: {:3f}, best_SSIM: {:3f}".format(
            self.checkpoint_dir + '/' + self.model_dir + '_best_PSNR.pt', checkpoint['last_epoch'] + 1,
            checkpoint['best_PSNR'], checkpoint['best_SSIM']))
        return checkpoint


class AverageClass(object):
    """ For convenience of averaging values """
    """ refer from "https://github.com/pytorch/examples/blob/master/imagenet/main.py" """

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} (avg:{avg' + self.fmt + '})'
        # Accm_Time[s]: 1263.517 (avg:639.701)    (<== if AverageClass('Accm_Time[s]:', ':6.3f'))
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """ For convenience of printing diverse values by using "AverageClass" """
    """ refer from "https://github.com/pytorch/examples/blob/master/imagenet/main.py" """

    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        # # Epoch: [0][  0/196]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    # Epoch: [45][480/885]	Batch_Time[s]:  1.430 (avg: 1.453)	Accm_Time[s]: 699.100 (avg:356.795)	trainLoss: 1.9964e-01 (avg:1.2468e-01)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def str2bool(x):
    return x.lower() in ('true')


def weights_init(m):
    classname = m.__class__.__name__
    if (classname.find('Conv2d') != -1) or (classname.find('Conv3d') != -1):
        # if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight)
        # init.kaiming_normal_(m.weight, nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            init.zeros_(m.bias)


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # RAFT
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            # elif isinstance(m, nn.BatchNorm2d):
            # 	init.constant_(m.weight, 1)
            # 	init.constant_(m.bias.data, 0.0) # ZSM
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # RAFT


def RGB_np2Tensor(imgIn, channel):
    if channel == 1:
        # rgb --> Y (gray)
        imgIn = np.sum(imgIn * np.reshape([65.481, 128.553, 24.966], [1, 1, 3]) / 255.0, axis=2, keepdims=True) + 16.0

    # to Tensor
    ts = (2, 0, 1)
    imgIn = torch.Tensor(imgIn.transpose(ts).astype(float)).mul_(1.0)

    # normalization [-1,1]
    imgIn = (imgIn / 255.0 - 0.5) * 2

    return imgIn


def RGBframes_np2Tensor(imgIn, channel):  
    ## input : T, H, W, C
    if channel == 1:
        # rgb --> Y (gray)
        imgIn = np.sum(imgIn * np.reshape([65.481, 128.553, 24.966], [1, 1, 1, 3]) / 255.0, axis=3,
                       keepdims=True) + 16.0

    # to Tensor
    ts = (3, 0, 1, 2)  ############# dimension order should be [C, T, H, W]
    imgIn = torch.Tensor(imgIn.transpose(ts).astype(float)).mul_(1.0)

    # normalization [-1,1]
    imgIn = (imgIn / 255.0 - 0.5) * 2

    return imgIn


""" Training """
def get_train_data(args):
    data_train = Adobe_Train(args)
    dataloader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size,
                                             drop_last=True, shuffle=True, num_workers=int(args.num_thrds),
                                             pin_memory=False)
    return dataloader


class Adobe_Train(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.t_step_size = args.t_step_size  # 8 "=K in paper"
        self.num_input_frames = 4
        self.t = np.linspace((1 / self.t_step_size), (1 - (1 / self.t_step_size)), (self.t_step_size - 1))

        self.framesPath, self.blurPath = make_2D_dataset_Adobe_Train(
            args.train_data_path)  # './Datasets/Adobe_240fps_blur'
        self.nScenes = len(self.framesPath)

        if self.nScenes == 0:
            raise (RuntimeError("Found 0 files in subfolders of: ./Datasets/Adobe_240fps_blur/ \n"))

    def __getitem__(self, idx):
        sharp_candidate_frames = self.framesPath[idx]  # each scene has different number of frames
        blur_candidate_frames = self.blurPath[idx]  # each scene has different number of frames

        blur_firstFrameIdx = random.randint(0 + 1, (len(blur_candidate_frames) - 2 - 1))
        interIdx = random.randint(1, self.t_step_size - 1)  # relative index, 1~self.t_step_size-1
        blur_absIdx = int(blur_candidate_frames[blur_firstFrameIdx].split(os.sep)[-1][:-4])  # ex) 00017.png => 17
        sharp_abs_interFrameIdx = blur_absIdx + interIdx - 1  # ex) 17(=blur_absIdx) + 1(=interIdx) - 1 => 17
        t_value = self.t[interIdx - 1]  # [0,1]
        sharp_abs_S0 = blur_absIdx - 1  # ex) 17(=blur_absIdx)  - 1 => [16]
        sharp_abs_S1 = blur_absIdx + 8 - 1  # ex) 17(=blur_absIdx) 8 - 1 => [24]

        sharp_abs_S_minus1 = blur_absIdx - 1 - 8  # ex) 17(=blur_absIdx)  - 1 - 8 => [8]
        sharp_abs_S2 = blur_absIdx + 8 - 1 + 8  # ex) 17(=blur_absIdx) 8 - 1 + 8 => [32]

        """ Randomly reverse frames """
        if (random.randint(0, 1)):
            frameRange =  [blur_firstFrameIdx, blur_firstFrameIdx + 1] + [blur_firstFrameIdx-1,blur_firstFrameIdx+2] + [sharp_abs_interFrameIdx] \
                         + [sharp_abs_S0, sharp_abs_S1] + [sharp_abs_S_minus1,sharp_abs_S2]
        else:
            frameRange = [blur_firstFrameIdx + 1, blur_firstFrameIdx] + [blur_firstFrameIdx+2, blur_firstFrameIdx-1] + [sharp_abs_interFrameIdx] \
                         + [sharp_abs_S1, sharp_abs_S0] + [sharp_abs_S2,sharp_abs_S_minus1]
            t_value = 1.0 - t_value
        frames = frames_loader_sharp_blur_train(self.args, sharp_candidate_frames, blur_candidate_frames,
                                                frameRange)  # including "np2Tensor [-1,1] normalized"
        # frames: [B0,B1,St,S0,S1]  (T, H, W, 3)
        B0 = (blur_candidate_frames[frameRange[0]])
        B1 = (blur_candidate_frames[frameRange[1]])
        St = (sharp_candidate_frames[frameRange[2]])
        S0 = (sharp_candidate_frames[frameRange[-2]])
        S1 = (sharp_candidate_frames[frameRange[-1]])

        return frames, np.expand_dims(np.array(t_value, dtype=np.float32), 0)

    def __len__(self):
        return self.nScenes


def make_2D_dataset_Adobe_Train(dir):
    """
	' folder format: "%s(112scenes)/%5d(N_frames)"'
	Creates a 2D list.
	2D List Structure:
	[[00000.png, 00001.png, ..., 00742.png]
	 [00000.png, 00001.png, ..., 00307.png],
	 :
	 [00000.png, 00001.png, ..., 00486.png]]
	 # total numbers: 123,532, average frames in each scene : 928
	"""
    sharp_dir = dir + '/train'
    blur_dir = dir + '/train_blur'

    # Find and loop over all the clips in root `dir`.
    sharp_framesPath = []
    for scene_folder in (sorted(os.listdir(sharp_dir))):
        scene_path = os.path.join(sharp_dir, scene_folder)
        frames_list = []
        for frame in sorted(os.listdir(scene_path)):
            frames_list.append(os.path.join(scene_path, frame))
        sharp_framesPath.append(frames_list)

    blur_framesPath = []
    for scene_folder in (sorted(os.listdir(blur_dir))):
        scene_path = os.path.join(blur_dir, scene_folder)
        frames_list = []
        for frame in sorted(os.listdir(scene_path)):
            frames_list.append(os.path.join(scene_path, frame))
        blur_framesPath.append(frames_list)

    return sharp_framesPath, blur_framesPath


def frames_loader_sharp_blur_train(args, sharp_candidate_frames, blur_candidate_frames, frameRange):
    frames = []
    for frameIndex in frameRange[:2+2]:  # first four
        frame = cv2.imread(blur_candidate_frames[frameIndex])  # blurry inputs
        frames.append(frame)
    frames.append(cv2.imread(sharp_candidate_frames[frameRange[2+2]]))  # sharp GT
    for frameIndex in frameRange[-2-2:]:  # last four
        frame = cv2.imread(sharp_candidate_frames[frameIndex])  # sharp inputs
        frames.append(frame)

    (ih, iw, c) = frame.shape
    frames = np.stack(frames, axis=0)  # (T, H, W, 3)
    if args.need_patch:
        ps = args.patch_size  # patch size, (=translation)
        ix = random.randrange(0, iw - ps + 1)
        iy = random.randrange(0, ih - ps + 1)
        frames = frames[:, iy:iy + ps, ix:ix + ps, :]  # (T, 512,512,3)

    if random.random() < 0.5:  # horizontal flip
        frames = frames[:, :, ::-1, :]  # (512,512,3)

    # No vertical flip
    rot = random.randint(0, 3)  # rotate
    frames = np.rot90(frames, rot, (1, 2))

    """ np2Tensor [-1,1] normalized """
    frames = RGBframes_np2Tensor(frames, args.img_ch)

    return frames


def get_test_data(args, multiple, center_flag, test_type):
    if args.phase =='test_custom':
        """ Testing for custom_path """
        data_test = Custom_Test(args)
        dataloader = torch.utils.data.DataLoader(data_test, batch_size=1,
                                                 drop_last=True, shuffle=False, pin_memory=False)
    else:
        """ Testing for Evaluation with GTs """
        data_test = diverse_Test(args, multiple, center_flag, test_type)
        dataloader = torch.utils.data.DataLoader(data_test, batch_size=1,
                                                 drop_last=True, shuffle=False, pin_memory=False)
    return dataloader

""" Testing for Evaluation with GTs """
class diverse_Test(data.Dataset):
    def __init__(self, args, multiple, center_flag, test_type):
        self.args = args
        self.multiple = multiple
        self.center_flag = center_flag
        self.testPath = make_2D_dataset_Test(
            self.args.test_data_path, multiple, test_type, args.t_step_size)  # './Datasets/Adobe_240fps_blur'
        self.nIterations = len(self.testPath)

        # Raise error if no images found in test_data_path.
        if len(self.testPath) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.args.test_data_path + "\n"))

    def __getitem__(self, idx):
        B0, B1, Bm1, B2, St, S0, S1, Sm1, S2, t_value, scene_name = self.testPath[idx]

        B0B1St_Path = [B0, B1, Bm1, B2, St]
        S0S1_Path = [S0, S1, Sm1, S2]

        """ Open frames using "cv2" """
        frames, S0S1_GT_frames = frames_loader_sharp_blur_test(self.args,
                                                               B0B1St_Path, self.center_flag, S0S1_Path)

        St_path = St.split(os.sep)[-1]
        S0_path = S0.split(os.sep)[-1]
        S1_path = S1.split(os.sep)[-1]

        # including "np2Tensor [-1,1] normalized"

        print("(x{}) Loading --- iterations: {}/{} --- [left, inter, right] indices: [{},{},{}] --- t_value: {}".format(
            int(self.multiple), idx + 1, self.nIterations,
            os.path.join(B0.split(os.sep)[-2], B0.split(os.sep)[-1]),
            os.path.join(St.split(os.sep)[-2], St.split(os.sep)[-1]),
            os.path.join(B1.split(os.sep)[-2], B1.split(os.sep)[-1]), t_value))

        return frames, np.expand_dims(np.array(t_value, dtype=np.float32),
                                      0), scene_name, [St_path, S0_path, S1_path], S0S1_GT_frames

    def __len__(self):
        return self.nIterations


def make_2D_dataset_Test(dir, multiple, test_type, t_step_size):
    if 'BlurLFR' in dir:
        if 'Adobe' in dir:
            sharp_dir = dir[:-3] + '_test_GT_zfill5'
        elif 'Gopro' in dir:
            sharp_dir = dir[:-3] + '_test_GT_zfill6'
        blur_dir = dir
    else:
        sharp_dir = dir + '/test'
        blur_dir = dir + '/test_blur'

    if 'GoPro_blur' in dir or 'Gopro' in dir:
        zfill_num = 6
    elif 'Adobe_240fps_blur' in dir or 'YouTube240_Scenes' in dir or 'Adobe' in dir:
        zfill_num = 5

    """ make [B0, B1, Bm1, B2, St, S0, S1, Sm1, S2, t_value, scene_name] """
    """ 1D (accumulated) """
    testPath = []
    t = np.linspace((1 / multiple), (1 - (1 / multiple)), (multiple - 1))
    for scene_folder in (sorted(os.listdir(blur_dir))):
        blur_scene_path = os.path.join(blur_dir, scene_folder)
        sharp_scene_path = os.path.join(sharp_dir, scene_folder)
        frame_folder = sorted(os.listdir(blur_scene_path))
        for idx in range(len(frame_folder)):
            if idx == len(frame_folder) - 2:
                break
            if not idx ==0:
                for mul in range(multiple - 1):
                    B0B1St_paths = []
                    B0B1St_paths.append(os.path.join(blur_scene_path, frame_folder[idx]))
                    B0B1St_paths.append(os.path.join(blur_scene_path, frame_folder[idx + 1]))
                    B0B1St_paths.append(os.path.join(blur_scene_path, frame_folder[idx-1]))
                    B0B1St_paths.append(os.path.join(blur_scene_path, frame_folder[idx + 2]))
                    sharp_str = str(int(int(frame_folder[idx][:-4]) + (t_step_size / multiple) * (mul + 1))).zfill(
                        zfill_num) + '.png'
                    B0B1St_paths.append(os.path.join(sharp_scene_path, sharp_str))
                    B0B1St_paths.append(os.path.join(sharp_scene_path, frame_folder[idx]))
                    B0B1St_paths.append(os.path.join(sharp_scene_path, frame_folder[idx + 1]))
                    B0B1St_paths.append(os.path.join(sharp_scene_path, frame_folder[idx-1]))
                    B0B1St_paths.append(os.path.join(sharp_scene_path, frame_folder[idx + 2]))
                    B0B1St_paths.append(t[mul])
                    B0B1St_paths.append(scene_folder)
                    testPath.append(B0B1St_paths)
            if test_type == 'valid_5_per_scene' and frame_folder[idx + 1] == '00057.png':
                # 00017.png, 00025.png,...,00057.png
                break

    return testPath


def frames_loader_sharp_blur_test(args, B0B1St_Path, center_flag, S0S1_GT_Path):  
    frames = []
    S0S1_GT_frames = []
    for path in B0B1St_Path:
        frame = cv2.imread(path)
        frames.append(frame)
    (ih, iw, c) = frame.shape
    frames = np.stack(frames, axis=0)  # (T, H, W, 3)

    for path in S0S1_GT_Path:
        frame = cv2.imread(path)
        S0S1_GT_frames.append(frame)
    S0S1_GT_frames = np.stack(S0S1_GT_frames, axis=0)  # (T, H, W, 3)

    if center_flag:
        ps = args.patch_size  # patch size, (=translation)
        ix = (iw - ps) // 2
        iy = (ih - ps) // 2
        frames = frames[:, iy:iy + ps, ix:ix + ps, :]  # (512,512,3)
        S0S1_GT_frames = S0S1_GT_frames[:, iy:iy + ps, ix:ix + ps, :]  # (512,512,3)

    """ np2Tensor [-1,1] normalized """
    frames = RGBframes_np2Tensor(frames, args.img_ch)
    S0S1_GT_frames = RGBframes_np2Tensor(S0S1_GT_frames, args.img_ch)

    return frames, S0S1_GT_frames


def frames_loader_Test(args, candidate_frames, frameRange, center_flag):
    frames = []
    for frameIndex in frameRange:
        frame = cv2.imread(candidate_frames[frameIndex])
        frames.append(frame)
    (ih, iw, c) = frame.shape

    if center_flag:
        ps = args.patch_size  # patch size, (=translation)
        ix = (iw - ps) // 2
        iy = (h - ps) // 2
        for fr_idx, frame in enumerate(frames):
            frames[fr_idx] = frame[iy:iy + ps, ix:ix + ps, :]  # (512,512,3)

    """ np2Tensor [-1,1] normalized """
    for fr_idx, frame in enumerate(frames):
        frames[fr_idx] = RGB_np2Tensor(frame, args.img_ch)

    return frames


""" Testing for custom_path (no GT) """
class Custom_Test(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.multiple_MFI = args.multiple_MFI
        self.testPath = make_2D_dataset_Custom_Test(args, self.args.custom_path, self.multiple_MFI)
        self.nIterations = len(self.testPath)

        # Raise error if no images found in test_data_path.
        if len(self.testPath) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.args.custom_path + "\n"))

    def __getitem__(self, idx):
        #I0, I1, It, t_value, scene_name = self.testPath[idx]
        B0, B1, Bm1, B2, St, S0, S1, t_value, scene_name = self.testPath[idx]

        B0B1Bm1B2_Path = [B0, B1, Bm1, B2]

        St_path = St.split(os.sep)[-1]
        S0_path = S0.split(os.sep)[-1]
        S1_path = S1.split(os.sep)[-1]

        frames = frames_loader_sharp_blur_custom_test(self.args, B0B1Bm1B2_Path)
        # including "np2Tensor [-1,1] normalized"


        return frames, np.expand_dims(np.array(t_value, dtype=np.float32), 0), scene_name, \
               [St_path, S0_path, S1_path]

    def __len__(self):
        return self.nIterations


def make_2D_dataset_Custom_Test(args, dir, multiple):
    """ make [B0, B1, Bm1, B2, St, S0, S1, t_value, scene_name] """
    """ 1D (accumulated) """
    testPath = []
    t = np.linspace((1 / multiple), (1 - (1 / multiple)), (multiple - 1))

    for scene_folder in sorted(glob.glob(os.path.join(dir, '*', ''))):  # [scene1, scene2, scene3, ...]
        frame_folder = sorted(glob.glob(scene_folder + '*.png'))  # ex) ['00000.png',...,'00123.png']
        converted_folder = os.path.join(dir, scene_folder+'_sharply_interpolated_x'+str(args.multiple_MFI))
        for idx in range(1, len(frame_folder)):
            if idx == len(frame_folder) - 2:
                break
            for suffix, mul in enumerate(range(multiple - 1)):
                B0B1St_paths = []
                B0B1St_paths.append(frame_folder[idx])  # B0 (fix)
                B0B1St_paths.append(frame_folder[idx + 1])  # B1 (fix)
                B0B1St_paths.append(frame_folder[idx-1])  # Bm1 (fix)
                B0B1St_paths.append(frame_folder[idx+2])  # B2 (fix)
                target_t_Idx = frame_folder[idx].split(os.sep)[-1].split('.')[0]+'_' + str(suffix).zfill(3) + '.png'
                # ex) target t name: 00017.png => '00017_1.png'
                B0B1St_paths.append(os.path.join(converted_folder, target_t_Idx))  # St
                B0B1St_paths.append(os.path.join(converted_folder, frame_folder[idx].split(os.sep)[-1]))  # S0
                B0B1St_paths.append(os.path.join(converted_folder, frame_folder[idx+1].split(os.sep)[-1]))  # S1
                B0B1St_paths.append(t[mul]) # t
                B0B1St_paths.append(frame_folder[idx].split(os.path.join(dir, ''))[-1].split(os.sep)[0])  # scene1
                testPath.append(B0B1St_paths)
    return testPath


def frames_loader_sharp_blur_custom_test(args, B0B1Bm1B2_Path):  
    frames = []
    for path in B0B1Bm1B2_Path:
        frame = cv2.imread(path)
        frames.append(frame)
    frames = np.stack(frames, axis=0)  # (T, H, W, 3)

    """ np2Tensor [-1,1] normalized """
    frames = RGBframes_np2Tensor(frames, args.img_ch)

    return frames


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        # self.eps = 1e-6
        self.epsilon = 1e-3

    def forward(self, X, Y):
        # diff = torch.add(X, -Y)
        # error = torch.sqrt(diff * diff + self.eps)
        # loss = torch.sum(error)
        loss = torch.mean(torch.sqrt((X - Y) ** 2 + self.epsilon ** 2))  # from AdaCoF

        return loss


def set_rec_loss(args):
    loss_type = args.loss_type
    if loss_type == 'MSE':
        lossfunction = nn.MSELoss()
    elif loss_type == 'L1':
        lossfunction = nn.L1Loss()
    elif loss_type == 'L1_Charbonnier_loss':
        lossfunction = L1_Charbonnier_loss()

    return lossfunction


def crop_8x8(img):
    ori_h = img.shape[0]
    ori_w = img.shape[1]

    h = (ori_h // 32) * 32
    w = (ori_w // 32) * 32

    while (h > ori_h - 16):
        h = h - 32
    while (w > ori_w - 16):
        w = w - 32

    y = (ori_h - h) // 2
    x = (ori_w - w) // 2
    # crop_img = img[y:y + h, x:x + w]
    # crop_img = img[y:y + h, x:x + w, :]
    # return crop_img, y, x
    return img, y, x


def to_uint8(x, vmin, vmax):
    ##### color space transform, originally from https://github.com/yhjo09/VSR-DUF #####
    x = x.astype('float32')
    x = (x - vmin) / (vmax - vmin) * 255  # 0~255
    return np.clip(np.round(x), 0, 255)


def psnr(img1, img2):
    ##### PSNR from BIN (PRF) #####
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim_matlab_func(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def ssim(img1, img2):
    '''calculate SSIM (matlab)
    the same outputs as MATLAB's
    img1, img2: [0, 255],
    size: [H,W,3]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim_matlab_func(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim_matlab_func(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim_matlab_func(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def denorm255(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1) * 255


def denorm255(x):
    out = (x + 1) / 2
    return torch.clamp(out, 0, 1) * 255


def denorm255_np(x):
    # numpy
    out = (x + 1) / 2
    return out.clip(0, 1) * 255


def set_lr(args, epoch, optimizer):
    lrDecay = args.lr_decay  # parser.add_argument('--lrDecay', type=int, default=0, help='epoch of half lr')
    lr_type = args.lr_type
    if lr_type == 'step':
        epoch_iter = (epoch + 1) // lrDecay
        lr = args.init_lr / (args.lr_decreasing_factor) ** epoch_iter
    elif lr_type == 'exp':
        k = math.log(2) / lrDecay
        lr = args.init_lr * math.exp(-k * epoch)
    elif lr_type == 'inv':
        k = 1 / lrDecay
        lr = args.init_lr / (1 + k * epoch)
    elif lr_type == 'linear_decay':
        lr = args.init_lr if epoch < lrDecay else args.lr * (args.epochs - epoch) / (
                args.epochs - lrDecay)
    elif lr_type == 'no_decay':
        lr = args.init_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def denorm255_01_np(x):
    # numpy
    # out = (x + 1) / 2
    return x.clip(0, 1) * 255


def get_batch_images(args, batch_size, save_images):
    height_num = batch_size
    width_num = (len(save_images)) + 1 + 3 + 1 + 1 + 1
    # S0S1_GT(+1),pred_Flows(+3),pred_Occ_0(+1), diff_maps (+1),
    log_img = np.zeros((height_num * args.patch_size, width_num * args.patch_size, 3), dtype=np.uint8)

    ovlp_B0B1_temp, pred_S0_prime_temp, pred_St_prime_temp, pred_S1_prime_temp, \
    pred_S0_final_temp, pred_St_final_temp, pred_S1_final_temp, \
    St_GT_temp, S0S1_GT_temp, pred_Flows_temp, pred_Occ_0_temp,\
        difference_maps_temp, pred_Flow_t0_t1_temp= save_images

    temp_S0S1_GT_temp = S0S1_GT_temp
    for b in range(height_num):
        ovlp_B0B1 = denorm255(ovlp_B0B1_temp[b, :])
        ovlp_B0B1 = np.transpose((ovlp_B0B1.detach().cpu().numpy()), [1, 2, 0]).astype(np.uint8)
        log_img[(b) * args.patch_size:(b + 1) * args.patch_size, 0 * args.patch_size:1 * args.patch_size,
        :] = ovlp_B0B1

        pred_S0_prime = denorm255(pred_S0_prime_temp[b, :])
        pred_S0_prime = np.transpose(pred_S0_prime.detach().cpu().numpy(), [1, 2, 0]).astype(np.uint8)
        log_img[(b) * args.patch_size:(b + 1) * args.patch_size, 1 * args.patch_size:2 * args.patch_size,
        :] = pred_S0_prime

        pred_St_prime = denorm255(pred_St_prime_temp[b, :])
        pred_St_prime = np.transpose(pred_St_prime.detach().cpu().numpy(), [1, 2, 0]).astype(np.uint8)
        log_img[(b) * args.patch_size:(b + 1) * args.patch_size, 2 * args.patch_size:3 * args.patch_size,
        :] = pred_St_prime

        pred_S1_prime = denorm255(pred_S1_prime_temp[b, :])
        pred_S1_prime = np.transpose(pred_S1_prime.detach().cpu().numpy(), [1, 2, 0]).astype(np.uint8)
        log_img[(b) * args.patch_size:(b + 1) * args.patch_size, 3 * args.patch_size:4 * args.patch_size,
        :] = pred_S1_prime

        pred_S0_final = denorm255(pred_S0_final_temp[b, :])
        pred_S0_final = np.transpose(pred_S0_final.detach().cpu().numpy(), [1, 2, 0]).astype(np.uint8)
        log_img[(b) * args.patch_size:(b + 1) * args.patch_size, 4 * args.patch_size:5 * args.patch_size,
        :] = pred_S0_final

        pred_St_final = denorm255(pred_St_final_temp[b, :])
        pred_St_final = np.transpose(pred_St_final.detach().cpu().numpy(), [1, 2, 0]).astype(np.uint8)
        log_img[(b) * args.patch_size:(b + 1) * args.patch_size, 5 * args.patch_size:6 * args.patch_size,
        :] = pred_St_final

        pred_S1_final = denorm255(pred_S1_final_temp[b, :])
        pred_S1_final = np.transpose(pred_S1_final.detach().cpu().numpy(), [1, 2, 0]).astype(np.uint8)
        log_img[(b) * args.patch_size:(b + 1) * args.patch_size, 6 * args.patch_size:7 * args.patch_size,
        :] = pred_S1_final

        S0S1_GT = denorm255(temp_S0S1_GT_temp[b, :, 0, :])
        S0S1_GT = np.transpose(S0S1_GT.detach().cpu().numpy(), [1, 2, 0]).astype(np.uint8)
        log_img[(b) * args.patch_size:(b + 1) * args.patch_size, 7 * args.patch_size:8 * args.patch_size,
        :] = S0S1_GT

        St_GT = denorm255(St_GT_temp[b, :])
        St_GT = np.transpose(St_GT.detach().cpu().numpy(), [1, 2, 0]).astype(np.uint8)
        log_img[(b) * args.patch_size:(b + 1) * args.patch_size, 8 * args.patch_size:9 * args.patch_size,
        :] = St_GT

        S0S1_GT = denorm255(temp_S0S1_GT_temp[b, :, 1, :])
        S0S1_GT = np.transpose(S0S1_GT.detach().cpu().numpy(), [1, 2, 0]).astype(np.uint8)
        log_img[(b) * args.patch_size:(b + 1) * args.patch_size, 9 * args.patch_size:10 * args.patch_size,
        :] = S0S1_GT

        output_flow_init_t0 = flow2img(np.transpose(pred_Flows_temp[0][b, :2, :, :].detach().cpu().numpy(), [1, 2, 0]))
        log_img[(b) * args.patch_size:(b + 1) * args.patch_size, 10 * args.patch_size:11 * args.patch_size,
        :] = output_flow_init_t0
        output_flow_final_t0 = flow2img(
            np.transpose(pred_Flows_temp[-1][b, :2, :, :].detach().cpu().numpy(), [1, 2, 0]))
        log_img[(b) * args.patch_size:(b + 1) * args.patch_size, 11 * args.patch_size:12 * args.patch_size,
        :] = output_flow_final_t0
        output_flow_init_t1 = flow2img(np.transpose(pred_Flows_temp[0][b, 2:, :, :].detach().cpu().numpy(), [1, 2, 0]))
        log_img[(b) * args.patch_size:(b + 1) * args.patch_size, 12 * args.patch_size:13 * args.patch_size,
        :] = output_flow_init_t1
        output_flow_final_t1 = flow2img(
            np.transpose(pred_Flows_temp[-1][b, 2:, :, :].detach().cpu().numpy(), [1, 2, 0]))
        log_img[(b) * args.patch_size:(b + 1) * args.patch_size, 13 * args.patch_size:14 * args.patch_size,
        :] = output_flow_final_t1

        occ_0_init = np.transpose(pred_Occ_0_temp[0][b, :].detach().cpu().numpy() * 255.0, [1, 2, 0]).astype(np.uint8)
        occ_0_init = np.concatenate([occ_0_init, occ_0_init, occ_0_init], axis=2)
        log_img[(b) * args.patch_size:(b + 1) * args.patch_size, 14 * args.patch_size:15 * args.patch_size,
        :] = occ_0_init
        occ_0_final = np.transpose(pred_Occ_0_temp[-1][b, :].detach().cpu().numpy() * 255.0, [1, 2, 0]).astype(np.uint8)
        occ_0_final = np.concatenate([occ_0_final, occ_0_final, occ_0_final], axis=2)
        log_img[(b) * args.patch_size:(b + 1) * args.patch_size, 15 * args.patch_size:16 * args.patch_size,
        :] = occ_0_final

        

        diff_minus1to0 = np.transpose(difference_maps_temp[0][b, :].detach().cpu().numpy() * 255.0, [1, 2, 0]).astype(np.uint8)
        diff_minus1to0 = np.concatenate([diff_minus1to0, diff_minus1to0, diff_minus1to0], axis=2)
        log_img[(b) * args.patch_size:(b + 1) * args.patch_size, 16 * args.patch_size:17 * args.patch_size,
        :] = diff_minus1to0
        diff_1to0 = np.transpose(difference_maps_temp[2][b, :].detach().cpu().numpy() * 255.0, [1, 2, 0]).astype(
            np.uint8)
        diff_1to0 = np.concatenate([diff_1to0, diff_1to0, diff_1to0], axis=2)
        log_img[(b) * args.patch_size:(b + 1) * args.patch_size, 17 * args.patch_size:18 * args.patch_size,
        :] = diff_1to0 # ["diff_1to0",diff_0to1,"diff_minus1to0",diff_2to1]

        flow_0m1_init = flow2img(np.transpose(pred_Flow_t0_t1_temp[0][0][b, :, :, :].detach().cpu().numpy(), [1, 2, 0]))
        log_img[(b) * args.patch_size:(b + 1) * args.patch_size, 18 * args.patch_size:19 * args.patch_size,
        :] = flow_0m1_init

        flow_01_init = flow2img(
            np.transpose(pred_Flow_t0_t1_temp[0][1][b, :, :, :].detach().cpu().numpy(), [1, 2, 0]))
        log_img[(b) * args.patch_size:(b + 1) * args.patch_size, 19 * args.patch_size:20 * args.patch_size,
        :] = flow_01_init


    return log_img


def visualizations(two_blurry_inputs_full,
                             Sharps_prime_t, Sharps_final_t, St_GT_full, flows_pred, occs_pred,
                             blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1):
    # h = save_size[1]
    # w = save_size[0]
    _, h, w = two_blurry_inputs_full.shape
    height_num = 3  # 1to0, 0to1
    width_num = 1 + 1 + 1 + 1 + 2 + 2  # 8
    width_num += 2  # diff_btw_GT
    log_img = np.zeros((height_num * h, width_num * w, 3), dtype=np.uint8)
    
    """ b = 0 """
    b = 0
    temp_img = np.transpose(denorm255_np(two_blurry_inputs_full), [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 0 * w: 1 * w,
    :] = temp_img

    temp_img = np.transpose(denorm255_np(Sharps_prime_t), [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 1 * w: 2 * w,
    :] = temp_img

    temp_img = np.transpose(denorm255_np(Sharps_final_t), [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 2 * w: 3 * w,
    :] = temp_img
    
    
    temp_img = np.transpose(denorm255_np(St_GT_full), [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 3 * w: 4 * w,
    :] = temp_img

    temp_img = flow2img(np.transpose(flows_pred[0][0], [1, 2, 0]))
    log_img[(b) * h:(b + 1) * h, 4 * w: 5 * w,
    :] = temp_img

    temp_img = flow2img(np.transpose(flows_pred[0][1], [1, 2, 0]))
    log_img[(b) * h:(b + 1) * h, 5 * w: 6 * w,
    :] = temp_img

    temp_img = np.transpose(occs_pred[0] * 255.0, [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 6 * w: 7 * w,
    :] = temp_img

    temp_img = np.transpose(occs_pred[1] * 255.0, [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 7 * w: 8 * w,
    :] = temp_img
    
    diff_prime = np.mean(np.abs(Sharps_prime_t - St_GT_full), axis=0, keepdims=True)

    temp_img = np.transpose(denorm255_01_np(diff_prime), [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 8 * w: 9 * w,
    :] = temp_img

    diff_sharp = np.mean(np.abs(Sharps_final_t - St_GT_full), axis=0, keepdims=True)
    temp_img = np.transpose(denorm255_01_np(diff_sharp), [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 9 * w: 10 * w,
    :] = temp_img

    """ b = 1 (1to0) """
    b = 1
    temp_img = np.transpose(
        denorm255_01_np(blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[0][0:1, :, :]), [1, 2, 0]).astype(
        np.uint8)
    log_img[(b) * h:(b + 1) * h, 0 * w: 1 * w,
    :] = temp_img

    temp_img = np.transpose(
        denorm255_01_np(blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[0][1:2, :, :]), [1, 2, 0]).astype(
        np.uint8)
    log_img[(b) * h:(b + 1) * h, 1 * w: 2 * w,
    :] = temp_img

    temp_img = np.transpose(
        denorm255_01_np(blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[0][2:3, :, :]), [1, 2, 0]).astype(
        np.uint8)
    log_img[(b) * h:(b + 1) * h, 2 * w: 3 * w,
    :] = temp_img

    temp_img = np.transpose(
        denorm255_01_np(blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[0][3:4, :, :]), [1, 2, 0]).astype(
        np.uint8)
    log_img[(b) * h:(b + 1) * h, 3 * w: 4 * w,
    :] = temp_img

    temp_img = np.transpose(
        denorm255_01_np(blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[0][4:5, :, :]),
        [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 4 * w: 5 * w,
    :] = temp_img

    temp_img = flow2img(np.transpose((blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[0][5:7, :, :]),
                                     [1, 2, 0]))
    log_img[(b) * h:(b + 1) * h, 5 * w: 6 * w,
    :] = temp_img

    temp_img = np.transpose(
        denorm255_01_np(blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[0][7:8, :, :]),
        [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 6 * w: 7 * w,
    :] = temp_img

    temp_img = np.transpose(
        denorm255_01_np(blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[0][8:9, :, :]),
        [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 7 * w: 8 * w,
    :] = temp_img

    diff_prime = np.mean(diff_prime, 0, keepdims=True)
    c, h, w = diff_prime.shape
    diff_prime = np.reshape(diff_prime, (1, -1))
    diff_prime -= diff_prime.min(1, keepdims=True)[0]
    diff_prime /= diff_prime.max(1, keepdims=True)[0]
    diff_prime = np.reshape(diff_prime, (1, h, w))

    temp_img = np.transpose(denorm255_01_np(diff_prime),
                            [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 8 * w: 9 * w,
    :] = temp_img

    diff_sharp = np.mean(diff_sharp, 0, keepdims=True)
    c, h, w = diff_sharp.shape
    diff_sharp = np.reshape(diff_sharp, (1, -1))
    diff_sharp -= diff_sharp.min(1, keepdims=True)[0]
    diff_sharp /= diff_sharp.max(1, keepdims=True)[0]
    diff_sharp = np.reshape(diff_sharp, (1, h, w))

    temp_img = np.transpose(denorm255_01_np(diff_sharp),
                            [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 9 * w: 10 * w,
    :] = temp_img

    """ b = 2 (0to1) """
    b = 2
    temp_img = np.transpose(
        denorm255_01_np(blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[1][0:1, :, :]),
        [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 0 * w: 1 * w,
    :] = temp_img

    temp_img = np.transpose(
        denorm255_01_np(blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[1][1:2, :, :]),
        [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 1 * w: 2 * w,
    :] = temp_img

    temp_img = np.transpose(
        denorm255_01_np(blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[1][2:3, :, :]),
        [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 2 * w: 3 * w,
    :] = temp_img

    temp_img = np.transpose(
        denorm255_01_np(blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[1][3:4, :, :]),
        [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 3 * w: 4 * w,
    :] = temp_img

    temp_img = np.transpose(
        denorm255_01_np(blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[1][4:5, :, :]),
        [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 4 * w: 5 * w,
    :] = temp_img

    temp_img = flow2img(np.transpose((blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[1][5:7, :, :]),
                                     [1, 2, 0]))
    log_img[(b) * h:(b + 1) * h, 5 * w: 6 * w,
    :] = temp_img

    temp_img = np.transpose(
        denorm255_01_np(blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[1][7:8, :, :]),
        [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 6 * w: 7 * w,
    :] = temp_img

    temp_img = np.transpose(
        denorm255_01_np(blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[1][8:9, :, :]),
        [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 7 * w: 8 * w,
    :] = temp_img
    return log_img


def visualizations_custom(two_blurry_inputs_full,
                   Sharps_prime_t, Sharps_final_t, St_GT_full, flows_pred, occs_pred,
                   blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1):
    # h = save_size[1]
    # w = save_size[0]
    _, h, w = two_blurry_inputs_full.shape
    height_num = 3  # 1to0, 0to1
    width_num = 1 + 1 + 1 + 1 + 2 + 2  # 8
    width_num += 2  # diff_btw_GT
    log_img = np.zeros((height_num * h, width_num * w, 3), dtype=np.uint8)

    """ b = 0 """
    b = 0
    temp_img = np.transpose(denorm255_np(two_blurry_inputs_full), [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 0 * w: 1 * w,
    :] = temp_img

    temp_img = np.transpose(denorm255_np(Sharps_prime_t), [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 1 * w: 2 * w,
    :] = temp_img

    temp_img = np.transpose(denorm255_np(Sharps_final_t), [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 2 * w: 3 * w,
    :] = temp_img

    if not St_GT_full == None:
        temp_img = np.transpose(denorm255_np(St_GT_full), [1, 2, 0]).astype(np.uint8)
        log_img[(b) * h:(b + 1) * h, 3 * w: 4 * w,
        :] = temp_img

    temp_img = flow2img(np.transpose(flows_pred[0][0], [1, 2, 0]))
    log_img[(b) * h:(b + 1) * h, 4 * w: 5 * w,
    :] = temp_img

    temp_img = flow2img(np.transpose(flows_pred[0][1], [1, 2, 0]))
    log_img[(b) * h:(b + 1) * h, 5 * w: 6 * w,
    :] = temp_img

    temp_img = np.transpose(occs_pred[0] * 255.0, [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 6 * w: 7 * w,
    :] = temp_img

    temp_img = np.transpose(occs_pred[1] * 255.0, [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 7 * w: 8 * w,
    :] = temp_img

    if not St_GT_full == None:
        diff_prime = np.mean(np.abs(Sharps_prime_t - St_GT_full), axis=0, keepdims=True)

        temp_img = np.transpose(denorm255_01_np(diff_prime), [1, 2, 0]).astype(np.uint8)
        log_img[(b) * h:(b + 1) * h, 8 * w: 9 * w,
        :] = temp_img

        diff_sharp = np.mean(np.abs(Sharps_final_t - St_GT_full), axis=0, keepdims=True)
        temp_img = np.transpose(denorm255_01_np(diff_sharp), [1, 2, 0]).astype(np.uint8)
        log_img[(b) * h:(b + 1) * h, 9 * w: 10 * w,
        :] = temp_img

    """ b = 1 (1to0) """
    b = 1
    temp_img = np.transpose(
        denorm255_01_np(blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[0][0:1, :, :]), [1, 2, 0]).astype(
        np.uint8)
    log_img[(b) * h:(b + 1) * h, 0 * w: 1 * w,
    :] = temp_img

    temp_img = np.transpose(
        denorm255_01_np(blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[0][1:2, :, :]), [1, 2, 0]).astype(
        np.uint8)
    log_img[(b) * h:(b + 1) * h, 1 * w: 2 * w,
    :] = temp_img

    temp_img = np.transpose(
        denorm255_01_np(blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[0][2:3, :, :]), [1, 2, 0]).astype(
        np.uint8)
    log_img[(b) * h:(b + 1) * h, 2 * w: 3 * w,
    :] = temp_img

    temp_img = np.transpose(
        denorm255_01_np(blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[0][3:4, :, :]), [1, 2, 0]).astype(
        np.uint8)
    log_img[(b) * h:(b + 1) * h, 3 * w: 4 * w,
    :] = temp_img

    temp_img = np.transpose(
        denorm255_01_np(blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[0][4:5, :, :]),
        [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 4 * w: 5 * w,
    :] = temp_img

    temp_img = flow2img(np.transpose((blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[0][5:7, :, :]),
                                     [1, 2, 0]))
    log_img[(b) * h:(b + 1) * h, 5 * w: 6 * w,
    :] = temp_img

    temp_img = np.transpose(
        denorm255_01_np(blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[0][7:8, :, :]),
        [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 6 * w: 7 * w,
    :] = temp_img

    temp_img = np.transpose(
        denorm255_01_np(blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[0][8:9, :, :]),
        [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 7 * w: 8 * w,
    :] = temp_img

    if not St_GT_full == None:
        diff_prime = np.mean(diff_prime, 0, keepdims=True)
        c, h, w = diff_prime.shape
        diff_prime = np.reshape(diff_prime, (1, -1))
        diff_prime -= diff_prime.min(1, keepdims=True)[0]
        diff_prime /= diff_prime.max(1, keepdims=True)[0]
        diff_prime = np.reshape(diff_prime, (1, h, w))

        temp_img = np.transpose(denorm255_01_np(diff_prime),
                                [1, 2, 0]).astype(np.uint8)
        log_img[(b) * h:(b + 1) * h, 8 * w: 9 * w,
        :] = temp_img

        diff_sharp = np.mean(diff_sharp, 0, keepdims=True)
        c, h, w = diff_sharp.shape
        diff_sharp = np.reshape(diff_sharp, (1, -1))
        diff_sharp -= diff_sharp.min(1, keepdims=True)[0]
        diff_sharp /= diff_sharp.max(1, keepdims=True)[0]
        diff_sharp = np.reshape(diff_sharp, (1, h, w))

        temp_img = np.transpose(denorm255_01_np(diff_sharp),
                                [1, 2, 0]).astype(np.uint8)
        log_img[(b) * h:(b + 1) * h, 9 * w: 10 * w,
        :] = temp_img

    """ b = 2 (0to1) """
    b = 2
    temp_img = np.transpose(
        denorm255_01_np(blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[1][0:1, :, :]),
        [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 0 * w: 1 * w,
    :] = temp_img

    temp_img = np.transpose(
        denorm255_01_np(blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[1][1:2, :, :]),
        [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 1 * w: 2 * w,
    :] = temp_img

    temp_img = np.transpose(
        denorm255_01_np(blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[1][2:3, :, :]),
        [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 2 * w: 3 * w,
    :] = temp_img

    temp_img = np.transpose(
        denorm255_01_np(blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[1][3:4, :, :]),
        [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 3 * w: 4 * w,
    :] = temp_img

    temp_img = np.transpose(
        denorm255_01_np(blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[1][4:5, :, :]),
        [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 4 * w: 5 * w,
    :] = temp_img

    temp_img = flow2img(np.transpose((blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[1][5:7, :, :]),
                                     [1, 2, 0]))
    log_img[(b) * h:(b + 1) * h, 5 * w: 6 * w,
    :] = temp_img

    temp_img = np.transpose(
        denorm255_01_np(blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[1][7:8, :, :]),
        [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 6 * w: 7 * w,
    :] = temp_img

    temp_img = np.transpose(
        denorm255_01_np(blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1[1][8:9, :, :]),
        [1, 2, 0]).astype(np.uint8)
    log_img[(b) * h:(b + 1) * h, 7 * w: 8 * w,
    :] = temp_img
    return log_img


def flow2img(flow, logscale=True, scaledown=6, output=False):
    """
	topleft is zero, u is horiz, v is vertical
	red is 3 o'clock, yellow is 6, light blue is 9, blue/purple is 12
	"""
    u = flow[:, :, 1]
    # u = flow[:, :, 0]
    v = flow[:, :, 0]
    # v = flow[:, :, 1]

    colorwheel = makecolorwheel()
    ncols = colorwheel.shape[0]

    radius = np.sqrt(u ** 2 + v ** 2)
    if output:
        print("Maximum flow magnitude: %04f" % np.max(radius))
    if logscale:
        radius = np.log(radius + 1)
        if output:
            print("Maximum flow magnitude (after log): %0.4f" % np.max(radius))
    radius = radius / scaledown
    if output:
        print("Maximum flow magnitude (after scaledown): %0.4f" % np.max(radius))
    # rot = np.arctan2(-v, -u) / np.pi
    rot = np.arctan2(v, u) / np.pi

    fk = (rot + 1) / 2 * (ncols - 1)  # -1~1 maped to 0~ncols
    k0 = fk.astype(np.uint8)  # 0, 1, 2, ..., ncols

    k1 = k0 + 1
    k1[k1 == ncols] = 0

    f = fk - k0

    ncolors = colorwheel.shape[1]
    img = np.zeros(u.shape + (ncolors,))
    for i in range(ncolors):
        tmp = colorwheel[:, i]
        col0 = tmp[k0]
        col1 = tmp[k1]
        col = (1 - f) * col0 + f * col1

        idx = radius <= 1
        # increase saturation with radius
        col[idx] = 1 - radius[idx] * (1 - col[idx])
        # out of range
        col[~idx] *= 0.75
        # img[:,:,i] = np.floor(255*col).astype(np.uint8)

        img[:, :, i] = np.clip(255 * col, 0.0, 255.0).astype(np.uint8)

    # return img.astype(np.uint8)
    return img


def makecolorwheel():
    # Create a colorwheel for visualization
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros((ncols, 3))

    col = 0
    # RY
    colorwheel[col:col + RY, 0] = 1
    colorwheel[col:col + RY, 1] = np.arange(0, 1, 1. / RY)
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = np.arange(1, 0, -1. / YG)
    colorwheel[col:col + YG, 1] = 1
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 1
    colorwheel[col:col + GC, 2] = np.arange(0, 1, 1. / GC)
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = np.arange(1, 0, -1. / CB)
    colorwheel[col:col + CB, 2] = 1
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 1
    colorwheel[col:col + BM, 0] = np.arange(0, 1, 1. / BM)
    col += BM

    # MR
    colorwheel[col:col + MR, 2] = np.arange(1, 0, -1. / MR)
    colorwheel[col:col + MR, 0] = 1

    return colorwheel


def patch_forward_DeFInet_itr(model_net, input_frames, frameT, t_value, num_update, patch, patch_boundary=0):
    # Shape of 'input_frames' : [1,C,T,H,W]
    """ considering patch & patch_boundary (with a wider margin)"""
    num_patch = patch  # due to memory capacity, we divide the whole image into small patches.
    patch_boundary = patch_boundary  # multiple of 32

    # crop img for 32 multiple (when using U-net)
    # h, w = args.test_input_size
    ori_B, ori_C, ori_T, ori_h, ori_w = input_frames.shape

    c = 3
    # for multiple of patch_boundary
    if not patch_boundary == 0:
        h_fill_remainder = (patch_boundary - (ori_h % patch_boundary))
        w_fill_remainder = (patch_boundary - (ori_w % patch_boundary))
        # h_fill_remainder = (patch_boundary - (ori_H % patch_boundary)) // 2
        # w_fill_remainder = (patch_boundary - (ori_W % patch_boundary)) // 2
        if h_fill_remainder == patch_boundary:
            h_fill_remainder = 0
        if w_fill_remainder == patch_boundary:
            w_fill_remainder = 0
        input_frames = input_frames.contiguous().view(ori_B, -1, ori_h,
                                                      ori_w)  # for reflection (it cannot work on 5D tensor)
        # input_frames = F.pad(input_frames,pad=[0,w_fill_remainder,0,h_fill_remainder]) # for DAVIS (can not use 'reflect')
        input_frames = F.pad(input_frames, pad=[0, w_fill_remainder, 0, h_fill_remainder], mode="reflect")
        input_frames = input_frames.view(ori_B, ori_C, ori_T, ori_h + h_fill_remainder,
                                         ori_w + w_fill_remainder).contiguous()
    # Pad to multiples of patch_boundary (now divisible), only add to "right, bottom"

    # input_frames = input_frames[:, :, :, :h, :w]  # now, it is divided by 32 with no remainder.
    # frameT = frameT[:, :, :h, :w]
    _, _, _, new_h, new_w = input_frames.shape

    S0_prime_full = np.zeros((c, new_h, new_w))
    S1_prime_full = np.zeros((c, new_h, new_w))
    St_prime_full = np.zeros((c, new_h, new_w))

    S0_final_full = np.zeros((c, new_h, new_w))
    S1_final_full = np.zeros((c, new_h, new_w))
    St_final_full = np.zeros((c, new_h, new_w))

    ft0_init_full = np.zeros((2, new_h, new_w))
    ft0_final_full = np.zeros((2, new_h, new_w))
    ft1_init_full = np.zeros((2, new_h, new_w))
    ft1_final_full = np.zeros((2, new_h, new_w))
    occ0_init_full = np.zeros((1, new_h, new_w))
    occ0_final_full = np.zeros((1, new_h, new_w))
    two_blurry_inputs_full = np.zeros((3, new_h, new_w))

    """ Divide & Process due to Limited Memory """
    for p in range(num_patch[0] * num_patch[1]):
        pH = p // num_patch[1]  # patch index (priority: w=>h)
        pW = p % num_patch[1]  # patch index
        sH = new_h // num_patch[0]  # patch size
        sW = new_w // num_patch[1]  # patch size

        # process data considering patch boundary
        H_low_ind, H_high_ind, W_low_ind, W_high_ind, add_H, add_W = \
            get_HW_boundary(patch_boundary, new_h, new_w, pH, sH, pW, sW)
        # compute output
        # _, test_Pred_patch, _, _, _ = model_net(input_frames[:, :, :, H_low_ind:H_high_ind, W_low_ind:W_high_ind],
        #                                         t_value, num_update)  # check for warm start and is_training
        Sharps_prime, Sharps_final, flow_predictions, occ0_predictions, two_blurry_inputs = model_net(input_frames[:, :, :, H_low_ind:H_high_ind, W_low_ind:W_high_ind],
                                                t_value, num_update)  # check for warm start and is_training

        # trim patch boundary
        Sharps_prime_trim = trim_patch_boundary(Sharps_prime, patch_boundary, new_h, new_w, pH, sH, pW,
                                                sW,
                                                sf=1)
        Sharps_final_trim = trim_patch_boundary(Sharps_final, patch_boundary, new_h, new_w, pH, sH, pW,
                                                sW,
                                                sf=1)
        flow_predictions_trim = trim_patch_boundary(flow_predictions, patch_boundary, new_h, new_w, pH, sH, pW,
                                             sW,
                                             sf=1)
        occ0_predictions_trim = trim_patch_boundary(occ0_predictions, patch_boundary, new_h, new_w, pH, sH, pW,
                                                sW,
                                                sf=1)
        two_blurry_inputs_trim = trim_patch_boundary(two_blurry_inputs, patch_boundary, new_h, new_w, pH, sH, pW,
                                                    sW,
                                                    sf=1)

        # store in pred_full
        S0_prime_full[:, pH * sH: (pH + 1) * sH, pW * sW: (pW + 1) * sW] = np.squeeze(
            Sharps_prime_trim[0].detach().cpu().numpy())
        S1_prime_full[:, pH * sH: (pH + 1) * sH, pW * sW: (pW + 1) * sW] = np.squeeze(
            Sharps_prime_trim[1].detach().cpu().numpy())
        St_prime_full[:, pH * sH: (pH + 1) * sH, pW * sW: (pW + 1) * sW] = np.squeeze(
            Sharps_prime_trim[-1].detach().cpu().numpy())

        S0_final_full[:, pH * sH: (pH + 1) * sH, pW * sW: (pW + 1) * sW] = np.squeeze(
            Sharps_final_trim[-1][0].detach().cpu().numpy()) # considering "list" of final
        S1_final_full[:, pH * sH: (pH + 1) * sH, pW * sW: (pW + 1) * sW] = np.squeeze(
            Sharps_final_trim[-1][1].detach().cpu().numpy())
        St_final_full[:, pH * sH: (pH + 1) * sH, pW * sW: (pW + 1) * sW] = np.squeeze(
            Sharps_final_trim[-1][-1].detach().cpu().numpy())

        ft0_init_full[:, pH * sH: (pH + 1) * sH, pW * sW: (pW + 1) * sW] = np.squeeze(
            flow_predictions_trim[0][:,:2,:,:].detach().cpu().numpy())
        ft0_final_full[:, pH * sH: (pH + 1) * sH, pW * sW: (pW + 1) * sW] = np.squeeze(
            flow_predictions_trim[-1][:,:2,:,:].detach().cpu().numpy())
        ft1_init_full[:, pH * sH: (pH + 1) * sH, pW * sW: (pW + 1) * sW] = np.squeeze(
            flow_predictions_trim[0][:,2:,:,:].detach().cpu().numpy())
        ft1_final_full[:, pH * sH: (pH + 1) * sH, pW * sW: (pW + 1) * sW] = np.squeeze(
            flow_predictions_trim[-1][:,2:,:,:].detach().cpu().numpy())
        occ0_init_full[:, pH * sH: (pH + 1) * sH, pW * sW: (pW + 1) * sW] = np.squeeze(
            occ0_predictions_trim[0].detach().cpu().numpy())
        occ0_final_full[:, pH * sH: (pH + 1) * sH, pW * sW: (pW + 1) * sW] = np.squeeze(
            occ0_predictions_trim[-1].detach().cpu().numpy())
        two_blurry_inputs_full[:, pH * sH: (pH + 1) * sH, pW * sW: (pW + 1) * sW] = np.squeeze(
            two_blurry_inputs_trim.detach().cpu().numpy())


    S0_prime_full = S0_prime_full[:, :ori_h, :ori_w]
    S1_prime_full = S1_prime_full[:, :ori_h, :ori_w]
    St_prime_full = St_prime_full[:, :ori_h, :ori_w]
    Sharps_prime = [S0_prime_full, S1_prime_full, St_prime_full]


    S0_final_full = S0_final_full[:, :ori_h, :ori_w]
    S1_final_full = S1_final_full[:, :ori_h, :ori_w]
    St_final_full = St_final_full[:, :ori_h, :ori_w]
    Sharps_final = [S0_final_full,S1_final_full,St_final_full]
    
    if not frameT==None:
        St_GT_full = np.squeeze(frameT)  # [c,h,w]
    else:
        St_GT_full = 0

    ft0_init_full = ft0_init_full[:, :ori_h, :ori_w]
    ft0_final_full = ft0_final_full[:, :ori_h, :ori_w]
    ft1_init_full = ft1_init_full[:, :ori_h, :ori_w]
    ft1_final_full = ft1_final_full[:, :ori_h, :ori_w]
    occ0_init_full = occ0_init_full[:, :ori_h, :ori_w]
    occ0_final_full = occ0_final_full[:, :ori_h, :ori_w]
    two_blurry_inputs_full = two_blurry_inputs_full[:, :ori_h, :ori_w]
    flows_pred = [[ft0_init_full,ft0_final_full],[ft1_init_full,ft1_final_full]]
    occs_pred = [occ0_init_full,occ0_final_full]
    return two_blurry_inputs_full, Sharps_prime, Sharps_final, St_GT_full, flows_pred, occs_pred


def patch_forward_DeFInet_w_diff(model_net, input_frames, frameT, t_value, num_update, patch, patch_boundary=0):
    # Shape of 'input_frames' : [1,C,T,H,W]
    """ considering patch & patch_boundary (with a wider margin)"""
    num_patch = patch  # due to memory capacity, we divide the whole image into small patches.
    patch_boundary = patch_boundary  # multiple of 32

    # crop img for 32 multiple (when using U-net)
    # h, w = args.test_input_size
    ori_B, ori_C, ori_T, ori_h, ori_w = input_frames.shape

    c = 3
    # for multiple of patch_boundary
    if not patch_boundary == 0:
        h_fill_remainder = (patch_boundary - (ori_h % patch_boundary))
        w_fill_remainder = (patch_boundary - (ori_w % patch_boundary))
        # h_fill_remainder = (patch_boundary - (ori_H % patch_boundary)) // 2
        # w_fill_remainder = (patch_boundary - (ori_W % patch_boundary)) // 2
        if h_fill_remainder == patch_boundary:
            h_fill_remainder = 0
        if w_fill_remainder == patch_boundary:
            w_fill_remainder = 0
        input_frames = input_frames.contiguous().view(ori_B, -1, ori_h,
                                                      ori_w)  # for reflection (it cannot work on 5D tensor)
        # input_frames = F.pad(input_frames,pad=[0,w_fill_remainder,0,h_fill_remainder]) # for DAVIS (can not use 'reflect')
        input_frames = F.pad(input_frames, pad=[0, w_fill_remainder, 0, h_fill_remainder], mode="reflect")
        input_frames = input_frames.view(ori_B, ori_C, ori_T, ori_h + h_fill_remainder,
                                         ori_w + w_fill_remainder).contiguous()
    # Pad to multiples of patch_boundary (now divisible), only add to "right, bottom"

    # input_frames = input_frames[:, :, :, :h, :w]  # now, it is divided by 32 with no remainder.
    # frameT = frameT[:, :, :h, :w]
    _, _, _, new_h, new_w = input_frames.shape

    S0_prime_full = np.zeros((c, new_h, new_w))
    S1_prime_full = np.zeros((c, new_h, new_w))
    St_prime_full = np.zeros((c, new_h, new_w))

    S0_final_full = np.zeros((c, new_h, new_w))
    S1_final_full = np.zeros((c, new_h, new_w))
    St_final_full = np.zeros((c, new_h, new_w))

    ft0_init_full = np.zeros((2, new_h, new_w))
    ft0_final_full = np.zeros((2, new_h, new_w))
    ft1_init_full = np.zeros((2, new_h, new_w))
    ft1_final_full = np.zeros((2, new_h, new_w))
    occ0_init_full = np.zeros((1, new_h, new_w))
    occ0_final_full = np.zeros((1, new_h, new_w))
    two_blurry_inputs_full = np.zeros((3, new_h, new_w))

    blws_blww_s1to0_r1to0_w1to0_flow01_diff1to0_FCW = np.zeros((1 + 1 + 1 + 1 + 1 + 2 + 1 + 1, new_h, new_w))
    blws_blww_s0to1_r0to1_w0to1_flow10_diff0to1_FCW = np.zeros((1 + 1 + 1 + 1 + 1 + 2 + 1 + 1, new_h, new_w))

    """ Divide & Process due to Limited Memory """
    for p in range(num_patch[0] * num_patch[1]):
        pH = p // num_patch[1]  # patch index (priority: w=>h)
        pW = p % num_patch[1]  # patch index
        sH = new_h // num_patch[0]  # patch size
        sW = new_w // num_patch[1]  # patch size

        # process data considering patch boundary
        H_low_ind, H_high_ind, W_low_ind, W_high_ind, add_H, add_W = \
            get_HW_boundary(patch_boundary, new_h, new_w, pH, sH, pW, sW)
        # compute output
        # _, test_Pred_patch, _, _, _ = model_net(input_frames[:, :, :, H_low_ind:H_high_ind, W_low_ind:W_high_ind],
        #                                         t_value, num_update)  # check for warm start and is_training
        Sharps_prime, Sharps_final, flow_predictions, occ0_predictions, two_blurry_inputs, \
        blending_weights, difference_maps = model_net(
            input_frames[:, :, :, H_low_ind:H_high_ind, W_low_ind:W_high_ind],
            t_value, num_update)  # check for warm start and is_training
        #  blending_weights =
        #  [[blending_weight,(1 - blending_weight),source_v,init_ref_k,final_warped_r2s,final_y]x4,[flow_01,flow_10]]
        #  difference_maps = [diff_1to0,diff_0to1,diff_1to0,diff_0to1]

        # trim patch boundary

        Sharps_prime_trim = trim_patch_boundary(Sharps_prime, patch_boundary, new_h, new_w, pH, sH, pW,
                                                sW,
                                                sf=1)
        Sharps_final_trim = trim_patch_boundary(Sharps_final, patch_boundary, new_h, new_w, pH, sH, pW,
                                                sW,
                                                sf=1)
        flow_predictions_trim = trim_patch_boundary(flow_predictions, patch_boundary, new_h, new_w, pH, sH, pW,
                                                    sW,
                                                    sf=1)
        occ0_predictions_trim = trim_patch_boundary(occ0_predictions, patch_boundary, new_h, new_w, pH, sH, pW,
                                                    sW,
                                                    sf=1)
        two_blurry_inputs_trim = trim_patch_boundary(two_blurry_inputs, patch_boundary, new_h, new_w, pH, sH, pW,
                                                     sW,
                                                     sf=1)
        """ 1to0"""
        blending_weights_s_1to0_trim = trim_patch_boundary(blending_weights[0][0], patch_boundary, new_h, new_w, pH, sH,
                                                           pW,
                                                           sW,
                                                           sf=1)

        blending_weights_w_1to0_trim = trim_patch_boundary(blending_weights[0][1], patch_boundary, new_h, new_w, pH, sH,
                                                           pW,
                                                           sW,
                                                           sf=1)
        source_v_1to0_trim = trim_patch_boundary(blending_weights[0][2], patch_boundary, new_h, new_w, pH, sH,
                                                 pW,
                                                 sW,
                                                 sf=1)
        init_ref_k_1to0_trim = trim_patch_boundary(blending_weights[0][3], patch_boundary, new_h, new_w, pH, sH,
                                                   pW,
                                                   sW,
                                                   sf=1)
        final_warped_r2s_1to0_trim = trim_patch_boundary(blending_weights[0][4], patch_boundary, new_h, new_w, pH, sH,
                                                         pW,
                                                         sW,
                                                         sf=1)

        FCW_flow_01_trim = trim_patch_boundary(blending_weights[-1][0], patch_boundary, new_h, new_w, pH, sH,
                                               pW,
                                               sW,
                                               sf=1)
        diff_1to0_trim = trim_patch_boundary(difference_maps[0], patch_boundary, new_h, new_w, pH, sH, pW,
                                             sW,
                                             sf=1)
        FCW_1to0_trim = trim_patch_boundary(blending_weights[0][5], patch_boundary, new_h, new_w, pH, sH,
                                            pW,
                                            sW,
                                            sf=1)
        """ 0to1"""
        blending_weights_s_0to1_trim = trim_patch_boundary(blending_weights[1][0], patch_boundary, new_h, new_w, pH, sH,
                                                           pW,
                                                           sW,
                                                           sf=1)

        blending_weights_w_0to1_trim = trim_patch_boundary(blending_weights[1][1], patch_boundary, new_h, new_w, pH, sH,
                                                           pW,
                                                           sW,
                                                           sf=1)
        source_v_0to1_trim = trim_patch_boundary(blending_weights[1][2], patch_boundary, new_h, new_w, pH, sH,
                                                 pW,
                                                 sW,
                                                 sf=1)
        init_ref_k_0to1_trim = trim_patch_boundary(blending_weights[1][3], patch_boundary, new_h, new_w, pH, sH,
                                                   pW,
                                                   sW,
                                                   sf=1)
        final_warped_r2s_0to1_trim = trim_patch_boundary(blending_weights[1][4], patch_boundary, new_h, new_w, pH, sH,
                                                         pW,
                                                         sW,
                                                         sf=1)
        FCW_flow_10_trim = trim_patch_boundary(blending_weights[-1][1], patch_boundary, new_h, new_w, pH, sH,
                                               pW,
                                               sW,
                                               sf=1)
        diff_0to1_trim = trim_patch_boundary(difference_maps[1], patch_boundary, new_h, new_w, pH, sH, pW,
                                             sW,
                                             sf=1)
        FCW_0to1_trim = trim_patch_boundary(blending_weights[1][5], patch_boundary, new_h, new_w, pH, sH,
                                            pW,
                                            sW,
                                            sf=1)

        # store in pred_full
        S0_prime_full[:, pH * sH: (pH + 1) * sH, pW * sW: (pW + 1) * sW] = np.squeeze(
            Sharps_prime_trim[0].detach().cpu().numpy())
        S1_prime_full[:, pH * sH: (pH + 1) * sH, pW * sW: (pW + 1) * sW] = np.squeeze(
            Sharps_prime_trim[1].detach().cpu().numpy())
        St_prime_full[:, pH * sH: (pH + 1) * sH, pW * sW: (pW + 1) * sW] = np.squeeze(
            Sharps_prime_trim[-1].detach().cpu().numpy())

        S0_final_full[:, pH * sH: (pH + 1) * sH, pW * sW: (pW + 1) * sW] = np.squeeze(
            Sharps_final_trim[-1][0].detach().cpu().numpy())  # considering "list" of final
        S1_final_full[:, pH * sH: (pH + 1) * sH, pW * sW: (pW + 1) * sW] = np.squeeze(
            Sharps_final_trim[-1][1].detach().cpu().numpy())
        St_final_full[:, pH * sH: (pH + 1) * sH, pW * sW: (pW + 1) * sW] = np.squeeze(
            Sharps_final_trim[-1][-1].detach().cpu().numpy())

        ft0_init_full[:, pH * sH: (pH + 1) * sH, pW * sW: (pW + 1) * sW] = np.squeeze(
            flow_predictions_trim[0][:, :2, :, :].detach().cpu().numpy())
        ft0_final_full[:, pH * sH: (pH + 1) * sH, pW * sW: (pW + 1) * sW] = np.squeeze(
            flow_predictions_trim[-1][:, :2, :, :].detach().cpu().numpy())
        ft1_init_full[:, pH * sH: (pH + 1) * sH, pW * sW: (pW + 1) * sW] = np.squeeze(
            flow_predictions_trim[0][:, 2:, :, :].detach().cpu().numpy())
        ft1_final_full[:, pH * sH: (pH + 1) * sH, pW * sW: (pW + 1) * sW] = np.squeeze(
            flow_predictions_trim[-1][:, 2:, :, :].detach().cpu().numpy())
        occ0_init_full[:, pH * sH: (pH + 1) * sH, pW * sW: (pW + 1) * sW] = np.squeeze(
            occ0_predictions_trim[0].detach().cpu().numpy())
        occ0_final_full[:, pH * sH: (pH + 1) * sH, pW * sW: (pW + 1) * sW] = np.squeeze(
            occ0_predictions_trim[-1].detach().cpu().numpy())
        two_blurry_inputs_full[:, pH * sH: (pH + 1) * sH, pW * sW: (pW + 1) * sW] = np.squeeze(
            two_blurry_inputs_trim.detach().cpu().numpy())

        """ 1to0"""
        blws_blww_s1to0_r1to0_w1to0_flow01_diff1to0_FCW[0:1, pH * sH: (pH + 1) * sH,
        pW * sW: (pW + 1) * sW] = np.squeeze(
            blending_weights_s_1to0_trim.detach().cpu().numpy())
        blws_blww_s1to0_r1to0_w1to0_flow01_diff1to0_FCW[1:2, pH * sH: (pH + 1) * sH,
        pW * sW: (pW + 1) * sW] = np.squeeze(
            blending_weights_w_1to0_trim.detach().cpu().numpy())
        blws_blww_s1to0_r1to0_w1to0_flow01_diff1to0_FCW[2:3, pH * sH: (pH + 1) * sH,
        pW * sW: (pW + 1) * sW] = np.squeeze(
            source_v_1to0_trim.detach().cpu().numpy())
        blws_blww_s1to0_r1to0_w1to0_flow01_diff1to0_FCW[3:4, pH * sH: (pH + 1) * sH,
        pW * sW: (pW + 1) * sW] = np.squeeze(
            init_ref_k_1to0_trim.detach().cpu().numpy())
        blws_blww_s1to0_r1to0_w1to0_flow01_diff1to0_FCW[4:5, pH * sH: (pH + 1) * sH,
        pW * sW: (pW + 1) * sW] = np.squeeze(
            final_warped_r2s_1to0_trim.detach().cpu().numpy())
        blws_blww_s1to0_r1to0_w1to0_flow01_diff1to0_FCW[5:7, pH * sH: (pH + 1) * sH,
        pW * sW: (pW + 1) * sW] = np.squeeze(
            FCW_flow_01_trim.detach().cpu().numpy())
        blws_blww_s1to0_r1to0_w1to0_flow01_diff1to0_FCW[7:8, pH * sH: (pH + 1) * sH,
        pW * sW: (pW + 1) * sW] = np.squeeze(
            diff_1to0_trim.detach().cpu().numpy())
        blws_blww_s1to0_r1to0_w1to0_flow01_diff1to0_FCW[8:9, pH * sH: (pH + 1) * sH,
        pW * sW: (pW + 1) * sW] = np.squeeze(
            FCW_1to0_trim.detach().cpu().numpy())

        """ 0to1"""
        blws_blww_s0to1_r0to1_w0to1_flow10_diff0to1_FCW[0:1, pH * sH: (pH + 1) * sH,
        pW * sW: (pW + 1) * sW] = np.squeeze(
            blending_weights_s_0to1_trim.detach().cpu().numpy())
        blws_blww_s0to1_r0to1_w0to1_flow10_diff0to1_FCW[1:2, pH * sH: (pH + 1) * sH,
        pW * sW: (pW + 1) * sW] = np.squeeze(
            blending_weights_w_0to1_trim.detach().cpu().numpy())
        blws_blww_s0to1_r0to1_w0to1_flow10_diff0to1_FCW[2:3, pH * sH: (pH + 1) * sH,
        pW * sW: (pW + 1) * sW] = np.squeeze(
            source_v_0to1_trim.detach().cpu().numpy())
        blws_blww_s0to1_r0to1_w0to1_flow10_diff0to1_FCW[3:4, pH * sH: (pH + 1) * sH,
        pW * sW: (pW + 1) * sW] = np.squeeze(
            init_ref_k_0to1_trim.detach().cpu().numpy())
        blws_blww_s0to1_r0to1_w0to1_flow10_diff0to1_FCW[4:5, pH * sH: (pH + 1) * sH,
        pW * sW: (pW + 1) * sW] = np.squeeze(
            final_warped_r2s_0to1_trim.detach().cpu().numpy())
        blws_blww_s0to1_r0to1_w0to1_flow10_diff0to1_FCW[5:7, pH * sH: (pH + 1) * sH,
        pW * sW: (pW + 1) * sW] = np.squeeze(
            FCW_flow_10_trim.detach().cpu().numpy())
        blws_blww_s0to1_r0to1_w0to1_flow10_diff0to1_FCW[7:8, pH * sH: (pH + 1) * sH,
        pW * sW: (pW + 1) * sW] = np.squeeze(
            diff_0to1_trim.detach().cpu().numpy())
        blws_blww_s0to1_r0to1_w0to1_flow10_diff0to1_FCW[8:9, pH * sH: (pH + 1) * sH,
        pW * sW: (pW + 1) * sW] = np.squeeze(
            FCW_0to1_trim.detach().cpu().numpy())
    S0_prime_full = S0_prime_full[:, :ori_h, :ori_w]
    S1_prime_full = S1_prime_full[:, :ori_h, :ori_w]
    St_prime_full = St_prime_full[:, :ori_h, :ori_w]
    Sharps_prime = [S0_prime_full, S1_prime_full, St_prime_full]

    S0_final_full = S0_final_full[:, :ori_h, :ori_w]
    S1_final_full = S1_final_full[:, :ori_h, :ori_w]
    St_final_full = St_final_full[:, :ori_h, :ori_w]
    Sharps_final = [S0_final_full, S1_final_full, St_final_full]

    if not frameT==None:
        St_GT_full = np.squeeze(frameT)  # [c,h,w]
    else:
        St_GT_full = 0

    ft0_init_full = ft0_init_full[:, :ori_h, :ori_w]
    ft0_final_full = ft0_final_full[:, :ori_h, :ori_w]
    ft1_init_full = ft1_init_full[:, :ori_h, :ori_w]
    ft1_final_full = ft1_final_full[:, :ori_h, :ori_w]
    occ0_init_full = occ0_init_full[:, :ori_h, :ori_w]
    occ0_final_full = occ0_final_full[:, :ori_h, :ori_w]
    two_blurry_inputs_full = two_blurry_inputs_full[:, :ori_h, :ori_w]
    flows_pred = [[ft0_init_full, ft0_final_full], [ft1_init_full, ft1_final_full]]
    occs_pred = [occ0_init_full, occ0_final_full]

    blws_blww_s1to0_r1to0_w1to0_flow01_diff1to0_FCW_full = blws_blww_s1to0_r1to0_w1to0_flow01_diff1to0_FCW[:, :ori_h,
                                                           :ori_w]
    blws_blww_s0to1_r0to1_w0to1_flow10_diff0to1_FCW_full = blws_blww_s0to1_r0to1_w0to1_flow10_diff0to1_FCW[:, :ori_h,
                                                           :ori_w]

    blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1 = \
        [blws_blww_s1to0_r1to0_w1to0_flow01_diff1to0_FCW_full, blws_blww_s0to1_r0to1_w0to1_flow10_diff0to1_FCW_full]

    return two_blurry_inputs_full, Sharps_prime, Sharps_final, St_GT_full, \
           flows_pred, occs_pred, \
           blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1


def get_HW_boundary(patch_boundary, h, w, pH, sH, pW, sW):
    H_low_ind = max(pH * sH - patch_boundary, 0)
    H_high_ind = min((pH + 1) * sH + patch_boundary, h)
    W_low_ind = max(pW * sW - patch_boundary, 0)
    W_high_ind = min((pW + 1) * sW + patch_boundary, w)

    add_H = 0
    add_W = 0
    if pH * sH >= patch_boundary:
        add_H = add_H + patch_boundary
    if (pH + 1) * sH + patch_boundary <= h:
        add_H = add_H + patch_boundary
    if pW * sW >= patch_boundary:
        add_W = add_W + patch_boundary
    if (pW + 1) * sW + patch_boundary <= w:
        add_W = add_W + patch_boundary

    return H_low_ind, H_high_ind, W_low_ind, W_high_ind, add_H, add_W


def trim_patch_boundary(img, patch_boundary, h, w, pH, sH, pW, sW, sf):
    if patch_boundary == 0:
        img = img
    else:
        if pH * sH < patch_boundary:
            img = img
        else:
            img = img[:, :, patch_boundary * sf:, :]
        if (pH + 1) * sH + patch_boundary > h:
            img = img
        else:
            img = img[:, :, :-patch_boundary * sf, :]
        if pW * sW < patch_boundary:
            img = img
        else:
            img = img[:, :, :, patch_boundary * sf:]
        if (pW + 1) * sW + patch_boundary > w:
            img = img
        else:
            img = img[:, :, :, :-patch_boundary * sf]

    return img





