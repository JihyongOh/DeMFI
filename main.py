"""
-----------------------------------------------------------------------------------------------------------------
<Official PyTorch Code>
Paper: Jihyong Oh and Munchurl Kim "DeMFI: Deep Joint Deblurring and Multi-Frame Interpolation 
		with Flow-Guided Attentive Correlation and Recursive Boosting", arXiv preprint arXiv: 2111.09985, 2021.

Written by Jihyong Oh (https://sites.google.com/view/ozbro), Contact: jhoh94@kaist.ac.kr
-----------------------------------------------------------------------------------------------------------------
"""
import argparse, os, shutil, time, random, torch, cv2, datetime, torch.nn.parallel, torch.utils.data, math
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torchinfo  import summary # new version
from collections import Counter
from tensorboardX import SummaryWriter

from DeMFInet import *
from utils import *

def parse_args():
	desc = "PyTorch implementation for DeMFI"
	parser = argparse.ArgumentParser(description=desc)
	parser.add_argument('--gpu', type=int, default=0, help='gpu index')
	parser.add_argument('--net_type', type=str, default='DeMFInet',
						choices=['DeMFInet'], help='The type of Net')
	parser.add_argument('--net_object', default=DeMFInet,
						choices=[DeMFInet], help='The type of Net')
	parser.add_argument('--exp_num', type=int, default=3, help='The experiment number')
	parser.add_argument('--phase', type=str, default='test',
						choices=['train', 'test', 'test_custom'])
	parser.add_argument('--test_epoch_point', default=False,
						help='testing epoch point for phase "test", if not "False"')
	parser.add_argument('--fine_tuning', default=False, help='finetuning the training')
	parser.add_argument('--fine_tuning_epoch_point', default=False,
						help='starting epoch point for finetuning, if not "False", ex) epoch=27 => 27')

	""" Directories Information"""
	parser.add_argument('--test_img_dir', type=str, default='./test_img_dir', help='test_img_dir path')
	parser.add_argument('--text_dir', type=str, default='./text_dir', help='text_dir path')
	parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint_dir', help='checkpoint_dir')
	parser.add_argument('--log_dir', type=str, default='./log_dir',
						help='Directory name to save training logs')

	""" Datasets Information """
	parser.add_argument('--train_data_path', type=str, default='./Datasets/Adobe_240fps_blur')
	parser.add_argument('--test_data_path', type=str, default='./Datasets/Adobe_240fps_blur',
						choices=['./Datasets/Adobe_240fps_blur', './Datasets/GoPro_blur',
								 './Datasets/YouTube240_Scenes'])  
	# parser.add_argument("--blur_window_size", type=int, default=11, help="number of frames to de average")
	# parser.add_argument("--blur_frame_idx", type=int, default=16, help="start of idx")
	parser.add_argument("--t_step_size", type=int, default=8, help="K in paper")

	""" 
						<Note>
		Adobe240 from PRF setting: 12186 samples
        (blurry_frame_idx) + i * (window_middle_delta) movement 
        caution: idx is '+1', ex) blurry_frame_idx = 16, window_middle_delta = 8
        ==> 00017.png, 00025.png, ...., 
        by averaging 'args.blur_window_size' frames
    """

	""" Hyperparameters for Training (when [phase=='train']) """
	# 1 epoch = 56 itrs., ===> total 420K itrs.
	parser.add_argument('--epochs', type=int, default=7500, help='The number of epochs to run')
	parser.add_argument('--freq_display', type=int, default=25,
						help='The iterations frequency for display')
	parser.add_argument('--epoch_freq_display', type=int, default=50,
						help='The epochs frequency for display')
	parser.add_argument('--epoch_freq_save', type=int, default=50,
						help='The epochs frequency for saving')
	parser.add_argument('--init_lr', type=float, default=1e-4, help='The initial learning rate')
	parser.add_argument('--lr_type', type=str, default='stair_decay',
						choices=['linear_decay', 'stair_decay', 'no_decay'])
	parser.add_argument('--lr_dec_fac', type=float, default=1/2, help='step - lr_decreasing_factor')
	parser.add_argument('--lr_milestones', type=int, default=[3750, 6250, 7250],
						help='When scheduler is MultiStepLR, lr decreases at lr_milestones') 
	parser.add_argument('--lr_dec_start', type=int, default=0,
						help='When scheduler is StepLR, lr decreases from epoch at lr_dec_start')
	parser.add_argument('--batch_size', type=int, default=2, help='The size of batch size.')
	parser.add_argument('--weight_decay', type=float, default=0,
						help='for optim., weight decay (default: 1e-4)')

	parser.add_argument('--need_patch', default=True, help='get patch form image')
	parser.add_argument('--img_ch', type=int, default=3, help='base number of channels for image')
	parser.add_argument('--nf', type=int, default=64, help='base number of channels for feature maps')
	parser.add_argument('--scale_factor', type=int, default=2,
						help='spatial reducetion for pixelshuffle layer')
	parser.add_argument('--patch_size', type=int, default=256, help='patch size')
	parser.add_argument('--num_thrds', type=int, default=8, help='number of threads for data loading')
	parser.add_argument('--loss_type', default='L1',
						choices=['L1','MSE','L1_Charbonnier_loss'], help='Loss type')

	""" Several Components """
	parser.add_argument('--num_ResB_FACFB', type=int, default=5)
	parser.add_argument('--num_ResB_Dec', type=int, default=5)
	parser.add_argument('--N_trn', type=int, default=5, help='total numbers of recursive boosting for training')
	parser.add_argument('--N_tst', type=int, default=3, help='total numbers of recursive boosting for testing')
	parser.add_argument('--shared_FGAC_flag', default=True)

	""" Weighting Parameters Lambda for Losses (when [phase=='train']) """
	parser.add_argument('--rec_D1_lambda', type=float, default=1.0, help='Lambda for D1 Reconstruction Loss')
	parser.add_argument('--rec_D2_lambda', type=float, default=1.0, help='Lambda for D2 Reconstruction Loss')
	
	""" Settings for Testing (when [phase=='test']) """
	parser.add_argument('--load_best_PSNR_flag', default=False, help='Note: We report final rusults for last epoch in DeMFI paper. (load_best_PSNR_flag = False)')
	parser.add_argument('--visualization_flag', default=False, help='[visualizations for diverse components]'
																	'1st row: ovlp_B0_B1, St_r(D1), St_Ntst(D2), GTt, ft0_r(fF), ft0_Ntst(fP_Ntst), ot0_r(fF), ot0_Ntst(fP_Ntst), |St_r-GTt|, |St_Ntst-GTt|'
																	'[for the 2nd and 3rd rows]'
																	'w_sr, (1-w_sr), F_s, Conv1(F_r), E_s, f_sr, |bolstered_F_s-F_s|, bolstered_F_s, None, None'
																	'2nd row: s=0, r=1, last rightmost two images are minmax(|St_r-GTt|), minmax(|St_Ntst-GTt|)'
																	'3rd row: s=1, r=0, last rightmost two images are nothing.')
	parser.add_argument('--test_patch', type=tuple, default=(1, 1),
						help='Divide img into patches in case of low memory')
	parser.add_argument('--patch_boundary', default=0,
						help='multiple of smallest spatial size for network & for margin of trimming, '
							 'also have to consider test_patch ex) (patch_boundary/test_patch[0] or patch_boundary/test_patch[1])')
	parser.add_argument('--multiple_MFI', type=int, default=8, help='temporal up-scaling factor x M (MFI),'
																	 'caution: when phase="test" (evaluation with GTs), only 2 or 8 are supported.')

	""" Settings for test_custom (when [phase=='test_custom']) """
	parser.add_argument('--custom_path', type=str, default='./custom_path',
						help='path for custom video containing frames')

	return check_args(parser.parse_args())


def check_args(args):
	# --checkpoint_dir
	check_folder(args.checkpoint_dir)

	# --text_dir
	check_folder(args.text_dir)

	# --log_dir
	check_folder(args.log_dir)

	# --test_img_dir
	check_folder(args.test_img_dir)

	return args


def main():
	args = parse_args()
	if args is None:
		exit()
	for arg in vars(args):
		print('# {} : {}'.format(arg, getattr(args, arg)))

	""" GPU Allocation, Important """
	# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
	# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # will be used as "x.to(device)"
	device = torch.device(
		'cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')  # will be used as "x.to(device)"
	torch.cuda.set_device(device)  # change allocation of current GPU
	"""
		Caution: if not "torch.cuda.set_device()":
			RuntimeError: grid_sampler(): expected input and grid to be on same device,
			but input is on cuda:1 and grid is on cuda:0
	"""
	print('Available devices: ', torch.cuda.device_count())
	print('Current cuda device: ', torch.cuda.current_device())
	print('Current cuda device name: ', torch.cuda.get_device_name(device))
	if args.gpu is not None:
		print("Use GPU: {} is used".format(args.gpu))

	print("Exp:", args.exp_num)
	model_dir = args.net_type + '_exp' + str(args.exp_num)  # ex) model_dir = "DeFInet_exp1"
	SM = save_manager(args)

	""" Initialize a model """
	model_net = args.net_object(args).apply(weights_init).to(device)

	criterion = [set_rec_loss(args).to(device)]
	optimizer = torch.optim.Adam(model_net.parameters(), lr=args.init_lr,
								 betas=(0.9, 0.999), weight_decay=args.weight_decay)  # optimizer
	# optimizer = torch.optim.Adam([{'params':model_net.parameters()},
	# 							  {'params':model_net.Init_Align.FGCW_F1toF0.w,'lr':1e-3},
	# 							  {'params':model_net.Init_Align.FGCW_F0toF1.w,'lr':1e-3}], lr=args.init_lr,
	# 							 betas=(0.9, 0.999), weight_decay=args.weight_decay)  # optimizer

	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_dec_fac)


	""" Analyze the model (number of model parameters (#P)) """
	summary(model_net, [(1, 3, 4, 32, 32), (1,1)], dtypes=[torch.float, torch.float]) 


	last_epoch = 0
	best_PSNR = 0
	# best_SSIM = 0
	# fine-tuning
	if args.fine_tuning:
		if args.fine_tuning_epoch_point == False:
			checkpoint = SM.load_model()
		else:
			checkpoint = SM.load_epc_model(args.fine_tuning_epoch_point)
		last_epoch = checkpoint['last_epoch'] + 1
		best_PSNR = checkpoint['best_PSNR']
		intp_testPSNR = checkpoint['intp_testPSNR']
		deblur_testPSNR = checkpoint['intp_testPSNR']
		testLoss = checkpoint['testLoss']
		deblur_testSSIM = checkpoint['deblur_testSSIM']
		intp_testSSIM = checkpoint['intp_testSSIM']
		# best_SSIM = checkpoint['best_SSIM']
		model_net.load_state_dict(checkpoint['state_dict_Model'])

		if 'state_dict_Optimizer' and 'state_dict_Scheduler' in checkpoint:
			print("Optimizer and Scheduler also have been reloaded. ")
			optimizer.load_state_dict(checkpoint['state_dict_Optimizer'])
			scheduler.load_state_dict(checkpoint['state_dict_Scheduler'])
	scheduler.milestones = Counter(args.lr_milestones)
	scheduler.gamma = args.lr_dec_fac
	start_epoch = last_epoch

	# to enable the inbuilt cudnn auto-tuner
	# to find the best algorithm to use for your hardware.
	# cudnn.benchmark = True

	""" Tensorboard """
	writer = SummaryWriter('log')
	valid_loader = get_test_data(args, multiple=2, center_flag=True,
								 test_type='valid_5_per_scene')  # center-cropped

	if args.phase == "train":
		SM.write_info(
			'Epoch\tintp_testPSNR\tdeblur_testPSNR\tavg_testPSNR\tbest_PSNR\n')
		print("[*] Training starts")
		start_time = time.time()
		# Main training loop for total epochs (start from 'epoch=0')
		for epoch in range(start_epoch, args.epochs):
			train_loader = get_train_data(args)
			trainLoss, trainLoss_rec_D1, trainLoss_rec_D2 = train(train_loader, model_net, criterion, optimizer,
								  scheduler, epoch, writer, args, device, model_dir)

			if (epoch + 1) % args.epoch_freq_display == 0 or epoch == 0:
				print(
					'evaluate on test set (validation while training) with multiple=2, center_flag=True, test_type=valid_5_per_scene')
				testLoss, intp_testPSNR, intp_testSSIM, deblur_testPSNR, \
				deblur_testSSIM, final_pred_save_path = test(valid_loader, model_net, criterion,epoch, writer, args, device, model_dir,
															 multiple=2, num_update=args.N_trn, patch=(1, 1),
															 visualization_flag=args.visualization_flag,post_fix='_x2_valid_5_intervals')
			# final_pred_save_path = './test_img_dir/DeFInet_exp1/epoch_00099_x4_center_first_32frames'

			# remember best best_PSNR and best_SSIM and save checkpoint
			best_PSNR_flag = (intp_testPSNR+deblur_testPSNR)/2 > best_PSNR
			best_PSNR = max((intp_testPSNR+deblur_testPSNR)/2, best_PSNR)

			# save checkpoint.
			combined_state_dict = {
				'net_type': args.net_type,
				'last_epoch': epoch,
				'epoch': epoch,
				'batch_size': args.batch_size,
				'learningRate': get_lr(optimizer),
				'trainLoss': trainLoss,
				'trainLoss_rec_D1': trainLoss_rec_D1,
				'trainLoss_rec_D2': trainLoss_rec_D2,
				'testLoss': testLoss,
				'intp_testPSNR': intp_testPSNR,
				'intp_testSSIM': intp_testSSIM,
				'deblur_testPSNR': deblur_testPSNR,
				'deblur_testSSIM': deblur_testSSIM,
				'best_PSNR': best_PSNR,
				'state_dict_Model': model_net.state_dict(),
				'state_dict_Optimizer': optimizer.state_dict(),
				'state_dict_Scheduler': scheduler.state_dict()}

			SM.save_best_model(combined_state_dict, best_PSNR_flag, False)

			if (epoch + 1) % args.epoch_freq_save ==0:
				SM.save_epc_model(combined_state_dict, epoch)
			# save information as ".txt" too.
			SM.write_info('{}\t{:.4}\t{:.4}\t{:.4}'
							  '\t{:.4}\n'.
							  format(epoch, intp_testPSNR,deblur_testPSNR,(intp_testPSNR+deblur_testPSNR)/2,best_PSNR))


			print(" ------------------------ time consumption: ", ((time.time() - start_time) / 60),
				" (minutes) ------------------------ ")

		print("------------------------- Training has been ended. -------------------------")
		print("information of model:", model_dir)
		print("best_PSNR of model:", best_PSNR)
		print("final predicted images for validation (x2, center-cropped) are saved in ", final_pred_save_path)

		print(" -------------------------- Final Test starts (Adobe240fps) after training ---------------------------- ")
		print('evaluate on Adobe240fps with multiple = %d on full resolution' \
			  % (args.multiple_MFI))
		final_test_loader = get_test_data(args, multiple=args.multiple_MFI,
										  center_flag=False, test_type=None)

		testLoss, intp_testPSNR, intp_testSSIM, deblur_testPSNR, \
		deblur_testSSIM, final_pred_save_path = test( final_test_loader, model_net, criterion, (args.epochs - 1), writer, args,
													  device, model_dir, multiple=args.multiple_MFI, num_update=args.N_tst,
													  patch=args.test_patch, visualization_flag=args.visualization_flag,
																  post_fix='_final_x' + str(args.multiple_MFI)
																		   + '_full_resolution'+'_Ntst'+str(args.N_tst))
		print("------------------------- Test has been ended. -------------------------")
		print("Exp:", args.exp_num)

	elif args.phase == "test" :
		if args.load_best_PSNR_flag == True:
			checkpoint = SM.load_best_PSNR_model()
		else:
			checkpoint = SM.load_model()

		if not args.test_epoch_point == False:
			checkpoint = SM.load_epc_model(args.test_epoch_point)
			args.epochs = args.test_epoch_point

		model_net.load_state_dict(checkpoint['state_dict_Model'])
		if 'stae_dict_Optimizer' and 'state_dict_Scheduler' in checkpoint:
			optimizer.load_state_dict(checkpoint['state_dict_Optimizer'])
			scheduler.load_state_dict(checkpoint['state_dict_Scheduler'])

		print(
			" -------------------------------------- Test starts -------------------------------------- ")
		print("Evaluating on {} with multiple = {} with full resolution".format(args.test_data_path, args.multiple_MFI))
		final_test_loader = get_test_data(args, multiple=args.multiple_MFI,
										  center_flag=False, test_type=None)

		testLoss, intp_testPSNR, intp_testSSIM, deblur_testPSNR, \
		deblur_testSSIM, final_pred_save_path = test(final_test_loader, model_net,
			 criterion, (args.epochs - 1),
			 writer, args, device, model_dir,
			 multiple=args.multiple_MFI,
			 num_update=args.N_tst,
			 patch=args.test_patch, visualization_flag=args.visualization_flag,
			 post_fix='_final_x' + str(
				 args.multiple_MFI) + '_full_resolution'+'_Ntst'+str(args.N_tst))

		print("------------------------- Test has been ended. -------------------------")
		print("Exp:", args.exp_num)


	elif args.phase =='test_custom':
		if args.load_best_PSNR_flag == True:
			checkpoint = SM.load_best_PSNR_model()
		else:
			checkpoint = SM.load_model()

		if not args.test_epoch_point == False:
			checkpoint = SM.load_epc_model(args.test_epoch_point)
			args.epochs = args.test_epoch_point

		model_net.load_state_dict(checkpoint['state_dict_Model'])
		if 'stae_dict_Optimizer' and 'state_dict_Scheduler' in checkpoint:
			optimizer.load_state_dict(checkpoint['state_dict_Optimizer'])
			scheduler.load_state_dict(checkpoint['state_dict_Scheduler'])

		print(
			" -------------------------------------- Custom Test starts -------------------------------------- ")
		print("Evaluating on {} with multiple = {} with full resolution".format(args.custom_path, args.multiple_MFI))
		final_test_loader = get_test_data(args, multiple=args.multiple_MFI,
										  center_flag=False, test_type=None)

		test_custom(final_test_loader, model_net,args, device,
			 multiple=args.multiple_MFI,num_update=args.N_tst,
			 patch=args.test_patch, visualization_flag=args.visualization_flag)


def train(train_loader, model_net, criterion, optimizer,
		  scheduler, epoch, writer, args, device, model_dir):
	## reset "AverageClass" per epoch
	batch_time = AverageClass('Batch_Time[s]:', ':6.3f')
	accm_time = AverageClass('Accm_Time[s]:', ':6.3f')
	losses = AverageClass('trainLoss:', ':.4e')
	rec_D1_losses = AverageClass('trainLoss_rec_D1:', ':.4e')
	rec_D2_losses = AverageClass('trainLoss_rec_D2:', ':.4e')
	progress = ProgressMeter(len(train_loader), batch_time, accm_time, losses,
							 rec_D1_losses, rec_D2_losses,
							 prefix="Epoch: [{}]".format(epoch))
	rec_loss_func0 = criterion[0] # Reconstruction Loss

	## switch to train mode
	model_net.train()
	fix_start_time = time.time()
	start_time = time.time()
	print('Start epoch {} at [{:s}], learning rate : [{}]'.format(epoch, (str(datetime.now())[:-7]),
																  optimizer.param_groups[0]['lr']))
	for trainIndex, (frames, t_value) in enumerate(train_loader):
		frames = frames.to(device)
		input_frames = frames[:, :, :2+2, :]  # [B, C, T, H, W]
		frameT = frames[:, :, 2+2, :]  # [B, C, H, W]
		input_frames_GT = frames[:, :, -2-2:, :]  # [B, C, T, H, W], t=0,1,-1,2

		## Getting the input and the target from the training set
		input_frames = Variable(input_frames.to(device))
		vis_input_frames = input_frames
		frameT = Variable(frameT.to(device))  # ground truth for frameT
		input_frames_GT = Variable(input_frames_GT.to(device))
		t_value = Variable(t_value.to(device))  # [B,1]
		optimizer.zero_grad()

		## compute output
		pred_Sharps_prime, pred_Sharps_final, pred_Flows, pred_Occ_0, ovlp_B0B1,\
		difference_maps, pred_flow_t0_t1  = model_net(input_frames, t_value, args.N_trn, is_training=True)

		rec_D1_loss = 0.0
		rec_D2_loss = 0.0
		for idx in range(3):
			if idx == 2:
				## last: for St
				rec_D1_loss += args.rec_D1_lambda * \
							rec_loss_func0(frameT, pred_Sharps_prime[idx])
				rec_D2_loss += args.rec_D2_lambda * \
							rec_loss_func0(frameT, pred_Sharps_final[0][idx])
				rec_D1_loss /= 3 # Eq.(9)
				rec_D2_loss /= 3 # Eq.(10) for i=1

			else:
				## idx = 0,1 => for S0,S1
				rec_D1_loss += args.rec_D1_lambda * \
							rec_loss_func0(input_frames_GT[:,:,idx,:], pred_Sharps_prime[idx])
				rec_D2_loss += args.rec_D2_lambda * \
							rec_loss_func0(input_frames_GT[:,:,idx,:], pred_Sharps_final[0][idx])

		""" for updater (N_trn), Eq.(10) for i=2,3,...,N_trn """
		for i in range(args.N_trn - 1):
			rec_D2_loss_temp = 0.0
			for idx in range(3):
				if idx == 2:
					# last: for St
					rec_D2_loss_temp += args.rec_D2_lambda * \
									   rec_loss_func0(frameT, pred_Sharps_final[i + 1][idx])
					rec_D2_loss_temp /= 3
					rec_D2_loss += rec_D2_loss_temp

				else:
					# idx = 0,1 => for S0,S1
					rec_D2_loss_temp += args.rec_D2_lambda * \
									   rec_loss_func0(input_frames_GT[:, :, idx, :], pred_Sharps_final[i + 1][idx])

		## rec_D2_loss /= args.N_trn
		total_loss = rec_D1_loss + rec_D2_loss

		## compute gradient and do SGD step
		total_loss.backward()  # Backpropagate
		optimizer.step()  # Optimizer update

		## measure accumulated time and update average "batch" time consumptions via "AverageClass"
		## update average values via "AverageClass"
		losses.update(total_loss.item(), 1)
		rec_D1_losses.update(rec_D1_loss.item(), 1)
		rec_D2_losses.update(rec_D2_loss.item(), 1)
		batch_time.update(time.time() - start_time)
		start_time = time.time()
		accm_time.update(time.time() - fix_start_time)

		if (trainIndex == args.freq_display-1):
			progress.print(trainIndex)
			# for tensorboard
			TB_count = trainIndex + epoch * (len(train_loader))
			writer.add_scalar('trainLoss', losses.val / args.freq_display, TB_count)
			writer.add_scalar('trainLoss_rec_D1', rec_D1_losses.val / args.freq_display, TB_count)
			writer.add_scalar('trainLoss_rec_D2', rec_D2_losses.val / args.freq_display, TB_count)

		if (trainIndex == 0 or trainIndex == args.freq_display-1 or trainIndex == (args.freq_display-1)*2):
			""" save Overlayed & Pred & GT while training """
			epoch_save_path = os.path.join(args.test_img_dir, model_dir,
										  'while_training')
			check_folder(epoch_save_path)
			## ex) './test_img_dir/DeFInet_exp1/epoch_00000_while_training'

			cv2.imwrite(os.path.join(epoch_save_path, 'trainIdx_' + str(trainIndex).zfill(5) + '_Ovld.png'),
						np.transpose(np.squeeze(denorm255_np(
							vis_input_frames[0, :, 0, :, :].detach().cpu().numpy())
												+ denorm255_np(
							vis_input_frames[0, :, 1, :, :].detach().cpu().numpy())) / 2,
									 [1, 2, 0]).astype(np.uint8))
			cv2.imwrite(os.path.join(epoch_save_path,'trainIdx_' + str(trainIndex).zfill(5) + '_S0_Prd.png'),
						np.transpose(np.squeeze(denorm255_np(
							pred_Sharps_final[-1][0][0, :, :, :].detach().cpu().numpy())
						), [1, 2, 0]).astype(np.uint8))
			cv2.imwrite(os.path.join(epoch_save_path,'trainIdx_' + str(trainIndex).zfill(5) + '_t_' + str(
				t_value[0, :].item()) + '_Prd.png'),
						np.transpose(np.squeeze(denorm255_np(
							pred_Sharps_final[-1][-1][0, :, :, :].detach().cpu().numpy())
						), [1, 2, 0]).astype(np.uint8))
			cv2.imwrite(os.path.join(epoch_save_path,'trainIdx_' + str(trainIndex).zfill(5) + '_S1_Prd.png'),
						np.transpose(np.squeeze(denorm255_np(
							pred_Sharps_final[-1][1][0, :, :, :].detach().cpu().numpy())
						), [1, 2, 0]).astype(np.uint8))

			cv2.imwrite(os.path.join(epoch_save_path,'trainIdx_' + str(trainIndex).zfill(5) + '_S0_GT.png'),
						np.transpose(np.squeeze(denorm255_np(
							input_frames_GT[0, :, 0, :].detach().cpu().numpy())
						), [1, 2, 0]).astype(np.uint8))
			cv2.imwrite(os.path.join(epoch_save_path,'trainIdx_' + str(trainIndex).zfill(5) + '_t_' + str(
				t_value[0, :].item()) + '_GT.png'),
						np.transpose(np.squeeze(denorm255_np(
							frameT[0, :, :, :].detach().cpu().numpy())
						), [1, 2, 0]).astype(np.uint8))
			cv2.imwrite(os.path.join(epoch_save_path,'trainIdx_' + str(trainIndex).zfill(5) + '_S1_GT.png'),
						np.transpose(np.squeeze(denorm255_np(
							input_frames_GT[0, :, 1, :].detach().cpu().numpy())
						), [1, 2, 0]).astype(np.uint8))

			batch_images = get_batch_images(args, args.batch_size,
											[ovlp_B0B1, pred_Sharps_prime[0], pred_Sharps_prime[-1], pred_Sharps_prime[1],
											 pred_Sharps_final[-1][0], pred_Sharps_final[-1][-1], pred_Sharps_final[-1][1],
											 frameT, input_frames_GT, pred_Flows, pred_Occ_0, difference_maps, pred_flow_t0_t1])
			cv2.imwrite(os.path.join(epoch_save_path , 'trainIdx_' + str(trainIndex).zfill(5) + '_Bx20.png'), batch_images)

	if epoch >= args.lr_dec_start:
		scheduler.step()
	return losses.avg, rec_D1_losses.avg, rec_D2_losses.avg


def test(test_loader, model_net, criterion, epoch, writer, args, device, model_dir, multiple, num_update, patch,
		 visualization_flag,
		 post_fix=''):
	assert multiple == 8 or multiple == 2
	batch_time = AverageClass('Time:', ':6.3f')
	accm_time = AverageClass('Accm_Time[s]:', ':6.3f')
	losses = AverageClass('testLoss:', ':.4e')

	intp_PSNRs_prime = AverageClass('intp_testPSNR_prime:', ':.4e')
	intp_SSIMs_prime = AverageClass('intp_testSSIM_prime:', ':.4e')
	deblur_PSNRs_prime = AverageClass('deblur_testPSNR_prime:', ':.4e')
	deblur_SSIMs_prime = AverageClass('deblur_testSSIM_prime:', ':.4e')

	intp_PSNRs = AverageClass('intp_testPSNR:', ':.4e')
	intp_SSIMs = AverageClass('intp_testSSIM:', ':.4e')
	deblur_PSNRs = AverageClass('deblur_testPSNR:', ':.4e')
	deblur_SSIMs = AverageClass('deblur_testSSIM:', ':.4e')

	progress = ProgressMeter(len(test_loader), batch_time, accm_time, losses,
							 intp_PSNRs_prime, intp_SSIMs_prime, deblur_PSNRs_prime, deblur_SSIMs_prime,
							 intp_PSNRs, intp_SSIMs,
							 deblur_PSNRs, deblur_SSIMs, prefix='Test after Epoch[{}, (caution: total average of total samples)]: '.format(epoch))

	rec_loss_func = criterion[0]

	""" 1. Total Performance Avg. of Scene Avg. (prime) """
	PSNR_1_prime = AverageClass('PSNR_1_prime:', ':.4e')
	PSNR_2_prime = AverageClass('PSNR_2_prime:', ':.4e')
	PSNR_3_prime = AverageClass('PSNR_3_prime:', ':.4e')
	PSNR_4_prime = AverageClass('PSNR_4_prime:', ':.4e')
	PSNR_5_prime = AverageClass('PSNR_5_prime:', ':.4e')
	PSNR_6_prime = AverageClass('PSNR_6_prime:', ':.4e')
	PSNR_7_prime = AverageClass('PSNR_7_prime:', ':.4e')
	PSNR_8_deblur_prime = AverageClass('PSNR_8_deblur_prime:', ':.4e')
	progress_PSNR_prime = ProgressMeter(len(test_loader), PSNR_1_prime, PSNR_2_prime, PSNR_3_prime, PSNR_4_prime,
										PSNR_5_prime, PSNR_6_prime, PSNR_7_prime, PSNR_8_deblur_prime,
										prefix='x8 MFI results [PSNR Stage I (7 intp, 1 dblr)] :')

	SSIM_1_prime = AverageClass('SSIM_1_prime:', ':.4e')
	SSIM_2_prime = AverageClass('SSIM_2_prime:', ':.4e')
	SSIM_3_prime = AverageClass('SSIM_3_prime:', ':.4e')
	SSIM_4_prime = AverageClass('SSIM_4_prime:', ':.4e')
	SSIM_5_prime = AverageClass('SSIM_5_prime:', ':.4e')
	SSIM_6_prime = AverageClass('SSIM_6_prime:', ':.4e')
	SSIM_7_prime = AverageClass('SSIM_7_prime:', ':.4e')
	SSIM_8_deblur_prime = AverageClass('SSIM_8_deblur_prime:', ':.4e')
	progress_SSIM_prime = ProgressMeter(len(test_loader), SSIM_1_prime, SSIM_2_prime, SSIM_3_prime, SSIM_4_prime,
										SSIM_5_prime, SSIM_6_prime, SSIM_7_prime, SSIM_8_deblur_prime,
										prefix='x8 MFI results [SSIM Stage I (7 intp, 1 dblr)] :')

	""" 2. Scene Avg. (prime) """
	PSNR_scene_1_prime = AverageClass('PSNR_scene_1_prime:', ':.4e')
	PSNR_scene_2_prime = AverageClass('PSNR_scene_2_prime:', ':.4e')
	PSNR_scene_3_prime = AverageClass('PSNR_scene_3_prime:', ':.4e')
	PSNR_scene_4_prime = AverageClass('PSNR_scene_4_prime:', ':.4e')
	PSNR_scene_5_prime = AverageClass('PSNR_scene_5_prime:', ':.4e')
	PSNR_scene_6_prime = AverageClass('PSNR_scene_6_prime:', ':.4e')
	PSNR_scene_7_prime = AverageClass('PSNR_scene_7_prime:', ':.4e')
	PSNR_scene_8_deblur_prime = AverageClass('PSNR_scene_8_deblur_prime:', ':.4e')
	SSIM_scene_1_prime = AverageClass('SSIM_scene_1_prime:', ':.4e')
	SSIM_scene_2_prime = AverageClass('SSIM_scene_2_prime:', ':.4e')
	SSIM_scene_3_prime = AverageClass('SSIM_scene_3_prime:', ':.4e')
	SSIM_scene_4_prime = AverageClass('SSIM_scene_4_prime:', ':.4e')
	SSIM_scene_5_prime = AverageClass('SSIM_scene_5_prime:', ':.4e')
	SSIM_scene_6_prime = AverageClass('SSIM_scene_6_prime:', ':.4e')
	SSIM_scene_7_prime = AverageClass('SSIM_scene_7_prime:', ':.4e')
	SSIM_scene_8_deblur_prime = AverageClass('SSIM_scene_8_deblur_prime:', ':.4e')

	""" 3. Total Performance Avg. of Scene Avg. (sharp) """
	PSNR_1 = AverageClass('PSNR_1:', ':.4e')
	PSNR_2 = AverageClass('PSNR_2:', ':.4e')
	PSNR_3 = AverageClass('PSNR_3:', ':.4e')
	PSNR_4 = AverageClass('PSNR_4:', ':.4e')
	PSNR_5 = AverageClass('PSNR_5:', ':.4e')
	PSNR_6 = AverageClass('PSNR_6:', ':.4e')
	PSNR_7 = AverageClass('PSNR_7:', ':.4e')
	PSNR_8_deblur = AverageClass('PSNR_8_deblur:', ':.4e')
	progress_PSNR = ProgressMeter(len(test_loader), PSNR_1, PSNR_2, PSNR_3, PSNR_4,
								  PSNR_5, PSNR_6, PSNR_7, PSNR_8_deblur,
								  prefix='x8 MFI results [PSNR Stage II (7 intp, 1 dblr)] :')

	SSIM_1 = AverageClass('SSIM_1:', ':.4e')
	SSIM_2 = AverageClass('SSIM_2:', ':.4e')
	SSIM_3 = AverageClass('SSIM_3:', ':.4e')
	SSIM_4 = AverageClass('SSIM_4:', ':.4e')
	SSIM_5 = AverageClass('SSIM_5:', ':.4e')
	SSIM_6 = AverageClass('SSIM_6:', ':.4e')
	SSIM_7 = AverageClass('SSIM_7:', ':.4e')
	SSIM_8_deblur = AverageClass('SSIM_8_deblur:', ':.4e')
	progress_SSIM = ProgressMeter(len(test_loader), SSIM_1, SSIM_2, SSIM_3, SSIM_4,
								  SSIM_5, SSIM_6, SSIM_7, SSIM_8_deblur,
								  prefix='x8 MFI results [SSIM Stage II (7 intp, 1 dblr)] :')

	""" 4. Scene Avg. (sharp)"""
	PSNR_scene_1 = AverageClass('PSNR_scene_1:', ':.4e')
	PSNR_scene_2 = AverageClass('PSNR_scene_2:', ':.4e')
	PSNR_scene_3 = AverageClass('PSNR_scene_3:', ':.4e')
	PSNR_scene_4 = AverageClass('PSNR_scene_4:', ':.4e')
	PSNR_scene_5 = AverageClass('PSNR_scene_5:', ':.4e')
	PSNR_scene_6 = AverageClass('PSNR_scene_6:', ':.4e')
	PSNR_scene_7 = AverageClass('PSNR_scene_7:', ':.4e')
	PSNR_scene_8_deblur = AverageClass('PSNR_scene_8_deblur:', ':.4e')
	SSIM_scene_1 = AverageClass('SSIM_scene_1:', ':.4e')
	SSIM_scene_2 = AverageClass('SSIM_scene_2:', ':.4e')
	SSIM_scene_3 = AverageClass('SSIM_scene_3:', ':.4e')
	SSIM_scene_4 = AverageClass('SSIM_scene_4:', ':.4e')
	SSIM_scene_5 = AverageClass('SSIM_scene_5:', ':.4e')
	SSIM_scene_6 = AverageClass('SSIM_scene_6:', ':.4e')
	SSIM_scene_7 = AverageClass('SSIM_scene_7:', ':.4e')
	SSIM_scene_8_deblur = AverageClass('SSIM_scene_8_deblur:', ':.4e')
	# switch to evaluate mode
	model_net.eval()

	print("------------------------------------------- Test ----------------------------------------------")
	with torch.no_grad():
		prev_scene_name = None
		fix_start_time = time.time()
		for testIndex, (frames, t_value, scene_name, frameRange, S0S1_GT_frames) in enumerate(test_loader):
			if prev_scene_name != scene_name[0]:
				""" scene change (assumption: at least each scene has more than 4 inputs.) """
				if testIndex != 0:
					""" Note: deblurred frame from the later sliding window is saved for simplicity. ("last sample" of each scene, S1)"""
					PSNR_scene_8_deblur_prime.update(test_psnr_S1_prime, 1)  # Stage I
					SSIM_scene_8_deblur_prime.update(test_ssim_S1_prime, 1)
					PSNR_scene_8_deblur.update(test_psnr_S1, 1)  # Stage II
					SSIM_scene_8_deblur.update(test_ssim_S1, 1)

					""" Total average of total samples """
					deblur_PSNRs_prime.update(test_psnr_S1_prime, 1)
					deblur_SSIMs_prime.update(test_ssim_S1_prime, 1)
					deblur_PSNRs.update(test_psnr_S1, 1)
					deblur_SSIMs.update(test_ssim_S1, 1)

					""" update average of scene (stage I)"""
					PSNR_1_prime.update(PSNR_scene_1_prime.avg, 1)
					PSNR_2_prime.update(PSNR_scene_2_prime.avg, 1)
					PSNR_3_prime.update(PSNR_scene_3_prime.avg, 1)
					PSNR_4_prime.update(PSNR_scene_4_prime.avg, 1)
					PSNR_5_prime.update(PSNR_scene_5_prime.avg, 1)
					PSNR_6_prime.update(PSNR_scene_6_prime.avg, 1)
					PSNR_7_prime.update(PSNR_scene_7_prime.avg, 1)
					PSNR_8_deblur_prime.update(PSNR_scene_8_deblur_prime.avg, 1)

					SSIM_1_prime.update(SSIM_scene_1_prime.avg, 1)
					SSIM_2_prime.update(SSIM_scene_2_prime.avg, 1)
					SSIM_3_prime.update(SSIM_scene_3_prime.avg, 1)
					SSIM_4_prime.update(SSIM_scene_4_prime.avg, 1)
					SSIM_5_prime.update(SSIM_scene_5_prime.avg, 1)
					SSIM_6_prime.update(SSIM_scene_6_prime.avg, 1)
					SSIM_7_prime.update(SSIM_scene_7_prime.avg, 1)
					SSIM_8_deblur_prime.update(SSIM_scene_8_deblur_prime.avg, 1)

					""" update average of scene (stage II)"""
					PSNR_1.update(PSNR_scene_1.avg, 1)
					PSNR_2.update(PSNR_scene_2.avg, 1)
					PSNR_3.update(PSNR_scene_3.avg, 1)
					PSNR_4.update(PSNR_scene_4.avg, 1)
					PSNR_5.update(PSNR_scene_5.avg, 1)
					PSNR_6.update(PSNR_scene_6.avg, 1)
					PSNR_7.update(PSNR_scene_7.avg, 1)
					PSNR_8_deblur.update(PSNR_scene_8_deblur.avg, 1)

					SSIM_1.update(SSIM_scene_1.avg, 1)
					SSIM_2.update(SSIM_scene_2.avg, 1)
					SSIM_3.update(SSIM_scene_3.avg, 1)
					SSIM_4.update(SSIM_scene_4.avg, 1)
					SSIM_5.update(SSIM_scene_5.avg, 1)
					SSIM_6.update(SSIM_scene_6.avg, 1)
					SSIM_7.update(SSIM_scene_7.avg, 1)
					SSIM_8_deblur.update(SSIM_scene_8_deblur.avg, 1)

				""" reset scene avg. (prime)"""
				PSNR_scene_1_prime.reset()
				PSNR_scene_2_prime.reset()
				PSNR_scene_3_prime.reset()
				PSNR_scene_4_prime.reset()
				PSNR_scene_5_prime.reset()
				PSNR_scene_6_prime.reset()
				PSNR_scene_7_prime.reset()
				PSNR_scene_8_deblur_prime.reset()
				SSIM_scene_1_prime.reset()
				SSIM_scene_2_prime.reset()
				SSIM_scene_3_prime.reset()
				SSIM_scene_4_prime.reset()
				SSIM_scene_5_prime.reset()
				SSIM_scene_6_prime.reset()
				SSIM_scene_7_prime.reset()
				SSIM_scene_8_deblur_prime.reset()

				""" reset scene avg. (sharp)"""
				PSNR_scene_1.reset()
				PSNR_scene_2.reset()
				PSNR_scene_3.reset()
				PSNR_scene_4.reset()
				PSNR_scene_5.reset()
				PSNR_scene_6.reset()
				PSNR_scene_7.reset()
				PSNR_scene_8_deblur.reset()
				SSIM_scene_1.reset()
				SSIM_scene_2.reset()
				SSIM_scene_3.reset()
				SSIM_scene_4.reset()
				SSIM_scene_5.reset()
				SSIM_scene_6.reset()
				SSIM_scene_7.reset()
				SSIM_scene_8_deblur.reset()

			# Getting the input and the target from the training set
			# Shape of 'frames' : [1,C,T+1,H,W]
			St_GT = frames[:, :, 2 + 2, :, :]  # [1,C,H,W]
			S0_GT = S0S1_GT_frames[:, :, 0, :, :]  # [1,C,2,H,W]
			S1_GT = S0S1_GT_frames[:, :, 1, :, :]  # [1,C,2,H,W]

			St_Path, S0_Path, S1_Path = frameRange  # tuple:string

			St_GT = Variable(St_GT.to(device))
			t_value = Variable(t_value.to(device))

			if (testIndex % (multiple - 1)) == 0:
				input_frames = frames[:, :, :-1, :, :]  # [1,C,T,H,W]
				input_frames = Variable(input_frames.to(device))

			start_time = time.time()
			""" Divide & Process due to Limited Memory """
			# compute output
			if not visualization_flag:
				two_blurry_inputs_full, Sharps_prime, Sharps_final, St_GT_full, flows_pred, occs_pred = \
					patch_forward_DeFInet_itr(model_net, input_frames, St_GT, t_value, num_update, patch,
											  args.patch_boundary)
			else:
				two_blurry_inputs_full, Sharps_prime, Sharps_final, St_GT_full, flows_pred, occs_pred,\
					blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1= \
					patch_forward_DeFInet_w_diff(model_net, input_frames, St_GT, t_value, num_update, patch,
											  args.patch_boundary)
			# pred_frameT_group = pred_frameT.data.float().cpu().squeeze(0)

			""" Measure Inference Time """
			batch_time.update(time.time() - start_time)
			start_time = time.time()
			accm_time.update(time.time() - fix_start_time)

			St_GT = St_GT_full.detach().cpu().numpy()

			""" D1 """
			pred_S0_prime, pred_S1_prime, pred_frameT_prime = Sharps_prime
			pred_frameT_group_prime = pred_frameT_prime

			""" Compute Interpolation (St) PSNR & SSIM (D1) """
			output_img = np.around(denorm255_np(
				np.transpose(pred_frameT_group_prime, [1, 2, 0])[:, :, ::-1]))  # [h,w,c] and BGR2RGB, [-1,1]2[0,255]
			target_img = denorm255_np(
				np.transpose(St_GT, [1, 2, 0])[:, :, ::-1])  # [h,w,c] and BGR2RGB, [-1,1]2[0,255]
			output_img_crop, _, _ = crop_8x8(output_img)
			target_img_crop, _, _ = crop_8x8(target_img)

			intp_test_psnr_prime = psnr(target_img_crop, output_img_crop)  # caution: calculation for RGB
			intp_test_ssim_prime = ssim(target_img_crop, output_img_crop)  # caution: calculation for RGB

			""" Compute Deblur (S0,S1) PSNR & SSIM (D1, Stage I) """
			output_img = np.around(denorm255_np(
				np.transpose(pred_S0_prime, [1, 2, 0])[:, :, ::-1]))  # [h,w,c] and BGR2RGB, [-1,1]2[0,255]
			target_img = denorm255_np(
				np.transpose(np.squeeze(S0_GT.detach().cpu().numpy()), [1, 2, 0])[:, :,
				::-1])  # [h,w,c] and BGR2RGB, [-1,1]2[0,255]
			output_img_crop, _, _ = crop_8x8(output_img)
			target_img_crop, _, _ = crop_8x8(target_img)

			test_psnr_S0_prime = psnr(target_img_crop, output_img_crop)  # caution: calculation for RGB
			test_ssim_S0_prime = ssim(target_img_crop, output_img_crop)  # caution: calculation for RGB

			output_img = np.around(denorm255_np(
				np.transpose(pred_S1_prime, [1, 2, 0])[:, :, ::-1]))  # [h,w,c] and BGR2RGB, [-1,1]2[0,255]
			target_img = denorm255_np(
				np.transpose(np.squeeze(S1_GT.detach().cpu().numpy()), [1, 2, 0])[:, :,
				::-1])  # [h,w,c] and BGR2RGB, [-1,1]2[0,255]
			output_img_crop, _, _ = crop_8x8(output_img)
			target_img_crop, _, _ = crop_8x8(target_img)

			test_psnr_S1_prime = psnr(target_img_crop, output_img_crop)  # caution: calculation for RGB
			test_ssim_S1_prime = ssim(target_img_crop, output_img_crop)  # caution: calculation for RGB


			""" D2 """
			pred_S0, pred_S1, pred_frameT = Sharps_final
			pred_frameT_group = pred_frameT

			# frameT = frameT.data.float().cpu().squeeze(0)
			# frameT = frameT.data.squeeze(0)
			test_loss = args.rec_D2_lambda * rec_loss_func(torch.from_numpy(pred_frameT_group).float(),
														  torch.from_numpy(St_GT).float())

			""" Compute Interpolation (St) PSNR & SSIM (D2, Stage II) """
			output_img = np.around(denorm255_np(
				np.transpose(pred_frameT_group, [1, 2, 0])[:, :, ::-1]))  # [h,w,c] and BGR2RGB, [-1,1]2[0,255]
			target_img = denorm255_np(
				np.transpose(St_GT, [1, 2, 0])[:, :, ::-1])  # [h,w,c] and BGR2RGB, [-1,1]2[0,255]
			output_img_crop, _, _ = crop_8x8(output_img)
			target_img_crop, _, _ = crop_8x8(target_img)

			intp_test_psnr = psnr(target_img_crop, output_img_crop)  # caution: calculation for RGB
			intp_test_ssim = ssim(target_img_crop, output_img_crop)  # caution: calculation for RGB

			""" Compute Deblur (S0,S1) PSNR & SSIM (D1, Stage II) """
			output_img = np.around(denorm255_np(
				np.transpose(pred_S0, [1, 2, 0])[:, :, ::-1]))  # [h,w,c] and BGR2RGB, [-1,1]2[0,255]
			target_img = denorm255_np(
				np.transpose(np.squeeze(S0_GT.detach().cpu().numpy()), [1, 2, 0])[:, :,
				::-1])  # [h,w,c] and BGR2RGB, [-1,1]2[0,255]
			output_img_crop, _, _ = crop_8x8(output_img)
			target_img_crop, _, _ = crop_8x8(target_img)

			test_psnr_S0 = psnr(target_img_crop, output_img_crop)  # caution: calculation for RGB
			test_ssim_S0 = ssim(target_img_crop, output_img_crop)  # caution: calculation for RGB

			output_img = np.around(denorm255_np(
				np.transpose(pred_S1, [1, 2, 0])[:, :, ::-1]))  # [h,w,c] and BGR2RGB, [-1,1]2[0,255]
			target_img = denorm255_np(
				np.transpose(np.squeeze(S1_GT.detach().cpu().numpy()), [1, 2, 0])[:, :,
				::-1])  # [h,w,c] and BGR2RGB, [-1,1]2[0,255]
			output_img_crop, _, _ = crop_8x8(output_img)
			target_img_crop, _, _ = crop_8x8(target_img)

			test_psnr_S1 = psnr(target_img_crop, output_img_crop)  # caution: calculation for RGB
			test_ssim_S1 = ssim(target_img_crop, output_img_crop)  # caution: calculation for RGB


			""" Saving Frames """
			if 'val' in post_fix:
				epoch_save_path = os.path.join(args.test_img_dir, model_dir, 'val_latest_' + post_fix)
			else:
				epoch_save_path = os.path.join(args.test_img_dir, model_dir, 'epoch_' + str(epoch).zfill(5) + post_fix)
			# './test_img_dir/DeFInet_exp1/epoch_00000_final_x8_first_960frames'
			check_folder(epoch_save_path)
			scene_save_path = os.path.join(epoch_save_path, scene_name[0])

			check_folder(scene_save_path)
			if (testIndex % (multiple - 1)) == 0:
				""" Save predicted S0_final, S1_final (x2) """
				cv2.imwrite(os.path.join(scene_save_path, S0_Path[0]),
							np.transpose(np.squeeze(denorm255_np(
								pred_S0)
							), [1, 2, 0]).astype(np.uint8))
				cv2.imwrite(os.path.join(scene_save_path, S1_Path[0]),
							np.transpose(np.squeeze(denorm255_np(
								pred_S1)
							), [1, 2, 0]).astype(np.uint8))

			""" Save predicted St_final """
			output_img = denorm255_np(
				np.transpose(pred_frameT_group, [1, 2, 0])[:, :, ::-1])  # [h,w,c] and BGR2RGB, [-1,1]2[0,255]
			cv2.imwrite(os.path.join(scene_save_path, St_Path[0]),
						output_img.astype(np.uint8)[:, :, ::-1])
			print("png for predicted St frame has been saved in [%s]" %
				  (os.path.join(scene_save_path, St_Path[0])))

			if visualization_flag:
				if 'val' in post_fix:
					OF_Occ_save_path = os.path.join(args.test_img_dir, model_dir, 'val_latest_OF_Occ' + post_fix)
				else:
					OF_Occ_save_path = os.path.join(args.test_img_dir, model_dir,
													'epoch_' + str(epoch).zfill(5) + '_OF_Occ' + post_fix)

				check_folder(OF_Occ_save_path)
				OF_Occ_scene_save_path = os.path.join(OF_Occ_save_path, scene_name[0])
				check_folder(OF_Occ_scene_save_path)
				OF_Occ_images = visualizations(two_blurry_inputs_full,
												  Sharps_prime[-1], Sharps_final[-1], St_GT, flows_pred, occs_pred,
														 blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1)
				cv2.imwrite(os.path.join(OF_Occ_scene_save_path, St_Path[0]),
							OF_Occ_images)

			# measure
			losses.update(test_loss.item(), 1)

			""" Save Performances for x8 MFI """
			if multiple == 8:
				if (testIndex % (multiple - 1)) == 0:
					PSNR_scene_1_prime.update(intp_test_psnr_prime, 1)  # Stage I
					SSIM_scene_1_prime.update(intp_test_ssim_prime, 1)
					PSNR_scene_1.update(intp_test_psnr, 1)  # Stage II
					SSIM_scene_1.update(intp_test_ssim, 1)

					""" Total average of total samples """
					intp_PSNRs_prime.update(intp_test_psnr_prime, 1)
					intp_SSIMs_prime.update(intp_test_ssim_prime, 1)
					intp_PSNRs.update(intp_test_psnr, 1)
					intp_SSIMs.update(intp_test_ssim, 1)

				elif (testIndex % (multiple - 1)) == 1:
					PSNR_scene_2_prime.update(intp_test_psnr_prime, 1)  # Stage I
					SSIM_scene_2_prime.update(intp_test_ssim_prime, 1)
					PSNR_scene_2.update(intp_test_psnr, 1)  # Stage II
					SSIM_scene_2.update(intp_test_ssim, 1)

					""" Total average of total samples """
					intp_PSNRs_prime.update(intp_test_psnr_prime, 1)
					intp_SSIMs_prime.update(intp_test_ssim_prime, 1)
					intp_PSNRs.update(intp_test_psnr, 1)
					intp_SSIMs.update(intp_test_ssim, 1)

				elif (testIndex % (multiple - 1)) == 2:
					PSNR_scene_3_prime.update(intp_test_psnr_prime, 1)  # Stage I
					SSIM_scene_3_prime.update(intp_test_ssim_prime, 1)
					PSNR_scene_3.update(intp_test_psnr, 1)  # Stage II
					SSIM_scene_3.update(intp_test_ssim, 1)

					""" Total average of total samples """
					intp_PSNRs_prime.update(intp_test_psnr_prime, 1)
					intp_SSIMs_prime.update(intp_test_ssim_prime, 1)
					intp_PSNRs.update(intp_test_psnr, 1)
					intp_SSIMs.update(intp_test_ssim, 1)


				elif (testIndex % (multiple - 1)) == 3:
					""" Center-Frame Interpolation (t=0.5) """
					PSNR_scene_4_prime.update(intp_test_psnr_prime, 1)  # Stage I
					SSIM_scene_4_prime.update(intp_test_ssim_prime, 1)
					PSNR_scene_4.update(intp_test_psnr, 1)  # Stage II
					SSIM_scene_4.update(intp_test_ssim, 1)
					"""" Note: For x8 MFI, save S0,S1 when interpolate frame at 0.5, without losing generality """
					cv2.imwrite(os.path.join(scene_save_path, S0_Path[0]),
								np.transpose(np.squeeze(denorm255_np(
									pred_S0)
								), [1, 2, 0]).astype(
									np.uint8))  # the deblurred frame from the later sliding window is saved for simplicity.
					cv2.imwrite(os.path.join(scene_save_path, S1_Path[0]),
								np.transpose(np.squeeze(denorm255_np(
									pred_S1)
								), [1, 2, 0]).astype(np.uint8))

					PSNR_scene_8_deblur_prime.update(test_psnr_S0_prime, 1)  # Stage I
					SSIM_scene_8_deblur_prime.update(test_ssim_S0_prime, 1)
					PSNR_scene_8_deblur.update(test_psnr_S0, 1)  # Stage II
					SSIM_scene_8_deblur.update(test_ssim_S0, 1)


					""" Total average of total samples """
					deblur_PSNRs_prime.update(test_psnr_S0_prime, 1)
					deblur_SSIMs_prime.update(test_ssim_S0_prime, 1)
					deblur_PSNRs.update(test_psnr_S0, 1)
					deblur_SSIMs.update(test_ssim_S0, 1)

					""" Total average of total samples """
					intp_PSNRs_prime.update(intp_test_psnr_prime, 1)
					intp_SSIMs_prime.update(intp_test_ssim_prime, 1)
					intp_PSNRs.update(intp_test_psnr, 1)
					intp_SSIMs.update(intp_test_ssim, 1)


				elif (testIndex % (multiple - 1)) == 4:
					PSNR_scene_5_prime.update(intp_test_psnr_prime, 1)  # Stage I
					SSIM_scene_5_prime.update(intp_test_ssim_prime, 1)
					PSNR_scene_5.update(intp_test_psnr, 1)  # Stage II
					SSIM_scene_5.update(intp_test_ssim, 1)

					""" Total average of total samples """
					intp_PSNRs_prime.update(intp_test_psnr_prime, 1)
					intp_SSIMs_prime.update(intp_test_ssim_prime, 1)
					intp_PSNRs.update(intp_test_psnr, 1)
					intp_SSIMs.update(intp_test_ssim, 1)

				elif (testIndex % (multiple - 1)) == 5:
					PSNR_scene_6_prime.update(intp_test_psnr_prime, 1)  # Stage I
					SSIM_scene_6_prime.update(intp_test_ssim_prime, 1)
					PSNR_scene_6.update(intp_test_psnr, 1)  # Stage II
					SSIM_scene_6.update(intp_test_ssim, 1)

					""" Total average of total samples """
					intp_PSNRs_prime.update(intp_test_psnr_prime, 1)
					intp_SSIMs_prime.update(intp_test_ssim_prime, 1)
					intp_PSNRs.update(intp_test_psnr, 1)
					intp_SSIMs.update(intp_test_ssim, 1)

				elif (testIndex % (multiple - 1)) == 6:
					PSNR_scene_7_prime.update(intp_test_psnr_prime, 1)  # Stage I
					SSIM_scene_7_prime.update(intp_test_ssim_prime, 1)
					PSNR_scene_7.update(intp_test_psnr, 1)  # Stage II
					SSIM_scene_7.update(intp_test_ssim, 1)

					""" Total average of total samples """
					intp_PSNRs_prime.update(intp_test_psnr_prime, 1)
					intp_SSIMs_prime.update(intp_test_ssim_prime, 1)
					intp_PSNRs.update(intp_test_psnr, 1)
					intp_SSIMs.update(intp_test_ssim, 1)

			elif multiple == 2:
				""" Center-Frame Interpolation (t=0.5) """
				PSNR_scene_4_prime.update(intp_test_psnr_prime, 1)  # Stage I
				SSIM_scene_4_prime.update(intp_test_ssim_prime, 1)
				PSNR_scene_4.update(intp_test_psnr, 1)  # Stage II
				SSIM_scene_4.update(intp_test_ssim, 1)
				"""" Note: For x8 MFI, save S0,S1 when interpolate frame at 0.5 without losing generality """
				cv2.imwrite(os.path.join(scene_save_path, S0_Path[0]),
							np.transpose(np.squeeze(denorm255_np(
								pred_S0)
							), [1, 2, 0]).astype(
								np.uint8))  # the deblurred frame from the later sliding window is saved for simplicity.
				cv2.imwrite(os.path.join(scene_save_path, S1_Path[0]),
							np.transpose(np.squeeze(denorm255_np(
								pred_S1)
							), [1, 2, 0]).astype(np.uint8))

				PSNR_scene_8_deblur_prime.update(test_psnr_S0_prime, 1)  # Stage I
				SSIM_scene_8_deblur_prime.update(test_ssim_S0_prime, 1)
				PSNR_scene_8_deblur.update(test_psnr_S0, 1)  # Stage II
				SSIM_scene_8_deblur.update(test_ssim_S0, 1)

				""" Total average of total samples """
				deblur_PSNRs_prime.update(test_psnr_S0_prime, 1)
				deblur_SSIMs_prime.update(test_ssim_S0_prime, 1)
				deblur_PSNRs.update(test_psnr_S0, 1)
				deblur_SSIMs.update(test_ssim_S0, 1)

				""" Total average of total samples """
				intp_PSNRs_prime.update(intp_test_psnr_prime, 1)
				intp_SSIMs_prime.update(intp_test_ssim_prime, 1)
				intp_PSNRs.update(intp_test_psnr, 1)
				intp_SSIMs.update(intp_test_ssim, 1)


			if (testIndex % (multiple - 1)) == multiple - 2:
				""" show results every last predicted frame in each interval """
				progress.print(testIndex)
				# for tensorboard
				TB_count = testIndex + epoch * (len(test_loader))
				writer.add_scalar('testLoss', losses.val / (multiple - 1), TB_count)
				writer.add_scalar('intp_testPSNR', intp_PSNRs.val / (multiple - 1), TB_count)
				writer.add_scalar('intp_testSSIM', intp_SSIMs.val / (multiple - 1), TB_count)
				writer.add_scalar('deblur_testPSNR', deblur_PSNRs.val / (multiple - 1), TB_count)
				writer.add_scalar('deblur_testSSIM', deblur_SSIMs.val / (multiple - 1), TB_count)

			prev_scene_name = scene_name[0]

		""" Note: deblurred frame from the later sliding window is saved for simplicity. """
		PSNR_scene_8_deblur_prime.update(test_psnr_S1_prime, 1)  # Stage I
		SSIM_scene_8_deblur_prime.update(test_ssim_S1_prime, 1)
		PSNR_scene_8_deblur.update(test_psnr_S1, 1)  # Stage II
		SSIM_scene_8_deblur.update(test_ssim_S1, 1)

		""" Total average of total samples """
		deblur_PSNRs_prime.update(test_psnr_S1_prime, 1)
		deblur_SSIMs_prime.update(test_ssim_S1_prime, 1)
		deblur_PSNRs.update(test_psnr_S1, 1)
		deblur_SSIMs.update(test_ssim_S1, 1)

		""" update last scene (Stage I) """
		PSNR_1_prime.update(PSNR_scene_1_prime.avg, 1)
		PSNR_2_prime.update(PSNR_scene_2_prime.avg, 1)
		PSNR_3_prime.update(PSNR_scene_3_prime.avg, 1)
		PSNR_4_prime.update(PSNR_scene_4_prime.avg, 1)
		PSNR_5_prime.update(PSNR_scene_5_prime.avg, 1)
		PSNR_6_prime.update(PSNR_scene_6_prime.avg, 1)
		PSNR_7_prime.update(PSNR_scene_7_prime.avg, 1)
		PSNR_8_deblur_prime.update(PSNR_scene_8_deblur_prime.avg, 1)

		SSIM_1_prime.update(SSIM_scene_1_prime.avg, 1)
		SSIM_2_prime.update(SSIM_scene_2_prime.avg, 1)
		SSIM_3_prime.update(SSIM_scene_3_prime.avg, 1)
		SSIM_4_prime.update(SSIM_scene_4_prime.avg, 1)
		SSIM_5_prime.update(SSIM_scene_5_prime.avg, 1)
		SSIM_6_prime.update(SSIM_scene_6_prime.avg, 1)
		SSIM_7_prime.update(SSIM_scene_7_prime.avg, 1)
		SSIM_8_deblur_prime.update(SSIM_scene_8_deblur_prime.avg, 1)

		""" update per scene (stage II)"""
		PSNR_1.update(PSNR_scene_1.avg, 1)
		PSNR_2.update(PSNR_scene_2.avg, 1)
		PSNR_3.update(PSNR_scene_3.avg, 1)
		PSNR_4.update(PSNR_scene_4.avg, 1)
		PSNR_5.update(PSNR_scene_5.avg, 1)
		PSNR_6.update(PSNR_scene_6.avg, 1)
		PSNR_7.update(PSNR_scene_7.avg, 1)
		PSNR_8_deblur.update(PSNR_scene_8_deblur.avg, 1)

		SSIM_1.update(SSIM_scene_1.avg, 1)
		SSIM_2.update(SSIM_scene_2.avg, 1)
		SSIM_3.update(SSIM_scene_3.avg, 1)
		SSIM_4.update(SSIM_scene_4.avg, 1)
		SSIM_5.update(SSIM_scene_5.avg, 1)
		SSIM_6.update(SSIM_scene_6.avg, 1)
		SSIM_7.update(SSIM_scene_7.avg, 1)
		SSIM_8_deblur.update(SSIM_scene_8_deblur.avg, 1)

		print("---------------------------------------- x8 MFI results--------------------------------------------")
		progress.print(testIndex)
		progress_PSNR_prime.print(testIndex)
		progress_SSIM_prime.print(testIndex)
		progress_PSNR.print(testIndex)
		progress_SSIM.print(testIndex)
		print(" Average Inference Time per Batch (1) :", batch_time.avg)
		print("----------------------------------------Test Ends--------------------------------------------")

	return losses.avg, intp_PSNRs.avg, intp_SSIMs.avg, deblur_PSNRs.avg, deblur_SSIMs.avg, epoch_save_path

def test_custom(test_loader, model_net, args, device, multiple, num_update, patch,
		 visualization_flag):
	batch_time = AverageClass('Time:', ':6.3f')
	accm_time = AverageClass('Accm_Time[s]:', ':6.3f')
	progress = ProgressMeter(len(test_loader), batch_time, accm_time, prefix='Custom Test for [{}]: '.format(args.custom_path))

	# switch to evaluate mode
	model_net.eval()

	print("------------------------------------------- Custom Test ----------------------------------------------")
	with torch.no_grad():
		fix_start_time = time.time()
		for testIndex, (frames, t_value, scene_name, frameRange) in enumerate(test_loader):
			# Getting the input and the target from the training set
			# Shape of 'frames' : [1,C,T+1,H,W]
			St_Path, S0_Path, S1_Path = frameRange  # tuple:string
			t_value = Variable(t_value.to(device))

			if (testIndex % (multiple - 1)) == 0:
				# input_frames = frames[:, :, :-1, :, :]  # [1,C,T,H,W]
				input_frames = frames # [1,C,T,H,W]
				input_frames = Variable(input_frames.to(device))

			start_time = time.time()

			""" Divide & Process due to Limited Memory """
			# compute output
			if not visualization_flag:
				two_blurry_inputs_full, Sharps_prime, Sharps_final, _, flows_pred, occs_pred = \
					patch_forward_DeFInet_itr(model_net, input_frames, None, t_value, num_update, patch,
											  args.patch_boundary)
			else:
				two_blurry_inputs_full, Sharps_prime, Sharps_final, _, flows_pred, occs_pred,\
					blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1= \
					patch_forward_DeFInet_w_diff(model_net, input_frames, None, t_value, num_update, patch,
											  args.patch_boundary)
			# pred_frameT_group = pred_frameT.data.float().cpu().squeeze(0)

			""" Measure Inference Time """
			batch_time.update(time.time() - start_time)
			start_time = time.time()
			accm_time.update(time.time() - fix_start_time)

			""" D2 """
			pred_S0, pred_S1, pred_frameT = Sharps_final
			pred_frameT_group = pred_frameT


			""" Saving Frames """
			scene_save_path = os.path.join(args.custom_path,
										   scene_name[0]+'_sharply_interpolated_x'+str(args.multiple_MFI))
			# './custom_path/Scene1__sharply_interpolated_xM'
			check_folder(scene_save_path)

			if (testIndex % (multiple - 1)) == 0:
				""" Save predicted S0_final, S1_final (x2) """
				cv2.imwrite(os.path.join(scene_save_path, S0_Path[0]),
							np.transpose(np.squeeze(denorm255_np(
								pred_S0)
							), [1, 2, 0]).astype(np.uint8))
				cv2.imwrite(os.path.join(scene_save_path, S1_Path[0]),
							np.transpose(np.squeeze(denorm255_np(
								pred_S1)
							), [1, 2, 0]).astype(np.uint8))

			""" Save predicted St_final """
			output_img = denorm255_np(
				np.transpose(pred_frameT_group, [1, 2, 0])[:, :, ::-1])  # [h,w,c] and BGR2RGB, [-1,1]2[0,255]
			cv2.imwrite(os.path.join(scene_save_path, St_Path[0]),
						output_img.astype(np.uint8)[:, :, ::-1])
			print("png for predicted St frame has been saved in [%s]" %
				  (os.path.join(scene_save_path, St_Path[0])))

			if visualization_flag:
				OF_Occ_save_path = scene_save_path + '_visualizations'
				check_folder(OF_Occ_save_path)
				OF_Occ_images = visualizations_custom(two_blurry_inputs_full,
												  Sharps_prime[-1], Sharps_final[-1], None, flows_pred, occs_pred,
														 blws_blww_source_ref_warped_flow1_diff_FCW_list_ch9_1to0_0to1)
				cv2.imwrite(os.path.join(OF_Occ_save_path, St_Path[0]),
							OF_Occ_images)

		print("---------------------------------------- test_custom results--------------------------------------------")
		progress.print(testIndex)
		print(" Average Inference Time per Batch (1) :", batch_time.avg)
		print("--------------------------------------- Custom Test Ends--------------------------------------------")

	return

if __name__ == '__main__':
	main()
