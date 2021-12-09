import functools, torch, random
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

"""
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@     Proposed Architecture: DeMFI-Net     @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""
class DeMFInet(nn.Module):
    # reference: torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    def __init__(self, args):
        super(DeMFInet, self).__init__()
        self.args = args
        self.device = torch.device(
            'cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')  # will be used as "x.to(device)"
        self.nf = args.nf
        self.scale_factor = args.scale_factor
        # self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.relu = nn.ReLU()

        """ Stage I: DeMFI-Net_bs (bs: baseline version) [Fig.3(a)] """
        self.FF_RDB_Module = FF_RDB(args)
        self.FAC_FB_Module = FAC_FB(args)
        self.Refine_Module = UNet(args)

        self.Dec_first = nn.Conv3d(self.nf, self.nf, [1, 3, 3], 1, [0, 1, 1], bias=True)
        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN_3D, nf=self.nf)
        self.Decoder_res = make_layer(ResidualBlock_noBN_f, args.num_ResB_Dec)
        self.Dec_last1 = nn.Conv3d(self.nf, self.nf, [1, 3, 3], 1, [0, 1, 1], bias=True)
        self.Dec_last2 = nn.Conv3d(self.nf, 3, [1, 3, 3], 1, [0, 1, 1], bias=True)

        """ Stage II: DeMFI-Net_rb (rb: recursive boosting) [Fig.3(c)] """
        self.Ch_Reducer = nn.Conv2d(self.nf * 3, self.nf, 7, padding=3, bias=True)
        self.Booster_Module = Booster(args)

        self.Dec_first_2 = nn.Conv2d(9 + self.nf + (4 * 2 + 1) + (2 * 2 + 1) + 12, self.nf, 3, 1, 1, bias=True)
        ResidualBlock_noBN_f_2 = functools.partial(ResidualBlock_noBN, nf=self.nf)
        self.Decoder_res_2 = make_layer(ResidualBlock_noBN_f_2, args.num_ResB_Dec)
        self.Dec_last1_2 = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)
        self.Dec_last2_2 = nn.Conv2d(self.nf, 9, 3, 1, 1, bias=True)  # only focus on 0,t,1

    def forward(self, x, t_value, num_update=None, is_training=None):
        '''
        x shape : [B,C,T,H,W]
        t_value shape : [B,1] ###############
        '''
        B, C, T, H, W = x.size()
        B0 = x[:, :, 0, :, :]
        B1 = x[:, :, 1, :, :]
        B_m1 = x[:, :, 2, :, :]
        B2 = x[:, :, 3, :, :]

        """ Stage I: Feature-Flow-based Warping and Blending (FWB), Features to Sharp Frames """
        ## Features (F) and Flows (f) Extraction, caution: F (tanh) """
        F0, F1, flow_01, flow_10, occ_0_logit = self.FF_RDB_Module(B0, B1, B_m1, B2)

        ## t-Alignment 
        # Ft: "feature"-based backward warping and blending by using occlusion maps
        t_value = torch.unsqueeze(torch.unsqueeze(t_value, -1), -1)  # [B, 1, 1, 1]
        flow_t0, flow_t1 = CFR_flow_t_align(self.device, flow_01, flow_10, t_value) # CFR: Complementary Flow Reversal
        
        occ_0 = torch.sigmoid(occ_0_logit)
        occ_1 = 1 - occ_0
        Ft = (1 - t_value) * occ_0 * \
             bwarp(self.device, F0, flow_t0) \
             + t_value * occ_1 * bwarp(self.device, F1, flow_t1)
        Ft = Ft / ((1 - t_value) * occ_0 + t_value * occ_1) # Eq.(2)

        ## FAC-FB Module
        aF0, aF1, blending_weights, difference_maps = self.FAC_FB_Module(F0, F1, flow_10, flow_01)

        ## Refinement
        Agg1 = torch.cat([aF0, aF1, Ft, flow_t0, flow_t1, flow_01, flow_10, occ_0_logit], dim=1)
        Agg1 = self.Refine_Module(Agg1) + torch.cat(
            [flow_t0, flow_t1, occ_0_logit, aF0, aF1],
            dim=1)
        rflow_t0 = Agg1[:, :2, :, :]
        rflow_t1 = Agg1[:, 2:4, :, :]
        occ_0_logit = Agg1[:, 4:5, :, :]
        occ_0 = torch.sigmoid(occ_0_logit)
        occ_1 = 1 - occ_0
        rF0_dec1 = torch.tanh(Agg1[:, 5: 5 + self.nf, :, :])
        rF1_dec1 = torch.tanh(Agg1[:, 5 + self.nf: 5 + self.nf * 2, :, :])

        ## Decoding Features into Sharp Frames with D1 """
        rFt_dec1 = (1 - t_value) * occ_0 * \
                   bwarp(self.device, rF0_dec1, rflow_t0) \
                   + t_value * occ_1 * bwarp(self.device, rF1_dec1, rflow_t1)
        rFt_dec1 = rFt_dec1 / ((1 - t_value) * occ_0 + t_value * occ_1)

        Dec_inputs = torch.stack([rF0_dec1, rF1_dec1, rFt_dec1], 2)  # [B,C,3,H,W]
        out = self.Decoder_res(self.relu(self.Dec_first(Dec_inputs)))
        out = self.relu(self.Dec_last1(out))
        out = self.Dec_last2(out)
        S0p = out[:, :, 0, :, :]
        S1p = out[:, :, 1, :, :]
        Stp = out[:, :, 2, :, :]
        Sharps_dec1 = [S0p, S1p, Stp]

        """ Stage II: Pixel-Flow-based Warping and Blending (PWB), Frames to Frames (residual learning) """
        flow_predictions = []
        occ0_predictions = []
        flow_t0_t1_init = torch.cat((rflow_t0, rflow_t1), dim=1)
        flow_predictions.append(flow_t0_t1_init)
        occ0_predictions.append(occ_0)
        flow_t0_t1_predictions = []
        flow_t0_t1_predictions.append([rflow_t0, rflow_t1])

        # rec
        F_rec = torch.tanh(self.Ch_Reducer(torch.cat((rF0_dec1, rF1_dec1, rFt_dec1), 1)))  # [-1,1] due to "tanh"

        # ref
        t_ref = torch.cat((flow_t0_t1_init, occ_0_logit), 1)  # [B,5,H,W]
        length1_ref = torch.cat((flow_10, flow_01), 1)  # [B,4,H,W]
        Sp_ref = torch.cat((S0p, S1p, Stp, B0, B1, B_m1, B2), 1)  # [B,21,H,W]
        ref_list = [Sp_ref, length1_ref, t_ref]  # [B,21,H,W], [B,4,H,W], [B,5,H,W]

        # del
        delta_list = [flow_t0_t1_init, occ_0_logit]  # t-related, # [B,5,H,W]
        Sharps_final = []

        if num_update == None:
            # for 'summary' in 'main.py'
            num_update = 1

        for itr in range(num_update):
            ## Update: feature-flows (f_F) -> pixel-flows (f_P)
            F_rec, delta_flow, delta_occ = \
                self.Booster_Module(F_rec, ref_list, delta_list)

            delta_list[0] = delta_list[0] + delta_flow  # + delta
            delta_list[1] = delta_list[1] + delta_occ  # + delta

            flow_t0_final = delta_list[0][:, :2, :, :]
            flow_t1_final = delta_list[0][:, 2:4, :, :]
            occ_0_final = torch.sigmoid(delta_list[1])
            occ_1_final = 1 - occ_0_final
            occ0_predictions.append(occ_0_final)

            ## Pixel-Flow-based Warping and Blending (PWB)
            flow_predictions.append(torch.cat((flow_t0_final, flow_t1_final), dim=1))
            St_new = (1 - t_value) * occ_0_final * \
                     bwarp(self.device, S0p, flow_t0_final) \
                     + t_value * occ_1_final * bwarp(self.device, S1p, flow_t1_final)
            St_new = St_new / ((1 - t_value) * occ_0_final + t_value * occ_1_final)

            Agg3 = torch.cat([S0p, S1p, St_new,
                                     F_rec,
                                     occ_0, rflow_t0, rflow_t1, flow_10, flow_01,
                                     flow_t0_final, flow_t1_final, occ_0_final,
                                     B0, B1, B_m1, B2], 1)  # [B,15+self.nf+(6*2+1)+(2*2+1)+12,H,W]
            
            ## Boosting Sharp Frames from D1 with D2 (residual learning) """
            out = self.Decoder_res_2(self.relu(self.Dec_first_2(Agg3)))
            out = self.relu(self.Dec_last1_2(out))
            out = self.Dec_last2_2(out)
            S0_final = out[:, 0:3, :, :] + S0p
            S1_final = out[:, 3:6, :, :] + S1p
            St_final = out[:, 6:9, :, :] + St_new

            Sharps_final.append([S0_final, S1_final, St_final])

        if self.args.visualization_flag:
            blending_weights.append([flow_01, flow_10])

        if is_training:
            return Sharps_dec1, Sharps_final, flow_predictions, occ0_predictions, \
                   torch.mean(x[:, :, 0:2, :, :], dim=2), difference_maps, flow_t0_t1_predictions

        elif (not is_training and self.args.visualization_flag):
            return Sharps_dec1, Sharps_final, flow_predictions, occ0_predictions, \
                   torch.mean(x[:, :, 0:2, :, :], dim=2), blending_weights, difference_maps
        else:
            return Sharps_dec1, Sharps_final, flow_predictions, occ0_predictions, torch.mean(x[:, :, 0:2, :, :],
                                                                                             dim=2)



"""
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@   Main Components    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""
""" [Stage I] DeMFI-Net_bs : baseline version """
class FF_RDB(nn.Module):
    def __init__(self, args,
                 G0=96,
                 num_RDB=12,
                 C=4,
                 G=32):
        super(FF_RDB, self).__init__()
        """ RDN_res-based FF_RDB_Module """
        self.args = args
        self.nf = args.nf
        self.scale_factor = self.args.scale_factor
        self.G0 = G0  # 64
        kSize = 3

        """ # of RDB blocks, conv layers, out channels """
        self.num_RDB = num_RDB  # 6
        self.C = C  # 4
        self.G = G  # 32

        """ Shallow Feature Extraction """
        self.SFENet1 = nn.Conv2d((3 + 3 + 3 + 3) * self.scale_factor * self.scale_factor,
                                 self.G0, 5, padding=2, stride=1)
        self.SFENet2 = nn.Conv2d(self.G0, self.G0, kSize, padding=(kSize - 1) // 2, stride=1)

        """ RDBs """
        self.RDBs = nn.ModuleList()
        for i in range(self.num_RDB):
            self.RDBs.append(
                RDB(growRate0=self.G0, growRate=self.G, nConvLayers=self.C)
            )

        """ Global Feature Fusion """
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.num_RDB * self.G0, self.G0, 1, padding=0, stride=1),
            nn.Conv2d(self.G0, self.G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        """ UP-sampling Net """
        self.UPNet = nn.Sequential(*[
            nn.Conv2d(self.G0, 256, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PixelShuffle(2),
            nn.Conv2d(64, self.nf * 2 + 4 + 1, kSize, padding=(kSize - 1) // 2, stride=1)
        ])  # modification

    def forward(self, B0, B1, Bm1, B2):
        cat_B0B1Bm1B2 = torch.cat((B0, B1, Bm1, B2), 1)
        B_shuffle = pixel_reshuffle(cat_B0B1Bm1B2, self.scale_factor)
        B_input = B_shuffle
        f__1 = self.SFENet1(B_input)
        x = self.SFENet2(f__1)
        RDBs_out = []
        for i in range(self.num_RDB):
            x = self.RDBs[i](x)
            RDBs_out.append(x)
        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1

        """ Modification """
        S = self.UPNet(x)
        F0F1 = torch.tanh(S[:, :2 * self.nf, :, :])
        flows = S[:, 2 * self.nf: 2 * self.nf + 4, :, :]
        occ = S[:, 2 * self.nf + 4: 2 * self.nf + 4 + 1, :, :]

        return F0F1[:, :self.nf, :, :], F0F1[:, self.nf:self.nf * 2, :, :], flows[:, 0:2, :, :], flows[:, 2:4, :,
                                                                                                 :], occ


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


def pixel_reshuffle(input, upscale_factor):
    r"""Rearranges elements in a tensor of shape ``[*, C, H, W]`` to a
    tensor of shape ``[C*r^2, H/r, W/r]``.

    See :class:`~torch.nn.PixelShuffle` for details.

    Args:
        input (Variable): Input
        upscale_factor (int): factor to increase spatial resolution by

    Examples:
        >>> input = autograd.Variable(torch.Tensor(1, 3, 12, 12))
        >>> output = pixel_reshuffle(input,2)
        >>> print(output.size())
        torch.Size([1, 12, 6, 6])
    """
    batch_size, channels, in_height, in_width = input.size()

    # // division is to keep data type unchanged. In this way, the out_height is still int type
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor
    input_view = input.contiguous().view(batch_size, channels, out_height, upscale_factor, out_width,
                                         upscale_factor)
    channels = channels * upscale_factor * upscale_factor

    shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return shuffle_out.view(batch_size, channels, out_height, out_width)


class FAC_FB(nn.Module):
    def __init__(self, args):
        super(FAC_FB, self).__init__()
        self.args = args
        self.nf = args.nf
        self.conv_first = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)
        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=self.nf)
        self.feature_extraction = make_layer(ResidualBlock_noBN_f, args.num_ResB_FACFB)
                
        if args.shared_FGAC_flag:
            self.shared_FGAC = FGAC(args)
        else:
            self.FGAC_F1toF0 = FGAC(args)
            self.FGAC_F0toF1 = FGAC(args)
        self.relu = nn.ReLU()

    def forward(self, F0, F1, flow_10, flow_01):
        # feature size: F0 = F1 = [B, nf, H, W]
        x = torch.stack([F0, F1], dim=1)
        # "torch.stack": Concatenates sequence of tensors along a "new" dimension.

        B, N, C, H, W = x.size()
        ## extract features
        enc_fea = self.relu(self.conv_first(x.view(-1, C, H, W)))  # (B*N) temporally shared for all frames
        enc_fea = self.feature_extraction(enc_fea)
        enc_fea = enc_fea.contiguous().view(B, N, -1, H, W)
        if self.args.shared_FGAC_flag:
            aligned_F0, blending_weight_F0, diff_1to0 = self.shared_FGAC(enc_fea[:, 1, :, :, :], enc_fea[:, 0, :, :, :],
                                                                         flow_01)  # F1 to F0
            aligned_F1, blending_weight_F1, diff_0to1 = self.shared_FGAC(enc_fea[:, 0, :, :, :], enc_fea[:, 1, :, :, :],
                                                                         flow_10)  # F0 to F1

        else:
            aligned_F0, blending_weight_F0, diff_1to0 = self.FGAC_F1toF0(enc_fea[:, 1, :, :, :], enc_fea[:, 0, :, :, :],
                                                                         flow_01)  # F1 to F0
            aligned_F1, blending_weight_F1, diff_0to1 = self.FGAC_F0toF1(enc_fea[:, 0, :, :, :], enc_fea[:, 1, :, :, :],
                                                                         flow_10)  # F0 to F1

        return aligned_F0, aligned_F1, [blending_weight_F0, blending_weight_F1, blending_weight_F0, blending_weight_F1], \
               [diff_1to0, diff_0to1, diff_1to0, diff_0to1]


class FGAC(nn.Module):
    def __init__(self, args):
        super(FGAC, self).__init__()
        """ Flow-Guided Attentive Correlation """
        self.args = args
        self.nf = args.nf
        self.scale = [1]

        self.conv_ref_k = nn.Conv2d(self.nf, self.nf, [1, 1], 1,[0, 0])
        self.conv_source_k = nn.Conv2d(self.nf, self.nf, [1, 1], 1, [0, 0])
        self.feature_ch = self.nf
        self.softmax = nn.Softmax(dim=1)

        self.w_gen = nn.Conv2d(self.nf * 2, self.nf, [3, 3], 1, [1, 1])
        self.w_gen_2 = nn.Conv2d(self.nf, 1, [3, 3], 1, [1, 1])
        self.relu = nn.ReLU()

        self.fusion = nn.Conv2d(self.nf, self.nf, [1, 1], 1, [0, 0])

        # self.w = torch.tensor([1.0], requires_grad=True, device=device)
        # optimizer = torch.optim.Adam([{'params':model_net.parameters()},
        # 							  {'params':model_net.FAC_FB_Module.FGAC_F1toF0.w,'lr':1e-3},
        # 							  {'params':model_net.FAC_FB_Module.FGAC_F0toF1.w,'lr':1e-3}], lr=args.init_lr,
        # 							 betas=(0.9, 0.999), weight_decay=args.weight_decay)  # optimizer in "main.py"

    def forward(self, ref, source, flow_s2r):
        init_ref_k = self.conv_ref_k(ref)
        init_source_k = self.conv_source_k(source)
        source_v = source

        ref_k = init_ref_k
        source_k = init_source_k

        flow_s2r = flow_s2r.contiguous().permute(0, 2, 3, 1).float()  # [B,H,W,2]
        f_bs, f_h, f_w, f_c = flow_s2r.shape
        
        """ 
            This is a generalized version when there are both radii for sources (sr) and ref. (rr) 
            For DeMFI, due to point-wise FGAC, we set rr=0 and sr=0.            
        """
        rr = 0
        sr = 0
        """ (i) make centroid based on flow_s2r, then bilinear sampling on ref_k """
        # (i-1): make grid
        dx = torch.linspace(-rr, rr, 2 * rr + 1)
        dy = torch.linspace(-rr, rr, 2 * rr + 1)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(flow_s2r.device)  # [B,2rr+1,2rr+1,2]
        delta_lvl = delta.contiguous().view(1, 1, 2 * rr + 1, 1, 2 * rr + 1, 2).repeat(1, f_h, 1, f_w, 1, 1). \
            contiguous().view(1, f_h * (2 * rr + 1), f_w * (2 * rr + 1), 2)  # [B, H*(2rr+1),W*(2rr+1),2]

        # (i-2): make centroid by using flow
        # flow_s2r = flow_s2r.contiguous().view(1, 1,f_h, 1,f_w, 2).repeat(1, 2*rr+1, 1, 2*rr+1, 1, 1)
        centroid_lvl = flow_s2r.repeat(1, 2 * rr + 1, 2 * rr + 1, 1)   # [B,H*(2rr+1),W*(2rr+1),2]

        # (i-3): make flow-grid and bilinear sampling
        flow_s2r_lvl = centroid_lvl + delta_lvl  # grid (including flow and coordinates): [B,H*(2rr+1),W*(2rr+1), 2]
        ref_k = F.avg_pool2d(ref_k, (2 * sr + 1, 2 * sr + 1), (1, 1), padding=sr)
        # gathering size of "source grid" in ref_k via average pooling.
        indexed_ref_k = bilinear_sampler(ref_k, flow_s2r_lvl)  # ref: [B,c,h,w], grid: [B,H*(2rr+1),W*(2rr+1), 2]
        # indexed_ref_k: [B,C,H*(2rr+1),W*(2rr+1)] (following dim. of grid)

        indexed_ref_k = indexed_ref_k.contiguous().view(f_bs, self.feature_ch, f_h, (2 * rr + 1), f_w,
                                                        (2 * rr + 1)).permute(0,1,3,2,5,4)
        indexed_ref_k = indexed_ref_k.contiguous().view(f_bs, self.feature_ch, (2 * rr + 1) * f_h,
                                                        (2 * rr + 1) * f_w)  # [batch,C,(2rr+1)*H,(2rr+1)*W]
        # caution: order is very important !
        indexed_ref_k = F.unfold(indexed_ref_k,
                                 kernel_size=((2 * rr + 1), (2 * rr + 1)),
                                 stride=((2 * rr + 1), (2 * rr + 1)), padding=rr)  # [batch, C*((2rr+1)**2), H, W]
        grid_sampled_ref_k = indexed_ref_k.contiguous().view(f_bs, self.feature_ch, (2 * rr + 1) ** 2, f_h, f_w)
        # [batch, C, (2rr+1)**2, H, W]

        """ (ii) unfold source_k for computing attentive correlation """
        source_k = F.avg_pool2d(source_k, (2 * sr + 1, 2 * sr + 1), (1, 1), padding=sr)
        # gathering size of "source grid" in source_k via average pooling.
        source_k = torch.unsqueeze(source_k, 2)
        # [batch, C, 1, H, W]
        corr_r2s_k = torch.sum(grid_sampled_ref_k * source_k, 1)  # ab
        # element-wise multiplication (source_k is broadcasted), then sum.
        # [batch, (2rr+1)**2, H, W]
        softmax_corr_r2s_k = torch.unsqueeze(self.softmax(corr_r2s_k), 1)
        # [batch, 1, (2rr+1)**2, H, W]
        FAC_sr = torch.sum(grid_sampled_ref_k * softmax_corr_r2s_k, 2)  # Eq.(3)
        # element-wise multiplication (softmax_corr_r2s_k is broadcasted)
        # [batch, C, H, W]

        
        E_s = self.fusion(FAC_sr) # right term of Eq.(4)
        w_sr = torch.sigmoid(self.w_gen_2(
            self.relu(self.w_gen(torch.cat([source_v, E_s], dim=1)))))  # spatially variant (adaptive)

        bolstered_F_s = w_sr * source_v + (1 - w_sr) * E_s # Eq.(4)
        
        """ min-max normalization for visualization of difference feature maps after applying Eq.(4) """
        # diff = torch.abs(bolstered_F_s) - torch.abs(source_v)
        diff = bolstered_F_s - source_v
        diff = torch.mean(torch.abs(diff), 1, keepdim=True)
        b, c, h, w = diff.shape
        diff = diff.view(b, -1)
        diff -= diff.min(1, keepdim=True)[0]
        diff /= diff.max(1, keepdim=True)[0]
        diff = diff.view(b, 1, h, w)

        if self.args.visualization_flag:
            E_s = torch.mean(torch.abs(E_s), 1, keepdim=True)
            b, c, h, w = E_s.shape
            E_s = E_s.view(b, -1)
            E_s -= E_s.min(1, keepdim=True)[0]
            E_s /= E_s.max(1, keepdim=True)[0]
            E_s = E_s.view(b, 1, h, w)

            source_v = torch.mean(torch.abs(source_v), 1, keepdim=True)
            b, c, h, w = source_v.shape
            source_v = source_v.view(b, -1)
            source_v -= source_v.min(1, keepdim=True)[0]
            source_v /= source_v.max(1, keepdim=True)[0]
            source_v = source_v.view(b, 1, h, w)

            init_ref_k = torch.mean(torch.abs(init_ref_k), 1, keepdim=True)
            b, c, h, w = init_ref_k.shape
            init_ref_k = init_ref_k.view(b, -1)
            init_ref_k -= init_ref_k.min(1, keepdim=True)[0]
            init_ref_k /= init_ref_k.max(1, keepdim=True)[0]            
            init_ref_k = init_ref_k.view(b, 1, h, w)
            
            bolstered_F_s_ch1 = torch.mean(torch.abs(bolstered_F_s), 1, keepdim=True)
            b, c, h, w = bolstered_F_s_ch1.shape
            bolstered_F_s_ch1 = bolstered_F_s_ch1.view(b, -1)
            bolstered_F_s_ch1 -= bolstered_F_s_ch1.min(1, keepdim=True)[0]
            bolstered_F_s_ch1 /= bolstered_F_s_ch1.max(1, keepdim=True)[0]            
            bolstered_F_s_ch1 = bolstered_F_s_ch1.view(b, 1, h, w)

            return bolstered_F_s, [w_sr, (1 - w_sr),
                             source_v, init_ref_k, E_s, bolstered_F_s_ch1], diff
        else:
            return bolstered_F_s, w_sr, diff


def bilinear_sampler(img, flow_s2r_lvl, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = flow_s2r_lvl.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    # img = F.grid_sample(img, grid, align_corners=True)
    img = F.grid_sample(img, grid, align_corners=True)  # check: align_corners

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN_3D(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN_3D, self).__init__()
        self.conv1 = nn.Conv3d(nf, nf, [1, 3, 3], 1, [0, 1, 1], bias=True)
        self.conv2 = nn.Conv3d(nf, nf, [1, 3, 3], 1, [0, 1, 1], bias=True)

    # initialization # check
    # initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    # initialization # check
    # initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class UNet(nn.Module):
    def __init__(self, args):
        super(UNet, self).__init__()
        self.args = args
        self.nf = args.nf

        self.relu = nn.ReLU()
        self.NN = nn.UpsamplingNearest2d(scale_factor=2)

        self.enc1 = nn.Conv2d((self.nf) * 3 + 4 * 2 + 1, self.nf, [4, 4], 2, [1, 1])
        self.enc2 = nn.Conv2d(self.nf, 2 * self.nf, [4, 4], 2, [1, 1])
        self.enc3 = nn.Conv2d(2 * self.nf, 4 * self.nf, [4, 4], 2, [1, 1])
        # self.enc4 = nn.Conv2d(4*self.nf, 4*self.nf, [4, 4], 2, [1, 1])

        self.dec0 = nn.Conv2d(4 * self.nf, 4 * self.nf, [3, 3], 1, [1, 1])
        self.dec1 = nn.Conv2d(4 * self.nf + 2 * self.nf, 2 * self.nf, [3, 3], 1,
                              [1, 1])  ## input concatenated with enc2
        self.dec2 = nn.Conv2d(2 * self.nf + self.nf, self.nf, [3, 3], 1, [1, 1])
        self.dec3 = nn.Conv2d(self.nf, 2 * 2 + 1 + (self.nf * 2), [3, 3], 1, [1, 1])

    def forward(self, concat):
        enc1 = self.relu(self.enc1(concat))
        enc2 = self.relu(self.enc2(enc1))
        out = self.relu(self.enc3(enc2))

        out = self.relu(self.dec0(out))
        out = self.NN(out)

        out = torch.cat((out, enc2), dim=1)
        out = self.relu(self.dec1(out))

        out = self.NN(out)
        out = torch.cat((out, enc1), dim=1)
        out = self.relu(self.dec2(out))

        out = self.NN(out)
        out = self.dec3(out)
        return out


def CFR_flow_t_align(device, flow_01, flow_10, t_value):
    """ modified from https://github.com/JihyongOh/XVFI/blob/main/XVFInet.py"""
    ## Feature warping
    flow_01, norm0 = fwarp(device, flow_01,
                           t_value * flow_01)  ## Actually, F (t) -> (t+1). Translation. Not normalized yet
    flow_10, norm1 = fwarp(device, flow_10, (
            1 - t_value) * flow_10)  ## Actually, F (1-t) -> (-t). Translation. Not normalized yet

    flow_t0 = -(1 - t_value) * (t_value) * flow_01 + (t_value) * (t_value) * flow_10
    flow_t1 = (1 - t_value) * (1 - t_value) * flow_01 - (t_value) * (1 - t_value) * flow_10

    norm = (1 - t_value) * norm0 + t_value * norm1
    mask_ = (norm.detach() > 0).type(norm.type())
    flow_t0 = (1 - mask_) * flow_t0 + mask_ * (flow_t0.clone() / (norm.clone() + (1 - mask_)))
    flow_t1 = (1 - mask_) * flow_t1 + mask_ * (flow_t1.clone() / (norm.clone() + (1 - mask_)))

    return flow_t0, flow_t1


def fwarp(device, img, flo):
    """
        -img: image (N, C, H, W)
        -flo: optical flow (N, 2, H, W)
        elements of flo is in [0, H] and [0, W] for dx, dy

    """

    # (x1, y1)		(x1, y2)
    # +---------------+
    # |				  |
    # |	o(x, y) 	  |
    # |				  |
    # |				  |
    # |				  |
    # |				  |
    # +---------------+
    # (x2, y1)		(x2, y2)

    N, C, _, _ = img.size()

    # translate start-point optical flow to end-point optical flow
    y = flo[:, 0:1:, :]
    x = flo[:, 1:2, :, :]

    x = x.repeat(1, C, 1, 1)
    y = y.repeat(1, C, 1, 1)

    # Four point of square (x1, y1), (x1, y2), (x2, y1), (y2, y2)
    x1 = torch.floor(x)
    x2 = x1 + 1
    y1 = torch.floor(y)
    y2 = y1 + 1

    # firstly, get gaussian weights
    w11, w12, w21, w22 = get_gaussian_weights(x, y, x1, x2, y1, y2)

    # secondly, sample each weighted corner
    img11, o11 = sample_one(device, img, x1, y1, w11)
    img12, o12 = sample_one(device, img, x1, y2, w12)
    img21, o21 = sample_one(device, img, x2, y1, w21)
    img22, o22 = sample_one(device, img, x2, y2, w22)

    imgw = img11 + img12 + img21 + img22
    o = o11 + o12 + o21 + o22

    return imgw, o


def get_gaussian_weights(x, y, x1, x2, y1, y2):
    w11 = torch.exp(-((x - x1) ** 2 + (y - y1) ** 2))
    w12 = torch.exp(-((x - x1) ** 2 + (y - y2) ** 2))
    w21 = torch.exp(-((x - x2) ** 2 + (y - y1) ** 2))
    w22 = torch.exp(-((x - x2) ** 2 + (y - y2) ** 2))

    return w11, w12, w21, w22


def sample_one(device, img, shiftx, shifty, weight):
    """
    Input:
        -img (N, C, H, W)
        -shiftx, shifty (N, c, H, W)
    """

    N, C, H, W = img.size()

    # flatten all (all restored as Tensors)
    flat_shiftx = shiftx.view(-1)
    flat_shifty = shifty.view(-1)
    flat_basex = torch.arange(0, H, requires_grad=False).view(-1, 1)[None, None].to(device).long().repeat(N, C,
                                                                                                          1,
                                                                                                          W).view(
        -1)
    flat_basey = torch.arange(0, W, requires_grad=False).view(1, -1)[None, None].to(device).long().repeat(N, C,
                                                                                                          H,
                                                                                                          1).view(
        -1)
    flat_weight = weight.view(-1)
    flat_img = img.contiguous().view(-1)

    # The corresponding positions in I1
    idxn = torch.arange(0, N, requires_grad=False).view(N, 1, 1, 1).to(device).long().repeat(1, C, H, W).view(
        -1)
    idxc = torch.arange(0, C, requires_grad=False).view(1, C, 1, 1).to(device).long().repeat(N, 1, H, W).view(
        -1)
    # ttype = flat_basex.type()
    idxx = flat_shiftx.long() + flat_basex
    idxy = flat_shifty.long() + flat_basey

    # recording the inside part the shifted
    mask = idxx.ge(0) & idxx.lt(H) & idxy.ge(0) & idxy.lt(W)

    # Mask off points out of boundaries
    ids = (idxn * C * H * W + idxc * H * W + idxx * W + idxy)
    ids_mask = torch.masked_select(ids, mask).clone().to(device)

    # Note here! accmulate fla must be true for proper bp
    img_warp = torch.zeros([N * C * H * W, ]).to(device)
    img_warp.put_(ids_mask, torch.masked_select(flat_img * flat_weight, mask), accumulate=True)

    one_warp = torch.zeros([N * C * H * W, ]).to(device)
    one_warp.put_(ids_mask, torch.masked_select(flat_weight, mask), accumulate=True)

    return img_warp.view(N, C, H, W), one_warp.view(N, C, H, W)


def bwarp(device, x, flo):
    '''
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    '''
    B, C, H, W = x.size()
    # mesh grid
    # xx = torch.arange(0,W).view(1,-1).repeat(H,1)
    # yy = torch.arange(0,H).view(-1,1).repeat(1,W)
    # xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    # yy = xx.view(1,1,H,W).repeat(B,1,1,1)
    xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
    yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.to(device)
    vgrid = torch.autograd.Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)  # [B,H,W,2]
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    mask = torch.autograd.Variable(torch.ones(x.size())).to(device)
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

    # mask[mask<0.9999] = 0
    # mask[mask>0] = 1
    mask = mask.masked_fill_(mask < 0.999, 0)
    mask = mask.masked_fill_(mask > 0, 1)

    return output * mask


""" [Stage II] DeMFI-Net_rb : recursive boosting """
class Booster(nn.Module):
    def __init__(self, args):
        super(Booster, self).__init__()
        self.args = args
        self.nf = args.nf
        self.Mixer = Mixer(args)
        self.GB = SepConvGRU(h_dim=args.nf, x_dim=args.nf)  # forward(self, h, x)
        self.flow_occ = FlowOcc(x_dim=args.nf, nf=args.nf)  # forward(self, x)

    def forward(self, F_rec, ref_list, delta_list):
        """
            :param F_rec: torch.cat((S0p, S1p, Stp), 1) # [B,9,H,W]
            :param ref_list: [Sp_ref, length1_ref, t_ref] # [B,21,H,W], [B,4,H,W], [B,5,H,W]
            :param delta_list:  [del_flow_t0_t1, del_occ_0_logit]  # t-related, # [B,5,H,W]
        """

        blend_enc = self.Mixer(ref_list, delta_list)  # Agg2

        F_rec = self.GB(F_rec, blend_enc)

        delta_flow_occ = self.flow_occ(F_rec)
        delta_flow = delta_flow_occ[:, :4, :, :]
        delta_occ = delta_flow_occ[:, 4:5, :, :]
        return F_rec, delta_flow, delta_occ


class Mixer(nn.Module):
    def __init__(self, args):
        super(Mixer, self).__init__()
        self.args = args
        self.conv_ref1 = nn.Conv2d(21 + 10 + 5 - 2 - 4, args.nf // 2, 7, padding=3)
        self.conv_ref2 = nn.Conv2d(args.nf // 2, args.nf // 2, 3, padding=1)

        self.conv_delta1 = nn.Conv2d(5, args.nf // 2, 7, padding=3)
        self.conv_delta2 = nn.Conv2d(args.nf // 2, args.nf // 2, 3, padding=1)

        self.conv_blend1 = nn.Conv2d(args.nf, args.nf // 2, 3, padding=1)
        self.conv_blend2 = nn.Conv2d(args.nf // 2, args.nf, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, ref_list, delta_list):
        """
            :param ref_list: [Sp_ref, length1_ref, t_ref] # [B,21,H,W], [B,4,H,W], # [B,5,H,W]
            :param delta_list:  [del_flow_t0_t1, del_occ_0_logit]  # t-related, # [B,5,H,W]
        """
        ref_enc = self.relu(self.conv_ref1(torch.cat(ref_list, 1)))
        ref_enc = self.relu(self.conv_ref2(ref_enc))

        delta_enc = self.relu(self.conv_delta1(torch.cat(delta_list, 1)))
        delta_enc = self.relu(self.conv_delta2(delta_enc))

        blend_enc = self.relu((self.conv_blend1(torch.cat([ref_enc, delta_enc], dim=1))))
        blend_enc = self.relu((self.conv_blend2(blend_enc)))

        return blend_enc


class SepConvGRU(nn.Module):
    def __init__(self, h_dim, x_dim):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(h_dim + x_dim, h_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(h_dim + x_dim, h_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(h_dim + x_dim, h_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(h_dim + x_dim, h_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(h_dim + x_dim, h_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(h_dim + x_dim, h_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        """
            :param h: F_rec # [B,64,H,W]
            :param x: blend_enc # [B,64,H,W]
        """
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class FlowOcc(nn.Module):
    def __init__(self, x_dim, nf):
        super(FlowOcc, self).__init__()
        self.conv1 = nn.Conv2d(x_dim, nf // 2, 3, padding=1)
        self.conv2 = nn.Conv2d(nf // 2, 5, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))



