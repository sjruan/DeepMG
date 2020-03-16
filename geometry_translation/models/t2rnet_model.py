from geometry_translation.models.networks import define_translator, init_net
from geometry_translation.models.base_model import BaseModel
from geometry_translation.models.losses import SoftDiceLoss
from geometry_translation.models.metrics import Metrics
import torch
import numpy as np
import torch.nn as nn
from itertools import chain


class T2RNetModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.visual_names = []
        self.visual_names.append('traj_pt')
        self.visual_names.append('traj_line')
        self.visual_names.append('pred_rn_img')
        self.visual_names.append('pred_road_img')
        self.visual_names.append('real_rn')
        self.loss_names = ['centerline', 'region', 'all']
        self.model_names = ['Trans']
        input_nc = 11
        self.model_names.append('EM')
        transit_nc = 128
        transit_embedding_dim = 8
        self.netEM = nn.Sequential(
            nn.Linear(transit_nc, 64),
            nn.ReLU(),
            nn.Linear(64, transit_embedding_dim),
            nn.ReLU()
        )
        self.netEM = init_net(self.netEM, init_type='normal', init_gain=0.02, gpu_ids=opt.gpu_ids)
        input_nc += transit_embedding_dim

        # add the output of dilation
        self.netTrans = define_translator(input_nc, opt.output_nc, opt.net_trans, gpu_ids=opt.gpu_ids)
        self.metrics = Metrics()
        self.criterion = SoftDiceLoss()
        if opt.is_train:
            self.optimizer = torch.optim.Adam(
                chain(self.netEM.parameters(), self.netTrans.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
        self.traj = None
        self.traj_pt = None
        self.traj_line = None
        self.transist = None
        self.real_rn = None
        self.pred_rn = None
        self.pred_rn_img = None
        self.pred_road_img = None
        self.loss_all = None
        self.loss_centerline = None
        self.loss_region = None
        self.real_road = None
        self.pred_road = None

    def set_input(self, inp):
        self.traj = inp['spatial_view'].to(self.device)
        self.real_rn = inp['centerline'].to(self.device)
        self.traj_pt = self.traj[:, 0:1, :, :]
        self.traj_line = self.traj[:, 1:2, :, :]
        self.transist = inp['transition_view'].to(self.device)
        self.real_road = inp['region'].to(self.device)
        self.image_paths = inp['img_path']

    def forward(self):
        embedded_trans = self.netEM(self.transist.reshape([-1, 256, 256, 128])).permute(0, 3, 1, 2)
        inp = torch.cat([self.traj, embedded_trans], 1)
        self.pred_road, self.pred_rn = self.netTrans(inp)
        self.pred_rn_img = T2RNetModel.pred2im(self.pred_rn)
        self.pred_road_img = T2RNetModel.pred2im(self.pred_road)

    def _backward(self):
        self.loss_centerline = self.criterion(self.real_rn, self.pred_rn)
        self.loss_region = self.criterion(self.real_road, self.pred_road)
        self.loss_all = (1 - self.opt.lam) * self.loss_centerline + self.opt.lam * self.loss_region
        self.loss_all.backward()

    def optimize_parameters(self):
        self.forward()
        metrics = self.metrics(self.pred_rn, self.real_rn)
        road_metrics = self.metrics(self.pred_road, self.real_road)

        self.optimizer.zero_grad()
        self._backward()
        self.optimizer.step()
        return self.loss_all, self.loss_region, self.loss_centerline, metrics, road_metrics

    def test(self):
        BaseModel.test(self)
        metrics = self.metrics(self.pred_rn, self.real_rn)
        road_metrics = self.metrics(self.pred_road, self.real_road)
        self.loss_centerline = self.criterion(self.real_rn, self.pred_rn)
        self.loss_region = self.criterion(self.real_road, self.pred_road)
        self.loss_all = (1 - self.opt.lam) * self.loss_centerline + self.opt.lam * self.loss_region
        return self.pred_rn, self.loss_all, metrics, road_metrics

    @staticmethod
    def pred2im(image_tensor):
        image_numpy = image_tensor[0].cpu().float().detach().numpy()  # convert it into a numpy array
        # handle sigmoid cases
        image_numpy[image_numpy > 0.5] = 1
        image_numpy[image_numpy <= 0.5] = 0
        image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255
        return image_numpy
