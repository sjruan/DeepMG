import sys
sys.path.append('../')
from geometry_translation.options.train_options import TrainOptions
from geometry_translation.utils.visualizer import Visualizer
from geometry_translation.models import create_model
from geometry_translation.data_loader import get_data_loader
from datetime import datetime
import os


class Trainer:
    def __init__(self, opt, model, train_dl, val_dl, visualizer):
        self.opt = opt
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.visualizer = visualizer

    def fit(self):
        best_f1_score = 0.0
        # training phase
        tot_iters = 0
        for epoch in range(1, self.opt.n_epochs + 1):
            print(f'epoch {epoch}/{self.opt.n_epochs}')
            ep_time = datetime.now()
            for i, data in enumerate(self.train_dl):
                self.model.train()
                self.model.set_input(data)
                iter_loss_all, iter_loss_road, iter_loss_cl, iter_metrics, iter_road_metrics = self.model.optimize_parameters()
                iter_metrics = iter_metrics.numpy()
                iter_road_metrics = iter_road_metrics.numpy()
                print("[Epoch %d/%d] [Batch %d/%d] [Loss: %f] [Road Loss: %f] [CL Loss: %f] [Precision: %f] [Recall: %f] [F1: %f] [Road IOU: %f] [CL IOU: %f]" % (epoch, opt.n_epochs, i, len(train_dl), iter_loss_all.item(), iter_loss_road.item(), iter_loss_cl.item(), iter_metrics[0], iter_metrics[1], iter_metrics[2], iter_road_metrics[3], iter_metrics[3]))
                tot_iters += 1

                if tot_iters % self.opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                    save_result = tot_iters % self.opt.update_html_freq == 0
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
                if tot_iters % self.opt.print_freq == 0:  # print training losses
                    losses = model.get_current_losses()
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, i / len(self.train_dl), losses)

                # validating phase
                if tot_iters % opt.sample_interval == 0:
                    self.model.eval()
                    tot_loss = 0
                    tot_metrics = 0
                    tot_road_metrics = 0
                    for i, data in enumerate(self.val_dl):
                        self.model.set_input(data)
                        _, iter_loss, iter_metrics, iter_road_metrics = self.model.test()
                        tot_loss += iter_loss.item()
                        tot_metrics += iter_metrics.numpy()
                        tot_road_metrics += iter_road_metrics.numpy()
                    tot_loss /= len(self.val_dl)
                    tot_metrics /= len(self.val_dl)
                    tot_road_metrics /= len(self.val_dl)
                    if tot_metrics[2] > best_f1_score:
                        best_f1_score = tot_metrics[2]
                        self.model.save_networks('latest')
                        self.model.save_networks(epoch)
                        with open(os.path.join(opt.checkpoints_dir, opt.name, 'results.txt'), 'a') as f:
                            f.write('epoch\t{}\titer\t{}\tloss\t{:.6f}\tprecision\t{:.4f}\trecall\t{:.4f}\tf1\t{:.4f}\troad_iou\t{:.4f}\tcl_iou\t{:.4f}\n'.format(epoch, tot_iters, tot_loss, tot_metrics[0], tot_metrics[1], tot_metrics[2], tot_road_metrics[3], tot_metrics[3]))
                            f.close()
            print('=================time cost: {}==================='.format(datetime.now() - ep_time))
            self.model.update_learning_rate()


if __name__ == '__main__':
    opt = TrainOptions().parse()
    model = create_model(opt)
    model.setup(opt)
    train_dl = get_data_loader(opt.dataroot, 'train')
    val_dl = get_data_loader(opt.dataroot, 'val')
    visualizer = Visualizer(opt)
    trainer = Trainer(opt, model, train_dl, val_dl, visualizer)
    trainer.fit()
