import sys
sys.path.append('../')
from geometry_translation.options.test_options import TestOptions
from geometry_translation.models import create_model
from geometry_translation.data_loader import get_data_loader
from geometry_translation.utils.visualizer import save_images
from geometry_translation.utils import html
import os


class Tester:
    def __init__(self, opt, model, test_dl):
        self.opt = opt
        self.model = model
        self.test_dl = test_dl

    def pred(self):
        self.model.eval()
        # create a website
        web_dir = os.path.join(self.opt.results_dir, self.opt.name, '%s_%s' % (self.opt.phase, self.opt.epoch))  # define the website directory
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
        tot_loss = 0
        tot_metrics = 0
        for i, data in enumerate(self.test_dl):
            self.model.set_input(data)
            _, iter_loss, iter_metrics, iter_road_metrics = self.model.test()
            tot_loss += iter_loss.item()
            tot_metrics += iter_metrics.numpy()
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()  # get image paths
            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        tot_loss /= len(self.test_dl)
        tot_metrics /= len(self.test_dl)
        print('loss\t{:.6f}\tprecision\t{:.4f}\trecall\t{:.4f}\tf1\t{:.4f}\tiou\t{:.4f}\n'.format(tot_loss, tot_metrics[0], tot_metrics[1], tot_metrics[2], tot_metrics[3]))
        webpage.save()  # save the HTML


if __name__ == '__main__':
    opt = TestOptions().parse()
    model = create_model(opt)
    model.setup(opt)
    test_dl = get_data_loader(opt.dataroot, 'test')
    tester = Tester(opt, model, test_dl)
    tester.pred()
