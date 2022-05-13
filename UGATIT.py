import time, itertools
from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from networks import *
from utils import *
from glob import glob


class UGATIT(object):
    def __init__(self, args):
        # whether is full version or light version
        self.isLight = args.isLight

        if self.isLight:
            self.model_name = 'UGATIT_light'
        else:
            self.model_name = 'UGATIT'

        # file path of saving results
        self.result_path = args.result_path

        # the name of dataset you want use
        self.dataset = args.dataset

        # the max limitation of training iteration
        self.iter = args.iter

        # whether to reduce the learning rate halfway through the training iteration
        self.reduce_lr = args.reduce_lr

        # batch samples randomly sampled from the training set in each iterations
        self.batch_size = args.batch_size

        # number of iterations required to print an image
        self.print_period = args.print_period

        # number of iterations required to save a model
        self.save_period = args.save_period

        # learning rate
        self.alpha = args.alpha

        # the weight decay parameter used in optimiser
        self.weight_decay = args.weight_decay

        # number of channels of the first convolutional layer used in neural network
        self.channel_number = args.channel_number

        # gan weight of loss
        self.gan_loss_weight = args.gan_loss_weight

        # cycle weight in loss
        self.cycle_loss_weight = args.cycle_loss_weight

        # identity weight in loss
        self.identity_loss_weight = args.identity_loss_weight

        # cam weight in loss
        self.cam_loss_weight = args.cam_loss_weight

        # resblock number
        self.resblock_num = args.resblock_num

        # size of image
        self.img_size = args.img_size

        # the channel number of image
        # self.img_channel_num = args.img_channel_num

        # device that the model is trained on (cpu or cuda)
        self.device = args.device

        # set benchmark on cudnn
        self.benchmark = args.benchmark

        # reload model from selected result path and dataset
        self.reload_model = args.reload_model

        if self.benchmark and torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = True
            print('set benchmark flag to true')

    def build_model(self):
        # transformer
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.img_size + 30, self.img_size + 30)),
            transforms.RandomCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        # dataloader
        self.train1_loader = DataLoader(
            ImageFolder(os.path.join('dataset', self.dataset, 'trainA'), train_transform),
            batch_size=self.batch_size, shuffle=True)
        self.train2_loader = DataLoader(
            ImageFolder(os.path.join('dataset', self.dataset, 'trainB'), train_transform),
            batch_size=self.batch_size, shuffle=True)
        self.test1_loader = DataLoader(
            ImageFolder(os.path.join('dataset', self.dataset, 'testA'), test_transform),
            batch_size=1, shuffle=False)
        self.test2_loader = DataLoader(
            ImageFolder(os.path.join('dataset', self.dataset, 'testB'), test_transform),
            batch_size=1, shuffle=False)

        # generator of changing image like 1 to image like 2
        self.gen1to2 = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.channel_number,
                                       n_blocks=self.resblock_num,
                                       img_size=self.img_size,
                                       light=self.isLight).to(self.device)

        # generator of changing image like 2 to image like 1
        self.gen2to1 = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.channel_number,
                                       n_blocks=self.resblock_num,
                                       img_size=self.img_size,
                                       light=self.isLight).to(self.device)

        # discriminator
        self.dis11 = Discriminator(input_nc=3, ndf=self.channel_number, n_layers=7).to(self.device)
        self.dis12 = Discriminator(input_nc=3, ndf=self.channel_number, n_layers=7).to(self.device)
        self.dis21 = Discriminator(input_nc=3, ndf=self.channel_number, n_layers=5).to(self.device)
        self.dis22 = Discriminator(input_nc=3, ndf=self.channel_number, n_layers=5).to(self.device)

        # loss function
        self.L1 = nn.L1Loss().to(self.device)
        self.MSE = nn.MSELoss().to(self.device)
        self.BCE = nn.BCEWithLogitsLoss().to(self.device)

        # optimiser
        self.opt_gen = torch.optim.Adam(itertools.chain(self.gen1to2.parameters(), self.gen2to1.parameters()),
                                        lr=self.alpha,
                                        betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.opt_dis = torch.optim.Adam(
            itertools.chain(self.dis11.parameters(), self.dis12.parameters(), self.dis21.parameters(),
                            self.dis22.parameters()), lr=self.alpha, betas=(0.5, 0.999), weight_decay=self.weight_decay)

        """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
        self.Rho_cutter = RhoClipper(0, 1)

    def train(self):
        self.gen1to2.train(), self.gen2to1.train(), self.dis11.train(), self.dis12.train(), self.dis21.train(), self.dis22.train()

        start_iter = 1
        if self.reload_model:
            model_list = glob(os.path.join(self.result_path, self.dataset, 'model', '*.pt'))
            if not len(model_list) == 0:
                model_list.sort()
                start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join(self.result_path, self.dataset, 'model'), start_iter)
                print(" [*] Load SUCCESS")
                if self.reduce_lr and start_iter > (self.iter // 2):
                    self.opt_gen.param_groups[0]['lr'] -= (self.alpha / (self.iter // 2)) * (
                            start_iter - self.iter // 2)
                    self.opt_dis.param_groups[0]['lr'] -= (self.alpha / (self.iter // 2)) * (
                            start_iter - self.iter // 2)

        # training loop
        print('training start !')
        start_time = time.time()
        for step in range(start_iter, self.iter + 1):
            if self.reduce_lr and step > (self.iter // 2):
                self.opt_gen.param_groups[0]['lr'] -= (self.alpha / (self.iter // 2))
                self.opt_dis.param_groups[0]['lr'] -= (self.alpha / (self.iter // 2))

            try:
                real_A, _ = trainA_iter.next()
            except:
                trainA_iter = iter(self.train1_loader)
                real_A, _ = trainA_iter.next()

            try:
                real_B, _ = trainB_iter.next()
            except:
                trainB_iter = iter(self.train2_loader)
                real_B, _ = trainB_iter.next()

            real_A, real_B = real_A.to(self.device), real_B.to(self.device)

            # Update D
            self.opt_dis.zero_grad()

            fake_A2B, _, _ = self.gen1to2(real_A)
            fake_B2A, _, _ = self.gen2to1(real_B)

            real_GA_logit, real_GA_cam_logit, _ = self.dis11(real_A)
            real_LA_logit, real_LA_cam_logit, _ = self.dis21(real_A)
            real_GB_logit, real_GB_cam_logit, _ = self.dis12(real_B)
            real_LB_logit, real_LB_cam_logit, _ = self.dis22(real_B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.dis11(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.dis21(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.dis12(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.dis22(fake_A2B)

            D_ad_loss_GA = self.MSE(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + self.MSE(
                fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
            D_ad_cam_loss_GA = self.MSE(real_GA_cam_logit,
                                        torch.ones_like(real_GA_cam_logit).to(self.device)) + self.MSE(
                fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit).to(self.device))
            D_ad_loss_LA = self.MSE(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) + self.MSE(
                fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
            D_ad_cam_loss_LA = self.MSE(real_LA_cam_logit,
                                        torch.ones_like(real_LA_cam_logit).to(self.device)) + self.MSE(
                fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit).to(self.device))
            D_ad_loss_GB = self.MSE(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) + self.MSE(
                fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
            D_ad_cam_loss_GB = self.MSE(real_GB_cam_logit,
                                        torch.ones_like(real_GB_cam_logit).to(self.device)) + self.MSE(
                fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit).to(self.device))
            D_ad_loss_LB = self.MSE(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)) + self.MSE(
                fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))
            D_ad_cam_loss_LB = self.MSE(real_LB_cam_logit,
                                        torch.ones_like(real_LB_cam_logit).to(self.device)) + self.MSE(
                fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit).to(self.device))

            D_loss_A = self.gan_loss_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
            D_loss_B = self.gan_loss_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

            Discriminator_loss = D_loss_A + D_loss_B
            Discriminator_loss.backward()
            self.opt_dis.step()

            # Update G
            self.opt_gen.zero_grad()

            fake_A2B, fake_A2B_cam_logit, _ = self.gen1to2(real_A)
            fake_B2A, fake_B2A_cam_logit, _ = self.gen2to1(real_B)

            fake_A2B2A, _, _ = self.gen2to1(fake_A2B)
            fake_B2A2B, _, _ = self.gen1to2(fake_B2A)

            fake_A2A, fake_A2A_cam_logit, _ = self.gen2to1(real_A)
            fake_B2B, fake_B2B_cam_logit, _ = self.gen1to2(real_B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.dis11(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.dis21(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.dis12(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.dis22(fake_A2B)

            G_ad_loss_GA = self.MSE(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
            G_ad_cam_loss_GA = self.MSE(fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit).to(self.device))
            G_ad_loss_LA = self.MSE(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
            G_ad_cam_loss_LA = self.MSE(fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit).to(self.device))
            G_ad_loss_GB = self.MSE(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
            G_ad_cam_loss_GB = self.MSE(fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit).to(self.device))
            G_ad_loss_LB = self.MSE(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))
            G_ad_cam_loss_LB = self.MSE(fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit).to(self.device))

            G_recon_loss_A = self.L1(fake_A2B2A, real_A)
            G_recon_loss_B = self.L1(fake_B2A2B, real_B)

            G_identity_loss_A = self.L1(fake_A2A, real_A)
            G_identity_loss_B = self.L1(fake_B2B, real_B)

            G_cam_loss_A = self.BCE(fake_B2A_cam_logit,
                                    torch.ones_like(fake_B2A_cam_logit).to(self.device)) + self.BCE(
                fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit).to(self.device))
            G_cam_loss_B = self.BCE(fake_A2B_cam_logit,
                                    torch.ones_like(fake_A2B_cam_logit).to(self.device)) + self.BCE(
                fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit).to(self.device))

            G_loss_A = self.gan_loss_weight * (
                    G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.cycle_loss_weight * G_recon_loss_A + self.identity_loss_weight * G_identity_loss_A + self.cam_loss_weight * G_cam_loss_A
            G_loss_B = self.gan_loss_weight * (
                    G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + self.cycle_loss_weight * G_recon_loss_B + self.identity_loss_weight * G_identity_loss_B + self.cam_loss_weight * G_cam_loss_B

            Generator_loss = G_loss_A + G_loss_B
            Generator_loss.backward()
            self.opt_gen.step()

            # clip parameter of AdaILN and ILN, applied after optimizer step
            self.gen1to2.apply(self.Rho_cutter)
            self.gen2to1.apply(self.Rho_cutter)

            print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (
                step, self.iter, time.time() - start_time, Discriminator_loss, Generator_loss))
            if step % self.print_period == 0:
                train_sample_num = 5
                test_sample_num = 5
                A2B = np.zeros((self.img_size * 7, 0, 3))
                B2A = np.zeros((self.img_size * 7, 0, 3))

                self.gen1to2.eval(), self.gen2to1.eval(), self.dis11.eval(), self.dis12.eval(), self.dis21.eval(), self.dis22.eval()
                for _ in range(train_sample_num):
                    try:
                        real_A, _ = trainA_iter.next()
                    except:
                        trainA_iter = iter(self.train1_loader)
                        real_A, _ = trainA_iter.next()

                    try:
                        real_B, _ = trainB_iter.next()
                    except:
                        trainB_iter = iter(self.train2_loader)
                        real_B, _ = trainB_iter.next()
                    real_A, real_B = real_A.to(self.device), real_B.to(self.device)

                    fake_A2B, _, fake_A2B_heatmap = self.gen1to2(real_A)
                    fake_B2A, _, fake_B2A_heatmap = self.gen2to1(real_B)

                    fake_A2B2A, _, fake_A2B2A_heatmap = self.gen2to1(fake_A2B)
                    fake_B2A2B, _, fake_B2A2B_heatmap = self.gen1to2(fake_B2A)

                    fake_A2A, _, fake_A2A_heatmap = self.gen2to1(real_A)
                    fake_B2B, _, fake_B2B_heatmap = self.gen1to2(real_B)

                    A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                               cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                               cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                               cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                    B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                               cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                               cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                               cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                for _ in range(test_sample_num):
                    try:
                        real_A, _ = testA_iter.next()
                    except:
                        testA_iter = iter(self.test1_loader)
                        real_A, _ = testA_iter.next()

                    try:
                        real_B, _ = testB_iter.next()
                    except:
                        testB_iter = iter(self.test2_loader)
                        real_B, _ = testB_iter.next()
                    real_A, real_B = real_A.to(self.device), real_B.to(self.device)

                    fake_A2B, _, fake_A2B_heatmap = self.gen1to2(real_A)
                    fake_B2A, _, fake_B2A_heatmap = self.gen2to1(real_B)

                    fake_A2B2A, _, fake_A2B2A_heatmap = self.gen2to1(fake_A2B)
                    fake_B2A2B, _, fake_B2A2B_heatmap = self.gen1to2(fake_B2A)

                    fake_A2A, _, fake_A2A_heatmap = self.gen2to1(real_A)
                    fake_B2B, _, fake_B2B_heatmap = self.gen1to2(real_B)

                    A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                               cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                               cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                               cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                    B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                               cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                               cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                               cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                cv2.imwrite(os.path.join(self.result_path, self.dataset, 'img', 'A2B_%07d.png' % step), A2B * 255.0)
                cv2.imwrite(os.path.join(self.result_path, self.dataset, 'img', 'B2A_%07d.png' % step), B2A * 255.0)
                self.gen1to2.train()
                self.gen2to1.train()
                self.dis11.train()
                self.dis12.train()
                self.dis21.train()
                self.dis22.train()

            if step % self.save_period == 0:
                self.save(os.path.join(self.result_path, self.dataset, 'model'), step)

            if step % 1000 == 0:
                params = {}
                params['genA2B'] = self.gen1to2.state_dict()
                params['genB2A'] = self.gen2to1.state_dict()
                params['disGA'] = self.dis11.state_dict()
                params['disGB'] = self.dis12.state_dict()
                params['disLA'] = self.dis21.state_dict()
                params['disLB'] = self.dis22.state_dict()
                torch.save(params, os.path.join(self.result_path, self.dataset + '_params_latest.pt'))

    def save(self, dir, step):
        params = {}
        params['genA2B'] = self.gen1to2.state_dict()
        params['genB2A'] = self.gen2to1.state_dict()
        params['disGA'] = self.dis11.state_dict()
        params['disGB'] = self.dis12.state_dict()
        params['disLA'] = self.dis21.state_dict()
        params['disLB'] = self.dis22.state_dict()
        torch.save(params, os.path.join(dir, self.dataset + '_params_%07d.pt' % step))

    def load(self, dir, step):
        params = torch.load(os.path.join(dir, self.dataset + '_params_%07d.pt' % step))
        self.gen1to2.load_state_dict(params['genA2B'])
        self.gen2to1.load_state_dict(params['genB2A'])
        self.dis11.load_state_dict(params['disGA'])
        self.dis12.load_state_dict(params['disGB'])
        self.dis21.load_state_dict(params['disLA'])
        self.dis22.load_state_dict(params['disLB'])

    def test(self):
        # load models
        models = glob(os.path.join(self.result_path, self.dataset, 'model', '*.pt'))
        if not len(models) == 0:
            models.sort()
            iter = int(models[-1].split('_')[-1].split('.')[0])
            self.load(os.path.join(self.result_path, self.dataset, 'model'), iter)
            print(" [*] Load SUCCESS")
        else:
            print(" [*] Load FAILURE")
            return

        # set to evaluation mode
        self.gen1to2.eval()
        self.gen2to1.eval()

        # start prediction
        for n, (image_1, _) in enumerate(self.test1_loader):
            # move to selected device
            image_1 = image_1.to(self.device)

            image_1to2, _, heatmap_1to2 = self.gen1to2(image_1)

            image_1to2to1, _, heatmap_1to2to1 = self.gen2to1(image_1to2)

            image_1to1, _, fake_A2A_heatmap = self.gen2to1(image_1)

            _1to2 = np.concatenate((RGB2BGR(tensor2numpy(denorm(image_1[0]))),
                                  cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(image_1to1[0]))),
                                  cam(tensor2numpy(heatmap_1to2[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(image_1to2[0]))),
                                  cam(tensor2numpy(heatmap_1to2to1[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(image_1to2to1[0])))), 0)

            cv2.imwrite(os.path.join(self.result_path, self.dataset, 'test', 'A2B_%d.png' % (n + 1)), _1to2 * 255.0)

        for n, (image_2, _) in enumerate(self.test2_loader):
            image_2 = image_2.to(self.device)

            image_2to1, _, heatmap_2to1 = self.gen2to1(image_2)

            image_2to1to2, _, heatmap_2to1to2 = self.gen1to2(image_2to1)

            image_2to2, _, heatmap_2to2 = self.gen1to2(image_2)

            _2to1 = np.concatenate((RGB2BGR(tensor2numpy(denorm(image_2[0]))),
                                  cam(tensor2numpy(heatmap_2to2[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(image_2to2[0]))),
                                  cam(tensor2numpy(heatmap_2to1[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(image_2to1[0]))),
                                  cam(tensor2numpy(heatmap_2to1to2[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(image_2to1to2[0])))), 0)

            cv2.imwrite(os.path.join(self.result_path, self.dataset, 'test', 'B2A_%d.png' % (n + 1)), _2to1 * 255.0)
