from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test')
        parser.add_argument('--nepoch', type=int, default=500, help='maximum epochs')#1000
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for optimizer')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum factor for SGD')
        parser.add_argument('--weight_decay', type=float, default=0.0005, help='momentum factor for optimizer')
        parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
        parser.add_argument('--lr_decay_iters', type=int, default=5000000, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--lr_decay_epochs', type=int, default=25, help='multiply by a gamma every lr_decay_epoch epochs')
        parser.add_argument('--lr_gamma', type=float, default=0.9, help='gamma factor for lr_scheduler')
        parser.add_argument('--data_augment', action='store_true',default=True,help='If true, use data_augment for training')
        parser.add_argument('--mirror_prob', type=float, default=0.0,help='The probability of mirror_prob augmentation')
        parser.add_argument('--rotate_prob',type=float, default=0.5,help='The probability of rotate_prob augmentation')

        self.isTrain = True
        return parser
