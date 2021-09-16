from .base_options import BaseOptions

class DetectOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./testresults/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test')
        parser.add_argument('--no_label', action='store_true', help='chooses if we have gt labels in testing phase')
        parser.add_argument('--view-img', action='store_true', default=True, help='show results')
        parser.add_argument('--save-video', action='store_true', default=True, help='if true, save video, otherwise save image results')
        parser.add_argument('--output_video_fn', type=str, default='detect', metavar='PATH',
                            help='the video filename if the output format is save-video')
        self.isTrain = False
        return parser
