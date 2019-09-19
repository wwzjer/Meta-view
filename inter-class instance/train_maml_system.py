import os
os.environ['DATASET_DIR'] = "datasets/"
from data import MetaLearningSystemDataLoader
from experiment_builder import ExperimentBuilder
from few_shot_learning_system import MAMLFewShotClassifier
from utils.parser_utils import get_args
from utils.dataset_tools import maybe_unzip_dataset

# Combines the arguments, model, data and experiment builders to run an experiment
args, device = get_args()
args.device = device

args.A = args.delta_M * args.delta_N

count_act = 0
args.act_to_delta = {}
args.delta_to_act = {}
for i in range(-(args.delta_N//2), args.delta_N//2+1):
    for j in range(-(args.delta_M//2), args.delta_M//2+1):
        args.act_to_delta[count_act] = (i, j)
        args.delta_to_act[(i, j)] = count_act
        count_act += 1

model = MAMLFewShotClassifier(args=args, device=device,
                              im_shape=(2, args.image_channels,
                                        args.image_height, args.image_width))
data = MetaLearningSystemDataLoader
maml_system = ExperimentBuilder(model=model, data=data, args=args, device=device)

maml_system.run_experiment()
