import argparse
import os
import torch

from attrdict import AttrDict

import numpy as np

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.utils import int_tuple, bool_flag
from sgan.utils import relative_to_abs, get_dset_path

# Function to load the trajectory generator
def get_generator(checkpoint):
    # Load the pretrained model
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=20,
        pred_len=20,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    
    generator.load_state_dict(checkpoint['g_best_state'])

    generator.cuda()
    generator.eval()

    return generator


# Function to generate trajectories using the pretrained model
def generate_trajectories(args, generator, loader, num_samples, name):  

    trajectories = []
    total_traj = 0
    
    with torch.no_grad():
        for batch in loader:        
            batch = [tensor.cuda() for tensor in batch]
            if batch is not None:
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
                non_linear_ped, loss_mask, seq_start_end) = batch
                              
                total_traj += pred_traj_gt.size(1)
                
                for _ in range(num_samples):
                    pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end)
                    predicted_traj = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                    
                    predicted_traj = predicted_traj.cpu().numpy()                
                    trajectories.append(predicted_traj)

    return trajectories


# Function to write trajectories to a text file
def write_trajectories_to_file(trajectories, output_file):
    frameID = 0
    personID = 0
    with open(output_file, 'w') as f:
        for traj in trajectories:
            for coords in traj[:,:,:]:
                for personID_temp, points in enumerate(coords):
                    x, y = points
                    line = f"{frameID*10}\t{personID+personID_temp:.1f}\t{x:.2f}\t{y:.2f}\n"
                    f.write(line)
                frameID += 1 
            personID += coords.shape[0]



def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]
    

    for path in paths:
        checkpoint = torch.load(path)#, map_location=torch.device('cpu'))
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        path = get_dset_path(
            os.path.join('/data/trajgan/datasets/datasets_synthetic/', args.dataset_name) , f'train_{args.dataset_name}'
            )
        _, loader = data_loader(_args, path)
        
        trajectories = generate_trajectories(_args, generator, loader, args.num_samples, f"{_args.tag}_{_args.dataset_name}")
        output_file = os.path.join(args.output_dir, f"{args.dataset_name}_{args.output_filename}")

        write_trajectories_to_file(trajectories, output_file)

        print(f"Trajectories written to {output_file}")
        break

if __name__ == '__main__':
    # Define the arguments for the script
    parser = argparse.ArgumentParser(description='Synt2real extraction')
    # Dataset options
    parser.add_argument('--dataset_name', default='zara2', type=str)
    parser.add_argument('--dataset_dir', default='/data', type=str)
    parser.add_argument('--dataset_dir_synth', default='/data', type=str)
    parser.add_argument('--delim', default=' ')
    parser.add_argument('--loader_num_workers', default=4, type=int)
    parser.add_argument('--obs_len', default=8, type=int)
    parser.add_argument('--pred_len', default=12, type=int)
    parser.add_argument('--skip', default=1, type=int)
    parser.add_argument('--tag', default="baseline", type=str)

    # Optimization
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_iterations', default=10000, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)

    # Model Options
    parser.add_argument('--model_path', default='./src/checkpoint/traj-gen', type=str)
    parser.add_argument('--num_samples', default=20, type=int)
    parser.add_argument('--embedding_dim', default=64, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--batch_norm', default=0, type=bool_flag)
    parser.add_argument('--mlp_dim', default=1024, type=int)

    # predictor Options
    parser.add_argument('--encoder_h_dim_g', default=64, type=int)
    parser.add_argument('--decoder_h_dim_g', default=128, type=int)
    parser.add_argument('--noise_dim', default=None, type=int_tuple)
    parser.add_argument('--noise_type', default='gaussian')
    parser.add_argument('--noise_mix_type', default='ped')
    parser.add_argument('--clipping_threshold_g', default=0, type=float)
    parser.add_argument('--g_learning_rate', default=1e-3, type=float)
    parser.add_argument('--g_steps', default=1, type=int)

    parser.add_argument('--p_steps', default=1, type=int)

    # Pooling Options
    parser.add_argument('--pooling_type', default='pool_net')
    parser.add_argument('--pool_every_timestep', default=1, type=bool_flag)

    # Pool Net Option
    parser.add_argument('--bottleneck_dim', default=1024, type=int)

    # Social Pooling Options
    parser.add_argument('--neighborhood_size', default=2.0, type=float)
    parser.add_argument('--grid_size', default=8, type=int)

    # Discriminator Options
    parser.add_argument('--d_type', default='local', type=str)
    parser.add_argument('--encoder_h_dim_d', default=64, type=int)
    parser.add_argument('--d_learning_rate', default=1e-3, type=float)
    parser.add_argument('--d_steps', default=2, type=int)
    parser.add_argument('--clipping_threshold_d', default=0, type=float)

    # Loss Options
    parser.add_argument('--l2_loss_weight', default=0, type=float)
    parser.add_argument('--best_k', default=1, type=int)

    # Output
    parser.add_argument('--output_dir', default='./src/output')
    parser.add_argument('--output_filename', default='traj.txt', type=str)
    parser.add_argument('--print_every', default=5, type=int)
    parser.add_argument('--checkpoint_every', default=100, type=int)
    parser.add_argument('--checkpoint_name', default='checkpoint')
    parser.add_argument('--checkpoint_start_from', default=None)
    parser.add_argument('--restore_from_checkpoint', default=1, type=int)
    parser.add_argument('--num_samples_check', default=5000, type=int)

    # Misc
    parser.add_argument('--use_gpu', default=1, type=int)
    parser.add_argument('--timing', default=0, type=int)
    parser.add_argument('--gpu_num', default="0", type=str)

    args = parser.parse_args()

    main(args)
