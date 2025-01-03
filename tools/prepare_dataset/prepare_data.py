import os
import sys
import shutil  # For file operations like copying and removing directories/files
import numpy as np  # For numerical computations
import json  # For working with JSON annotation files
import pickle  # To save and load processed data efficiently
import argparse  # For command-line argument parsing

# Import specific modules from the CRUW dataset library
from cruw import CRUW  # CRUW is the dataset object
from cruw.annotation.init_json import init_meta_json  # For initializing metadata for annotations
from cruw.mapping import ra2idx  # Maps range-angle coordinates to grid indices

# Import RODNet-specific utilities
from rodnet.core.confidence_map import generate_confmap, normalize_confmap, add_noise_channel
from rodnet.utils.load_configs import load_configs_from_file, update_config_dict
from rodnet.utils.visualization import visualize_confmap  # For visualizing confidence maps

# Predefined splits of the dataset
SPLITS_LIST = ['train', 'valid', 'test', 'demo']

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Prepare RODNet data.')
    parser.add_argument('--config', type=str, dest='config', help='Path to the configuration file')
    parser.add_argument('--data_root', type=str, help='Root directory of the dataset (overrides config)')
    parser.add_argument('--sensor_config', type=str, default='sensor_config_rod2021', help='Sensor configuration file')
    parser.add_argument('--split', type=str, dest='split', default='', help='Dataset split: train, valid, test, demo')
    parser.add_argument('--out_data_dir', type=str, default='./data', help='Directory to save prepared data')
    parser.add_argument('--overwrite', action="store_true", help="Overwrite prepared data if it exists")
    args = parser.parse_args()
    return args

# Function to load annotations from a .txt file and convert them to metadata
def load_anno_txt(txt_path, n_frame, dataset):
    # Map folder names to camera and radar data
    folder_name_dict = dict(
        cam_0='IMAGES_0',
        rad_h='RADAR_RA_H'
    )
    # Initialize metadata structure for annotations
    anno_dict = init_meta_json(n_frame, folder_name_dict)
    
    # Read annotation file line by line
    with open(txt_path, 'r') as f:
        data = f.readlines()
    
    # Process each annotation line
    for line in data:
        frame_id, r, a, class_name = line.rstrip().split()
        frame_id = int(frame_id)
        r = float(r)  # Range value
        a = float(a)  # Angle value
        
        # Convert range and angle to grid indices
        rid, aid = ra2idx(r, a, dataset.range_grid, dataset.angle_grid)
        
        # Update metadata with object information
        anno_dict[frame_id]['rad_h']['n_objects'] += 1
        anno_dict[frame_id]['rad_h']['obj_info']['categories'].append(class_name)
        anno_dict[frame_id]['rad_h']['obj_info']['centers'].append([r, a])
        anno_dict[frame_id]['rad_h']['obj_info']['center_ids'].append([rid, aid])
        anno_dict[frame_id]['rad_h']['obj_info']['scores'].append(1.0)  # Confidence score

    return anno_dict

# Function to generate confidence maps based on metadata
def generate_confmaps(metadata_dict, n_class, viz):
    confmaps = []  # List to store confidence maps for all frames
    
    # Iterate over each frame's metadata
    for metadata_frame in metadata_dict:
        n_obj = metadata_frame['rad_h']['n_objects']  # Number of objects in the frame
        obj_info = metadata_frame['rad_h']['obj_info']  # Object details
        
        # If no objects, initialize an empty confidence map with a noise channel
        if n_obj == 0:
            confmap_gt = np.zeros(
                (n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']),
                dtype=float
            )
            confmap_gt[-1, :, :] = 1.0  # Noise channel
        else:
            # Generate confidence map for the objects
            confmap_gt = generate_confmap(n_obj, obj_info, dataset, config_dict)
            confmap_gt = normalize_confmap(confmap_gt)  # Normalize the confidence map
            confmap_gt = add_noise_channel(confmap_gt, dataset, config_dict)  # Add noise channel
        
        # Ensure the map shape is correct
        assert confmap_gt.shape == (
            n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']
        )
        
        # Optionally visualize the confidence map
        if viz:
            visualize_confmap(confmap_gt)
        
        confmaps.append(confmap_gt)  # Add confidence map to the list

    confmaps = np.array(confmaps)  # Convert to a NumPy array for efficiency
    return confmaps

# Function to prepare dataset for training/testing
def prepare_data(dataset, config_dict, data_dir, split, save_dir, viz=False, overwrite=False):
    """
    Prepare pickle data for RODNet training and testing.
    This function processes radar and camera data, and saves them along with annotations and confidence maps.
    """
    camera_configs = dataset.sensor_cfg.camera_cfg
    radar_configs = dataset.sensor_cfg.radar_cfg
    n_chirp = radar_configs['n_chirps']  # Number of chirps per frame
    n_class = dataset.object_cfg.n_class  # Number of object categories

    # Determine paths and sequences based on the split
    data_root = config_dict['dataset_cfg']['data_root']
    anno_root = config_dict['dataset_cfg']['anno_root']
    if split is None:
        set_cfg = {
            'subdir': '',
            'seqs': sorted(os.listdir(data_root))
        }
        sets_seqs = sorted(os.listdir(data_root))
    else:
        set_cfg = config_dict['dataset_cfg'][split]
        if 'seqs' not in set_cfg:
            sets_seqs = sorted(os.listdir(os.path.join(data_root, set_cfg['subdir'])))
        else:
            sets_seqs = set_cfg['seqs']

    # Optionally overwrite existing data
    if overwrite:
        if os.path.exists(os.path.join(data_dir, split)):
            shutil.rmtree(os.path.join(data_dir, split))
        os.makedirs(os.path.join(data_dir, split))

    for seq in sets_seqs:
        seq_path = os.path.join(data_root, set_cfg['subdir'], seq)
        seq_anno_path = os.path.join(anno_root, set_cfg['subdir'], seq + config_dict['dataset_cfg']['anno_ext'])
        save_path = os.path.join(save_dir, seq + '.pkl')
        print("Processing sequence %s and saving to %s" % (seq_path, save_path))

        try:
            # Skip existing files if not overwriting
            if not overwrite and os.path.exists(save_path):
                print("%s already exists, skipping." % save_path)
                continue

            # Prepare paths to radar and camera data, handle various formats
            # ...

            # Load annotations and confidence maps
            # ...

            # Save processed data to a pickle file
            pickle.dump(data_dict, open(save_path, 'wb'))

        except Exception as e:
            print("Error while processing %s: %s" % (seq_path, e))

# Main entry point of the script
if __name__ == "__main__":
    args = parse_args()
    data_root = args.data_root
    splits = args.split.split(',') if args.split else None
    out_data_dir = args.out_data_dir
    os.makedirs(out_data_dir, exist_ok=True)
    overwrite = args.overwrite

    dataset = CRUW(data_root=data_root, sensor_config_name=args.sensor_config)
    config_dict = load_configs_from_file(args.config)
    config_dict = update_config_dict(config_dict, args)
    radar_configs = dataset.sensor_cfg.radar_cfg

    if splits is None:
        prepare_data(dataset, config_dict, out_data_dir, split=None, save_dir=out_data_dir, viz=False, overwrite=overwrite)
    else:
        for split in splits:
            if split not in SPLITS_LIST:
                raise ValueError(f"Split '{split}' not recognized.")
            save_dir = os.path.join(out_data_dir, split)
            os.makedirs(save_dir, exist_ok=True)
            print(f'Preparing {split} split...')
            prepare_data(dataset, config_dict, out_data_dir, split, save_dir, viz=False, overwrite=overwrite)
