import numpy as np
import os
import torch
import cv2
from utils import load_config, create_output_dir_and_save_config
from dataset import  read_rgb_img, get_patch_info_one_img
from model import MTRoad
import time
from tqdm import tqdm
from argparse import ArgumentParser
import glob
os.environ["CUDA_VISIBLE_DEVICES"]='1'

def get_img_paths(root_dir, image_indices):
    img_paths = []

    for ind in image_indices:
        img_paths.append(os.path.join(root_dir, f"region_{ind}_sat.png"))

    return img_paths

def crop_img_patch(img, x0, y0, x1, y1):
    return img[y0:y1, x0:x1, :]


def get_batch_img_patches(img, batch_patch_info):
    patches = []
    for _, (x0, y0), (x1, y1) in batch_patch_info:
        patch = crop_img_patch(img, x0, y0, x1, y1)
        patches.append(torch.tensor(patch, dtype=torch.float32))
    batch = torch.stack(patches, 0).contiguous()
    return batch

def infer_one_img(net, img, config):
    # TODO(congrui): centralize these configs

    image_size = img.shape[0]
    batch_size = config.INFER_BATCH_SIZE
    all_patch_info = get_patch_info_one_img(
        0, image_size, config.SAMPLE_MARGIN, config.PATCH_SIZE, config.INFER_PATCHES_PER_EDGE)
    patch_num = len(all_patch_info)
    batch_num = (
        patch_num // batch_size
        if patch_num % batch_size == 0
        else patch_num // batch_size + 1
    )

    # [IMG_H, IMG_W]
    pixel_counter = torch.zeros(img.shape[0:2], dtype=torch.float32).to(args.device, non_blocking=False)
    fused_surface= torch.zeros(img.shape[0:2], dtype=torch.float32).to(args.device, non_blocking=False)


    for batch_index in range(batch_num):
        offset = batch_index * batch_size
        batch_patch_info = all_patch_info[offset : offset + batch_size]
        # tensor [B, H, W, C]
        batch_img_patches = get_batch_img_patches(img, batch_patch_info)

        with torch.no_grad():
            batch_img_patches = batch_img_patches.to(args.device, non_blocking=False)
            patch_surfaces = net.infer_masks_and_img_features(batch_img_patches)


        for patch_index, patch_info in enumerate(batch_patch_info):
            _, (x0, y0), (x1, y1) = patch_info

            patch_surface=patch_surfaces[patch_index, :, :, 0]

            pixel_counter[y0:y1, x0:x1] += torch.ones(patch_surface.shape[0:2], dtype=torch.float32, device=args.device)

            fused_surface[y0:y1, x0:x1] += patch_surface
    

    fused_surface /= pixel_counter

    fused_surface = torch.where(fused_surface > 0.5, 1, 0)
    fused_surface = (fused_surface * 255).to(torch.uint8).cpu().numpy()
    return fused_surface


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        # "--checkpoint", default='massachusetts/result/a_train_road_result/train_weight/epoch-173-f10.78.ckpt',#
        "--checkpoint", default='CHN6_CUG/result/a_train_road_result/train_weight/epoch-119-f10.77.ckpt',#
    )
    parser.add_argument(
        # "--config", default='config/config_512_massachusetts.yaml', help="model config."
        "--config", default='config/config_512_CHN6_CUG.yaml', help="model config."
    )
    parser.add_argument(
        # "--output_dir", default='massachusetts/result/b_inferencer_result',
        "--output_dir", default='CHN6_CUG/result/b_inferencer_result',
    )
    parser.add_argument("--device", default="cuda", help="device to use for training")#"cuda"
    args = parser.parse_args()
    config = load_config(args.config)

    device = torch.device("cuda") if args.device == "cuda" else torch.device("cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    net = MTRoad(config)

    # load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    print(f'##### Loading Trained CKPT {args.checkpoint} #####')
    net.load_state_dict(checkpoint["state_dict"], strict=True)
    net.eval()
    net.to(device)

    if config.DATASET == 'massachusetts':
        image_paths = sorted(glob.glob(os.path.join("./massachusetts/test/image", "*.tiff")), key=lambda x: (
            int(os.path.basename(x).split("_")[0]), int(os.path.basename(x).split("_")[1].split(".")[0])))
        mask_paths = sorted(glob.glob(os.path.join("./massachusetts/test/mask", "*.tif")), key=lambda x: (
            int(os.path.basename(x).split("_")[0]), int(os.path.basename(x).split("_")[1].split(".")[0])))

    elif config.DATASET == 'CHN6_CUG':
        image_paths = sorted(glob.glob(os.path.join(f"./CHN6_CUG/test/image", "*.jpg")),
                                  key=lambda x: (
                                      int(os.path.basename(x).split("_")[0][2:])))  # [:20]
        mask_paths = sorted(glob.glob(os.path.join(f"./CHN6_CUG/test/mask", "*.png")),
                                 key=lambda x: (
                                     int(os.path.basename(x).split("_")[0][2:])))  # [:20]
    if args.output_dir:
        output_dir = create_output_dir_and_save_config( config, specified_dir=f'./{args.output_dir}')
    else:
        output_dir = create_output_dir_and_save_config(config)
    
    total_inference_seconds = 0.0

    i_num=1

    for img_id in tqdm(range(len(image_paths))):
        img = read_rgb_img(image_paths[img_id])
        if config.DATASET == 'massachusetts':
            true_surface = cv2.imread(mask_paths[img_id], cv2.IMREAD_GRAYSCALE)[..., None]
            image_name = os.path.basename(image_paths[img_id]).split(".")[0]
        elif config.DATASET == 'CHN6_CUG':
            file_name, base_name = os.path.split(image_paths[img_id])
            path_mask = os.path.join(file_name.replace("image", "mask"), base_name.replace("_sat.jpg", "_mask.png"))
            true_surface = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)[..., None]
            image_name = os.path.basename(image_paths[img_id]).split("_sat")[0]
        start_seconds = time.time()
        surface = infer_one_img(net, img, config)

        end_seconds = time.time()
        print(f"{img_id} image result output need {end_seconds-start_seconds}s")
        total_inference_seconds += (end_seconds - start_seconds)

        viz_img = np.copy(img)
        viz_img_gt = np.copy(img)
        img_size = viz_img.shape[0]

        surface_save_dir = os.path.join(output_dir, 'surface')
        if not os.path.exists(surface_save_dir):
            os.makedirs(surface_save_dir)
        cv2.imwrite(os.path.join(surface_save_dir, f'{image_name}_pre.png'), surface)
        cv2.imwrite(os.path.join(surface_save_dir, f'{image_name}_gt.png'), true_surface)

    time_txt = f'Inference completed for {args.config} in {total_inference_seconds} seconds.'
    print(time_txt)
    with open(os.path.join(output_dir, 'inference_time.txt'), 'w') as f:
        f.write(time_txt)
