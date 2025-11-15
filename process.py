from network import U2NET

import os
from PIL import Image
import cv2
import gdown
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from collections import OrderedDict
from options import opt


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("----No checkpoints at given path----")
        return
    model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print("----checkpoints loaded from path: {}----".format(checkpoint_path))
    return model


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask. """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


class Normalize_image(object):
    def __init__(self, mean, std):
        assert isinstance(mean, (float))
        if isinstance(mean, float):
            self.mean = mean
        if isinstance(std, float):
            self.std = std

        self.normalize_1 = transforms.Normalize(self.mean, self.std)
        self.normalize_3 = transforms.Normalize([self.mean] * 3, [self.std] * 3)
        self.normalize_18 = transforms.Normalize([self.mean] * 18, [self.std] * 18)

    def __call__(self, image_tensor):
        if image_tensor.shape[0] == 1:
            return self.normalize_1(image_tensor)
        elif image_tensor.shape[0] == 3:
            return self.normalize_3(image_tensor)
        elif image_tensor.shape[0] == 18:
            return self.normalize_18(image_tensor)
        else:
            assert "Please set proper channels! Normalization implemented only for 1, 3 and 18"


def apply_transform(img):
    transforms_list = []
    transforms_list += [transforms.ToTensor()]
    transforms_list += [Normalize_image(0.5, 0.5)]
    transform_rgb = transforms.Compose(transforms_list)
    return transform_rgb(img)


def generate_mask(input_image, net, palette, device='cpu'):
    """Generate segmentation mask - returns PIL image"""
    img = input_image
    img_size = img.size
    img = img.resize((768, 768), Image.BICUBIC)
    image_tensor = apply_transform(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    with torch.no_grad():
        output_tensor = net(image_tensor.to(device))
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()

    # Create final segmentation
    cloth_seg = Image.fromarray(output_arr[0].astype(np.uint8), mode='P')
    cloth_seg.putpalette(palette)
    cloth_seg = cloth_seg.resize(img_size, Image.BICUBIC)
    
    return cloth_seg


def apply_mask_extraction(original_cv, segmentation_pil):
    """Apply mask to extract dress with black background"""
    # Convert PIL to CV2
    seg_cv = cv2.cvtColor(np.array(segmentation_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
    
    # Resize segmentation to match original
    seg_resized = cv2.resize(seg_cv, (original_cv.shape[1], original_cv.shape[0]))
    
    # Create mask
    seg_gray = cv2.cvtColor(seg_resized, cv2.COLOR_BGR2GRAY)
    unique, counts = np.unique(seg_gray, return_counts=True)
    bg_value = unique[np.argmax(counts)]
    mask = (seg_gray != bg_value).astype(np.uint8) * 255
    
    # Apply mask
    result = cv2.bitwise_and(original_cv, original_cv, mask=mask)
    
    return result


def check_or_download_model(file_path):
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        url = "https://drive.google.com/uc?id=11xTBALOeUkyuaK3l60CpkYHLTmv7k3dY"
        gdown.download(url, file_path, quiet=False)
        print("Model downloaded successfully.")
    else:
        print("Model already exists.")


def load_seg_model(checkpoint_path, device='cpu'):
    net = U2NET(in_ch=3, out_ch=4)
    check_or_download_model(checkpoint_path)
    net = load_checkpoint(net, checkpoint_path)
    net = net.to(device)
    net = net.eval()
    return net


def process_batch(args):
    """Process all images in input folder"""
    
    device = 'cuda:0' if args.cuda else 'cpu'
    print(f"Using device: {device}")
    
    # Load model once
    print("Loading segmentation model...")
    model = load_seg_model(args.checkpoint_path, device=device)
    palette = get_palette(4)
    
    # Setup folders
    input_folder = args.input_folder
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_folder).glob(ext))
    
    print(f"\n📂 Found {len(image_files)} images to process")
    print(f"📥 Input folder: {input_folder}")
    print(f"📤 Output folder: {output_folder}\n")
    
    # Process each image
    for idx, img_path in enumerate(image_files, 1):
        try:
            print(f"[{idx}/{len(image_files)}] Processing: {img_path.name}...", end=" ")
            
            # Load original image
            original_cv = cv2.imread(str(img_path))
            original_pil = Image.open(str(img_path)).convert('RGB')
            
            # Generate segmentation
            segmentation_pil = generate_mask(original_pil, model, palette, device)
            
            # Extract dress
            result = apply_mask_extraction(original_cv, segmentation_pil)
            
            # Save result
            output_path = os.path.join(output_folder, img_path.name)
            cv2.imwrite(output_path, result)
            
            print(f"✅")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            continue
    
    print(f"\n🎉 All done! Processed {len(image_files)} images")
    print(f"📁 Check output at: {output_folder}")


def main(args):
    process_batch(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch process cloth segmentation and extraction.')
    parser.add_argument('--input_folder', type=str, required=True, 
                        help='Path to input folder with images')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Path to output folder for extracted dresses')
    parser.add_argument('--cuda', action='store_true', 
                        help='Enable CUDA (default: False)')
    parser.add_argument('--checkpoint_path', type=str, default='model/cloth_segm.pth',
                        help='Path to the checkpoint file')
    args = parser.parse_args()

    main(args)
