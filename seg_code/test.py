import torch
import numpy as np
import torchvision.transforms as T
from . import special_transforms as SegT
from .models import UNet, my_FCN, DeepLabv3_plus, save_model, model_factory
from .utils import load_data, ConfusionMatrix, SELECT_LABEL_NAMES, N_CLASSES
import torch.utils.tensorboard as tb
from skimage.segmentation import find_boundaries
from skimage.morphology import binary_dilation, disk
from skimage.transform import resize

def calculate_rom_rum(pred_mask, gt_mask):
    """
    Calculate ROM (Region-wise Over-segmentation Measure) and RUM (Region-wise Under-segmentation Measure).
    
    Args:
        pred_mask (np.array): Predicted segmentation mask.
        gt_mask (np.array): Ground truth segmentation mask.
    
    Returns:
        rom (float): ROM score.
        rum (float): RUM score.
    """
    from skimage.measure import label, regionprops
    if pred_mask.shape != gt_mask.shape:
        # Resize prediction to match ground truth
        pred_mask = resize(pred_mask, gt_mask.shape, order=0, preserve_range=True, anti_aliasing=False)
        
    # Label connected regions in predicted and ground truth masks
    pred_labels = label(pred_mask)
    gt_labels = label(gt_mask)

    # Initialize counters for over- and under-segmentation
    over_seg_count = 0
    under_seg_count = 0

    # Calculate over-segmentation (ROM)
    for region in regionprops(gt_labels):
        gt_region = gt_labels == region.label
        overlapping_pred_regions = np.unique(pred_labels[gt_region])
        if len(overlapping_pred_regions) > 1:
            over_seg_count += 1

    # Calculate under-segmentation (RUM)
    for region in regionprops(pred_labels):
        pred_region = pred_labels == region.label
        overlapping_gt_regions = np.unique(gt_mask[pred_region])
        if len(overlapping_gt_regions) > 1:
            under_seg_count += 1

    # Normalize ROM and RUM by the total number of regions
    rom = over_seg_count / len(np.unique(gt_labels))
    rum = under_seg_count / len(np.unique(pred_labels))

    return rom, rum

def test(args):
    from os import path
    model = model_factory[args.model]()
    test_logger = None
    if args.log_dir:
        test_logger = tb.SummaryWriter(path.join(args.log_dir, 'test'), flush_secs=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), f'{args.model}.th')))

    test_data = load_data('small_dataset/test', num_workers=4)
    test_conf = ConfusionMatrix()
    
    # Initialize ROM and RUM accumulators
    total_rom = 0.0
    total_rum = 0.0
    num_samples = 0

    model.eval()
    with torch.no_grad():
        for img, label, msk in test_data:
            img, label = img.to(device), label.to(device).long()
            logit = model(img)
            
            # Upsample predictions to match original mask size
            pred_mask = torch.nn.functional.interpolate(
                logit.argmax(1, keepdim=True).float(),
                size=label.shape[-2:],  # Use label's resolution (e.g., 256x256)
                mode='nearest'
            ).squeeze().cpu().numpy()
            # Convert ground truth to numpy
            gt_mask = label.squeeze().cpu().numpy()

            # Calculate ROM and RUM
            rom, rum = calculate_rom_rum(pred_mask, gt_mask)
            total_rom += rom
            total_rum += rum
            num_samples += 1

            # Update confusion matrix
            test_conf.add(logit.argmax(1), label)

    # Calculate average ROM and RUM
    avg_rom = total_rom / num_samples
    avg_rum = total_rum / num_samples

    # Calculate class-wise metrics
    class_iou = test_conf.class_iou.cpu().numpy()
    class_acc = test_conf.class_accuracy.cpu().numpy()
    
    # Print results
    print('\nClass-wise Metrics:')
    for i in range(N_CLASSES):
        print(f'{SELECT_LABEL_NAMES[i]:<15}: IoU = {class_iou[i]:.3f}, Acc = {class_acc[i]:.3f}')
    
    print('\nSummary Metrics:')
    print(f'Global Accuracy: {test_conf.global_accuracy:.3f}')
    print(f'Mean IoU: {test_conf.iou:.3f}')
    print(f'ROM (Over-segmentation): {avg_rom:.3f}')
    print(f'RUM (Under-segmentation): {avg_rum:.3f}')

    # Log to TensorBoard
    if test_logger:
        # Class-wise metrics
        for i in range(N_CLASSES):
            test_logger.add_scalar(f'class_iou/{SELECT_LABEL_NAMES[i]}', class_iou[i], 0)
            test_logger.add_scalar(f'class_acc/{SELECT_LABEL_NAMES[i]}', class_acc[i], 0)
        
        # Summary metrics
        test_logger.add_scalar('global_accuracy', test_conf.global_accuracy, 0)
        test_logger.add_scalar('mean_iou', test_conf.iou, 0)
        test_logger.add_scalar('rom', avg_rom, 0)
        test_logger.add_scalar('rum', avg_rum, 0)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    parser.add_argument('-m', '--model', choices=['my_fcn', 'unet', 'deeplab'], default='my_fcn')
    args = parser.parse_args()
    test(args)
