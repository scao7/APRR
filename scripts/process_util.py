import cv2
import os
import torch
import torchvision.transforms as T
import numpy as np

def video_to_images(video_path, output_folder, frame_interval=1,downsample_rate = 4):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Loop through frames and save images
    frame_count = 0
    while frame_count < total_frames:
        ret, frame = cap.read()
        
        if downsample_rate != 1:
            frame= cv2.resize(frame, (frame.shape[1]//downsample_rate, frame.shape[0]//downsample_rate))


        # Save frame as an image
        if frame_count % frame_interval == 0:
            image_filename = f"{output_folder}/{frame_count:08d}.png"
            cv2.imwrite(image_filename, frame)

        frame_count += 1

    # Release the video capture object
    cap.release()

    print(f"Conversion completed. {frame_count} frames saved to {output_folder}")

def calculate_overlapped_area(mask1, mask2):
    # Step 1: Convert masks to binary images
    binary_mask1 = (mask1 > 0).astype(np.uint8)
    binary_mask2 = (mask2 > 0).astype(np.uint8)

    # Step 2: Element-wise multiplication
    overlapped_mask = np.multiply(binary_mask1, binary_mask2)

    # Step 3: Sum the overlapped area
    overlapped_area = np.sum(overlapped_mask)

    return overlapped_area
def calculate_mismatched_area(mask1, mask2):
    # Step 1: Convert masks to binary images
    binary_mask1 = (mask1 > 0).astype(np.uint8)
    binary_mask2 = (mask2 > 0).astype(np.uint8)

    # Step 2: Element-wise XOR
    mismatched_mask = np.logical_xor(binary_mask1, binary_mask2).astype(np.uint8)

    # Step 3: Sum the mismatched area
    mismatched_area = np.sum(mismatched_mask)

    return mismatched_area
def video_to_selected(video_path, output_folder, frame_interval=1,downsample_rate = 4,match_threshold =0.5 ):
    print('selcted')
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    settings = romp.main.default_settings 
    # settings is just a argparse Namespace. To change it, for instance, you can change mode via
    # settings.mode='video'
    settings.mode = 'image'
    settings.render_mesh=True
    settings.cal_smpl = True
    # settings.onnx = True
    settings.show_largest = True
    settings.show_items='SMPL_mask'
    settings.renderer = 'sim3dr'
    settings.show_largest=True
    romp_model = romp.ROMP(settings)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Loop through frames and save images
    frame_count = 0
    while frame_count < total_frames:
        ret, frame = cap.read()
        
        if downsample_rate != 1:
            frame= cv2.resize(frame, (frame.shape[1]//downsample_rate, frame.shape[0]//downsample_rate))
            

        # direct mask 
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = transform(frame)
        input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

        # Perform inference
        with torch.no_grad():
            output = model(input_batch)['out'][0]
        
        output_predictions = output.argmax(0)

        # Convert PyTorch tensor to NumPy array
        direct_mask = output_predictions.byte().cpu().numpy()
        outputs= romp_model(frame)
        SMPL_mask = outputs['SMPL_mask']


        # print(direct_mask.shape,SMPL_mask.shape)
        
        try:
            SMPL_mask.shape
        except:
            print("no human")
            continue

        print(calculate_mismatched_area(direct_mask,SMPL_mask)/calculate_overlapped_area(direct_mask,SMPL_mask))
        print(calculate_mismatched_area(direct_mask,SMPL_mask))

        if calculate_mismatched_area(direct_mask,SMPL_mask)/calculate_overlapped_area(direct_mask,SMPL_mask) >match_threshold:
            continue


        # Save frame as an image
        if frame_count % frame_interval == 0:
            image_filename = f"{output_folder}/{frame_count:08d}.png"
            cv2.imwrite(image_filename, frame)

        frame_count += 1

    # Release the video capture object
    cap.release()

    print(f"Conversion completed. {frame_count} frames saved to {output_folder}")

# Load the pre-trained DeepLabv3 model
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
model.eval()

from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image
def mask_rcnn(image_path):

    # Load a pre-trained Mask R-CNN model
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # Load and preprocess the input image
    image_path = image_path
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([T.ToTensor()])
    input_image = transform(image)
    input_batch = input_image.unsqueeze(0)  # Add a batch dimension

    # Perform inference
    with torch.no_grad():
        prediction = model(input_batch)

    # Access the results
    masks = prediction[0]['masks']
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']

    # Identify humans (label 1 corresponds to 'person' in COCO)
    human_indices = (labels == 1).nonzero().squeeze()

    # Find the index of the largest person instance based on area
    largest_person_index = torch.argmax((boxes[human_indices, 2] - boxes[human_indices, 0]) * (boxes[human_indices, 3] - boxes[human_indices, 1]))

    # Separate and visualize the largest person
    largest_person_mask = masks[human_indices[largest_person_index], 0].cpu().numpy()
    largest_person_image = input_image.permute(1, 2, 0).cpu().numpy() * largest_person_mask[:, :, None]

    return largest_person_mask

# Define a function to apply segmentation
def segment_human(image_path):
    # Load and preprocess the input image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
 
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    

    


    output_predictions = output.argmax(0)
    output_predictions =output_predictions == 15

    # Convert PyTorch tensor to NumPy array
    mask_np = output_predictions.byte().cpu().numpy()

    return mask_np


# Define a function to visualize the mask on the original image
def visualize_mask(image, mask):
    # Apply mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Display the original image and the masked image
    cv2.imshow('Original Image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imshow('Human Segmentation Mask', cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Define a function to save the mask as a PNG file
def save_mask_as_png(mask, output_path):
    # Save the mask as a PNG file
    cv2.imwrite(output_path, mask * 255)  # Multiply by 255 to convert binary mask to 0 and 255 values

import romp
def rompEstimation(input_path,focal_length,imsize):
    print("estimating the romp parameters")
    settings = romp.main.default_settings 
    # settings is just a argparse Namespace. To change it, for instance, you can change mode via
    # settings.mode='video'
    romp_model = romp.ROMP(settings)
    
    # crate humannerf metadata based on the romp estimation
    metadata = {}
    for image_path in sorted(os.listdir(input_path)):
        print(image_path)
        image_path_noextension = image_path.split(".")[0]
        # processing the images 
        # img = cv2.imread(os.path.join(self.input_path,image_path))
        outputs = romp_model(cv2.imread(os.path.join(input_path,image_path))) # please note that we take the input image in BGR format (cv2.imread).
    
        metadata[image_path_noextension] = {}
        # print("pose shape",outputs["smpl_thetas"].shape)
        metadata[image_path_noextension]["poses"] = outputs["smpl_thetas"].tolist()[0]
        metadata[image_path_noextension]["betas"] = outputs["smpl_betas"].tolist()[0]
        # print(outputs.keys())
        # print(outputs['cam'])
        # focu length H/2 * 1/(tan(FOV/2)) = 1920/2. * 1./np.tan(np.radians(30)) = 1662.768
        # tra_pred = romp.utils.estimate_translation_cv2(outputs['joints'],outputs['pj2d_org'],1662.768,np.array([1920.,1080.]))
        # print(tra_pred)
        # print(outputs["cam_trans"])

        metadata[image_path_noextension]["cam_intrinsics"] = [
        [focal_length, 0.0,imsize[0]//2], 
        [0.0, focal_length, imsize[1]//2 ],
        [0.0, 0.0, 1.0]
        ]
        metadata[image_path_noextension]["cam_extrinsics"] = np.eye(4)
        metadata[image_path_noextension]["cam_extrinsics"][:3, 3] = outputs["cam_trans"][0]
        metadata[image_path_noextension]["cam_extrinsics"] = metadata[image_path_noextension]["cam_extrinsics"].tolist() 
        # https://github.com/Arthur151/ROMP/issues/300 cam extrinsic calculation
        key = cv2.waitKey(10)
        if key == 27:
            break
    return metadata

def rompSelection(input_path,focal_length,imsize):
    print("estimating the romp parameters")
    settings = romp.main.default_settings 
    # settings is just a argparse Namespace. To change it, for instance, you can change mode via
    # settings.mode='video'
    romp_model = romp.ROMP(settings)
    
    # crate humannerf metadata based on the romp estimation
    metadata = {}
    for image_path in sorted(os.listdir(input_path)):
        print(image_path)
        image_path_noextension = image_path.split(".")[0]
        # processing the images 
        # img = cv2.imread(os.path.join(self.input_path,image_path))
        
        outputs = romp_model(cv2.imread(os.path.join(input_path,image_path))) # please note that we take the input image in BGR format (cv2.imread).
        
        try:
            outputs['smpl_thetas'].tolist()[0]
        except:
            
            continue
        metadata[image_path_noextension] = {}
        # print("pose shape",outputs["smpl_thetas"].shape)
    
        metadata[image_path_noextension]["poses"] = outputs["smpl_thetas"].tolist()[0]
        metadata[image_path_noextension]["betas"] = outputs["smpl_betas"].tolist()[0]


        # print(outputs.keys())
        # print(outputs['cam'])
        # focu length H/2 * 1/(tan(FOV/2)) = 1920/2. * 1./np.tan(np.radians(30)) = 1662.768
        # tra_pred = romp.utils.estimate_translation_cv2(outputs['joints'],outputs['pj2d_org'],1662.768,np.array([1920.,1080.]))
        # print(tra_pred)
        # print(outputs["cam_trans"])
        metadata[image_path_noextension]["cam_intrinsics"] = [
        [focal_length, 0.0,imsize[0]//2], 
        [0.0, focal_length, imsize[1]//2 ],
        [0.0, 0.0, 1.0]
        ]
        metadata[image_path_noextension]["cam_extrinsics"] = np.eye(4)
        metadata[image_path_noextension]["cam_extrinsics"][:3, 3] = outputs["cam_trans"][0]
        metadata[image_path_noextension]["cam_extrinsics"] = metadata[image_path_noextension]["cam_extrinsics"].tolist() 
        # https://github.com/Arthur151/ROMP/issues/300 cam extrinsic calculation
        key = cv2.waitKey(10)


        if key == 27:
            break
      
    return metadata