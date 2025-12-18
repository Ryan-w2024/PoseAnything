import cv2
from pathlib import Path
from PIL import Image

        
def resize(frame, target_max_size=512):

    height, width = frame.shape[:2]
    aspect_ratio = width / height
    
    if width >= height:
        new_width = target_max_size
        new_height = int(target_max_size / aspect_ratio)
    else:
        new_height = target_max_size
        new_width = int(target_max_size * aspect_ratio)
    
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized_frame
        
def video_to_images(video_path, output_dir, frame_interval=1):

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    video_name = video_path.stem
    video_output_folder = output_dir / video_name
    video_output_folder.mkdir(exist_ok=True, parents=True)
    
    error_log_path = output_dir / "error.txt"
    images_list = []
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            error_msg = f"can not open: {video_path}"
            print(error_msg)
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_index = 0
        save_index = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_index % frame_interval == 0:
                # resize frame
                resized_frame = resize(frame, 512)
                # BGR to RGB
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame).convert("RGB")
                images_list.append(pil_image)

                output_path = video_output_folder / f"{save_index:03d}.png"
                pil_image.save(str(output_path))
                save_index += 1
                    
            frame_index += 1
            
        cap.release()
        return images_list
        
    except Exception as e:
        error_msg = f"handle {video_path} error: {str(e)}"
        print(error_msg)
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        return []


def read_png_as_pil(png_path):

    png_path = Path(png_path)

    if not png_path.exists():
        raise FileNotFoundError(f"file not exist: {png_path}")
    if png_path.suffix.lower() != ".png":
        raise ValueError(f"not PNG file: {png_path}")

    frame = cv2.imread(str(png_path))
    if frame is None:
        raise IOError(f"cannot read: {png_path}")

    resized_frame = resize(frame, 512)

    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame).convert("RGB")

    return pil_image