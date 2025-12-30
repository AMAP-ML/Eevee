import os, torch, torchvision, imageio, random, csv
from PIL import Image


class EeveeDataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        dresses_dataset_base_path = None,
        dresses_dataset_metadata_path = None,
        lower_dataset_base_path = None,
        lower_dataset_metadata_path = None,
        upper_dataset_base_path = None,
        upper_dataset_metadata_path = None,
        target_height = None,
        target_width = None,
        num_frames = None,
    ): 
        self.dresses_dataset_base_path = dresses_dataset_base_path
        self.dresses_dataset_metadata_path = dresses_dataset_metadata_path
        self.lower_dataset_base_path = lower_dataset_base_path
        self.lower_dataset_metadata_path = lower_dataset_metadata_path
        self.upper_dataset_base_path = upper_dataset_base_path
        self.upper_dataset_metadata_path = upper_dataset_metadata_path
        self.target_height = target_height
        self.target_width = target_width
        self.num_frames = num_frames

        self.dresses_data = self.process_csv(dresses_dataset_metadata_path, dresses_dataset_base_path)
        self.lower_data = self.process_csv(lower_dataset_metadata_path, lower_dataset_base_path)
        self.upper_data = self.process_csv(upper_dataset_metadata_path, upper_dataset_base_path)

        self.data = self.dresses_data + self.lower_data + self.upper_data
    

    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()

        caption_path = os.path.join(data["path"], "caption.txt")
        vace_reference_image_path = os.path.join(data["path"], "in.png")        
        if random.random()<0.5:
            vace_video_path = os.path.join(data["path"], "video_0_agnostic.mp4")
            vace_video_mask_path = os.path.join(data["path"], "video_0_agnostic_mask.mp4")
            video_path = os.path.join(data["path"], "video_0.mp4")
        else:
            vace_video_path = os.path.join(data["path"], "video_1_agnostic.mp4")
            vace_video_mask_path = os.path.join(data["path"], "video_1_agnostic_mask.mp4")
            video_path = os.path.join(data["path"], "video_1.mp4")

        with open(caption_path, 'r', encoding='utf-8') as file:
            data["prompt"] = "Model is wearing " + file.read()
        data["vace_reference_image"] = self.process_image(vace_reference_image_path)
        data["video"] = self.process_video(video_path)
        data["vace_video_mask"] = self.process_video(vace_video_mask_path)
        data["vace_video"] = self.process_video(vace_video_path)
        
        return data


    def __len__(self):
        return len(self.data)
        
    def process_csv(self, csv_path, dataset_base_path):
        data = []
        with open(csv_path, mode='r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                case_name = row["Person"]
                data.append({"path": os.path.join(dataset_base_path, case_name)})
        return data


    def process_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.image_crop_and_resize(image)
        return [image]
    

    def process_video(self, file_path):
        reader = imageio.get_reader(file_path)
        frames = []
        for frame_id in range(self.num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.image_crop_and_resize(frame)
            frames.append(frame)
        reader.close()
        return frames


    def image_crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.target_width / width, self.target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (self.target_height, self.target_width))
        return image



if __name__ == "__main__":
    dataset = EeveeDataset(
        dresses_dataset_base_path = "/mnt/xmap_nas_ml/zengjianhao/Data/Eevee4/dresses",
        dresses_dataset_metadata_path = "/mnt/xmap_nas_ml/zengjianhao/Data/Eevee4/dresses_train.csv",
        lower_dataset_base_path = "/mnt/xmap_nas_ml/zengjianhao/Data/Eevee4/lower_body",
        lower_dataset_metadata_path = "/mnt/xmap_nas_ml/zengjianhao/Data/Eevee4/lower_train.csv",
        upper_dataset_base_path = "/mnt/xmap_nas_ml/zengjianhao/Data/Eevee4/upper_body",
        upper_dataset_metadata_path = "/mnt/xmap_nas_ml/zengjianhao/Data/Eevee4/upper_train.csv",
        target_height=816,
        target_width=1088,
        num_frames=49
    )
    data = dataset[0]
    print(data["vace_reference_image"][0].size)
    print(len(data["video"]))
    print(data["video"][0].size)
    print(data["prompt"])
    