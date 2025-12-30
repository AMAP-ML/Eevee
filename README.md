<div align="center">

<h1>Eevee: Towards Close-up High-resolution Video-based Virtual Try-on</h1>

<div>
    <a href="https://zengjianhao.github.io/" target="_blank">Jianhao Zeng</a><sup>1,*</sup>,
    <a href="https://scholar.google.com.hk/citations?user=Ilx8WNkAAAAJ&hl=en&oi=ao" target="_blank">Yancheng Bai</a><sup>1,*</sup>,
    <a href="https://littlefatshiba.github.io/" target="_blank">Ruidong Chen</a><sup>1,2</sup>,
    <a href="https://scholar.google.com.hk/citations?user=EzPr96kAAAAJ&hl=en&oi=ao" target="_blank">Xuanpu Zhang</a><sup>2</sup>,
    <a href="https://allylei.github.io/" target="_blank">Lei Sun</a><sup>1</sup>
</div>
<div>
    <a href="https://scholar.google.com.hk/citations?user=1xA5KxAAAAAJ&hl=en&oi=ao" target="_blank">Dongyang Jin</a><sup>1</sup>,
    <a href="https://scholar.google.com.hk/citations?hl=en&user=MDrO_twAAAAJ" target="_blank">Ryan Xu</a><sup>1</sup>,
    <a href="https://scholar.google.com.hk/citations?hl=en&user=sshKuUMAAAAJ" target="_blank">Nannan Zhang</a><sup>3,#</sup>,
    <a href="https://scholar.google.com.hk/citations?user=G-mHRrEAAAAJ&hl=en&oi=ao" target="_blank">Dan Song</a><sup>2</sup>,
    <a href="https://cxxgtxy.github.io/" target="_blank">Xiangxiang Chu</a><sup>1</sup>
</div>

<div>
    <sup>1</sup>Amap, Alibaba Group &emsp; <sup>2</sup>Tianjin University
</div>
<div>
    <sup>3</sup>Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences
</div>
</div>



</br>

[![Arxiv](https://img.shields.io/badge/arXiv-2507.19946-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2511.18957)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue)](https://huggingface.co/JianhaoZeng/Eevee)

## Abstract

> Video virtual try-on technology provides a cost-effective solution for creating marketing videos in fashion e-commerce. However, its practical adoption is hindered by two critical limitations. First, the reliance on a single garment image as input in current virtual try-on datasets limits the accurate capture of realistic texture details. Second, most existing methods focus solely on generating full-shot virtual try-on videos, neglecting the business's demand for videos that also provide detailed close-ups. To address these challenges, we introduce a high-resolution dataset for video-based virtual try-on. This dataset offers two key features. First, it provides more detailed information on the garments, which includes high-fidelity images with detailed close-ups and textual descriptions; Second, it uniquely includes full-shot and close-up try-on videos of real human models. Furthermore, accurately assessing consistency becomes significantly more critical for the close-up videos, which demand high-fidelity preservation of garment details. To facilitate such fine-grained evaluation, we propose a new garment consistency metric VGID (Video Garment Inception Distance) that quantifies the preservation of both texture and structure. Our experiments validate these contributions. We demonstrate that by utilizing the detailed images from our dataset, existing video generation models can extract and incorporate texture features, significantly enhancing the realism and detail fidelity of virtual try-on results. Furthermore, we conduct a comprehensive benchmark of recent models. The benchmark effectively identifies the texture and structural preservation problems among current methods.

## Dataset Access

1. Sets the environment variable to point to a mirror site for faster and more stable Hugging Face connections.
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

2. Downloads the snapshot from Huggingface and saves it to the local data directory.
```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="JianhaoZeng/Eevee",
    local_dir="./data",
    repo_type="model"
)
```

3. Merges the split multi-part files into a single zip archive and extracts the contents.
```bash
cd ./data
cat Eevee.zip.part* > Eevee.zip
unzip Eevee.zip -d ./Eevee
```

##  Data Organization

As illustrated in ./Eevee, the following data should be provided.

```
|-- dresses
|   |-- 00030
|   |   |-- garment_caption.txt
|   |   |-- garment_detail.png
|   |   |-- garment_line.png
|   |   |-- garment_mask.png
|   |   |-- garment.png
|   |   |-- person_agnostic.png
|   |   |-- person_mask.png
|   |   |-- person.png
|   |   |-- video_0_agnostic_sam.mp4
|   |   |-- video_0_agnostic.mp4
|   |   |-- video_0_densepose.mp4
|   |   |-- video_0_mask.mp4
|   |   |-- video_0.mp4
|   |   |-- video_1_agnostic_sam.mp4
|   |   |-- video_1_agnostic.mp4
|   |   |-- video_1_densepose.mp4
|   |   |-- video_1_mask.mp4
|   |   |-- video_1.mp4
|   |-- 00032
|   ...
|-- lower_body
|   |-- 00003
|   |-- 00006
|   ...
|-- upper_bdoy
|   |-- 00000
|   |-- 00001
|   ...
|   dresses_test.csv
|   dresses_train.csv
|   lower_test.csv
|   lower_train.csv
|   upper_test.csv
|   upper_train.csv
```


<table>
  <thead>
    <tr>
      <th>File Name</th>
      <th>Source</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="3"><strong>--- Garment Data ---</strong></td>
    </tr>
    <tr>
      <td>garment.png</td><td>Raw data</td>
      <td>In-shop garment image</td>
    </tr>
    <tr>
      <td>garment_detail.png</td><td>Raw data</td>
      <td>Dataied garment image</td>
    </tr>
    <tr>
      <td>garment_caption.txt</td><td>Qwen-VL-MAX</td>
      <td>Detailed text description of garment image generated by Qwen-vl-max</td>
    </tr>
    <tr>
        <td>garment_line.png </td><td>AniLines</td>
        <td>Lineart of garment image generated by AniLines</td>
    </tr>
    <tr>
        <td>garment_mask.png</td><td>Grounded SAM-2</td>
        <td>Binary mask of garment image generated by Grounded SAM-2</td>
    <tr>
      <td colspan="3"><strong>--- Person Data ---</strong></td>
    </tr>
    </tr>
        <td>person.png</td><td>Raw data</td>
        <td>Image of a person wearing the corresponding garment</td>
    </tr>
    </tr>
        <td>person_mask.png</td><td>Grounded SAM-2</td>
        <td>Binary mask of the garment area on the person image generated by Grounded SAM-2</td>
    </tr>
    </tr>
        <td>person_agnostic.png</td><td>Multiplication</td>
        <td>Person image with garment area masked out generated by pixel-wise multiplication</td>
    </tr>
    <tr>
      <td colspan="3"><strong>--- Full-shot person video Data ---</strong></td>
    </tr>
    </tr>
        <td>video_0.mp4</td><td>Raw data</td>
        <td>Full-shot person video</td>
    </tr>
    </tr>
        <td>video_0_mask.mp4</td><td>OpenPose</td>
        <td>Binary mask of the garment area on the full-shot person video generated by OpenPose</td>
    </tr>
    </tr>
        <td>video_0_agnostic.mp4</td><td>Multiplication</td>
        <td>Full-shot person video with garment area masked out generated by pixel-wise multiplication</td>
    </tr>
    </tr>
        <td>video_0_agnostic_sam.mp4</td><td>Grounded SAM-2</td>
        <td>Full-shot person video with garment area masked out generated by Grounded SAM-2</td>
    </tr>
    </tr>
        <td>video_0_densepose.mp4</td><td>Detectron2</td>
        <td>DensePose UV coordinates for the human body of full-shot person video generated by Detectron2</td>
    </tr>
    <tr>
      <td colspan="3"><strong>--- Close-up person video Data ---</strong></td>
    </tr>
    </tr>
        <td>video_1.mp4 </td><td>Raw data</td>
        <td>Close-up person video</td>
    </tr>
    </tr>
        <td>video_1_mask.mp4</td><td> Grounded SAM-2</td>
        <td>Binary mask of the garment area on the Close-up person video generated by Grounded SAM-2</td>
    </tr>
    </tr>
        <td>video_1_agnostic.mp4</td><td>Multiplication</td>
        <td>Close-up person video with garment area masked out generated by pixel-wise multiplication</td>
    </tr>
    </tr>
        <td>video_1_agnostic_sam.mp4</td><td>Grounded SAM-2</td>
        <td>Close-up person video with garment area masked out generated by Grounded SAM-2</td>
    </tr>
    </tr>
        <td>video_1_densepose.mp4</td><td>Detectron2</td>
        <td>DensePose UV coordinates for the human body of close-up person video generated by Detectron2</td>
    </tr>
  </tbody>
</table>



## Data Processing 




## Contact

If you have any questions, please reach out via email at jh_zeng@tju.edu.cn

## Citation

If you find this work useful for your research, please cite our paper:

```
@article{zeng2025eevee,
  title={Eevee: Towards Close-up High-resolution Video-based Virtual Try-on},
  author={Zeng, Jianhao and Bai, Yancheng and Chen, Ruidong and Zhang, Xuanpu and Sun, Lei and Jin, Dongyang and Xu, Ryan and Zhang, Nannan and Song, Dan and Chu, Xiangxiang},
  journal={arXiv preprint arXiv:2511.18957},
  year={2025}
}
```
