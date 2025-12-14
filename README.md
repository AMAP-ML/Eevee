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

[![Paper](https://img.shields.io/badge/arXiv-2507.19946-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2511.18957)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue)](https://huggingface.co/JianhaoZeng/Eevee)

</br>


## Abstract

> Video virtual try-on technology provides a cost-effective solution for creating marketing videos in fashion e-commerce. However, its practical adoption is hindered by two critical limitations. First, the reliance on a single garment image as input in current virtual try-on datasets limits the accurate capture of realistic texture details. Second, most existing methods focus solely on generating full-shot virtual try-on videos, neglecting the business's demand for videos that also provide detailed close-ups. To address these challenges, we introduce a high-resolution dataset for video-based virtual try-on. This dataset offers two key features. First, it provides more detailed information on the garments, which includes high-fidelity images with detailed close-ups and textual descriptions; Second, it uniquely includes full-shot and close-up try-on videos of real human models. Furthermore, accurately assessing consistency becomes significantly more critical for the close-up videos, which demand high-fidelity preservation of garment details. To facilitate such fine-grained evaluation, we propose a new garment consistency metric VGID (Video Garment Inception Distance) that quantifies the preservation of both texture and structure. Our experiments validate these contributions. We demonstrate that by utilizing the detailed images from our dataset, existing video generation models can extract and incorporate texture features, significantly enhancing the realism and detail fidelity of virtual try-on results. Furthermore, we conduct a comprehensive benchmark of recent models. The benchmark effectively identifies the texture and structural preservation problems among current methods.

## Dataset Access

You can easily load the dataset with one line of code:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

```python
from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('JianhaoZeng/Eevee', cache_dir='./data')
```

```bash
cd ./data
cat Eevee.zip.part* > Eevee.zip
```

## Citation
```
@article{zeng2025eevee,
  title={Eevee: Towards Close-up High-resolution Video-based Virtual Try-on},
  author={Zeng, Jianhao and Bai, Yancheng and Chen, Ruidong and Zhang, Xuanpu and Sun, Lei and Jin, Dongyang and Xu, Ryan and Zhang, Nannan and Song, Dan and Chu, Xiangxiang},
  journal={arXiv preprint arXiv:2511.18957},
  year={2025}
}
```
