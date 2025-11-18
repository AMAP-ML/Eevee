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


## Abstract

> Video virtual try-on technology provides a cost-effective solution for creating marketing videos in fashion e-commerce. However, its practical adoption is hindered by two critical limitations. First, the reliance on a single garment image as input in current virtual try-on datasets limits the accurate capture of realistic texture details. Second, most existing methods focus solely on generating full-shot virtual try-on videos, neglecting the business's demand for videos that also provide detailed close-ups. To address these challenges, we introduce a high-resolution dataset for video-based virtual try-on. This dataset offers two key features. First, it provides more detailed information on the garments, which includes high-fidelity images with detailed close-ups and textual descriptions; Second, it uniquely includes full-shot and close-up try-on videos of real human models. Furthermore, accurately assessing consistency becomes significantly more critical for the close-up videos, which demand high-fidelity preservation of garment details. To facilitate such fine-grained evaluation, we propose a new garment consistency metric VGID (Video Garment Inception Distance) that quantifies the preservation of both texture and structure. Our experiments validate these contributions. We demonstrate that by utilizing the detailed images from our dataset, existing video generation models can extract and incorporate texture features, significantly enhancing the realism and detail fidelity of virtual try-on results. Furthermore, we conduct a comprehensive benchmark of recent models. The benchmark effectively identifies the texture and structural preservation problems among current methods.
