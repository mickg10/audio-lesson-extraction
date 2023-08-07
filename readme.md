Setup ⚙️
Tested for PyTorch 2.0, Python 3.10 (use other versions at your own risk!)
GPU execution requires the NVIDIA libraries cuBLAS 11.x and cuDNN 8.x to be installed on the system. Please refer to the CTranslate2 documentation.

1. Create Python3.10 environment
```
conda create --name whisperx2 python=3.10
conda activate whisperx2
```

2. Install PyTorch2.0, e.g. for Linux and Windows CUDA11.8:
```
conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia ffmpeg
```


3. Install pip dependecies
```
pip install -r requirements.txt
```


4. Run like
```
python decode_whisper.py  --outdir /mnt/c/s/mickg/audio_lessons `cat youtube_links.txt`
```
