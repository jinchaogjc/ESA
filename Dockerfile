FROM pytorch/pytorch:latest

RUN apt-get update -y --allow-unauthenticated 
RUN apt-get upgrade -y --allow-unauthenticated
RUN apt-get install gcc -y --allow-unauthenticated
RUN apt-get install zip unzip -y --allow-unauthenticated
RUN apt-get install git -y --allow-unauthenticated
RUN apt-get install vim -y --allow-unauthenticated
RUN apt-get install ffmpeg libsm6 libxext6 -y --allow-unauthenticated
# RUN pip install wandb
# RUN pip install showdata
# RUN pip install tqdm


COPY requirements.txt /code/requirements.txt
WORKDIR /code

RUN conda install -c anaconda python=3.8 -y
RUN conda update --all -y
RUN conda install -y ipython pip
RUN pip install -f https://download.pytorch.org/whl/cu110/torch_stable.html torch==1.7.1+cu110 torchvision==0.8.2
RUN pip install -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html mmcv_full==1.3.10
RUN pip install -r requirements.txt
# RUN wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
# RUN ln -s resnet101-5d3b4d8f.pth /root/.cache/torch/hub/checkpoints/resnet101-5d3b4d8f.pth 