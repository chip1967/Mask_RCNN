FROM tensorflow/tensorflow:1.7.0-py3

RUN apt-get update && \
    apt-get install -y python3-tk git wget unzip tar && \
    apt-get install -y libsm6 && \ 
    apt-get install -y gtk3.0


COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
# RUN pip3 install pycocotools

WORKDIR "/build"
# RUN git clone https://github.com/chip1967/Mask_RCNN.git
#WORKDIR "/build/Mask_RNN"
# RUN perl setup.py install

