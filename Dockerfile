FROM tensorflow/tensorflow:1.7.0

#   FROM tensorflow/tensorflow:1.7.0-py3
#   libosmesa6-dev is for osmesa headless 3d rendering 

RUN apt-get update && \
    apt-get install -y python3-tk git wget unzip tar && \
    apt-get install -y libsm6 && \
    apt-get install -y gtk3.0 &&    \
    apt-get install -y python-opengl libosmesa6-dev
    
#     apt-get install -y libgl1-mesa-glx
#     apt-get install libOSMesa
#     apt-get install -y libglfw3-dev
#     COPY requirements.txt /tmp/requirements.txt
#     RUN pip3 install -r /tmp/requirements.txt
# RUN pip3 install pycocotools

WORKDIR "/build"
# RUN git clone https://github.com/chip1967/Mask_RCNN.git
#WORKDIR "/build/Mask_RNN"
# RUN perl setup.py install
# PYOPENGL_PLATFORM=osmesa

