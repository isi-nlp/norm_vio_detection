FROM python:3.9
RUN pip install --upgrade pip
RUN mkdir -p /isi_darma/norm_vio_detect_api/
ADD evaluator.py dataset.py models.py requirements.txt api-inference.py api-test-data.json /isi_darma/norm_vio_detect_api/
RUN mkdir -p /isi_darma/norm_vio_detect_api/ckps/BERTRNN/
ADD ckps/BERTRNN/model_1.pt /isi_darma/norm_vio_detect_api/ckps/BERTRNN/
RUN mkdir -p /isi_darma/norm_vio_detect_api/results/BERTRNN/1/
ADD results/BERTRNN/1/config.json /isi_darma/norm_vio_detect_api/results/BERTRNN/1/
RUN cd /isi_darma/norm_vio_detect_api/ && pip install -r requirements.txt
WORKDIR /isi_darma/norm_vio_detect_api/
CMD ["python", "api-inference.py"]
