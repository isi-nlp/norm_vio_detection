FROM python:3.9
RUN pip install --upgrade pip
RUN mkdir -p /isi_darma/norm_vio_detect_api/
RUN mkdir -p /isi_darma/norm_vio_detect_api/results/clf/bert-base-uncased/1/seed=2022
RUN mkdir -p /isi_darma/norm_vio_detect_api/results/prompt/t5-base/1/seed=2022
RUN mkdir -p /isi_darma/norm_vio_detect_api/ckps/clf/bert-base-uncased/1/seed=2022
RUN mkdir -p /isi_darma/norm_vio_detect_api/ckps/prompt/t5-base/1/seed=2022
ADD api-test-data.json api-inference.py dataset.py dataset_inference.py evaluator.py evaluator_prompt.py models.py requirements.txt /isi_darma/norm_vio_detect_api/
ADD ckps/clf/bert-base-uncased/1/seed=2022/model.pt /isi_darma/norm_vio_detect_api/ckps/clf/bert-base-uncased/1/seed=2022/model.pt
ADD ckps/prompt/t5-base/1/seed=2022/model.pt /isi_darma/norm_vio_detect_api/ckps/prompt/t5-base/1/seed=2022/model.pt
ADD results/clf/bert-base-uncased/1/seed=2022/config.json /isi_darma/norm_vio_detect_api/results/clf/bert-base-uncased/1/seed=2022/config.json
ADD results/prompt/t5-base/1/seed=2022/config.json /isi_darma/norm_vio_detect_api/results/prompt/t5-base/1/seed=2022/config.json
RUN cd /isi_darma/norm_vio_detect_api/ && pip install -r requirements.txt
WORKDIR /isi_darma/norm_vio_detect_api/
CMD ["python", "api-inference.py", "--task=prompt", "--model_name=t5-base"]
