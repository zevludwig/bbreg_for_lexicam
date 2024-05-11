FROM python:3.9

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip3 install --no-cache-dir --upgrade --trusted-host pypi.python.org -r requirements.txt
RUN pip3 install torch==2.1.2+cpu torchvision==0.16.2+cpu -f https://download.pytorch.org/whl/torch_stable.html


COPY main.py /app/main.py
COPY detect_test.txt /app/detect_test.txt
COPY curlformat.txt /app/curlformat.txt
COPY out_dyn.onnx /app/out_dyn.onnx

COPY testfiles /app/testfiles

ENV PORT 5005

EXPOSE 5005

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5005"]



