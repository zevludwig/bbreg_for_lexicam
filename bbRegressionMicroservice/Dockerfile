FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

WORKDIR /app

COPY requirements.txt /app/

RUN pip3 install --trusted-host pypi.python.org -r requirements.txt
RUN pip3 install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html


COPY . /app

COPY testfiles /app/testfiles

ENV PORT 5005

EXPOSE 5005





