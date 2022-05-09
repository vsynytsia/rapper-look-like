FROM python:3.8

RUN mkdir -p /root/rapper-face-similarity/
WORKDIR /root/rapper-face-similarity/

COPY . /root/rapper-face-similarity/
RUN pip install -r requirements.txt

CMD ["python", "app.py"]