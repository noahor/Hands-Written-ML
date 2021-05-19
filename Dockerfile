FROM python:3.6
WORKDIR /app
COPY requirements.txt /app
COPY 4.png /app
COPY HandsWritten_CheckYourImage.py /app
RUN pip install -r ./requirements.txt
CMD ["python", "HandsWritten_CheckYourImage.py"]~
