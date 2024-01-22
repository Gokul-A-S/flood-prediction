FROM python:3.9
ADD main.py /
RUN pip install tensorflow==2.14.0 pandas numpy
CMD [ "python", "main.py" ]