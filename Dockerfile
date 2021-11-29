# set base image (host OS)
FROM python:3.8

# set the working directory in the container
WORKDIR /workdir

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY nlpService.py .

EXPOSE 5000

# command to run on container start
CMD [ "python", "nlpService.py" ]