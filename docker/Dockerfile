FROM c-adas-dldi-baseimages-docker-v.eu.artifactory.conti.de/nvidia/tensorflow:21.12-tf2-py3-continental.v1

RUN apt-get update && apt-get install -y python3-opencv dos2unix
RUN mkdir /app && chgrp users /app

# Install project requirements
COPY --chown=root:users requirements.txt /app/requirements.txt
COPY --chown=root:users requirements_no_deps.txt /app/requirements_no_deps.txt
RUN pip install --no-input -r /app/requirements.txt && \
	pip install --no-input -r /app/requirements_no_deps.txt --no-deps

# Install project libs & scripts
# COPY --chown=root:users docker /app/docker
COPY --chown=root:users docker/entrypoint /entrypoint
RUN dos2unix /entrypoint
# Your existing Dockerfile commands

# Copy the 'dev.json' file into the container
#COPY docker/dev.json /app/docker/dev.json
#COPY --chown=root:users docker/dev.json /app/dev.json

COPY --chown=root:users shift /app/shift
WORKDIR /app

RUN chmod o+rx /entrypoint
RUN chmod -R o+rx /app

ENV PYTHONPATH /app:$PYTHONPATH

CMD ["/entrypoint"]