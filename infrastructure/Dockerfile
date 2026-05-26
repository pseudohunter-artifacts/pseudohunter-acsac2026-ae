FROM python:3.11-slim

WORKDIR /artifact

COPY . /artifact

RUN python -m pip install --upgrade pip \
    && python -m pip install -e "artifact/.[dev,metrics]"

CMD ["bash", "claims/claim4_smoke_pipeline/run.sh"]

