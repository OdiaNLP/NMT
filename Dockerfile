FROM python:3.8-slim as build_base

RUN apt-get update \
    && apt-get upgrade --yes

ENV DEBIAN_FRONTEND noninteractive

RUN apt install --yes \
    apt-utils lsb-release ncompress \
    xz-utils net-tools apt-utils \
    autoconf automake build-essential

ENV USERNAME shua

RUN /usr/sbin/groupadd -g 10010 $USERNAME \
    && /usr/sbin/useradd -u 10010 -g 10010 --create-home --home /home/$USERNAME $USERNAME \
    && chown -R $USERNAME.$USERNAME /home/$USERNAME \
    && chsh -s /bin/bash $USERNAME

WORKDIR /home/$USERNAME

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix /home/$USERNAME/.local -r requirements.txt

FROM python:3.8-slim

ENV USERNAME shua

RUN /usr/sbin/groupadd -g 10010 $USERNAME \
    && /usr/sbin/useradd -u 10010 -g 10010 --create-home --home /home/$USERNAME $USERNAME \
    && chown -R $USERNAME.$USERNAME /home/$USERNAME \
    && chsh -s /bin/bash $USERNAME

WORKDIR /home/$USERNAME

USER $USERNAME

COPY --from=build_base  /home/$USERNAME/.local /home/$USERNAME/.local
COPY nmt ./nmt/
COPY run.sh .

ENV PATH=/home/$USERNAME/.local/bin:$PATH

EXPOSE 31137

ENTRYPOINT ["./run.sh"]
