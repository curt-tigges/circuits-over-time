FROM pytorch/pytorch:latest

# Set the DEBIAN_FRONTEND variable to noninteractive to suppress prompts
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    graphviz \
    libgraphviz-dev \
    pkg-config \
    tzdata  # Include tzdata in the same RUN command

# Reset the DEBIAN_FRONTEND variable to its default value
ENV DEBIAN_FRONTEND=

# Install GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install -y gh

# Install Python packages
RUN pip install plotly einops protobuf==3.20.* jaxtyping==0.2.13 torchtyping jupyterlab scikit-learn ipywidgets matplotlib kaleido openai \
    transformer_lens \
    typeguard==2.13.3 \
    circuitsvis \
    imgkit \
    dill==0.3.4 \
    jupyter ipykernel pytest pytest-doctestplus nbval pytest-cov \
    git+https://github.com/neelnanda-io/neel-plotly.git \
    --upgrade jax jaxlib \
    pygraphviz

ENTRYPOINT ["/bin/bash", "-c", "tail -f /dev/null"]
