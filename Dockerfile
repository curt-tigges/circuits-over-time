# Use the FROM instruction to pull other images to base your new one on
FROM pytorch/pytorch:latest

# Use the RUN instruction to make the image do a terminal command like behavior
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    graphviz \
    libgraphviz-dev \
    pkg-config \
    libgl1-mesa-glx

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
    imgkit \
    dill==0.3.4 \
    jupyterlab ipykernel pytest pytest-doctestplus nbval pytest-cov \
    git+https://github.com/neelnanda-io/neel-plotly.git \
    --upgrade jax jaxlib \
    pygraphviz \
    cmapy

# Use EXPOSE to instruct the image to expose ports as needed
EXPOSE 8888


# The main purpose of a CMD is to provide defaults for an executing container
# This CMD opens the jupyter notebook when you run the image
CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --ip 0.0.0.0 --no-browser --allow-root"]
