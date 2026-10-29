# Vercel Container Runtime image (see docs/DEPLOY_VERCEL.md). Runs the Studio dashboard behind its
# bearer-token gate; API keys come from Vercel project environment variables, never the image.
FROM python:3.12-slim

# opencv-python links against libGL and glib even when no window is opened.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY courtvision ./courtvision
RUN pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1 PORT=80
# Function instances have no persistent disk: footage and analyses live in /tmp for the instance's life.
# Studio refuses to listen beyond localhost without a token, so an unset COURTVISION_TOKEN fails closed.
# Requests arrive with the deployment's *.vercel.app Host; Vercel only routes this project's hosts here.
CMD ["sh", "-c", "exec python -m courtvision studio --host 0.0.0.0 --port \"$PORT\" --data-dir /tmp/courtvision --auth-token \"$COURTVISION_TOKEN\" --set 'server.allowed_hosts=[\"*.vercel.app\"]'"]
