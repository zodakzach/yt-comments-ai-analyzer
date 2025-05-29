# YouTube Comments AI Analyzer FastAPI App

## Overview

This project is a FastAPI application that summarizes YouTube comments using AI models. It fetches comments from YouTube, summarizes them, vectorizes the comments for similarity search, and allows users to query for specific information.  
**All main endpoints are HTMX endpoints and return rendered HTML templates, not raw JSON.**

## Project Structure

```
yt-comments-ai-analyzer/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logging.py
│   │   ├── openai_client.py
│   │   └── redis_client.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   └── services/
│       ├── __init__.py
│       ├── errors.py
│       ├── fetch_comments.py
│       ├── qa.py
│       ├── session.py
│       ├── summarize.py
│       └── vectorize.py
├── mvp/
│   └── mvp.ipynb
├── static/
│   ├── css/
│   │   ├── input.tailwind.css
│   │   └── tailwind.css
│   └── js/
│       ├── bundle.js
│       ├── bundle.js.map
│       └── index.js
├── templates/
│   ├── base.html
│   ├── chat.html
│   ├── index.html
│   └── summary.html
├── tests/
│   └── test_routes.py
├── .env
├── .env.example
├── package.json
├── package-lock.json
├── postcss.config.mjs
├── pyproject.toml
├── README.md
├── rollup.config.mjs
├── uv.lock
```

## Setup Instructions

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd yt-comments-ai-analyzer
   ```

2. **Create a virtual environment and install dependencies with [uv](https://github.com/astral-sh/uv)**

   ```bash
   uv venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   uv pip install -r pyproject.toml
   ```

3. **Set up environment variables**

   - Copy `.env.example` to `.env` and fill in the required variables:
     - `YOUTUBE_API_KEY`
     - `THREAD_OPENAI_API_KEY`
     - `UPSTASH_REDIS_REST_URL`
     - `UPSTASH_REDIS_REST_TOKEN`

4. **Run the application**

   ```bash
   uvicorn app.main:app --reload
   ```

5. **Run tests**

   ```bash
   uv pip install pytest  # if not already installed
   pytest
   ```

## Usage

- The API provides HTMX endpoints that return HTML templates:
  - `/summarize/` - Summarize YouTube comments and start a session (returns a summary template).
  - `/question/` - Ask a question about the summarized comments using your session (returns an answer template).
  - `/health` - Health check endpoint
  
## Testing

- Tests are located in the `tests/` directory.
- Use `pytest` to run the test suite.
- Currently no tests lol

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
