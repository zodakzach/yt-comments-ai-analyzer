# YouTube Comments AI Analyzer FastAPI App

## Overview

This project is a FastAPI application that summarizes YouTube comments using AI models. It fetches comments from YouTube, summarizes them, vectorizes the comments for similarity search, and allows users to query for specific information.

## Project Structure

```
Thread_summarizer/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── openai_client.py
│   │   └── redis_client.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── fetch_comments.py
│   │   ├── summarize.py
│   │   ├── vectorize.py
│   │   └── search.py
│   └── models/
│       ├── __init__.py
│       └── schemas.py
├── tests/
│   └── test_routes.py
├── requirements.txt
├── .env
├── .env.example
├── README.md
├── package.json
```

## Setup Instructions

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd Thread_summarizer
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   - Copy `.env` or `.env.example` and fill in the required variables:
     - `YOUTUBE_API_KEY`
     - `THREAD_OPENAI_API_KEY`
     - `UPSTASH_REDIS_REST_URL`
     - `UPSTASH_REDIS_REST_TOKEN`

5. **Run the application**

   ```bash
   uvicorn app.main:app --reload
   ```

6. **Run tests**
   ```bash
   pytest
   ```

## Usage

- The API provides endpoints for:
  - `/summarize/` - Summarize YouTube comments and start a session.
  - `/question/` - Ask a question about the summarized comments using your session.
  - `/health` - Health check endpoint.

## Testing

- Tests are located in the `tests/` directory.
- Use `pytest` to run the test suite.
- Example test file: `tests/test_routes.py` includes basic endpoint tests.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
