import os

import uvicorn

from codedebug_env.server.app import app


def main() -> None:
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "7860")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
