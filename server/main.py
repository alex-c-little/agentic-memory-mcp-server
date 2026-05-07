import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("server.app:combined_app", host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
