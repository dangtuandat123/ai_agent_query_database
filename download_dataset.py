import kagglehub


def main() -> None:
    # Download latest version
    path = kagglehub.dataset_download("neilclack/nyc-taxi-trip-data-google-public-data")
    print("Path to dataset files:", path)


if __name__ == "__main__":
    main()
