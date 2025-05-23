def download_data(data_url=None, local_file=None):
    os.makedirs("data", exist_ok=True)
    response = requests.get(data_url)
    pdf_path = "data/sample.pdf"

    with open(pdf_path, "wb") as f:
        f.write(response.content)
    print("Downloaded and saved into data directory")
