 
import argparse

 


def parse_args():
    parser = argparse.ArgumentParser(description="Google Cloud Platform.")

    parser.add_argument( "--project_id", type=str, default=None,  required=True, help="Project ID")
    parser.add_argument( "--location", type=str,  default='us-central1', help="Location" )
    parser.add_argument( "--bucket_uri", type=str,  default=None, help="Bucket URI" )

    args = parser.parse_args()

    return args



def main(args):
    print(args)
    PROJECT_ID = args.project_id
    LOCATION = args.location
    BUCKET_URI=args.bucket_uri

 
    
if __name__ == "__main__":
    args = parse_args()
    main(args) 
