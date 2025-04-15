 
import argparse

 


def parse_args():
    parser = argparse.ArgumentParser(description="Google Cloud Platform.")
    parser.add_argument(
        "--project_id",
        type=str,
        default=None,
        required=True,
        help="Project ID",
    )

    args = parser.parse_args()

    return args



def main(args):
    print(args.project_id)

 
    
if __name__ == "__main__":
    args = parse_args()
    main(args) 
