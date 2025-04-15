import argparse
from google.cloud import aiplatform
 


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

    # Model Training
    aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)
    print(aiplatform)
    # my_job = aiplatform.CustomContainerTrainingJob(
    #     display_name = 'flower-sdk-job',
    #     container_uri = 'us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-9.py310:latest',
    #     staging_bucket = args.bucket_uri
    # )

    # my_job.run(
    #     replica_count=1,
    #     machine_type='n1-standard-8',
    #     accelerator_type='NVIDIA_TESLA_V100',
    #     accelerator_count=1
    # )

    # Upload Model to Registry"""

    # my_model = aiplatform.Model.upload(display_name='flower-model',
    #                                   artifact_uri='gs://{PROJECT_ID}-bucket/model_output',
    #                                   serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-14:latest')

    # Deploy Model to Endpoint"""

    # MODEL_ID = '8090889353896656896'
    # PROJECT_NUMBER = '866955933784'

    # my_model = aiplatform.Model("projects/{PROJECT_NUMBER}/locations/us-central1/models/{MODEL_ID}")

    # endpoint = my_model.deploy(
    #      deployed_model_display_name='my-endpoint',
    #      traffic_split={"0": 100},
    #      machine_type="n1-standard-4",
    #      accelerator_count=0,
    #      min_replica_count=1,
    #      max_replica_count=1,
    #    )

 
    
if __name__ == "__main__":
    args = parse_args()
    main(args) 
