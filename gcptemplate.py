import argparse
from google.cloud import aiplatform
 


def parse_args():
    parser = argparse.ArgumentParser(description="Google Cloud Platform.")

    parser.add_argument( "--project_id", type=str, default=None,  required=True, help="Project ID")
    parser.add_argument( "--location", type=str,  default='us-central1', help="Location" )
    parser.add_argument( "--bucket_uri", type=str,  default=None, help="Bucket URI" )
    parser.add_argument( "--script_path", type=str,  default=None, help="Script Path" )
    parser.add_argument( "--job_name", type=str,  default='flower-sdk-job', help="Custom Job Name" )

    args = parser.parse_args()

    return args



def main(args):
    
    PROJECT_ID = args.project_id
    LOCATION = args.location
    BUCKET_URI=args.bucket_uri
    SCRIPT_PATH = args.script_path
    JOB_NAME = args.job_name

    # Model Training
    aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)
    
    job = aiplatform.CustomJob.from_local_script(
        display_name=JOB_NAME,
        script_path=SCRIPT_PATH,
        container_uri= 'us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-9.py310:latest',
        enable_autolog=False,
    )

    job.run()

    
    

    # Upload Model to Registry"""

    # my_model = aiplatform.Model.upload(display_name='flower-model',
    #                                   artifact_uri='gs://{PROJECT_ID}-bucket/model_output',
    #                                   serving_container_image_uri='us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-6:latest')

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
